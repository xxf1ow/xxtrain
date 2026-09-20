import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from xxtrain.business_tasks.definition import AnnotationPolicy, StepDefinition, TaskDefinition
from xxtrain.platform.config import WorkspaceConfig
from xxtrain.platform.contracts import PlatformConflictError
from xxtrain.platform.runtime import RuntimeCache
from xxtrain.platform.service import AnnotationService
from xxtrain.platform.training_contracts import ExecutionView, TrainingRun
from xxtrain.platform.training_service import TrainingService
from xxtrain.platform.training_store import TrainingRunStore
from xxtrain.workspace_data import WorkspaceData
from xxtrain.workspace_data.inputs import AxisAlignedRectangleInputs, OriginalImageInputs


class _ExecutionFacts:
    def __init__(self) -> None:
        self.views: dict[str, ExecutionView] = {}
        self.observe_calls: list[str] = []

    def observe(self, task_id: str) -> ExecutionView:
        self.observe_calls.append(task_id)
        return self.views[task_id]


def dependency_task() -> TaskDefinition:
    rectangles = AxisAlignedRectangleInputs()
    return TaskDefinition(
        (
            StepDefinition(
                'regions',
                frozenset({'rectangle'}),
                frozenset({'part'}),
                frozenset(),
                frozenset(),
                annotation=AnnotationPolicy('rectangle', 'STANDARD'),
                input_adapter=OriginalImageInputs(),
            ),
            StepDefinition(
                'kind',
                frozenset({'classification'}),
                frozenset({'a', 'b'}),
                frozenset({'regions'}),
                frozenset(),
                annotation=AnnotationPolicy('tag', 'TAGS', maximum_annotations=1),
                input_adapter=rectangles,
            ),
            StepDefinition(
                'needles',
                frozenset({'polyline'}),
                frozenset({'line'}),
                frozenset({'regions'}),
                frozenset(),
                annotation=AnnotationPolicy('polyline', 'STANDARD', point_count=2),
                input_adapter=rectangles,
            ),
        ),
        key='dependency-lock-test',
        display_name='Dependency lock test',
    )


class TrainingDependencyLocksTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix='xxtrain-training-locks-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        workspace = self.root / 'workspace'
        (workspace / 'images').mkdir(parents=True)
        self.config = WorkspaceConfig(
            'line-1',
            'Line 1',
            17,
            workspace,
            self.root / 'runtime',
            'http://cvat.test',
            'test.test_training_dependency_locks:dependency_task',
        )
        data = WorkspaceData(workspace, dependency_task())
        self.annotations = AnnotationService(self.config, data, object(), RuntimeCache(self.config.runtime_dir))
        self.store = TrainingRunStore(self.root / 'training-runs.db')
        self.execution = _ExecutionFacts()
        self.training = TrainingService(
            self.config, self.annotations, self.store, self.execution, self.config.runtime_dir / 'cache'
        )

    def add_run(
        self,
        target: str,
        *,
        user_id: int = 17,
        desired_action: str | None = 'execute',
        status: str | None = 'running',
        task_entry: str | None = None,
    ) -> TrainingRun:
        task_id = None if status == 'pending' else f'task-{uuid4()}'
        run = TrainingRun(
            str(uuid4()),
            user_id,
            self.config.workspace_id,
            self.config.display_name,
            target,
            uuid4().hex,
            f'cache/{uuid4().hex}/{target}',
            '2026-09-19T00:00:00+00:00',
            None,
            task_id,
            desired_action,
            task_entry or self.config.task_entry,
        )
        self.store.create(run)
        if task_id is not None:
            active = status in {'queued', 'running', 'unknown'}
            self.execution.views[task_id] = ExecutionView(task_id, status, active, None, None, None, None, False, None)
        return run

    def test_active_run_locks_only_target_and_ancestors_and_projects_union_once(self) -> None:
        self.add_run('needles')

        self.training.require_editable(self.config.workspace_id, 'kind')
        with self.assertRaises(PlatformConflictError):
            self.training.require_editable(self.config.workspace_id, 'regions')
        with self.assertRaises(PlatformConflictError):
            self.training.require_editable(self.config.workspace_id, 'needles')
        with self.assertRaises(PlatformConflictError):
            self.training.require_editable(self.config.workspace_id, None)

        before_observations = len(self.execution.observe_calls)
        view = self.training.workspace_view(self.config.owner_user_id)
        self.assertFalse(view['can_upload'])
        self.assertNotIn('editable', view)
        self.assertEqual({'regions': False, 'kind': True, 'needles': False}, view['target_editable'])
        self.assertEqual(1, len(self.execution.observe_calls) - before_observations)

    def test_active_states_and_cancel_request_remain_locked_but_terminal_states_release(self) -> None:
        for status in ('pending', 'queued', 'running', 'unknown'):
            with self.subTest(status=status):
                self.store = TrainingRunStore(self.root / f'{status}.db')
                self.training.store = self.store
                self.add_run('kind', status=status)
                with self.assertRaises(PlatformConflictError):
                    self.training.require_editable(self.config.workspace_id, 'regions')
                with self.assertRaises(PlatformConflictError):
                    self.training.require_editable(self.config.workspace_id, 'kind')
                self.training.require_editable(self.config.workspace_id, 'needles')

        for status in ('completed', 'failed', 'cancelled'):
            with self.subTest(status=status):
                self.store = TrainingRunStore(self.root / f'{status}.db')
                self.training.store = self.store
                self.add_run('kind', status=status)
                self.training.require_editable(self.config.workspace_id, None)
                for target in ('regions', 'kind', 'needles'):
                    self.training.require_editable(self.config.workspace_id, target)

        self.store = TrainingRunStore(self.root / 'cancel-request.db')
        self.training.store = self.store
        self.add_run('kind', desired_action='cancel', status='running')
        with self.assertRaises(PlatformConflictError):
            self.training.require_editable(self.config.workspace_id, 'kind')

    def test_union_includes_every_workspace_user_and_unknown_target_is_conservative(self) -> None:
        self.add_run('kind', user_id=18)
        self.add_run('needles', user_id=19)

        for target in ('regions', 'kind', 'needles'):
            with self.assertRaises(PlatformConflictError):
                self.training.require_editable(self.config.workspace_id, target)

        store = TrainingRunStore(self.root / 'unknown.db')
        self.training.store = store
        self.store = store
        self.add_run('retired-step')
        with self.assertRaises(PlatformConflictError):
            self.training.require_editable(self.config.workspace_id, 'needles')
        with self.assertRaises(PlatformConflictError):
            self.training.require_editable(self.config.workspace_id, None)

        store = TrainingRunStore(self.root / 'unknown-task.db')
        self.training.store = store
        self.store = store
        self.add_run('kind', task_entry='missing.module:factory')
        with self.assertRaises(PlatformConflictError):
            self.training.require_editable(self.config.workspace_id, 'needles')


if __name__ == '__main__':
    unittest.main()
