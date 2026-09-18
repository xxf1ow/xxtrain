import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from test.platform_fixture import create_fixture
from xxtrain.business_tasks.point import point_task_definition
from xxtrain.platform.config import load_config
from xxtrain.platform.contracts import AnnotationChanges, PlatformError
from xxtrain.platform.runtime import RuntimeCache
from xxtrain.platform.service import AnnotationService
from xxtrain.platform.training_contracts import DownloadFile, ExecutionView
from xxtrain.platform.training_service import TrainingService
from xxtrain.platform.training_store import TrainingRunStore
from xxtrain.workspace_data import WorkspaceData
from xxtrain.workspace_data.repository import AnnotationRepository


class _UnusedCvat:
    pass


class ControlledClearML:
    def __init__(self) -> None:
        self.tasks = {}
        self.create_calls = 0
        self.enqueue_calls = 0
        self.create_failure = None
        self.enqueue_failure = None

    def find(self, run_id):
        matches = [task_id for task_id, task in self.tasks.items() if task['run_id'] == run_id]
        return matches[0] if matches else None

    def create(self, run):
        self.create_calls += 1
        task_id = f'task-{self.create_calls}'
        self.tasks[task_id] = {'run_id': run.id, 'status': 'created'}
        if self.create_failure is not None:
            error = self.create_failure
            self.create_failure = None
            raise error
        return task_id

    def enqueue(self, task_id):
        self.enqueue_calls += 1
        self.tasks[task_id]['status'] = 'queued'
        if self.enqueue_failure is not None:
            error = self.enqueue_failure
            self.enqueue_failure = None
            raise error

    def observe(self, task_id):
        status = self.tasks[task_id]['status']
        active = status in {'created', 'queued', 'running', 'unknown'}
        public_status = 'unknown' if status == 'created' else status
        return ExecutionView(task_id, public_status, active, None, None, None, None, status == 'completed', None)

    def cancel(self, task_id):
        self.tasks[task_id]['status'] = 'cancelled'
        return self.observe(task_id)

    def download(self, task_id, destination):
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / 'model.onnx'
        path.write_bytes(b'model')
        return DownloadFile(path, path.name, 'application/octet-stream')

    def complete(self, task_id):
        self.tasks[task_id]['status'] = 'completed'


class TrainingServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        receipt = create_fixture(self.root, owner_user_id=17, cvat_internal_url='http://cvat.test')
        self.config = load_config(Path(receipt['config_path']))
        self.workspace_id = self.config.workspace_id
        self.owner = self.config.owner_user_id
        self.data = WorkspaceData(self.config.workspace_dir)
        self.annotations = AnnotationService(
            self.config, self.data, _UnusedCvat(), RuntimeCache(self.config.runtime_dir)
        )
        for target in ('detect', 'classify', 'segment'):
            self.annotations.generate_target_cache(self.owner, target)
        self.store_path = self.root / 'platform' / 'training-runs.db'
        self.backend = ControlledClearML()
        self.service = self._service()

    def _service(self):
        return TrainingService(
            self.config,
            self.annotations,
            TrainingRunStore(self.store_path),
            self.backend,
            self.config.runtime_dir / 'cache',
        )

    def test_lock_depends_on_all_runs_not_last_clicked_target(self):
        runs = [self.service.submit(self.owner, target, str(uuid4())) for target in ('detect', 'classify', 'segment')]
        self.backend.complete(runs[0].run.clearml_task_id)
        self.backend.complete(runs[1].run.clearml_task_id)
        with self.assertRaises(PlatformError):
            self.service.require_editable(self.workspace_id)
        self.backend.complete(runs[2].run.clearml_task_id)
        self.service.require_editable(self.workspace_id)

    def test_cancelled_run_is_unlocked_after_query_and_fresh_service(self):
        run = self.service.submit(self.owner, 'detect', str(uuid4()))
        cancelled = self.service.cancel(self.owner, run.run.id)
        self.assertEqual('cancelled', cancelled.execution.status)

        restarted = self._service()
        self.assertEqual('cancelled', restarted.get_run(self.owner, run.run.id).execution.status)
        restarted.require_editable(self.workspace_id)

    def test_same_request_recovers_task_when_create_return_was_lost(self):
        request_id = str(uuid4())
        self.backend.create_failure = ConnectionError('response lost')

        recovered = self.service.submit(self.owner, 'detect', request_id)
        repeated = self.service.submit(self.owner, 'detect', request_id)

        self.assertEqual(recovered.run, repeated.run)
        self.assertEqual(1, self.backend.create_calls)
        self.assertEqual('queued', repeated.execution.status)

    def test_uncertain_create_without_remote_match_is_not_retried_blindly(self):
        request_id = str(uuid4())

        def missing_create(run):
            self.backend.create_calls += 1
            raise ConnectionError('request state unknown')

        self.backend.create = missing_create
        with self.assertRaisesRegex(PlatformError, 'not confirmed'):
            self.service.submit(self.owner, 'detect', request_id)
        with self.assertRaisesRegex(PlatformError, 'not confirmed'):
            self.service.submit(self.owner, 'detect', request_id)
        self.assertEqual(1, self.backend.create_calls)

    def test_uncertain_enqueue_reuses_bound_task(self):
        request_id = str(uuid4())
        self.backend.enqueue_failure = ConnectionError('response lost')
        with self.assertRaisesRegex(PlatformError, 'queue'):
            self.service.submit(self.owner, 'detect', request_id)

        recovered = self.service.submit(self.owner, 'detect', request_id)

        self.assertEqual('task-1', recovered.run.clearml_task_id)
        self.assertEqual(1, self.backend.create_calls)

    def test_unknown_remote_status_keeps_workspace_locked(self):
        run = self.service.submit(self.owner, 'detect', str(uuid4()))

        def unavailable(task_id):
            raise ConnectionError('offline')

        self.backend.observe = unavailable
        self.assertEqual('unknown', self.service.get_run(self.owner, run.run.id).execution.status)
        with self.assertRaises(PlatformError):
            self.service.require_editable(self.workspace_id)

    def test_retry_uses_retained_original_cache_after_current_annotations_change(self):
        original = self.service.submit(self.owner, 'classify', str(uuid4()))
        self.backend.complete(original.run.clearml_task_id)
        repository = AnnotationRepository(self.config.workspace_dir / 'annotations.db', point_task_definition())
        removed = repository.annotations(step_key='classify')[0]
        repository.apply_changes(AnnotationChanges((), frozenset({removed.id}), frozenset()))

        retried = self.service.retry(self.owner, original.run.id, str(uuid4()))

        self.assertEqual(original.run.fingerprint, retried.run.fingerprint)
        self.assertEqual(original.run.cache_relative_path, retried.run.cache_relative_path)
        self.assertNotEqual(original.run.id, retried.run.id)

    def test_referenced_cache_damage_is_not_rebuilt_over_historical_input(self):
        original = self.service.submit(self.owner, 'detect', str(uuid4()))
        self.backend.complete(original.run.clearml_task_id)
        cache = self.config.runtime_dir / 'cache' / original.run.cache_relative_path
        (cache / 'dataset.yaml').unlink()

        with self.assertRaisesRegex(PlatformError, 'cannot be replaced'):
            self.service.submit(self.owner, 'detect', str(uuid4()))

        self.assertFalse((cache / 'dataset.yaml').exists())


if __name__ == '__main__':
    unittest.main()
