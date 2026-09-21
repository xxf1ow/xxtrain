import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid4

from PIL import Image

from xxtrain.business_tasks.point import point_task_definition
from xxtrain.platform.config import WorkspaceConfig
from xxtrain.platform.contracts import AnnotationRecord, ImageRecord, PlatformConflictError, PlatformError
from xxtrain.platform.runtime import RuntimeCache
from xxtrain.platform.service import AnnotationService
from xxtrain.platform.training_contracts import ExecutionView, TrainingRun
from xxtrain.platform.training_input_compat import initialize_input_compatibility
from xxtrain.platform.training_service import TrainingService
from xxtrain.platform.training_store import TrainingRunStore
from xxtrain.workspace_data import WorkspaceData
from xxtrain.workspace_data.inputs import AxisAlignedRectangleInputs
from xxtrain.workspace_data.legacy_fingerprints import legacy_point_fingerprint
from xxtrain.workspace_data.repository import AnnotationRepository


class _NoClearMLWrites:
    def __init__(self) -> None:
        self.created: list[str] = []

    @staticmethod
    def observe(task_id: str) -> ExecutionView:
        return ExecutionView(task_id, 'completed', False, None, None, None, None, True, None)


class _RecordingClearML(_NoClearMLWrites):
    def create(self, run: TrainingRun) -> str:
        self.created.append(run.id)
        return 'clearml-new'

    @staticmethod
    def enqueue(task_id: str, run: TrainingRun) -> None:
        del task_id, run


class _PaddedAxisAlignedRectangleInputs(AxisAlignedRectangleInputs):
    def mappings(self, images, records, step):
        return tuple(
            replace(mapping, bounds=(mapping.bounds[0] - 1, mapping.bounds[1] - 1, *mapping.bounds[2:]))
            for mapping in super().mappings(images, records, step)
        )


class TrainingInputCompatibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix='xxtrain-training-compat-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / 'workspace'
        (self.workspace / 'images').mkdir(parents=True)
        image_path = self.workspace / 'images' / 'source.png'
        with Image.new('RGB', (64, 48), (10, 20, 30)) as image:
            image.save(image_path)
        self.task = point_task_definition()
        self.data = WorkspaceData(self.workspace, self.task)
        self.repository = AnnotationRepository(self.workspace / 'annotations.db', self.task)
        self.image = ImageRecord('a' * 64, 'images/source.png', 64, 48, 0)
        self.repository.register_images((self.image,))
        parent = AnnotationRecord(UUID(int=1), self.image.id, 'detect', None, 'rectangle', 'Point', [[1, 2], [30, 40]])
        category = AnnotationRecord(UUID(int=2), self.image.id, 'classify', parent.id, 'classification', 'tl', None)
        line = AnnotationRecord(UUID(int=3), self.image.id, 'segment', parent.id, 'polyline', '1', [[2, 3], [4, 5]])
        self.repository.save_annotations((parent, category, line))
        self.config = WorkspaceConfig('line-1', 'Line 1', 17, self.workspace, self.root / 'runtime', 'http://cvat.test')
        self.annotations = AnnotationService(self.config, self.data, object(), RuntimeCache(self.config.runtime_dir))
        self.store_path = self.root / 'metadata' / 'training-runs.db'
        self.clearml = _NoClearMLWrites()

    def _legacy_fingerprint(self, target: str) -> str:
        return legacy_point_fingerprint(target, self.data.images(), self.repository.annotations())

    def _old_run(self, target: str = 'segment', *, fingerprint: str | None = None) -> TrainingRun:
        fingerprint = fingerprint or self._legacy_fingerprint(target)
        return TrainingRun(
            str(uuid4()),
            self.config.owner_user_id,
            self.config.workspace_id,
            self.config.display_name,
            target,
            fingerprint,
            f'{fingerprint}/{target}',
            '2026-09-17T00:00:00+00:00',
            None,
            'clearml-old',
            None,
            'point',
        )

    def _write_prechange_run(self, run: TrainingRun) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.store_path)) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE training_runs (
                  id TEXT PRIMARY KEY,
                  user_id INTEGER NOT NULL,
                  workspace_id TEXT NOT NULL,
                  workspace_name TEXT NOT NULL,
                  target TEXT NOT NULL,
                  fingerprint TEXT NOT NULL,
                  cache_relative_path TEXT NOT NULL,
                  submitted_at TEXT NOT NULL,
                  create_attempted_at TEXT,
                  clearml_task_id TEXT UNIQUE,
                  desired_action TEXT
                );
                """
            )
            connection.execute(
                'INSERT INTO training_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    run.id,
                    run.user_id,
                    run.workspace_id,
                    run.workspace_name,
                    run.target,
                    run.fingerprint,
                    run.cache_relative_path,
                    run.submitted_at,
                    run.create_attempted_at,
                    run.clearml_task_id,
                    run.desired_action,
                ),
            )

    def _publish(self, run: TrainingRun, *, manifest: bool = True, complete: bool = True) -> Path:
        target = self.config.runtime_dir / 'cache' / run.cache_relative_path
        target.mkdir(parents=True)
        if complete:
            (target / 'dataset.yaml').write_text(
                f'path: {target.parent}\ntrain: {run.target}/train.txt\nval: {run.target}/val.txt\n', encoding='utf-8'
            )
            (target / 'train.txt').write_text(f'{run.target}/sample.png\n', encoding='utf-8')
            (target / 'val.txt').write_text(f'{run.target}/sample.png\n', encoding='utf-8')
            (target / 'sample.png').touch()
        if manifest:
            (target.parent / 'manifest.json').write_text(
                json.dumps({'fingerprint': run.fingerprint, 'target': run.target}), encoding='utf-8'
            )
        return target

    def _service(self, store: TrainingRunStore) -> TrainingService:
        return TrainingService(self.config, self.annotations, store, self.clearml, self.config.runtime_dir / 'cache')

    def test_startup_aliases_exact_old_input_without_rewriting_run_or_clearml(self) -> None:
        old_run = self._old_run()
        self._write_prechange_run(old_run)
        self._publish(old_run)
        store = TrainingRunStore(self.store_path)
        service = self._service(store)
        new_fingerprint = self.data.training_fingerprint('segment')

        initialize_input_compatibility(service)
        initialize_input_compatibility(service)

        associated = store.find_input(old_run.user_id, old_run.workspace_id, 'segment', new_fingerprint)
        self.assertEqual(old_run.id, associated.id)
        stored = store.get(old_run.user_id, old_run.id)
        self.assertEqual(old_run.fingerprint, stored.fingerprint)
        self.assertEqual(old_run.cache_relative_path, stored.cache_relative_path)
        self.assertEqual('point', stored.task_entry)
        self.assertEqual([], self.clearml.created)
        with self.assertRaises(PlatformError):
            service.require_cache_rebuild(old_run.workspace_id, old_run.target, new_fingerprint)

    def test_classification_change_after_startup_keeps_segment_alias(self) -> None:
        old_run = self._old_run()
        self._write_prechange_run(old_run)
        self._publish(old_run)
        store = TrainingRunStore(self.store_path)
        service = self._service(store)
        initialize_input_compatibility(service)
        before = self.data.training_fingerprint('segment')
        (category,) = self.repository.annotations(step_key='classify')

        self.repository.save_annotations((replace(category, label='tc'),))

        self.assertEqual(before, self.data.training_fingerprint('segment'))
        self.assertEqual(old_run.id, store.find_input(17, 'line-1', 'segment', before).id)
        self.assertEqual(old_run.id, service.workspace_view(17)['training']['segment'].run.id)
        self.assertEqual(old_run.id, service.submit(17, 'segment').run.id)
        self.assertEqual([], self.clearml.created)

    def test_broken_publication_does_not_prevent_exact_input_alias_or_trigger_rebuild(self) -> None:
        old_run = self._old_run()
        self._write_prechange_run(old_run)
        self._publish(old_run, complete=False)
        store = TrainingRunStore(self.store_path)

        initialize_input_compatibility(self._service(store))

        self.assertEqual(
            old_run.id, store.find_input(17, 'line-1', 'segment', self.data.training_fingerprint('segment')).id
        )
        self.assertEqual([], self.clearml.created)
        self.assertFalse((self.config.runtime_dir / 'cache' / old_run.cache_relative_path / 'dataset.yaml').exists())

    def test_changed_conversion_or_new_detect_named_task_cannot_claim_legacy_identity(self) -> None:
        old_run = self._old_run('detect')
        self._write_prechange_run(old_run)
        self._publish(old_run, manifest=False)
        store = TrainingRunStore(self.store_path)
        detect = self.task.step('detect')
        changed_training = replace(detect.training, conversion_key='different-detect-semantics-v2')
        changed_task = replace(
            self.task, key='new-task', steps=(replace(detect, training=changed_training), *self.task.steps[1:])
        )
        changed_data = WorkspaceData(self.workspace, changed_task)
        changed_annotations = AnnotationService(
            self.config, changed_data, object(), RuntimeCache(self.config.runtime_dir)
        )
        service = TrainingService(
            self.config, changed_annotations, store, self.clearml, self.config.runtime_dir / 'cache'
        )

        initialize_input_compatibility(service)

        self.assertIsNone(store.find_input(17, 'line-1', 'detect', changed_data.training_fingerprint('detect')))

    def test_changed_input_projection_is_not_aliased_and_submit_creates_a_new_run(self) -> None:
        second_parent = AnnotationRecord(
            UUID(int=4), self.image.id, 'detect', None, 'rectangle', 'Point', [[31, 2], [50, 40]]
        )
        second_category = AnnotationRecord(
            UUID(int=5), self.image.id, 'classify', second_parent.id, 'classification', 'tc', None
        )
        self.repository.save_annotations((second_parent, second_category))
        old_run = self._old_run('classify')
        self._write_prechange_run(old_run)
        self._publish(old_run, complete=False)
        store = TrainingRunStore(self.store_path)
        classify = self.task.step('classify')
        changed_task = replace(
            self.task,
            steps=tuple(
                replace(step, input_adapter=_PaddedAxisAlignedRectangleInputs())
                if step.key == 'classify'
                else replace(step, minimum_samples=0)
                if step.key == 'detect'
                else step
                for step in self.task.steps
            ),
        )
        changed_data = WorkspaceData(self.workspace, changed_task)
        changed_annotations = AnnotationService(
            self.config, changed_data, object(), RuntimeCache(self.config.runtime_dir)
        )
        clearml = _RecordingClearML()
        service = TrainingService(self.config, changed_annotations, store, clearml, self.config.runtime_dir / 'cache')
        current_fingerprint = changed_data.training_fingerprint(classify.key)
        self.assertNotEqual(self.data.training_fingerprint(classify.key), current_fingerprint)

        initialize_input_compatibility(service)
        submitted = service.submit(17, classify.key)

        self.assertNotEqual(old_run.id, submitted.run.id)
        self.assertEqual(current_fingerprint, submitted.run.fingerprint)
        self.assertEqual(2, len(store.list_user(17)))
        self.assertEqual([submitted.run.id], clearml.created)

    def test_manifest_free_detection_does_not_weaken_runtime_manifest_requirement(self) -> None:
        old_run = self._old_run('detect')
        self._write_prechange_run(old_run)
        target = self._publish(old_run, manifest=False)
        store = TrainingRunStore(self.store_path)

        with self.assertRaises(ValueError):
            RuntimeCache(self.config.runtime_dir).cache_path('detect', old_run.fingerprint)
        initialize_input_compatibility(self._service(store))

        current = self.data.training_fingerprint('detect')
        self.assertEqual(old_run.id, store.find_input(17, 'line-1', 'detect', current).id)
        self.assertEqual(target, self.config.runtime_dir / 'cache' / old_run.cache_relative_path)

    def test_associate_input_rejects_wrong_owners_targets_and_conflicting_identities(self) -> None:
        store = TrainingRunStore(self.store_path)
        first = store.create(self._old_run())
        second = store.create(
            replace(
                self._old_run(),
                fingerprint='e' * 64,
                cache_relative_path=f'{"e" * 64}/segment',
                clearml_task_id='clearml-second',
            )
        )
        alias = 'f' * 64

        with self.assertRaises(ValueError):
            store.associate_input(18, first.workspace_id, first.target, alias, first.id)
        with self.assertRaises(ValueError):
            store.associate_input(first.user_id, first.workspace_id, 'classify', alias, first.id)

        store.associate_input(first.user_id, first.workspace_id, first.target, alias, first.id)
        store.associate_input(first.user_id, first.workspace_id, first.target, alias, first.id)
        self.assertEqual(first.id, store.find_input(first.user_id, first.workspace_id, first.target, alias).id)
        with self.assertRaises(PlatformConflictError):
            store.associate_input(second.user_id, second.workspace_id, second.target, alias, second.id)

        candidate = replace(
            first,
            id=str(uuid4()),
            fingerprint=alias,
            cache_relative_path=f'{alias}/segment',
            submitted_at='2026-09-20T00:00:00+00:00',
        )
        self.assertEqual(first.id, store.create(candidate).id)


if __name__ == '__main__':
    unittest.main()
