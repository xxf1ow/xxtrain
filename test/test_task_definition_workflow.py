from __future__ import annotations

import copy
import random
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from PIL import Image

from test.task_definitions import workflow_task_definition
from test.test_platform_point_workflow import HttpxCvatFixture
from xxtrain.business_tasks.loader import load_task_definition
from xxtrain.business_tasks.point import point_task_definition
from xxtrain.integrations.clearml.client import ClearMLClient
from xxtrain.integrations.clearml.worker import main as worker_main
from xxtrain.platform.config import WorkspaceConfig
from xxtrain.platform.contracts import AnnotationRecord, PlatformConflictError, PlatformError
from xxtrain.platform.runtime import RuntimeCache
from xxtrain.platform.service import AnnotationService
from xxtrain.platform.training_contracts import TrainingRun
from xxtrain.platform.training_input_compat import initialize_input_compatibility
from xxtrain.platform.training_service import TrainingService
from xxtrain.platform.training_store import TrainingRunStore
from xxtrain.training.settings import TrainingResult
from xxtrain.workspace_data import WorkspaceData
from xxtrain.workspace_data.legacy_fingerprints import legacy_point_fingerprint
from xxtrain.workspace_data.repository import AnnotationRepository


class _TaskAdapter:
    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, object]] = {}
        self.create_calls = 0
        self.enqueue_calls = 0
        self.last_selected_entry: str | None = None

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.create_calls += 1
        task_id = f'task-{self.create_calls}'
        arguments = dict(kwargs['argparse_args'])
        self.last_selected_entry = arguments['task']
        self.tasks[task_id] = {'id': task_id, 'status': 'created', 'last_worker': None, 'parameters': {}}
        return SimpleNamespace(id=task_id)

    def find(self, **kwargs: object) -> list[SimpleNamespace]:
        return []

    def prepare(self, task_id: str, **kwargs: object) -> bool:
        return True

    def get(self, task_id: str) -> dict[str, object]:
        return copy.deepcopy(self.tasks[task_id])

    def enqueue(self, task_id: str, **kwargs: object) -> None:
        self.enqueue_calls += 1
        self.tasks[task_id]['status'] = 'queued'

    def has_artifact(self, task_id: str, name: str, **kwargs: object) -> bool:
        return False


class TaskDefinitionWorkflowTest(unittest.TestCase):
    configured_entry = 'test.task_definitions:workflow_task_definition'

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix='xxtrain-task-workflow-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / 'workspace'
        (self.workspace / 'images').mkdir(parents=True)
        self.config = WorkspaceConfig(
            'synthetic-workflow',
            'Synthetic workflow',
            17,
            self.workspace,
            self.root / 'runtime',
            'http://cvat.test',
            self.configured_entry,
        )
        self.task = load_task_definition(self.configured_entry)
        self.data = WorkspaceData(self.workspace, self.task)
        self.repository = AnnotationRepository(self.workspace / 'annotations.db', self.task)
        for index in range(2):
            path = self.root / f'input-{index}.png'
            rng = random.Random(700 + index)
            image = Image.new('RGB', (48, 40))
            image.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(48 * 40)])
            image.save(path)
            self.assertEqual(1, self.data.admit((path,)).accepted_count)
        self.cvat = HttpxCvatFixture()
        self.addCleanup(self.cvat.close)
        self.service = AnnotationService(
            self.config, self.data, self.cvat.client, RuntimeCache(self.config.runtime_dir)
        )

    def _begin(self, target: str):
        frames = self.data.target_frames(target, self.config.runtime_dir)
        self.cvat.expect(frames)
        self.service.begin_target(self.config.owner_user_id, target)
        job = max(self.cvat.jobs.values(), key=lambda item: item['id'])
        self.cvat.tasks[job['task_id']]['target'] = target
        return frames, job

    def _tags(self, target: str, count: int, label: str, start: int) -> list[dict]:
        label_id = self.cvat.label_id(target, label)
        return [{'id': start + frame, 'frame': frame, 'label_id': label_id, 'attributes': []} for frame in range(count)]

    def _rectangles(self, target: str, count: int, points: list[int], start: int) -> list[dict]:
        label_id = self.cvat.label_id(target, 'part')
        return [
            {
                'id': start + frame,
                'type': 'rectangle',
                'frame': frame,
                'label_id': label_id,
                'points': points,
                'attributes': [],
            }
            for frame in range(count)
        ]

    def test_configured_definition_supports_the_cross_component_workflow(self) -> None:
        region_frames, _ = self._begin('regions')
        self.cvat.set_annotations(
            'regions', shapes=self._rectangles('regions', len(region_frames), [10, 10, 35, 35], 100)
        )
        self.service.sync_target(17, 'regions')
        saved_outer_box = next(
            record
            for record in self.repository.annotations(step_key='regions')
            if record.image_id == region_frames[0].mapping.image_id
        )

        kind_frames, _ = self._begin('kind')
        kind_tags = self._tags('kind', len(kind_frames), 'a', 200)
        self.cvat.set_annotations('kind', tags=kind_tags)
        self.service.sync_target(17, 'kind')

        needle_frames, _ = self._begin('needles')
        needle_label = self.cvat.label_id('needles', 'line')
        needle_shapes = [
            {
                'id': 300 + frame,
                'type': 'polyline',
                'frame': frame,
                'label_id': needle_label,
                'points': [3, 4, 12, 14],
                'attributes': [],
            }
            for frame in range(len(needle_frames))
        ]
        self.cvat.set_annotations('needles', shapes=needle_shapes)
        self.service.sync_target(17, 'needles')

        needles_before = self.repository.annotations(step_key='needles')
        needle_fingerprint = self.data.target_fingerprint('needles')
        changed_tags = copy.deepcopy(kind_tags)
        changed_tags[0]['label_id'] = self.cvat.label_id('kind', 'b')
        self.cvat.set_annotations('kind', tags=changed_tags)
        self.service.sync_target(17, 'kind')
        self.assertEqual(needles_before, self.repository.annotations(step_key='needles'))
        self.assertEqual(needle_fingerprint, self.data.target_fingerprint('needles'))

        subregion_frames, _ = self._begin('subregions')
        self.cvat.set_annotations(
            'subregions', shapes=self._rectangles('subregions', len(subregion_frames), [2, 4, 18, 20], 400)
        )
        self.service.sync_target(17, 'subregions')
        saved_inner_box = next(
            record
            for record in self.repository.annotations(step_key='subregions')
            if record.parent_id == saved_outer_box.id
        )

        detail_frames, detail_job = self._begin('details')
        detail_tags = self._tags('details', len(detail_frames), 'x', 500)
        self.cvat.set_annotations('details', tags=detail_tags)
        database_path = self.workspace / 'annotations.db'
        with closing(sqlite3.connect(database_path)) as connection, connection:
            connection.execute(
                f"""CREATE TRIGGER reject_detail_binding BEFORE INSERT ON cvat_annotation_map
                WHEN NEW.job_id = {detail_job['id']} AND NEW.object_id = 500
                BEGIN SELECT RAISE(ABORT, 'controlled detail rollback'); END"""
            )
        with self.assertRaises(PlatformError):
            self.service.sync_target(17, 'details')
        self.assertEqual((), self.repository.annotations(step_key='details'))
        with closing(sqlite3.connect(database_path)) as connection, connection:
            connection.execute('DROP TRIGGER reject_detail_binding')

        restarted_task = load_task_definition(self.configured_entry)
        self.data = WorkspaceData(self.workspace, restarted_task)
        self.service = AnnotationService(
            self.config, self.data, self.cvat.client, RuntimeCache(self.config.runtime_dir)
        )
        self.service.sync_target(17, 'details')
        saved_detail = next(
            record
            for record in self.repository.annotations(step_key='details')
            if record.parent_id == saved_inner_box.id
        )
        self.assertEqual(saved_detail.parent_id, saved_inner_box.id)
        self.assertEqual(saved_inner_box.parent_id, saved_outer_box.id)
        self.assertEqual(saved_inner_box.geometry, [[12.0, 14.0], [28.0, 30.0]])

        cache_path = self.service.ensure_target_cache(17, 'regions')
        details_cache = self.service.ensure_target_cache(17, 'details')
        self.assertTrue((cache_path / 'dataset.yaml').is_file())
        self.assertTrue(any(path.suffix == '.png' for path in details_cache.rglob('*')))

        task_adapter = _TaskAdapter()
        clearml = ClearMLClient(
            'xxtrain',
            'training',
            self.root / 'xxtrain-worker.py',
            self.config.runtime_dir / 'cache',
            run_root=self.root / 'runs',
            sdk=task_adapter,
        )
        training = TrainingService(
            self.config,
            self.service,
            TrainingRunStore(self.root / 'metadata' / 'training-runs.db'),
            clearml,
            self.config.runtime_dir / 'cache',
        )
        self.service.require_editable = training.require_editable
        self.service.require_cache_rebuild = training.require_cache_rebuild
        region_run = training.submit(17, 'regions').run
        task_adapter.tasks[region_run.clearml_task_id]['status'] = 'completed'
        detail_run = training.submit(17, 'details').run

        self.assertEqual(task_adapter.last_selected_entry, self.configured_entry)
        training.require_editable(self.config.workspace_id, 'kind')
        training.require_editable(self.config.workspace_id, 'needles')
        for locked in ('regions', 'subregions', 'details'):
            with self.subTest(locked=locked), self.assertRaises(PlatformConflictError):
                training.require_editable(self.config.workspace_id, locked)
        with self.assertRaises(PlatformConflictError):
            training.require_editable(self.config.workspace_id)

        worker_task = Mock()
        worker_task.get_logger.return_value = Mock()
        worker_task.upload_artifact.return_value = True
        training_result = TrainingResult(self.root / 'model.onnx', {0: 'x', 1: 'y'}, {'synthetic/details': 0.75})
        delivery = self.root / 'delivery.zip'
        run_root = self.root / 'worker-runs'
        with (
            patch('xxtrain.integrations.clearml.worker._task_init', return_value=worker_task),
            patch('xxtrain.integrations.clearml.worker._disable_ultralytics_clearml'),
            patch('xxtrain.integrations.clearml.worker.train_prepared', return_value=training_result) as train,
            patch('xxtrain.integrations.clearml.worker.build_delivery', return_value=delivery) as build,
        ):
            worker_main(
                [
                    '--task',
                    detail_run.task_entry,
                    '--target',
                    detail_run.target,
                    '--cache-relative-path',
                    detail_run.cache_relative_path,
                    '--shared-root',
                    str(self.config.runtime_dir / 'cache'),
                    '--run-id',
                    detail_run.id,
                    '--run-root',
                    str(run_root),
                ]
            )
        definition = workflow_task_definition().step('details').training
        assert definition is not None
        self.assertEqual(definition.settings, train.call_args.args[0])
        self.assertIs(training_result, build.call_args.args[0])
        self.assertEqual(definition.delivery, build.call_args.args[1])
        self.assertEqual(details_cache, build.call_args.args[2])
        self.assertEqual(run_root / detail_run.id / 'delivery', build.call_args.args[3])

    def test_legacy_alias_survives_missing_publication_without_external_writes(self) -> None:
        workspace = self.root / 'legacy-workspace'
        (workspace / 'images').mkdir(parents=True)
        task = point_task_definition()
        data = WorkspaceData(workspace, task)
        source = self.root / 'legacy.png'
        Image.new('RGB', (32, 24), (10, 20, 30)).save(source)
        self.assertEqual(1, data.admit((source,)).accepted_count)
        image_id = data.images()[0].sample_id
        repository = AnnotationRepository(workspace / 'annotations.db', task)
        parent = AnnotationRecord(uuid4(), image_id, 'detect', None, 'rectangle', 'Point', [[2, 3], [26, 20]])
        category = AnnotationRecord(uuid4(), image_id, 'classify', parent.id, 'classification', 'tl', None)
        line = AnnotationRecord(uuid4(), image_id, 'segment', parent.id, 'polyline', '1', [[4, 5], [12, 14]])
        repository.save_annotations((parent, category, line))
        config = WorkspaceConfig(
            'legacy-workflow', 'Legacy workflow', 17, workspace, self.root / 'legacy-runtime', 'http://cvat.test'
        )
        annotations = AnnotationService(config, data, object(), RuntimeCache(config.runtime_dir))
        store = TrainingRunStore(self.root / 'legacy-metadata' / 'training-runs.db')
        legacy_fingerprint = legacy_point_fingerprint('segment', data.images(), repository.annotations())
        historical = store.create(
            TrainingRun(
                str(uuid4()),
                17,
                config.workspace_id,
                config.display_name,
                'segment',
                legacy_fingerprint,
                f'{legacy_fingerprint}/segment',
                '2026-09-17T00:00:00+00:00',
                None,
                'legacy-task',
                None,
                'point',
            )
        )
        marker = config.runtime_dir / 'cache' / historical.cache_relative_path / 'preserve-marker'
        marker.parent.mkdir(parents=True)
        marker.write_bytes(b'preserve')
        task_adapter = _TaskAdapter()
        task_adapter.tasks['legacy-task'] = {
            'id': 'legacy-task',
            'status': 'completed',
            'last_worker': None,
            'parameters': {},
        }
        training = TrainingService(
            config,
            annotations,
            store,
            ClearMLClient(
                'xxtrain', 'training', self.root / 'xxtrain-worker.py', config.runtime_dir / 'cache', sdk=task_adapter
            ),
            config.runtime_dir / 'cache',
        )

        initialize_input_compatibility(training)
        initialize_input_compatibility(training)
        submitted = training.submit(17, 'segment')

        self.assertEqual(historical.id, submitted.run.id)
        self.assertEqual(historical, store.get(17, historical.id))
        associated = store.find_input(17, config.workspace_id, 'segment', data.training_fingerprint('segment'))
        self.assertIsNotNone(associated)
        self.assertEqual(historical.id, associated.id)
        self.assertEqual(1, len(store.list_user(17)))
        self.assertEqual((0, 0), (task_adapter.create_calls, task_adapter.enqueue_calls))
        self.assertTrue(marker.is_file())
        self.assertFalse((marker.parent / 'dataset.yaml').exists())


if __name__ == '__main__':
    unittest.main()
