import json
import os
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from xxtrain.data import Bbox
from xxtrain.integrations.cvat import PreparationState
from xxtrain.platform.config import WorkspaceConfig, load_config
from xxtrain.platform.contracts import DetectionBox, FrameResult, JobRef, PlatformAccessError, PlatformError
from xxtrain.platform.runtime import RuntimeCache
from xxtrain.platform.service import AnnotationService
from xxtrain.platform.state import StateStore
from xxtrain.workspace_data import WorkspaceData

PrepareBehavior = Callable[[PreparationState, Callable[[PreparationState], None] | None], JobRef]


class FakeCvatClient:
    def __init__(self, results: tuple[FrameResult, ...]) -> None:
        self.results = results
        self.calls: list[tuple[str, object]] = []
        self.create_errors: list[PlatformError] = []
        self.fetch_errors: list[PlatformError] = []
        self.prepare_behaviors: list[PrepareBehavior] = []
        self.job_path_error: PlatformError | None = None

    def create_task(self, name: str, labels: tuple[str, ...]) -> int:
        self.calls.append(('create_task', (name, labels)))
        if self.create_errors:
            raise self.create_errors.pop(0)
        return 41

    def prepare_task(
        self,
        task_id: int,
        images: tuple[object, ...],
        user_id: int,
        *,
        preparation: PreparationState = PreparationState(),
        checkpoint: Callable[[PreparationState], None] | None = None,
    ) -> JobRef:
        self.calls.append(('prepare_task', (task_id, len(images), user_id, preparation)))
        if self.prepare_behaviors:
            return self.prepare_behaviors.pop(0)(preparation, checkpoint)
        assert checkpoint is not None
        for state in (
            PreparationState('uploading'),
            PreparationState('uploading', 'upload-41'),
            PreparationState('uploaded'),
            PreparationState('initializing'),
            PreparationState('initialized'),
        ):
            checkpoint(state)
        return JobRef(41, 73, ('a', 'b'))

    def fetch_detection(self, ref: JobRef) -> tuple[FrameResult, ...]:
        self.calls.append(('fetch_detection', ref))
        if self.fetch_errors:
            raise self.fetch_errors.pop(0)
        return self.results

    def job_path(self, ref: JobRef) -> str:
        self.calls.append(('job_path', ref))
        if self.job_path_error is not None:
            raise self.job_path_error
        return f'/tasks/{ref.task_id}/jobs/{ref.job_id}'


class PlatformConfigAndStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_config_loads_exact_json_fields_and_resolves_paths_from_its_directory(self) -> None:
        config_path = self.root / 'deploy' / 'workspace.json'
        config_path.parent.mkdir()
        config_path.write_text(
            json.dumps(
                {
                    'workspace_id': 'line-3',
                    'display_name': 'Line 3',
                    'owner_user_id': 17,
                    'images_dir': '../site/images',
                    'annotations_dir': '../site/annotations',
                    'state_path': 'state/workspace.json',
                    'cvat_internal_url': 'http://cvat.internal:8080',
                }
            ),
            encoding='utf-8',
        )

        config = load_config(config_path)

        self.assertEqual(
            WorkspaceConfig(
                workspace_id='line-3',
                display_name='Line 3',
                owner_user_id=17,
                images_dir=(self.root / 'site' / 'images').resolve(),
                annotations_dir=(self.root / 'site' / 'annotations').resolve(),
                state_path=(self.root / 'deploy' / 'state' / 'workspace.json').resolve(),
                cvat_internal_url='http://cvat.internal:8080',
            ),
            config,
        )

        payload = json.loads(config_path.read_text(encoding='utf-8'))
        payload['unexpected'] = True
        config_path.write_text(json.dumps(payload), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'fields'):
            load_config(config_path)

    def test_state_store_round_trips_an_object_and_rejects_other_json_roots(self) -> None:
        state_path = self.root / 'state.json'
        state = StateStore(state_path)

        self.assertEqual({}, state.load())
        state.save({'status': 'preparing', 'task_id': 41})
        self.assertEqual({'status': 'preparing', 'task_id': 41}, state.load())

        state_path.write_text('[]', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'object'):
            state.load()

    def test_runtime_job_map_is_atomic_and_keyed_by_target_and_fingerprint(self) -> None:
        cache = RuntimeCache(self.root / 'runtime')
        ref = JobRef(41, 73, ('a',))

        cache.remember_job('detect', 'a' * 64, ref)

        self.assertEqual(ref, RuntimeCache(self.root / 'runtime').job_for('detect', 'a' * 64))
        self.assertIsNone(cache.job_for('detect', 'b' * 64))
        self.assertFalse(cache.has_detection_cache('a' * 64))
        (self.root / 'runtime' / 'cache' / ('a' * 64) / 'detect').mkdir(parents=True)
        self.assertTrue(cache.has_detection_cache('a' * 64))

    def test_runtime_job_map_rejects_a_non_job_mapping_shape(self) -> None:
        runtime = self.root / 'runtime'
        runtime.mkdir()
        (runtime / 'jobs.json').write_text('{"detect": {"fingerprint": []}}', encoding='utf-8')

        with self.assertRaisesRegex(ValueError, 'Runtime job map'):
            RuntimeCache(runtime).job_for('detect', 'fingerprint')


class AnnotationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images_dir = self.root / 'images'
        self.annotations_dir = self.root / 'annotations'
        self.images_dir.mkdir()
        self.annotations_dir.mkdir()
        for name in ('a.jpg', 'b.jpg'):
            Image.new('RGB', (64, 48), 'white').save(self.images_dir / name)
        self.a_path = self._write_annotation('a', 'old-a')
        self.b_path = self._write_annotation('b', 'old-b')
        self.state_path = self.root / 'state.json'
        self.config = WorkspaceConfig(
            workspace_id='line-3',
            display_name='Line 3',
            owner_user_id=17,
            images_dir=self.images_dir,
            annotations_dir=self.annotations_dir,
            state_path=self.state_path,
            cvat_internal_url='http://cvat.internal:8080',
        )
        self.data = WorkspaceData(self.images_dir, self.annotations_dir)
        self.results = (
            FrameResult('a', (DetectionBox(Bbox(label='tl', x1=1, y1=2, x2=20, y2=30)),)),
            FrameResult('b', (DetectionBox(Bbox(label='cc', x1=3, y1=4, x2=22, y2=31)),)),
        )
        self.cvat = FakeCvatClient(self.results)
        self.state = StateStore(self.state_path)
        self.service = AnnotationService(self.config, self.data, self.cvat, self.state)

    def _write_annotation(self, stem: str, marker: str) -> Path:
        path = self.annotations_dir / f'{stem}.json'
        path.write_text(
            json.dumps(
                {
                    'marker': marker,
                    'shapes': [{'label': 'mask', 'shape_type': 'polygon', 'points': [[0, 0], [1, 0], [1, 1]]}],
                }
            ),
            encoding='utf-8',
        )
        return path

    def _call_names(self) -> list[str]:
        return [name for name, _ in self.cvat.calls]

    def test_first_begin_persists_every_checkpoint_and_restart_continues_the_job(self) -> None:
        observed_checkpoints = []

        def observe_checkpoints(
            preparation: PreparationState, checkpoint: Callable[[PreparationState], None] | None
        ) -> JobRef:
            self.assertEqual(PreparationState(), preparation)
            assert checkpoint is not None
            for current in (
                PreparationState('uploading'),
                PreparationState('uploading', 'upload-41'),
                PreparationState('uploaded'),
                PreparationState('initializing'),
                PreparationState('initialized'),
            ):
                checkpoint(current)
                observed_checkpoints.append(self.state.load()['preparation'])
            return JobRef(41, 73, ('a', 'b'))

        self.cvat.prepare_behaviors.append(observe_checkpoints)
        self.assertEqual('pending', self.service.view(17).status)

        self.assertEqual('/tasks/41/jobs/73', self.service.begin(user_id=17))

        self.assertEqual(
            [
                {'stage': 'uploading', 'request_id': None},
                {'stage': 'uploading', 'request_id': 'upload-41'},
                {'stage': 'uploaded', 'request_id': None},
                {'stage': 'initializing', 'request_id': None},
                {'stage': 'initialized', 'request_id': None},
            ],
            observed_checkpoints,
        )
        saved = self.state.load()
        self.assertEqual('annotating', saved['status'])
        self.assertEqual({'stage': 'initialized', 'request_id': None}, saved['preparation'])
        self.assertEqual({'task_id': 41, 'job_id': 73, 'sample_ids': ['a', 'b']}, saved['job'])
        self.assertEqual(1, self._call_names().count('create_task'))

        restarted = AnnotationService(self.config, self.data, self.cvat, StateStore(self.state_path))
        self.assertEqual('/tasks/41/jobs/73', restarted.begin(user_id=17))
        self.assertEqual(1, self._call_names().count('create_task'))
        self.assertEqual(1, self._call_names().count('prepare_task'))

    def test_known_upload_request_resumes_without_recreating_the_task(self) -> None:
        def stop_after_request(
            preparation: PreparationState, checkpoint: Callable[[PreparationState], None] | None
        ) -> JobRef:
            self.assertEqual(PreparationState(), preparation)
            assert checkpoint is not None
            checkpoint(PreparationState('uploading'))
            checkpoint(PreparationState('uploading', 'known-request'))
            raise PlatformError('polling interrupted')

        def resume_request(
            preparation: PreparationState, checkpoint: Callable[[PreparationState], None] | None
        ) -> JobRef:
            self.assertEqual(PreparationState('uploading', 'known-request'), preparation)
            assert checkpoint is not None
            checkpoint(PreparationState('uploaded'))
            checkpoint(PreparationState('initializing'))
            checkpoint(PreparationState('initialized'))
            return JobRef(41, 73, ('a', 'b'))

        self.cvat.prepare_behaviors.extend((stop_after_request, resume_request))

        with self.assertRaisesRegex(PlatformError, 'polling interrupted'):
            self.service.begin(17)
        self.assertEqual({'stage': 'uploading', 'request_id': 'known-request'}, self.state.load()['preparation'])

        self.assertEqual('/tasks/41/jobs/73', self.service.begin(17))
        self.assertEqual(1, self._call_names().count('create_task'))

    def test_uncertain_creation_is_not_repeated_and_is_visible_for_reconciliation(self) -> None:
        self.cvat.create_errors.append(PlatformError('connection ended after POST'))

        with self.assertRaisesRegex(PlatformError, 'administrator reconciliation'):
            self.service.begin(17)

        saved = self.state.load()
        self.assertEqual('preparing', saved['status'])
        self.assertIn('administrator', saved['error'])
        self.assertEqual('creating', saved['task_creation'])

        restarted = AnnotationService(self.config, self.data, self.cvat, StateStore(self.state_path))
        with self.assertRaisesRegex(PlatformError, 'administrator reconciliation'):
            restarted.begin(17)
        self.assertEqual(1, self._call_names().count('create_task'))

    def test_ambiguous_initialization_checkpoint_survives_restart(self) -> None:
        def uncertain_initialization(
            preparation: PreparationState, checkpoint: Callable[[PreparationState], None] | None
        ) -> JobRef:
            assert checkpoint is not None
            checkpoint(PreparationState('uploaded'))
            checkpoint(PreparationState('initializing'))
            raise PlatformError('initialization result is unknown')

        def reject_repeat(
            preparation: PreparationState, checkpoint: Callable[[PreparationState], None] | None
        ) -> JobRef:
            self.assertEqual(PreparationState('initializing'), preparation)
            raise PlatformError('administrator reconciliation is required')

        self.cvat.prepare_behaviors.extend((uncertain_initialization, reject_repeat))

        with self.assertRaisesRegex(PlatformError, 'unknown'):
            self.service.begin(17)
        self.assertEqual({'stage': 'initializing', 'request_id': None}, self.state.load()['preparation'])

        restarted = AnnotationService(self.config, self.data, self.cvat, StateStore(self.state_path))
        with self.assertRaisesRegex(PlatformError, 'administrator reconciliation'):
            restarted.begin(17)
        self.assertEqual(1, self._call_names().count('create_task'))

    def test_sync_failure_is_retryable_without_recreating_the_job(self) -> None:
        self.service.begin(17)
        self.cvat.fetch_errors.append(PlatformError('temporary CVAT read failure'))

        with self.assertRaisesRegex(PlatformError, '无法取回或保存标注，请重试。'):
            self.service.sync(17)

        self.assertEqual('sync_failed', self.service.view(17).status)
        self.assertEqual('无法取回或保存标注，请重试。', self.service.view(17).error)
        self.assertEqual('saved', self.service.sync(17).status)
        self.assertEqual('tl', json.loads(self.a_path.read_text(encoding='utf-8'))['shapes'][1]['label'])
        self.assertEqual(1, self._call_names().count('create_task'))

    def test_partial_two_file_save_keeps_earlier_replacement_then_retry_converges(self) -> None:
        self.service.begin(17)
        old_b = self.b_path.read_bytes()
        real_replace = os.replace

        def fail_second_annotation(source: str | bytes, destination: str | bytes) -> None:
            if Path(destination) == self.b_path:
                raise OSError('disk full')
            real_replace(source, destination)

        with patch('xxtrain.workspace_data.store.os.replace', side_effect=fail_second_annotation):
            with self.assertRaisesRegex(PlatformError, '无法取回或保存标注，请重试。'):
                self.service.sync(17)

        self.assertEqual('tl', json.loads(self.a_path.read_text(encoding='utf-8'))['shapes'][1]['label'])
        self.assertEqual(old_b, self.b_path.read_bytes())
        self.assertEqual('sync_failed', self.service.view(17).status)

        result = self.service.sync(17)

        self.assertEqual('saved', result.status)
        self.assertEqual('tl', json.loads(self.a_path.read_text(encoding='utf-8'))['shapes'][1]['label'])
        self.assertEqual('cc', json.loads(self.b_path.read_text(encoding='utf-8'))['shapes'][1]['label'])
        self.assertEqual(2, self._call_names().count('fetch_detection'))

    def test_state_save_failure_cannot_report_sync_success(self) -> None:
        self.service.begin(17)
        real_replace = os.replace

        def fail_state(source: str | bytes, destination: str | bytes) -> None:
            if Path(destination) == self.state_path:
                raise OSError('state disk full')
            real_replace(source, destination)

        with patch('xxtrain.platform.state.os.replace', side_effect=fail_state):
            with self.assertRaisesRegex(OSError, 'state disk full'):
                self.service.sync(17)

        self.assertEqual('annotating', self.service.view(17).status)
        self.assertEqual('saved', self.service.sync(17).status)

    def test_begin_on_saved_job_persists_annotating_before_returning_to_editor(self) -> None:
        self.service.begin(17)
        self.service.sync(17)
        self.cvat.job_path_error = PlatformError('cannot render path')

        with self.assertRaisesRegex(PlatformError, 'cannot render path'):
            self.service.begin(17)

        self.assertEqual('annotating', self.service.view(17).status)

    def test_repeated_sync_overwrites_the_same_files_without_history(self) -> None:
        self.service.begin(17)
        self.service.sync(17)
        self.service.sync(17)

        self.assertEqual(['a.json', 'b.json'], sorted(path.name for path in self.annotations_dir.iterdir()))
        self.assertEqual(2, self._call_names().count('fetch_detection'))

    def test_wrong_user_and_disabled_target_calls_have_no_side_effects(self) -> None:
        original_annotations = (self.a_path.read_bytes(), self.b_path.read_bytes())
        for operation in (self.service.view, self.service.begin, self.service.sync):
            with (
                self.subTest(operation=operation.__name__),
                self.assertRaisesRegex(PlatformAccessError, 'Workspace access denied'),
            ):
                operation(99)

        for target in ('classify', 'segment'):
            with self.subTest(target=target), self.assertRaises(TypeError):
                self.service.begin(user_id=17, target=target)

        self.assertFalse(self.state_path.exists())
        self.assertEqual(original_annotations, (self.a_path.read_bytes(), self.b_path.read_bytes()))
        self.assertEqual([], self.cvat.calls)


if __name__ == '__main__':
    unittest.main()
