import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from xxtrain.data import Bbox
from xxtrain.platform.config import WorkspaceConfig, load_config
from xxtrain.platform.contracts import DetectionBox, FrameResult, JobRef, PlatformAccessError, PlatformError
from xxtrain.platform.runtime import RuntimeCache
from xxtrain.platform.service import AnnotationService
from xxtrain.workspace_data import WorkspaceData


class FakeCvatClient:
    def __init__(self):
        self.unfinished = True
        self.create_task_calls = 0
        self.prepare_error = None
        self.check_error = None
        self.results = ()

    def create_task(self, name, labels):
        self.create_task_calls += 1
        return 40 + self.create_task_calls

    def prepare_task(self, task_id, images, user_id):
        if self.prepare_error:
            raise self.prepare_error
        return JobRef(task_id, task_id + 32, tuple(image.sample_id for image in images))

    def job_is_unfinished(self, ref):
        if self.check_error:
            raise self.check_error
        return self.unfinished

    def fetch_detection(self, ref):
        return self.results

    def job_path(self, ref):
        return f'/tasks/{ref.task_id}/jobs/{ref.job_id}'


class PlatformConfigTest(unittest.TestCase):
    def test_config_loads_exact_fields_and_resolves_both_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / 'workspace.json'
            payload = {
                'workspace_id': 'line-3',
                'display_name': 'Line 3',
                'owner_user_id': 17,
                'workspace_dir': 'site',
                'runtime_dir': 'runtime',
                'cvat_internal_url': 'http://cvat.test',
            }
            config_path.write_text(json.dumps(payload), encoding='utf-8')
            config = load_config(config_path)
            self.assertEqual((root / 'site').resolve(), config.workspace_dir)
            self.assertEqual((root / 'runtime').resolve(), config.runtime_dir)
            for field, value in (('unexpected', True), ('owner_user_id', True), ('workspace_dir', '')):
                with self.subTest(field=field):
                    config_path.write_text(json.dumps(payload | {field: value}), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        load_config(config_path)

    def test_runtime_job_map_is_atomic_and_keyed_by_target_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = RuntimeCache(Path(directory))
            ref = JobRef(41, 73, ('a',))
            cache.remember_job('detect', 'a' * 64, ref)
            self.assertEqual(ref, RuntimeCache(Path(directory)).job_for('detect', 'a' * 64))
            self.assertIsNone(cache.job_for('detect', 'b' * 64))
            self.assertIsNone(cache.job_for('classify', 'a' * 64))
            self.assertFalse(cache.has_detection_cache('a' * 64))
            (Path(directory) / 'cache' / ('a' * 64) / 'detect').mkdir(parents=True)
            self.assertTrue(cache.has_detection_cache('a' * 64))

    def test_runtime_job_map_rejects_a_non_job_mapping_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'jobs.json').write_text('{"detect": {"fingerprint": []}}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Runtime job map'):
                RuntimeCache(Path(directory)).job_for('detect', 'fingerprint')


class AnnotationServiceTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / 'workspace'
        self.images = self.workspace / 'images'
        self.annotations = self.workspace / 'annotations'
        self.images.mkdir(parents=True)
        self.annotations.mkdir()
        self.config = WorkspaceConfig('line-3', 'Line 3', 17, self.workspace, self.root / 'runtime', 'http://cvat.test')
        self.data = WorkspaceData(self.workspace)
        self.runtime = RuntimeCache(self.config.runtime_dir)
        self.cvat = FakeCvatClient()
        self.service = AnnotationService(self.config, self.data, self.cvat, self.runtime)

    def create_workspace(self, boxed=0, negatives=0, incomplete=0):
        for index in range(boxed + negatives + incomplete):
            Image.new('RGB', (64, 48), (index, 10, 20)).save(self.images / f'{index:03}.png')
        for index in range(boxed + negatives):
            document = (
                {'shapes': [{'label': 'tl', 'shape_type': 'rectangle', 'points': [[1, 2], [20, 30]]}]}
                if index < boxed
                else {'shapes': [], 'flags': {'xxtrain_detection_negative': True}}
            )
            (self.annotations / f'{index:03}.json').write_text(json.dumps(document), encoding='utf-8')

    def test_view_is_derived_without_a_status_file_and_gate_requires_50_boxed_images(self):
        self.create_workspace(boxed=49, negatives=1)
        view = self.service.view(17)
        self.assertEqual((50, 50, 49), (view.image_count, view.annotated_image_count, view.boxed_image_count))
        self.assertFalse(view.can_generate_detection_cache)
        self.assertFalse(view.detection_cache_ready)
        restarted = AnnotationService(
            self.config, WorkspaceData(self.workspace), self.cvat, RuntimeCache(self.config.runtime_dir)
        )
        self.assertEqual(view, restarted.view(17))
        self.assertFalse((self.root / 'state.json').exists())
        self.assertFalse(self.config.runtime_dir.exists())

    def test_cache_gate_rejects_empty_or_incomplete_inputs(self):
        self.assertFalse(self.service.view(17).can_generate_detection_cache)
        with self.assertRaises(PlatformError):
            self.service.generate_detection_cache(17)
        self.create_workspace(boxed=50, incomplete=1)
        self.assertFalse(self.service.view(17).can_generate_detection_cache)
        with self.assertRaises(PlatformError):
            self.service.generate_detection_cache(17)
        self.assertFalse(self.config.runtime_dir.exists())

    def test_cache_generation_publishes_dataset_and_changed_files_invalidate_readiness(self):
        self.create_workspace(boxed=50, negatives=1)
        self.assertTrue(self.service.view(17).can_generate_detection_cache)
        view = self.service.generate_detection_cache(17)
        self.assertTrue(view.detection_cache_ready)
        cache = self.config.runtime_dir / 'cache' / self.data.detection_fingerprint() / 'detect'
        labels = list((cache / 'workspace').glob('*.txt'))
        self.assertEqual(51, len(labels))
        self.assertEqual(50, sum(bool(path.read_text(encoding='utf-8')) for path in labels))
        self.assertEqual(view, self.service.generate_detection_cache(17))
        (self.annotations / '000.json').write_text('{"shapes": []}', encoding='utf-8')
        self.assertFalse(self.service.view(17).detection_cache_ready)
        self.assertFalse(self.service.view(17).can_generate_detection_cache)

    def test_same_input_reuses_only_an_unfinished_job(self):
        self.create_workspace(incomplete=1)
        first = self.service.begin_detection(17)
        restarted = AnnotationService(self.config, self.data, self.cvat, RuntimeCache(self.config.runtime_dir))
        self.assertEqual(first, restarted.begin_detection(17))
        self.assertEqual(1, self.cvat.create_task_calls)
        self.cvat.unfinished = False
        self.assertNotEqual(first, self.service.begin_detection(17))
        self.assertEqual(2, self.cvat.create_task_calls)

    def test_changed_formal_detection_creates_a_new_job_and_sync_requires_current_fingerprint(self):
        self.create_workspace(incomplete=1)
        first = self.service.begin_detection(17)
        (self.annotations / '000.json').write_text(
            '{"shapes": [], "flags": {"xxtrain_detection_negative": true}}', encoding='utf-8'
        )
        with self.assertRaisesRegex(PlatformError, 'job'):
            self.service.sync_detection(17)
        self.assertNotEqual(first, self.service.begin_detection(17))

    def test_failed_preparation_never_persists_a_job_and_retry_creates_a_fresh_task(self):
        self.create_workspace(incomplete=1)
        self.cvat.prepare_error = PlatformError('preparation failed')
        with self.assertRaisesRegex(PlatformError, 'preparation failed'):
            self.service.begin_detection(17)
        self.assertIsNone(self.runtime.job_for('detect', self.data.detection_fingerprint()))
        self.cvat.prepare_error = None
        self.assertEqual('/tasks/42/jobs/74', self.service.begin_detection(17))

    def test_job_check_failure_surfaces_without_creating_a_replacement(self):
        self.create_workspace(incomplete=1)
        self.service.begin_detection(17)
        self.cvat.check_error = PlatformError('CVAT unavailable')
        with self.assertRaisesRegex(PlatformError, 'CVAT unavailable'):
            self.service.begin_detection(17)
        self.assertEqual(1, self.cvat.create_task_calls)

    def test_sync_overwrites_detection_and_derives_fresh_counts(self):
        self.create_workspace(incomplete=1)
        annotation = self.annotations / '000.json'
        annotation.write_text(
            json.dumps(
                {
                    'marker': 'kept',
                    'shapes': [{'label': 'mask', 'shape_type': 'polygon', 'points': [[0, 0], [1, 0], [1, 1]]}],
                }
            ),
            encoding='utf-8',
        )
        self.service.begin_detection(17)
        self.cvat.results = (FrameResult('000', (DetectionBox(Bbox(label='cc', x1=3, y1=4, x2=22, y2=31)),)),)
        view = self.service.sync_detection(17)
        self.assertEqual((1, 1), (view.annotated_image_count, view.boxed_image_count))
        document = json.loads(annotation.read_text(encoding='utf-8'))
        self.assertEqual('kept', document['marker'])
        self.assertEqual(['mask', 'cc'], [shape['label'] for shape in document['shapes']])
        self.assertEqual(['000.json'], [path.name for path in self.annotations.iterdir()])

    def test_successful_sync_is_repeatable_and_completed_job_is_not_reopened(self):
        self.create_workspace(incomplete=1)
        first = self.service.begin_detection(17)
        self.cvat.results = (FrameResult('000', (DetectionBox(Bbox(label='tl', x1=1, y1=2, x2=20, y2=30)),)),)
        saved = self.service.sync_detection(17)
        before = (self.annotations / '000.json').read_bytes()
        self.assertEqual(saved, self.service.sync_detection(17))
        self.assertEqual(before, (self.annotations / '000.json').read_bytes())
        self.cvat.unfinished = False
        self.assertNotEqual(first, self.service.begin_detection(17))
        self.assertEqual(2, self.cvat.create_task_calls)

    def test_sync_save_failure_is_visible_and_lock_is_released(self):
        self.create_workspace(incomplete=1)
        self.service.begin_detection(17)
        self.cvat.results = (FrameResult('000', (DetectionBox(Bbox(label='tl', x1=1, y1=2, x2=20, y2=30)),)),)
        with patch('xxtrain.workspace_data.store.os.replace', side_effect=OSError('disk full')):
            with self.assertRaises(PlatformError):
                self.service.sync_detection(17)
        self.assertEqual(0, self.service.view(17).annotated_image_count)
        self.assertEqual(1, self.service.sync_detection(17).annotated_image_count)

    def test_upload_admits_staged_images_and_returns_current_view(self):
        staged = self.root / 'upload.png'
        Image.new('RGB', (64, 48), 'white').save(staged)
        view = self.service.upload(17, (staged,))
        self.assertEqual((1, 0, 0), (view.image_count, view.annotated_image_count, view.boxed_image_count))
        self.assertFalse(staged.exists())

    def test_partial_sync_failure_restores_documents_and_fingerprint_for_retry(self):
        self.create_workspace(boxed=2)
        before = {path: path.read_bytes() for path in self.annotations.iterdir()}
        fingerprint = self.data.detection_fingerprint()
        self.service.begin_detection(17)
        self.cvat.results = tuple(FrameResult(f'{index:03}', ()) for index in range(2))
        real_replace = os.replace

        def fail_second(source, destination):
            if Path(destination) == self.annotations / '001.json':
                raise OSError('disk full')
            return real_replace(source, destination)

        with patch('xxtrain.workspace_data.store.os.replace', side_effect=fail_second):
            with self.assertRaises(PlatformError):
                self.service.sync_detection(17)
        self.assertEqual(before, {path: path.read_bytes() for path in self.annotations.iterdir()})
        self.assertEqual(fingerprint, self.data.detection_fingerprint())
        self.assertEqual(0, self.service.sync_detection(17).boxed_image_count)
        self.assertEqual(1, self.cvat.create_task_calls)

    def test_writes_reject_concurrency_and_wrong_owner_without_side_effects(self):
        operations = (
            lambda user: self.service.upload(user, ()),
            self.service.begin_detection,
            self.service.sync_detection,
            self.service.generate_detection_cache,
        )
        for operation in (*operations, self.service.view):
            with self.assertRaises(PlatformAccessError):
                operation(99)
        with self.service.lock:
            for operation in operations:
                with self.assertRaisesRegex(PlatformError, 'progress'):
                    operation(17)
        self.assertEqual(0, self.cvat.create_task_calls)
        self.assertFalse(self.config.runtime_dir.exists())


if __name__ == '__main__':
    unittest.main()
