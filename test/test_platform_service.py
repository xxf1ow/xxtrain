import json
import random
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from PIL import Image

from xxtrain.business_tasks.loader import DEFAULT_TASK_ENTRY
from xxtrain.business_tasks.point import point_task_definition
from xxtrain.data import Bbox
from xxtrain.platform.config import WorkspaceConfig, load_config
from xxtrain.platform.contracts import (
    AnnotationRecord,
    CvatBinding,
    DetectionBox,
    EditJob,
    FrameMapping,
    FrameResult,
    JobRef,
    PlatformAccessError,
    PlatformError,
    PreparedJob,
)
from xxtrain.platform.runtime import RuntimeCache
from xxtrain.platform.service import AnnotationService
from xxtrain.workspace_data import WorkspaceData
from xxtrain.workspace_data.repository import AnnotationRepository


class FakeCvatClient:
    def __init__(self):
        self.unfinished = True
        self.create_task_calls = 0
        self.prepare_error = None
        self.check_error = None
        self.results = None
        self.images = ()
        self.prepared = None

    def create_task(self, name, labels):
        self.create_task_calls += 1
        return 40 + self.create_task_calls

    def prepare_task(self, task_id, images, user_id):
        if self.prepare_error:
            raise self.prepare_error
        self.images = images
        ref = JobRef(task_id, task_id + 32, tuple(image.sample_id for image in images))
        image_boxes = ((image, box) for image in images for box in image.boxes)
        bindings = tuple(
            CvatBinding(image.sample_id, 'shape', 100 + index, box.geometry.id)
            for index, (image, box) in enumerate(image_boxes, start=1)
        )
        self.prepared = PreparedJob(ref, bindings)
        return self.prepared

    def job_is_unfinished(self, ref):
        if self.check_error:
            raise self.check_error
        return self.unfinished

    def fetch_detection(self, ref):
        if self.results is not None:
            return self.results
        binding_by_annotation = {binding.annotation_id: binding.object_id for binding in self.prepared.bindings}
        return tuple(
            FrameResult(
                image.sample_id,
                tuple(
                    DetectionBox(box.geometry, box.extra, binding_by_annotation[box.geometry.id]) for box in image.boxes
                ),
            )
            for image in self.images
        )

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
            self.assertEqual(DEFAULT_TASK_ENTRY, config.task_entry)
            payload['task_entry'] = 'test.task_definitions:synthetic_task_definition'
            config_path.write_text(json.dumps(payload), encoding='utf-8')
            self.assertEqual(payload['task_entry'], load_config(config_path).task_entry)
            for value in ('', 3):
                with self.subTest(task_entry=value):
                    config_path.write_text(json.dumps(payload | {'task_entry': value}), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        load_config(config_path)
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
            self.assertIsNone(cache.edit_job_for('detect', 'a' * 64))
            self.assertFalse(cache.has_detection_cache('a' * 64))

    def test_runtime_edit_job_map_round_trips_repeated_original_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent_a = uuid4()
            parent_b = uuid4()
            parent_c = uuid4()
            ref = JobRef(41, 73, ('original-a', 'original-b'))
            job = EditJob(
                ref,
                (
                    FrameMapping(str(parent_a), 'original-a', parent_a, (0, 1, 6, 8)),
                    FrameMapping(str(parent_b), 'original-a', parent_b, (2, 3, 9, 10)),
                    FrameMapping(str(parent_c), 'original-b', parent_c, (1, 2, 7, 9)),
                ),
            )

            RuntimeCache(root).remember_edit_job('segment', 'fingerprint', job)

            restarted = RuntimeCache(root)
            self.assertEqual(job, restarted.edit_job_for('segment', 'fingerprint'))
            self.assertEqual(ref, restarted.job_for('segment', 'fingerprint'))
            entry = json.loads((root / 'jobs.json').read_text(encoding='utf-8'))['segment']['fingerprint']
            self.assertEqual({'task_id', 'job_id', 'sample_ids', 'frames'}, set(entry))

    def test_runtime_edit_job_map_rejects_invalid_source_mappings(self):
        parent = uuid4()
        valid_frame = {
            'frame_id': str(parent),
            'image_id': 'original-a',
            'parent_id': str(parent),
            'bounds': [0, 1, 6, 8],
        }
        invalid_frames = {
            'duplicate frame IDs': [valid_frame, valid_frame],
            'unknown original': [valid_frame | {'image_id': 'missing'}],
            'malformed bounds': [valid_frame | {'bounds': [0, 1, 0, 8]}],
            'malformed parent': [valid_frame | {'parent_id': 'not-a-uuid'}],
        }
        for name, frames in invalid_frames.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                payload = {
                    'segment': {
                        'fingerprint': {'task_id': 41, 'job_id': 73, 'sample_ids': ['original-a'], 'frames': frames}
                    }
                }
                (Path(directory) / 'jobs.json').write_text(json.dumps(payload), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Runtime edit job map'):
                    RuntimeCache(Path(directory)).edit_job_for('segment', 'fingerprint')

    def test_runtime_rejects_nonunique_original_job_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = uuid4()
            job = EditJob(
                JobRef(41, 73, ('original-a', 'original-a')),
                (FrameMapping(str(parent), 'original-a', parent, (0, 1, 6, 8)),),
            )

            with self.assertRaisesRegex(ValueError, 'unique original image IDs'):
                RuntimeCache(Path(directory)).remember_edit_job('segment', 'fingerprint', job)

    def test_runtime_forgets_every_fingerprint_for_only_the_invalidated_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = RuntimeCache(Path(directory))
            detect = JobRef(1, 2, ('a',))
            cache.remember_job('detect', 'input', detect)
            cache.remember_job('classify', 'old', JobRef(3, 4, ('a',)))
            classify_parent = uuid4()
            classify = EditJob(
                JobRef(5, 6, ('a',)), (FrameMapping(str(classify_parent), 'a', classify_parent, (0, 0, 4, 5)),)
            )
            segment_parent = uuid4()
            segment = EditJob(
                JobRef(7, 8, ('a',)), (FrameMapping(str(segment_parent), 'a', segment_parent, (0, 0, 4, 5)),)
            )
            cache.remember_edit_job('classify', 'new', classify)
            cache.remember_edit_job('segment', 'input', segment)

            cache.forget_targets(frozenset({'classify', 'segment'}))

            restarted = RuntimeCache(Path(directory))
            self.assertEqual(detect, restarted.job_for('detect', 'input'))
            self.assertIsNone(restarted.job_for('classify', 'old'))
            self.assertIsNone(restarted.job_for('classify', 'new'))
            self.assertIsNone(restarted.job_for('segment', 'input'))
            self.assertIsNone(restarted.edit_job_for('classify', 'new'))
            self.assertIsNone(restarted.edit_job_for('segment', 'input'))

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
        self.images.mkdir(parents=True)
        self.config = WorkspaceConfig('line-3', 'Line 3', 17, self.workspace, self.root / 'runtime', 'http://cvat.test')
        self.data = WorkspaceData(self.workspace)
        self.repository = AnnotationRepository(self.workspace / 'annotations.db', point_task_definition())
        self.runtime = RuntimeCache(self.config.runtime_dir)
        self.cvat = FakeCvatClient()
        self.service = AnnotationService(self.config, self.data, self.cvat, self.runtime)

    def create_workspace(self, *, boxed=0, negatives=0, incomplete=0):
        staged = []
        for index in range(boxed + negatives + incomplete):
            path = self.root / f'candidate-{index:03}.png'
            rng = random.Random(index + 19)
            with Image.new('RGB', (64, 48)) as image:
                for x in range(64):
                    for y in range(48):
                        image.putpixel((x, y), (rng.randrange(256), rng.randrange(256), rng.randrange(256)))
                image.save(path)
            staged.append(path)
        result = self.data.admit(tuple(staged))
        self.assertEqual(len(staged), result.accepted_count)
        samples = self.data.images()
        records = []
        for sample in samples[:boxed]:
            records.append(
                AnnotationRecord(uuid4(), sample.sample_id, 'detect', None, 'rectangle', 'tl', [[1, 2], [20, 30]])
            )
        for sample in samples[boxed : boxed + negatives]:
            records.append(AnnotationRecord(uuid4(), sample.sample_id, 'detect', None, 'negative', None, None))
        self.repository.save_annotations(tuple(records))
        return samples

    def test_view_is_derived_from_sqlite_and_gate_requires_50_boxed_images(self):
        self.create_workspace(boxed=49, negatives=1)
        view = self.service.view(17)
        self.assertEqual((50, 50, 49), (view.image_count, view.annotated_image_count, view.boxed_image_count))
        self.assertFalse(view.can_generate_detection_cache)
        restarted = AnnotationService(
            self.config, WorkspaceData(self.workspace), self.cvat, RuntimeCache(self.config.runtime_dir)
        )
        self.assertEqual(view, restarted.view(17))

    def test_fifty_boxes_on_one_image_do_not_satisfy_the_boxed_image_gate(self):
        (sample,) = self.create_workspace(boxed=1)
        self.repository.save_annotations(
            tuple(
                AnnotationRecord(
                    uuid4(), sample.sample_id, 'detect', None, 'rectangle', 'tl', [[index + 1, 2], [60, 40]]
                )
                for index in range(49)
            )
        )

        view = self.service.view(17)

        self.assertEqual((1, 1, 1), (view.image_count, view.annotated_image_count, view.boxed_image_count))
        self.assertFalse(view.can_generate_detection_cache)
        with self.assertRaisesRegex(PlatformError, 'at least 50 boxed images'):
            self.service.generate_detection_cache(17)

    def test_one_incomplete_image_blocks_an_otherwise_eligible_workspace(self):
        self.create_workspace(boxed=50, incomplete=1)

        view = self.service.view(17)

        self.assertEqual((51, 50, 50), (view.image_count, view.annotated_image_count, view.boxed_image_count))
        self.assertFalse(view.can_generate_detection_cache)
        with self.assertRaisesRegex(PlatformError, 'all images annotated'):
            self.service.generate_detection_cache(17)

    def test_sync_keeps_fifty_boxed_plus_one_incomplete_image_ineligible(self):
        self.create_workspace(boxed=50, incomplete=1)
        self.service.begin_detection(17)

        view = self.service.sync_detection(17)

        self.assertEqual((51, 50, 50), (view.image_count, view.annotated_image_count, view.boxed_image_count))
        self.assertFalse(view.can_generate_detection_cache)
        with self.assertRaisesRegex(PlatformError, 'all images annotated'):
            self.service.generate_detection_cache(17)

    def test_cache_generation_publishes_registered_database_samples(self):
        self.create_workspace(boxed=50, negatives=1)
        view = self.service.generate_detection_cache(17)
        self.assertTrue(view.detection_cache_ready)
        output = self.config.runtime_dir / 'cache' / self.data.detection_fingerprint() / 'detect'
        labels = list((output / 'workspace').glob('*.txt'))
        self.assertEqual(51, len(labels))
        self.assertEqual(50, sum(bool(path.read_text(encoding='utf-8')) for path in labels))

    def test_same_input_reuses_only_an_unfinished_bound_job(self):
        self.create_workspace(incomplete=1)
        first = self.service.begin_detection(17)
        self.assertEqual(self.cvat.prepared.bindings, self.repository.bindings(self.cvat.prepared.ref))
        restarted = AnnotationService(self.config, self.data, self.cvat, RuntimeCache(self.config.runtime_dir))
        self.assertEqual(first, restarted.begin_detection(17))
        self.assertEqual(1, self.cvat.create_task_calls)
        self.cvat.unfinished = False
        self.assertNotEqual(first, self.service.begin_detection(17))

    def test_failed_preparation_or_binding_never_publishes_an_editable_job(self):
        self.create_workspace(boxed=1)
        fingerprint = self.data.detection_fingerprint()
        self.cvat.prepare_error = PlatformError('preparation failed')
        with self.assertRaisesRegex(PlatformError, 'preparation failed'):
            self.service.begin_detection(17)
        self.assertIsNone(self.runtime.job_for('detect', fingerprint))
        self.cvat.prepare_error = None
        with patch.object(self.data, 'bind_job', side_effect=PlatformError('binding failed')):
            with self.assertRaisesRegex(PlatformError, 'binding failed'):
                self.service.begin_detection(17)
        self.assertIsNone(self.runtime.job_for('detect', fingerprint))

    def test_sync_prepublishes_result_fingerprint_and_survives_process_restart(self):
        (sample,) = self.create_workspace(incomplete=1)
        self.service.begin_detection(17)
        ref = self.cvat.prepared.ref
        self.cvat.results = (
            FrameResult(sample.sample_id, (DetectionBox(Bbox(label='cc', x1=3, y1=4, x2=22, y2=31), cvat_id=501),)),
        )

        view = self.service.sync_detection(17)

        self.assertEqual((1, 1), (view.annotated_image_count, view.boxed_image_count))
        fingerprint = self.data.detection_fingerprint()
        self.assertEqual(ref, RuntimeCache(self.config.runtime_dir).job_for('detect', fingerprint))
        restarted = AnnotationService(
            self.config, WorkspaceData(self.workspace), self.cvat, RuntimeCache(self.config.runtime_dir)
        )
        self.assertEqual(view, restarted.view(17))
        self.assertEqual('/tasks/41/jobs/73', restarted.begin_detection(17))

    def test_sync_invalidates_empty_downstream_jobs_and_reverting_does_not_restore_them(self):
        (sample,) = self.create_workspace(boxed=1)
        self.service.begin_detection(17)
        original = self.data.images()[0].boxes[0]
        old_classify = JobRef(80, 81, (sample.sample_id,))
        old_segment = JobRef(82, 83, (sample.sample_id,))
        self.runtime.remember_job('classify', 'old', old_classify)
        self.runtime.remember_job('segment', 'old', old_segment)
        self.cvat.results = (
            FrameResult(
                sample.sample_id,
                (DetectionBox(Bbox(label=original.geometry.label, x1=4, y1=2, x2=20, y2=30), cvat_id=101),),
            ),
        )
        self.service.sync_detection(17)
        self.assertIsNone(self.runtime.job_for('classify', 'old'))
        self.assertIsNone(self.runtime.job_for('segment', 'old'))

        self.cvat.results = (FrameResult(sample.sample_id, (DetectionBox(original.geometry, cvat_id=101),)),)
        self.service.sync_detection(17)
        self.assertIsNone(self.runtime.job_for('classify', 'old'))
        self.assertIsNone(self.runtime.job_for('segment', 'old'))

    def test_duplicate_upload_does_not_invalidate_downstream_jobs(self):
        (sample,) = self.create_workspace(incomplete=1)
        downstream = JobRef(80, 81, (sample.sample_id,))
        self.runtime.remember_job('classify', 'input', downstream)
        duplicate = self.root / 'duplicate.png'
        duplicate.write_bytes(sample.image_path.read_bytes())

        self.service.upload(17, (duplicate,))

        self.assertEqual(downstream, self.runtime.job_for('classify', 'input'))

    def test_forget_and_remember_failures_do_not_modify_annotations(self):
        (sample,) = self.create_workspace(boxed=1)
        self.service.begin_detection(17)
        before = self.repository.annotations()
        self.cvat.results = (
            FrameResult(sample.sample_id, (DetectionBox(Bbox(label='tl', x1=4, y1=2, x2=20, y2=30), cvat_id=101),)),
        )
        with patch.object(self.runtime, 'forget_targets', side_effect=OSError('runtime disk full')):
            with self.assertRaises(PlatformError):
                self.service.sync_detection(17)
        self.assertEqual(before, self.repository.annotations())
        with patch.object(self.runtime, 'remember_job', side_effect=OSError('runtime disk full')):
            with self.assertRaises(PlatformError):
                self.service.sync_detection(17)
        self.assertEqual(before, self.repository.annotations())

    def test_database_mapping_failure_preserves_data_and_retry_does_not_duplicate(self):
        (sample,) = self.create_workspace(incomplete=1)
        self.service.begin_detection(17)
        ref = self.cvat.prepared.ref
        before = self.repository.annotations()
        self.cvat.results = (
            FrameResult(sample.sample_id, (DetectionBox(Bbox(label='tc', x1=3, y1=4, x2=22, y2=31), cvat_id=501),)),
        )

        def fail_transaction(changes, *, ref=None, bindings=()):
            raise PlatformError('Annotation storage operation failed') from sqlite3.OperationalError('injected')

        with patch.object(self.data._repository, 'apply_changes', side_effect=fail_transaction):
            with self.assertRaisesRegex(PlatformError, '取回或保存'):
                self.service.sync_detection(17)
        self.assertEqual(before, self.repository.annotations())
        self.assertEqual((), self.repository.bindings(ref))

        view = self.service.sync_detection(17)

        self.assertEqual(1, view.boxed_image_count)
        self.assertEqual(1, len(self.repository.annotations(step_key='detect')))
        self.assertEqual(1, len(self.repository.bindings(ref)))

    def test_repository_calls_from_another_thread_do_not_use_thread_affine_connections(self):
        self.create_workspace(incomplete=1)
        results = []
        thread = threading.Thread(target=lambda: results.append(self.service.view(17)))
        thread.start()
        thread.join()

        self.assertEqual(1, results[0].image_count)

    def test_legacy_detection_and_target_writes_share_owner_and_operation_lock(self):
        operations = (
            ('upload', lambda user: self.service.upload(user, ())),
            ('legacy detection start', self.service.begin_detection),
            ('legacy detection sync', self.service.sync_detection),
            ('legacy detection cache', self.service.generate_detection_cache),
            ('target start', lambda user: self.service.begin_target(user, 'classify')),
            ('target sync', lambda user: self.service.sync_target(user, 'segment')),
            ('target cache', lambda user: self.service.generate_target_cache(user, 'classify')),
        )
        for name, operation in (*operations, ('view', self.service.view)):
            with self.subTest(name=name, gate='owner'), self.assertRaises(PlatformAccessError):
                operation(99)
        with self.service.lock:
            for name, operation in operations:
                with self.subTest(name=name, gate='lock'), self.assertRaisesRegex(PlatformError, 'progress'):
                    operation(17)
        self.assertEqual(0, self.cvat.create_task_calls)

    def test_server_side_edit_guard_covers_upload_begin_and_sync_but_not_cache_generation(self):
        guarded = AnnotationService(
            self.config,
            self.data,
            self.cvat,
            self.runtime,
            require_editable=lambda workspace_id: (_ for _ in ()).throw(PlatformError('training active')),
        )
        operations = (
            ('upload', lambda: guarded.upload(17, ())),
            ('legacy begin', lambda: guarded.begin_detection(17)),
            ('target begin', lambda: guarded.begin_target(17, 'classify')),
            ('legacy sync', lambda: guarded.sync_detection(17)),
            ('target sync', lambda: guarded.sync_target(17, 'segment')),
        )
        for name, operation in operations:
            with self.subTest(name=name), self.assertRaisesRegex(PlatformError, 'training active'):
                operation()

        self.create_workspace(boxed=50)
        guarded.generate_target_cache(17, 'detect')


if __name__ == '__main__':
    unittest.main()
