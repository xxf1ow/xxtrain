import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from PIL import Image

from xxtrain.business_tasks.point import point_task_definition
from xxtrain.platform.config import WorkspaceConfig
from xxtrain.platform.contracts import (
    AnnotationRecord,
    CvatBinding,
    EditAnnotation,
    EditFrameResult,
    JobRef,
    PlatformError,
    PreparedJob,
)
from xxtrain.platform.runtime import RuntimeCache
from xxtrain.platform.service import AnnotationService
from xxtrain.workspace_data import WorkspaceData
from xxtrain.workspace_data.repository import AnnotationRepository


class FakeEditCvat:
    def __init__(self) -> None:
        self.create_calls: list[tuple[str, tuple[str, ...], str]] = []
        self.frames = ()
        self.results = None
        self.unfinished = True

    def create_task(self, name, labels, policy):
        self.create_calls.append((name, labels, policy.cvat_type))
        return 100 + len(self.create_calls)

    def prepare_task(self, task_id, frames, user_id, policy):
        self.frames = frames
        sample_ids = tuple(dict.fromkeys(frame.mapping.image_id for frame in frames))
        bindings = tuple(
            CvatBinding(
                frame.mapping.image_id, 'tag' if annotation.kind == 'classification' else 'shape', index, annotation.id
            )
            for index, frame in enumerate(frames, start=1)
            for annotation in frame.annotations
        )
        return PreparedJob(JobRef(task_id, task_id + 100, sample_ids), bindings)

    def job_is_unfinished(self, ref):
        return self.unfinished

    def fetch_annotations(self, job, policy):
        if self.results is not None:
            return self.results
        return tuple(EditFrameResult(frame.frame_id, ()) for frame in job.frames)

    def job_path(self, ref):
        return f'/tasks/{ref.task_id}/jobs/{ref.job_id}'


class DownstreamServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / 'workspace'
        (self.workspace / 'images').mkdir(parents=True)
        self.config = WorkspaceConfig('line-3', 'Line 3', 17, self.workspace, self.root / 'runtime', 'http://cvat.test')
        self.data = WorkspaceData(self.workspace, point_task_definition())
        self.repository = AnnotationRepository(self.workspace / 'annotations.db', point_task_definition())
        self.runtime = RuntimeCache(self.config.runtime_dir)
        self.cvat = FakeEditCvat()
        self.service = AnnotationService(self.config, self.data, self.cvat, self.runtime)

    def _admit(self, count: int, *, start: int = 0):
        staged = [self._candidate(index) for index in range(start, start + count)]
        result = self.data.admit(tuple(staged))
        self.assertEqual(count, result.accepted_count)
        return self.data.images()

    def _candidate(self, index: int) -> Path:
        path = self.root / f'candidate-{index:03}.png'
        rng = random.Random(index + 103)
        image = Image.new('RGB', (32, 24))
        image.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(32 * 24)])
        image.save(path)
        return path

    def _box_all(self, samples):
        boxes = tuple(
            AnnotationRecord(uuid4(), sample.sample_id, 'detect', None, 'rectangle', 'tl', [[1, 1], [25, 20]])
            for sample in samples
        )
        self.repository.save_annotations(boxes)
        return boxes

    @staticmethod
    def _targets(view):
        return {target.id: target for target in view.targets}

    def test_gates_are_recomputed_from_original_and_crop_facts(self) -> None:
        samples = self._admit(49)
        boxes = self._box_all(samples)

        view = self.service.view(17)
        self.assertTrue(hasattr(view, 'targets'), 'WorkspaceView does not expose target facts')
        targets = self._targets(view)
        self.assertFalse(targets['classify'].can_annotate)
        self.assertEqual((49, 0), (targets['classify'].sample_count, targets['classify'].annotated_sample_count))

        samples = self._admit(1, start=49)
        final_box = self._box_all(samples[-1:])[0]
        boxes = (*boxes, final_box)
        targets = self._targets(self.service.view(17))
        self.assertTrue(targets['classify'].can_annotate)
        self.assertFalse(targets['classify'].can_generate_cache)
        self.assertTrue(targets['segment'].can_annotate)

        classifications = tuple(
            AnnotationRecord(uuid4(), box.image_id, 'classify', box.id, 'classification', 'tc', None) for box in boxes
        )
        self.repository.save_annotations(classifications)
        targets = self._targets(self.service.view(17))
        self.assertTrue(targets['classify'].can_generate_cache)
        self.assertTrue(targets['segment'].can_annotate)
        self.assertFalse(targets['classify'].cache_ready)

        lines = [
            AnnotationRecord(uuid4(), box.image_id, 'segment', box.id, 'polyline', '1', [[3, 3], [10, 10]])
            for box in boxes
        ]
        lines.append(
            AnnotationRecord(uuid4(), boxes[0].image_id, 'segment', boxes[0].id, 'polyline', '1', [[4, 3], [11, 10]])
        )
        self.repository.save_annotations(tuple(lines))
        targets = self._targets(self.service.view(17))
        self.assertEqual((50, 50), (targets['segment'].sample_count, targets['segment'].annotated_sample_count))
        self.assertTrue(targets['segment'].can_generate_cache)

        negative_sample = self._admit(1, start=50)[-1]
        self.repository.save_annotations(
            (AnnotationRecord(uuid4(), negative_sample.sample_id, 'detect', None, 'negative', None, None),)
        )
        targets = self._targets(self.service.view(17))
        self.assertEqual(50, targets['classify'].sample_count)
        self.assertTrue(targets['classify'].can_annotate)

        view = self.service.upload(17, (self._candidate(51),))
        targets = self._targets(view)
        self.assertFalse(targets['classify'].can_annotate)
        self.assertFalse(targets['segment'].can_annotate)

    def test_target_jobs_sync_missing_annotations_and_reuse_restart_mapping(self) -> None:
        boxes = self._box_all(self._admit(50))
        self.assertTrue(hasattr(self.service, 'begin_target'), 'shared target service methods are not implemented')

        annotation_url = self.service.begin_target(17, 'classify')

        self.assertEqual('/tasks/101/jobs/201?defaultWorkspace=TAGS', annotation_url)
        fingerprint = self.data.target_fingerprint('classify')
        job = self.runtime.edit_job_for('classify', fingerprint)
        self.assertIsNotNone(job)
        self.assertEqual(tuple(dict.fromkeys(box.image_id for box in boxes)), job.ref.sample_ids)
        self.assertEqual(tuple(str(box.id) for box in boxes), tuple(frame.frame_id for frame in job.frames))
        restarted = AnnotationService(
            self.config,
            WorkspaceData(self.workspace, point_task_definition()),
            self.cvat,
            RuntimeCache(self.config.runtime_dir),
        )
        self.assertEqual(annotation_url, restarted.begin_target(17, 'classify'))
        self.assertEqual(1, len(self.cvat.create_calls))

        view = restarted.sync_target(17, 'classify')

        self.assertEqual(0, self._targets(view)['classify'].annotated_sample_count)
        self.assertEqual((), self.repository.annotations(step_key='classify'))

    def test_invalid_completed_job_stays_correctable_without_overwriting_valid_data(self) -> None:
        boxes = self._box_all(self._admit(50))
        existing = tuple(
            AnnotationRecord(uuid4(), box.image_id, 'classify', box.id, 'classification', 'tl', None) for box in boxes
        )
        self.repository.save_annotations(existing)
        self.service.begin_target(17, 'classify')
        job = self.runtime.edit_job_for('classify', self.data.target_fingerprint('classify'))
        self.assertIsNotNone(job)
        self.cvat.unfinished = False
        self.cvat.results = (
            EditFrameResult(
                job.frames[0].frame_id,
                (
                    EditAnnotation(None, 'classification', 'tc', None, 501),
                    EditAnnotation(None, 'classification', 'cc', None, 502),
                ),
            ),
            *(EditFrameResult(frame.frame_id, ()) for frame in job.frames[1:]),
        )

        with self.assertRaises(PlatformError) as raised:
            self.service.sync_target(17, 'classify')

        self.assertEqual(existing, self.repository.annotations(step_key='classify'))
        self.assertEqual(job, self.runtime.edit_job_for('classify', self.data.target_fingerprint('classify')))
        self.assertEqual(
            '/tasks/101/jobs/201?defaultWorkspace=TAGS&frame=0', getattr(raised.exception, 'annotation_url', None)
        )
        self.assertNotIn(str(job.frames[0].parent_id), str(raised.exception))

    def test_stale_job_and_runtime_publication_failure_leave_database_authoritative(self) -> None:
        boxes = self._box_all(self._admit(50))
        self.service.begin_target(17, 'classify')
        old_fingerprint = self.data.target_fingerprint('classify')
        old_job = self.runtime.edit_job_for('classify', old_fingerprint)
        self.assertIsNotNone(old_job)
        changed = boxes[0]
        self.repository.save_annotations(
            (
                AnnotationRecord(
                    changed.id, changed.image_id, 'detect', None, 'rectangle', changed.label, [[2, 1], [25, 20]]
                ),
            )
        )

        with self.assertRaisesRegex(PlatformError, 'not ready'):
            self.service.sync_target(17, 'classify')
        self.assertEqual(old_job, self.runtime.edit_job_for('classify', old_fingerprint))

        self.service.begin_target(17, 'classify')
        current_job = self.runtime.edit_job_for('classify', self.data.target_fingerprint('classify'))
        self.assertIsNotNone(current_job)
        self.cvat.results = tuple(
            EditFrameResult(frame.frame_id, (EditAnnotation(None, 'classification', 'tl', None, index),))
            for index, frame in enumerate(current_job.frames, start=1)
        )
        before = self.repository.annotations(step_key='classify')
        with patch.object(self.runtime, 'remember_edit_job', side_effect=OSError('runtime disk full')):
            with self.assertRaisesRegex(PlatformError, '取回或保存'):
                self.service.sync_target(17, 'classify')
        self.assertEqual(before, self.repository.annotations(step_key='classify'))

        view = self.service.sync_target(17, 'classify')
        self.assertEqual(50, self._targets(view)['classify'].annotated_sample_count)

    def test_full_classification_opens_segment_without_cache_and_target_caches_publish(self) -> None:
        self._box_all(self._admit(50))
        self.service.begin_target(17, 'classify')
        classify_job = self.runtime.edit_job_for('classify', self.data.target_fingerprint('classify'))
        self.assertIsNotNone(classify_job)
        self.cvat.results = tuple(
            EditFrameResult(frame.frame_id, (EditAnnotation(None, 'classification', 'cc', None, index),))
            for index, frame in enumerate(classify_job.frames, start=1)
        )
        view = self.service.sync_target(17, 'classify')
        targets = self._targets(view)
        self.assertTrue(targets['segment'].can_annotate)
        self.assertFalse(targets['classify'].cache_ready)

        self.cvat.results = None
        segment_url = self.service.begin_target(17, 'segment')
        self.assertEqual('/tasks/102/jobs/202', segment_url)
        segment_job = self.runtime.edit_job_for('segment', self.data.target_fingerprint('segment'))
        self.assertIsNotNone(segment_job)
        self.cvat.results = tuple(
            EditFrameResult(frame.frame_id, (EditAnnotation(None, 'polyline', '1', [[8, 8], [16, 12]], index),))
            for index, frame in enumerate(segment_job.frames, start=1)
        )
        view = self.service.sync_target(17, 'segment')
        self.assertTrue(self._targets(view)['segment'].can_generate_cache)

        classify_view = self.service.generate_target_cache(17, 'classify')
        segment_view = self.service.generate_target_cache(17, 'segment')

        self.assertTrue(self._targets(classify_view)['classify'].cache_ready)
        self.assertTrue(self._targets(segment_view)['segment'].cache_ready)


if __name__ == '__main__':
    unittest.main()
