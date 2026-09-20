import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from uuid import UUID

from PIL import Image

from xxtrain.business_tasks.point import point_task_definition
from xxtrain.platform.contracts import (
    AnnotationRecord,
    CvatBinding,
    EditAnnotation,
    EditFrameResult,
    EditJob,
    ImageRecord,
    JobRef,
    PlatformError,
    PreparedJob,
    TargetSummary,
)
from xxtrain.workspace_data import WorkspaceData
from xxtrain.workspace_data.repository import AnnotationRepository


class PlatformDownstreamDataTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix='xxtrain-downstream-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace_root = self.root / 'workspace'
        images_root = self.workspace_root / 'images'
        images_root.mkdir(parents=True)
        image_path = images_root / 'original.png'
        with Image.new('RGB', (100, 80)) as image:
            for y in range(image.height):
                for x in range(image.width):
                    image.putpixel((x, y), (x, y, (x + y) % 256))
            image.save(image_path)

        self.data = WorkspaceData(self.workspace_root, point_task_definition())
        self.repository = AnnotationRepository(self.workspace_root / 'annotations.db', point_task_definition())
        self.image = ImageRecord('a' * 64, 'images/original.png', 100, 80, 0)
        self.repository.register_images((self.image,))
        self.first = AnnotationRecord(
            UUID('10000000-0000-0000-0000-000000000001'),
            self.image.id,
            'detect',
            None,
            'rectangle',
            'Point',
            [[10, 20], [50, 60]],
        )
        self.second = AnnotationRecord(
            UUID('10000000-0000-0000-0000-000000000002'),
            self.image.id,
            'detect',
            None,
            'rectangle',
            'Point',
            [[60, 10], [90, 40]],
        )
        self.repository.save_annotations((self.first, self.second))

    def add_targets(self) -> tuple[AnnotationRecord, AnnotationRecord, AnnotationRecord, AnnotationRecord]:
        category = AnnotationRecord(
            UUID('20000000-0000-0000-0000-000000000001'),
            self.image.id,
            'classify',
            self.first.id,
            'classification',
            'tl',
            None,
        )
        other_category = AnnotationRecord(
            UUID('20000000-0000-0000-0000-000000000002'),
            self.image.id,
            'classify',
            self.second.id,
            'classification',
            'cc',
            None,
        )
        first_line = AnnotationRecord(
            UUID('30000000-0000-0000-0000-000000000001'),
            self.image.id,
            'segment',
            self.first.id,
            'polyline',
            '1',
            [[11.0, 22.0], [13.0, 24.0]],
        )
        second_line = AnnotationRecord(
            UUID('30000000-0000-0000-0000-000000000002'),
            self.image.id,
            'segment',
            self.first.id,
            'polyline',
            '1',
            [[14.0, 25.0], [16.0, 27.0]],
        )
        self.repository.save_annotations((category, other_category, first_line, second_line))
        return category, other_category, first_line, second_line

    def edit_job(self, target: str, job_id: int) -> tuple[EditJob, tuple]:
        frames = self.data.target_frames(target, self.root / f'runtime-{target}-{job_id}')
        sample_ids = tuple(dict.fromkeys(frame.mapping.image_id for frame in frames))
        return EditJob(JobRef(job_id - 1, job_id, sample_ids), tuple(frame.mapping for frame in frames)), frames

    def bind_existing(self, job: EditJob, object_type: str, ids: tuple[int, ...], frames: tuple) -> None:
        annotations = tuple(annotation for frame in frames for annotation in frame.annotations)
        parents = {
            annotation.id: frame.mapping.image_id
            for frame in frames
            for annotation in frame.annotations
            if annotation.id is not None
        }
        bindings = tuple(
            CvatBinding(parents[annotation.id], object_type, object_id, annotation.id)
            for annotation, object_id in zip(annotations, ids, strict=True)
            if annotation.id is not None
        )
        self.data.bind_job(PreparedJob(job.ref, bindings))

    def test_projects_target_frames_summaries_and_local_segment_geometry(self) -> None:
        category, other_category, first_line, second_line = self.add_targets()

        classify = self.data.target_frames('classify', self.root / 'classify')
        segment = self.data.target_frames('segment', self.root / 'segment')

        self.assertEqual((self.first.id, self.second.id), tuple(frame.mapping.parent_id for frame in classify))
        self.assertEqual(
            ((category.id,), (other_category.id,)), tuple(tuple(a.id for a in f.annotations) for f in classify)
        )
        self.assertEqual((None, None), tuple(frame.annotations[0].geometry for frame in classify))
        self.assertEqual((first_line.id, second_line.id), tuple(a.id for a in segment[0].annotations))
        self.assertEqual(
            ([[1.0, 2.0], [3.0, 4.0]], [[4.0, 5.0], [6.0, 7.0]]),
            tuple(annotation.geometry for annotation in segment[0].annotations),
        )
        self.assertEqual((), segment[1].annotations)
        self.assertEqual(TargetSummary(2, 2, 2), self.data.target_summary('classify'))
        self.assertEqual(TargetSummary(2, 1, 1), self.data.target_summary('segment'))

    def test_segment_sync_copies_preserves_and_deletes_native_identities(self) -> None:
        _, _, first_line, second_line = self.add_targets()
        job, frames = self.edit_job('segment', 20)
        self.bind_existing(job, 'shape', (101, 102), frames)
        results = (
            EditFrameResult(
                frames[0].mapping.frame_id,
                (
                    EditAnnotation(None, 'polyline', '1', [[1, 2], [3, 4]], 101),
                    EditAnnotation(None, 'polyline', '1', [[4, 5], [6, 7]], 102),
                    EditAnnotation(None, 'polyline', '1', [[7, 8], [9, 10]], 103),
                ),
            ),
            EditFrameResult(frames[1].mapping.frame_id, ()),
        )

        sync = self.data.prepare_target_sync('segment', job, results)
        self.data.commit_target_sync(job, sync)

        records = self.repository.annotations(step_key='segment')
        self.assertEqual(
            {first_line.id, second_line.id},
            {record.id for record in records if record.id in {first_line.id, second_line.id}},
        )
        copied = next(record for record in records if record.id not in {first_line.id, second_line.id})
        self.assertEqual(self.first.id, copied.parent_id)
        self.assertEqual([[17.0, 28.0], [19.0, 30.0]], copied.geometry)
        self.assertEqual(sync.fingerprint, self.data.target_fingerprint('segment'))

        delete_sync = self.data.prepare_target_sync(
            'segment',
            job,
            (
                EditFrameResult(
                    frames[0].mapping.frame_id, (EditAnnotation(None, 'polyline', '1', [[1, 2], [3, 4]], 101),)
                ),
                EditFrameResult(frames[1].mapping.frame_id, ()),
            ),
        )
        self.data.commit_target_sync(job, delete_sync)

        self.assertEqual(
            (first_line.id,), tuple(record.id for record in self.repository.annotations(step_key='segment'))
        )

    def test_classification_change_and_noop_preserve_parallel_segment_lines(self) -> None:
        category, other_category, first_line, second_line = self.add_targets()
        other_line = AnnotationRecord(
            UUID('30000000-0000-0000-0000-000000000003'),
            self.image.id,
            'segment',
            self.second.id,
            'polyline',
            '1',
            [[61.0, 12.0], [63.0, 14.0]],
        )
        self.repository.save_annotations((other_line,))
        job, frames = self.edit_job('classify', 30)
        self.bind_existing(job, 'tag', (101, 102), frames)
        unchanged = tuple(
            EditFrameResult(
                frame.mapping.frame_id,
                (EditAnnotation(None, 'classification', frame.annotations[0].label, None, object_id),),
            )
            for frame, object_id in zip(frames, (101, 102), strict=True)
        )

        no_op = self.data.prepare_target_sync('classify', job, unchanged)
        self.assertEqual(
            ((), frozenset(), frozenset()),
            (no_op.changes.upserts, no_op.changes.delete_ids, no_op.changes.invalidated_steps),
        )
        self.data.commit_target_sync(job, no_op)
        self.assertEqual(
            {first_line.id, second_line.id, other_line.id},
            {record.id for record in self.repository.annotations(step_key='segment')},
        )

        changed = (
            EditFrameResult(frames[0].mapping.frame_id, (EditAnnotation(None, 'classification', 'tc', None, 101),)),
            unchanged[1],
        )
        sync = self.data.prepare_target_sync('classify', job, changed)
        self.data.commit_target_sync(job, sync)

        self.assertEqual('tc', self.repository.annotations(step_key='classify')[0].label)
        self.assertEqual(other_category.id, self.repository.annotations(step_key='classify')[1].id)
        self.assertEqual(
            {first_line.id, second_line.id, other_line.id},
            {record.id for record in self.repository.annotations(step_key='segment')},
        )
        self.assertEqual(category.id, self.repository.annotations(step_key='classify')[0].id)

    def test_native_numeric_id_is_scoped_by_job_and_object_type(self) -> None:
        category, _, first_line, _ = self.add_targets()
        classify_job, classify_frames = self.edit_job('classify', 40)
        segment_job, segment_frames = self.edit_job('segment', 41)
        self.bind_existing(classify_job, 'tag', (101, 102), classify_frames)
        self.bind_existing(segment_job, 'shape', (101, 102), segment_frames)

        classify_sync = self.data.prepare_target_sync(
            'classify',
            classify_job,
            tuple(
                EditFrameResult(
                    frame.mapping.frame_id,
                    (EditAnnotation(None, 'classification', frame.annotations[0].label, None, object_id),),
                )
                for frame, object_id in zip(classify_frames, (101, 102), strict=True)
            ),
        )
        segment_sync = self.data.prepare_target_sync(
            'segment',
            segment_job,
            (
                EditFrameResult(
                    segment_frames[0].mapping.frame_id,
                    (
                        EditAnnotation(None, 'polyline', '1', [[1, 2], [3, 4]], 101),
                        EditAnnotation(None, 'polyline', '1', [[4, 5], [6, 7]], 102),
                    ),
                ),
                EditFrameResult(segment_frames[1].mapping.frame_id, ()),
            ),
        )

        self.assertEqual(category.id, classify_sync.bindings[0].annotation_id)
        self.assertEqual(first_line.id, segment_sync.bindings[0].annotation_id)

    def test_sync_rejects_incomplete_results_invalid_content_and_cross_frame_moves(self) -> None:
        self.add_targets()
        job, frames = self.edit_job('segment', 50)
        self.bind_existing(job, 'shape', (101, 102), frames)

        with self.assertRaisesRegex(ValueError, 'cover'):
            self.data.prepare_target_sync(
                'segment', job, (EditFrameResult(frames[0].mapping.frame_id, frames[0].annotations),)
            )
        invalid = (
            EditFrameResult(
                frames[0].mapping.frame_id, (EditAnnotation(None, 'polyline', '1', [[1, 2], [41, 4]], 101),)
            ),
            EditFrameResult(frames[1].mapping.frame_id, ()),
        )
        with self.assertRaisesRegex(ValueError, 'bounds'):
            self.data.prepare_target_sync('segment', job, invalid)
        moved = (
            EditFrameResult(frames[0].mapping.frame_id, ()),
            EditFrameResult(
                frames[1].mapping.frame_id, (EditAnnotation(None, 'polyline', '1', [[1, 2], [3, 4]], 101),)
            ),
        )
        with self.assertRaisesRegex(ValueError, 'frame'):
            self.data.prepare_target_sync('segment', job, moved)

        classify_job, classify_frames = self.edit_job('classify', 51)
        multiple = (
            EditFrameResult(
                classify_frames[0].mapping.frame_id,
                (
                    EditAnnotation(None, 'classification', 'tl', None, 201),
                    EditAnnotation(None, 'classification', 'tc', None, 202),
                ),
            ),
            EditFrameResult(classify_frames[1].mapping.frame_id, ()),
        )
        with self.assertRaisesRegex(ValueError, 'at most 1'):
            self.data.prepare_target_sync('classify', classify_job, multiple)

    def test_old_mapping_cannot_authorize_same_content_replacement(self) -> None:
        job, frames = self.edit_job('classify', 60)
        replacement = AnnotationRecord(
            UUID('10000000-0000-0000-0000-000000000003'),
            self.image.id,
            'detect',
            None,
            'rectangle',
            self.first.label,
            self.first.geometry,
        )
        self.repository.save_annotations((replacement,), delete_ids=frozenset({self.first.id}))
        empty_results = tuple(EditFrameResult(frame.mapping.frame_id, ()) for frame in frames)

        with self.assertRaisesRegex(ValueError, 'current'):
            self.data.prepare_target_sync('classify', job, empty_results)

    def test_restart_recovers_bindings_and_empty_sync_is_incomplete(self) -> None:
        category, _, _, _ = self.add_targets()
        job, frames = self.edit_job('classify', 70)
        self.bind_existing(job, 'tag', (101, 102), frames)
        reopened = WorkspaceData(self.workspace_root, point_task_definition())
        results = (
            EditFrameResult(frames[0].mapping.frame_id, ()),
            EditFrameResult(
                frames[1].mapping.frame_id,
                (EditAnnotation(None, 'classification', frames[1].annotations[0].label, None, 102),),
            ),
        )

        sync = reopened.prepare_target_sync('classify', job, results)
        reopened.commit_target_sync(job, sync)

        self.assertNotIn(category.id, {record.id for record in self.repository.annotations(step_key='classify')})
        self.assertEqual(TargetSummary(2, 1, 1), reopened.target_summary('classify'))

    def test_transaction_failure_keeps_annotations_and_bindings_equal(self) -> None:
        self.add_targets()
        job, frames = self.edit_job('segment', 80)
        self.bind_existing(job, 'shape', (101, 102), frames)
        results = (
            EditFrameResult(
                frames[0].mapping.frame_id,
                (
                    EditAnnotation(None, 'polyline', '1', [[1, 2], [3, 4]], 101),
                    EditAnnotation(None, 'polyline', '1', [[8, 9], [10, 11]], 999),
                ),
            ),
            EditFrameResult(frames[1].mapping.frame_id, ()),
        )
        sync = self.data.prepare_target_sync('segment', job, results)
        before_annotations = self.repository.annotations()
        before_bindings = self.repository.bindings(job.ref)
        with closing(sqlite3.connect(self.workspace_root / 'annotations.db')) as connection:
            connection.execute(
                """CREATE TRIGGER reject_downstream_binding BEFORE INSERT ON cvat_annotation_map
                WHEN NEW.object_id = 999
                BEGIN SELECT RAISE(ABORT, 'injected downstream failure'); END"""
            )

        with self.assertRaises(PlatformError):
            self.data.commit_target_sync(job, sync)

        self.assertEqual(before_annotations, self.repository.annotations())
        self.assertEqual(before_bindings, self.repository.bindings(job.ref))

    def test_fingerprint_tracks_inputs_and_is_independent_of_record_order_and_runtime_files(self) -> None:
        self.add_targets()
        before = self.data.target_fingerprint('segment')
        self.data.target_frames('segment', self.root / 'runtime-one')
        self.data.target_frames('segment', self.root / 'runtime-two')
        self.assertEqual(before, self.data.target_fingerprint('segment'))

        ordered = self.repository.annotations()
        second_root = self.root / 'second-workspace'
        (second_root / 'images').mkdir(parents=True)
        (second_root / 'images' / 'original.png').write_bytes(
            (self.workspace_root / 'images' / 'original.png').read_bytes()
        )
        other_repository = AnnotationRepository(second_root / 'annotations.db', point_task_definition())
        other_repository.register_images((self.image,))
        other_repository.save_annotations(tuple(reversed(ordered)))
        self.assertEqual(before, WorkspaceData(second_root, point_task_definition()).target_fingerprint('segment'))

        third_root = self.root / 'third-workspace'
        (third_root / 'images').mkdir(parents=True)
        (third_root / 'images' / 'original.png').write_bytes(
            (self.workspace_root / 'images' / 'original.png').read_bytes()
        )
        third_repository = AnnotationRepository(third_root / 'annotations.db', point_task_definition())
        third_repository.register_images((self.image, ImageRecord('b' * 64, 'images/second.png', 100, 80, 1)))
        third_repository.save_annotations(ordered)
        self.assertEqual(before, WorkspaceData(third_root, point_task_definition()).target_fingerprint('segment'))

        category = next(
            record for record in self.repository.annotations(step_key='classify') if record.parent_id == self.first.id
        )
        self.repository.save_annotations((replace(category, label='tc'),))
        self.assertEqual(before, self.data.target_fingerprint('segment'))


if __name__ == '__main__':
    unittest.main()
