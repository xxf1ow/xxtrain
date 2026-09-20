import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from uuid import uuid4

from xxtrain.business_tasks import target_complete, validate_target_annotations
from xxtrain.business_tasks.point import point_task_definition
from xxtrain.platform.contracts import (
    AnnotationRecord,
    EditAnnotation,
    EditFrame,
    EditFrameResult,
    EditJob,
    FrameMapping,
    ImageRecord,
    JobRef,
    TargetSummary,
)
from xxtrain.workspace_data.repository import AnnotationRepository


class PlatformEditContractsTest(unittest.TestCase):
    def test_shared_edit_records_are_frozen_and_preserve_values(self) -> None:
        parent_id = uuid4()
        annotation = EditAnnotation(uuid4(), 'polyline', '1', [[1, 2], [3, 4]], 17)
        mapping = FrameMapping(str(parent_id), 'a' * 64, parent_id, (10, 20, 30, 40))
        frame = EditFrame(mapping, Path('crop.png'), 20, 20, (annotation,))
        result = EditFrameResult(mapping.frame_id, (annotation,))
        job = EditJob(JobRef(11, 12, (mapping.image_id,)), (mapping,))
        summary = TargetSummary(3, 2, 2)

        self.assertEqual((annotation,), frame.annotations)
        self.assertEqual(mapping.frame_id, result.frame_id)
        self.assertEqual((mapping,), job.frames)
        self.assertEqual(
            (3, 2, 2), (summary.sample_count, summary.annotated_sample_count, summary.positive_sample_count)
        )
        with self.assertRaises(FrozenInstanceError):
            setattr(annotation, 'label', 'tl')

    def test_classification_completion_and_cardinality(self) -> None:
        self.assertFalse(target_complete('classify', ()))
        self.assertTrue(target_complete('classify', (EditAnnotation(None, 'classification', 'tl', None),)))
        with self.assertRaises(ValueError):
            validate_target_annotations(
                'classify',
                (
                    EditAnnotation(None, 'classification', 'tl', None),
                    EditAnnotation(None, 'classification', 'cc', None),
                ),
            )

    def test_classification_rejects_wrong_kind_label_and_geometry(self) -> None:
        invalid = (
            EditAnnotation(None, 'negative', 'tl', None),
            EditAnnotation(None, 'polyline', 'tl', None),
            EditAnnotation(None, 'classification', 'missing', None),
            EditAnnotation(None, 'classification', 'tl', []),
        )

        for annotation in invalid:
            with self.subTest(annotation=annotation), self.assertRaises(ValueError):
                validate_target_annotations('classify', (annotation,))

    def test_segment_completion_accepts_multiple_two_point_lines(self) -> None:
        self.assertFalse(target_complete('segment', ()))
        self.assertTrue(
            target_complete(
                'segment',
                (
                    EditAnnotation(None, 'polyline', '1', [[1, 2], [3, 4]]),
                    EditAnnotation(None, 'polyline', '1', [[2, 3], [4, 5]]),
                ),
            )
        )

    def test_segment_rejects_wrong_type_label_and_line_geometry(self) -> None:
        invalid = (
            EditAnnotation(None, 'classification', '1', None),
            EditAnnotation(None, 'negative', '1', None),
            EditAnnotation(None, 'polyline', 'Point', [[1, 2], [3, 4]]),
            EditAnnotation(None, 'polyline', '1', [[1, 2]]),
            EditAnnotation(None, 'polyline', '1', [[1, 2], [3, 4], [5, 6]]),
            EditAnnotation(None, 'polyline', '1', [[1, 2], [1, 2]]),
            EditAnnotation(None, 'polyline', '1', [[1, 2], [float('inf'), 4]]),
            EditAnnotation(None, 'polyline', '1', [[1, 2], [float('nan'), 4]]),
            EditAnnotation(None, 'polyline', '1', [[True, 2], [3, 4]]),
            EditAnnotation(None, 'polyline', '1', [(1, 2), [3, 4]]),
        )

        for annotation in invalid:
            with self.subTest(annotation=annotation), self.assertRaises(ValueError):
                validate_target_annotations('segment', (annotation,))

    def test_unknown_target_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_target_annotations('detect', ())
        with self.assertRaises(ValueError):
            target_complete('missing', ())

    def test_point_classification_and_line_round_trip_through_repository(self) -> None:
        with tempfile.TemporaryDirectory(prefix='xxtrain-edit-contracts-') as temporary:
            path = Path(temporary) / 'annotations.db'
            task = point_task_definition()
            repository = AnnotationRepository(path, task)
            image = ImageRecord('a' * 64, 'a.png', 100, 80, 0)
            repository.register_images((image,))
            parent = AnnotationRecord(
                uuid4(), image.id, 'detect', None, 'rectangle', 'Point', [[1.5, 2.5], [30.5, 40.5]]
            )
            classification = AnnotationRecord(uuid4(), image.id, 'classify', parent.id, 'classification', 'tc', None)
            line = AnnotationRecord(
                uuid4(), image.id, 'segment', parent.id, 'polyline', '1', [[2.25, 3.5], [8.75, 12.5]]
            )

            repository.save_annotations((line, classification, parent))
            reopened = AnnotationRepository(path, task)

            self.assertEqual((classification,), reopened.annotations(image_id=image.id, step_key='classify'))
            self.assertEqual((line,), reopened.annotations(image_id=image.id, step_key='segment'))
            self.assertEqual([[2.25, 3.5], [8.75, 12.5]], reopened.annotations(step_key='segment')[0].geometry)


if __name__ == '__main__':
    unittest.main()
