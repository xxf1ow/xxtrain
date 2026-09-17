import os
import tempfile
import unittest
from pathlib import Path
from uuid import UUID

from PIL import Image

from xxtrain.business_tasks.point import point_task_definition
from xxtrain.data import Bbox
from xxtrain.platform.contracts import AnnotationRecord, DetectionBox, FrameMapping, ImageInput
from xxtrain.workspace_data import WorkspaceData
from xxtrain.workspace_data.crops import crop_frames, to_local, to_original
from xxtrain.workspace_data.repository import AnnotationRepository


class PlatformCropsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image_path = self.root / 'source.png'
        with Image.new('RGB', (10, 8)) as image:
            for y in range(image.height):
                for x in range(image.width):
                    image.putpixel((x, y), (x * 20, y * 30, (x + y) * 10))
            image.save(self.image_path)

    @staticmethod
    def box(annotation_id: str, *, label: str = 'tl', bounds=(-0.4, 1.2, 5.1, 7.7)):
        return DetectionBox(
            Bbox(id=UUID(annotation_id), label=label, x1=bounds[0], y1=bounds[1], x2=bounds[2], y2=bounds[3])
        )

    def image(self, *boxes: DetectionBox, width=10, height=8):
        return ImageInput('original-sha', self.image_path, width, height, boxes)

    def test_materializes_clamped_lossless_rgb_crops_per_source_object(self):
        first_id = '11111111-1111-1111-1111-111111111111'
        second_id = '22222222-2222-2222-2222-222222222222'
        negative = ImageInput('negative-sha', self.image_path, 10, 8, ())

        frames = crop_frames((negative, self.image(self.box(first_id), self.box(second_id))), self.root / 'runtime')

        self.assertEqual((first_id, second_id), tuple(frame.mapping.frame_id for frame in frames))
        self.assertEqual((UUID(first_id), UUID(second_id)), tuple(frame.mapping.parent_id for frame in frames))
        self.assertEqual(((0, 1, 6, 8), (0, 1, 6, 8)), tuple(frame.mapping.bounds for frame in frames))
        self.assertEqual(((6, 7), (6, 7)), tuple((frame.width, frame.height) for frame in frames))
        self.assertEqual(((), ()), tuple(frame.annotations for frame in frames))
        self.assertEqual(frames[0].image_path, frames[1].image_path)
        with Image.open(self.image_path) as source, Image.open(frames[0].image_path) as actual:
            expected = source.convert('RGB').crop((0, 1, 6, 8))
            self.assertEqual('RGB', actual.mode)
            self.assertEqual(expected.size, actual.size)
            self.assertEqual(expected.tobytes(), actual.tobytes())

    def test_cache_reuses_label_independent_crop_and_changes_with_geometry(self):
        annotation_id = '11111111-1111-1111-1111-111111111111'
        first = crop_frames((self.image(self.box(annotation_id)),), self.root / 'runtime')[0]
        labelled = crop_frames((self.image(self.box(annotation_id, label='cc')),), self.root / 'runtime')[0]
        changed = crop_frames(
            (self.image(self.box(annotation_id, bounds=(1.1, 1.2, 5.1, 7.7))),), self.root / 'runtime'
        )[0]

        self.assertEqual(first.image_path, labelled.image_path)
        self.assertNotEqual(first.image_path, changed.image_path)
        self.assertEqual((1, 1, 6, 8), changed.mapping.bounds)

    def test_classification_and_line_changes_reuse_the_detection_crop(self):
        workspace = self.root / 'workspace'
        (workspace / 'images').mkdir(parents=True)
        candidate = self.root / 'candidate.png'
        candidate.write_bytes(self.image_path.read_bytes())
        data = WorkspaceData(workspace)
        data.admit((candidate,))
        (sample,) = data.images()
        detection_id = UUID('11111111-1111-1111-1111-111111111111')
        repository = AnnotationRepository(workspace / 'annotations.db', point_task_definition())
        repository.save_annotations(
            (AnnotationRecord(detection_id, sample.sample_id, 'detect', None, 'rectangle', 'tl', [[1, 1], [7, 7]]),)
        )
        first = crop_frames(data.images(), self.root / 'runtime')[0]
        cached_time = 1_700_000_000_000_000_000
        os.utime(first.image_path, ns=(cached_time, cached_time))

        repository.save_annotations(
            (
                AnnotationRecord(UUID(int=2), sample.sample_id, 'classify', detection_id, 'classification', 'cc', None),
                AnnotationRecord(
                    UUID(int=3), sample.sample_id, 'segment', detection_id, 'polyline', '1', [[2, 2], [5, 5]]
                ),
            )
        )
        second = crop_frames(data.images(), self.root / 'runtime')[0]

        self.assertEqual(first.image_path, second.image_path)
        self.assertEqual(cached_time, second.image_path.stat().st_mtime_ns)

    def test_missing_crop_is_regenerated_from_source(self):
        frame = crop_frames((self.image(self.box('11111111-1111-1111-1111-111111111111')),), self.root / 'runtime')[0]
        expected = frame.image_path.read_bytes()
        frame.image_path.unlink()

        regenerated = crop_frames(
            (self.image(self.box('11111111-1111-1111-1111-111111111111')),), self.root / 'runtime'
        )[0]

        self.assertEqual(frame.image_path, regenerated.image_path)
        self.assertEqual(expected, regenerated.image_path.read_bytes())

    def test_geometry_translation_uses_integer_crop_origin(self):
        mapping = FrameMapping(
            '11111111-1111-1111-1111-111111111111',
            'original-sha',
            UUID('11111111-1111-1111-1111-111111111111'),
            (3, 4, 9, 12),
        )
        local = [[1.0, 1.0], [4.0, 5.0]]

        self.assertEqual([[4.0, 5.0], [7.0, 9.0]], to_original(mapping, local))
        self.assertEqual(local, to_local(mapping, to_original(mapping, local)))
        self.assertIsNone(to_local(mapping, None))

    def test_rejects_empty_clamped_crops_with_frame_identity(self):
        annotation_id = '11111111-1111-1111-1111-111111111111'
        outside = self.box(annotation_id, bounds=(10.2, 1.0, 12.0, 3.0))

        with self.assertRaisesRegex(ValueError, annotation_id):
            crop_frames((self.image(outside),), self.root / 'runtime')

    def test_rejects_declared_dimensions_that_do_not_match_the_image(self):
        box = self.box('11111111-1111-1111-1111-111111111111')

        with self.assertRaisesRegex(ValueError, 'dimensions'):
            crop_frames((self.image(box, width=9),), self.root / 'runtime')


if __name__ == '__main__':
    unittest.main()
