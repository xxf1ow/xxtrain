import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from xxtrain.data import Bbox
from xxtrain.platform.contracts import DetectionBox, FrameResult, PlatformAccessError
from xxtrain.workspace_data import WorkspaceData
from xxtrain.workspace_data.labelme import merge_detection


class PlatformDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images_dir = self.root / 'images'
        self.annotations_dir = self.root / 'annotations'
        self.images_dir.mkdir()
        self.annotations_dir.mkdir()

    def make_image(self, name: str = 'a.jpg', *, size: tuple[int, int] = (64, 48)) -> Path:
        path = self.images_dir / name
        Image.new('RGB', size, 'white').save(path)
        return path

    def write_annotation(self, stem: str, document: dict[str, object]) -> Path:
        path = self.annotations_dir / f'{stem}.json'
        path.write_text(json.dumps(document), encoding='utf-8')
        return path

    def test_detection_edit_preserves_other_annotations(self) -> None:
        polygon = {'label': '1', 'shape_type': 'line', 'points': [[1, 2], [3, 4]], 'flags': {'keep': True}}
        source = {'imagePath': '../images/a.jpg', 'custom': 'keep', 'shapes': [polygon]}
        box = DetectionBox(
            Bbox(label='tl', x1=1.25, y1=2, x2=30, y2=40),
            {'description': 'keep', 'label': 'wrong', 'points': [], 'shape_type': 'polygon'},
        )

        result = merge_detection(source, (box,))

        self.assertEqual(result['shapes'][0], polygon)
        self.assertEqual(result['shapes'][1]['label'], 'tl')
        self.assertEqual(result['shapes'][1]['points'][0], [1.25, 2.0])
        self.assertEqual(result['shapes'][1]['description'], 'keep')
        self.assertEqual(result['custom'], 'keep')
        self.assertEqual(source['shapes'], [polygon])

    def test_images_and_save_support_an_empty_annotation(self) -> None:
        image_path = self.make_image(size=(37, 23))
        original_image = image_path.read_bytes()
        workspace = WorkspaceData(self.images_dir, self.annotations_dir)

        images = workspace.images()

        self.assertEqual(1, len(images))
        self.assertEqual('a', images[0].sample_id)
        self.assertEqual(image_path, images[0].image_path)
        self.assertEqual((37, 23), (images[0].width, images[0].height))
        self.assertEqual((), images[0].boxes)

        workspace.save_detection((FrameResult(sample_id='a', boxes=()),))

        self.assertEqual(
            {
                'version': '5.0.0',
                'flags': {},
                'shapes': [],
                'imagePath': '../images/a.jpg',
                'imageData': None,
                'imageHeight': 23,
                'imageWidth': 37,
            },
            json.loads((self.annotations_dir / 'a.json').read_text(encoding='utf-8')),
        )
        self.assertEqual(original_image, image_path.read_bytes())

    def test_existing_rectangle_label_fractional_geometry_and_metadata_round_trip(self) -> None:
        self.make_image()
        source = {
            'version': '5.4.1',
            'shapes': [
                {
                    'label': 'tl',
                    'shape_type': 'rectangle',
                    'points': [[1.25, 2], [30, 40]],
                    'group_id': 7,
                    'flags': {'reviewed': True},
                    'description': 'keep',
                }
            ],
            'imagePath': '../images/a.jpg',
            'imageHeight': 48,
            'imageWidth': 64,
        }
        self.write_annotation('a', source)
        workspace = WorkspaceData(self.images_dir, self.annotations_dir)

        image = workspace.images()[0]

        self.assertEqual('tl', image.boxes[0].geometry.label)
        self.assertEqual((1.25, 2.0, 30.0, 40.0), image.boxes[0].geometry.bbox)
        self.assertEqual({'group_id': 7, 'flags': {'reviewed': True}, 'description': 'keep'}, image.boxes[0].extra)

        workspace.save_detection((FrameResult(sample_id='a', boxes=image.boxes),))

        saved_shape = json.loads((self.annotations_dir / 'a.json').read_text(encoding='utf-8'))['shapes'][0]
        self.assertEqual('tl', saved_shape['label'])
        self.assertEqual([[1.25, 2.0], [30.0, 40.0]], saved_shape['points'])
        self.assertEqual('rectangle', saved_shape['shape_type'])
        self.assertEqual(7, saved_shape['group_id'])
        self.assertEqual({'reviewed': True}, saved_shape['flags'])
        self.assertEqual('keep', saved_shape['description'])

    def test_deleting_all_rectangles_preserves_lines_polygons_and_root_fields(self) -> None:
        line = {'label': 'line', 'shape_type': 'line', 'points': [[1, 2], [3, 4]]}
        polygon = {'label': 'mask', 'shape_type': 'polygon', 'points': [[1, 2], [3, 4], [5, 2]]}
        source = {'custom': {'keep': True}, 'shapes': [line, {'label': 'tl', 'shape_type': 'rectangle'}, polygon]}

        result = merge_detection(source, ())

        self.assertEqual([line, polygon], result['shapes'])
        self.assertEqual({'keep': True}, result['custom'])

    def test_images_reject_invalid_rectangle_geometry(self) -> None:
        self.make_image()
        self.write_annotation(
            'a', {'shapes': [{'label': 'Point', 'shape_type': 'rectangle', 'points': [[1, 2], [1, 4]]}]}
        )

        with self.assertRaisesRegex(ValueError, 'Bbox'):
            WorkspaceData(self.images_dir, self.annotations_dir).images()

    def test_save_requires_each_image_exactly_once(self) -> None:
        self.make_image('a.jpg')
        self.make_image('b.png')
        workspace = WorkspaceData(self.images_dir, self.annotations_dir)
        valid_box = DetectionBox(Bbox(label='Point', x1=1, y1=2, x2=3, y2=4))

        invalid_results = (
            (FrameResult(sample_id='a', boxes=(valid_box,)),),
            (FrameResult(sample_id='a', boxes=()), FrameResult(sample_id='a', boxes=())),
            (FrameResult(sample_id='a', boxes=()), FrameResult(sample_id='unknown', boxes=())),
        )
        for results in invalid_results:
            with self.subTest(results=results), self.assertRaises(ValueError):
                workspace.save_detection(results)

        self.assertEqual([], list(self.annotations_dir.iterdir()))

    def test_startup_rejects_duplicate_stems_and_missing_directories(self) -> None:
        self.make_image('same.jpg')
        self.make_image('same.png')

        with self.assertRaisesRegex(ValueError, 'Duplicate image stem'):
            WorkspaceData(self.images_dir, self.annotations_dir)
        with self.assertRaises(PlatformAccessError):
            WorkspaceData(self.root / 'missing-images', self.annotations_dir)
        with self.assertRaises(PlatformAccessError):
            WorkspaceData(self.images_dir, self.root / 'missing-annotations')

    def test_all_documents_are_encoded_before_any_file_is_replaced(self) -> None:
        self.make_image('a.jpg')
        self.make_image('b.jpg')
        a_path = self.write_annotation('a', {'marker': 'old-a', 'shapes': []})
        b_path = self.write_annotation('b', {'marker': 'old-b', 'shapes': []})
        old_a = a_path.read_bytes()
        old_b = b_path.read_bytes()
        workspace = WorkspaceData(self.images_dir, self.annotations_dir)
        invalid_box = DetectionBox(Bbox(label='Point', x1=1, y1=2, x2=3, y2=4), {'confidence': float('nan')})

        with self.assertRaises(ValueError):
            workspace.save_detection(
                (FrameResult(sample_id='a', boxes=()), FrameResult(sample_id='b', boxes=(invalid_box,)))
            )

        self.assertEqual(old_a, a_path.read_bytes())
        self.assertEqual(old_b, b_path.read_bytes())

    def test_replace_failure_does_not_truncate_old_json_and_cleans_temporary_file(self) -> None:
        self.make_image()
        annotation_path = self.write_annotation('a', {'marker': 'old', 'shapes': []})
        old_content = annotation_path.read_bytes()
        workspace = WorkspaceData(self.images_dir, self.annotations_dir)

        with patch('xxtrain.workspace_data.store.os.replace', side_effect=OSError('replace failed')):
            with self.assertRaisesRegex(OSError, 'replace failed'):
                workspace.save_detection((FrameResult(sample_id='a', boxes=()),))

        self.assertEqual(old_content, annotation_path.read_bytes())
        self.assertEqual([annotation_path], list(self.annotations_dir.iterdir()))

        workspace.save_detection((FrameResult(sample_id='a', boxes=()),))

        self.assertEqual({'marker': 'old', 'shapes': []}, json.loads(annotation_path.read_text(encoding='utf-8')))


if __name__ == '__main__':
    unittest.main()
