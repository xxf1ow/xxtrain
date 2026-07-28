import json
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import xxtrain.data.formats as formats
from xxtrain.data import Annotation, Bbox, Circle, ImageInfo, Keypoint, Points, Polygon, Polyline, Pose
from xxtrain.data.formats import (
    decode_detect,
    decode_pose,
    decode_segment,
    read_coco,
    read_labelimg,
    read_labelme,
    write_coco,
    write_labelimg,
    write_labelme,
)

FIXTURES_PATH = Path(__file__).resolve().parent / 'fixtures'


class UnsupportedAnnotation(Annotation):
    def _validate_geometry(self) -> None:
        pass


class AnnotationFormatsTest(unittest.TestCase):
    def test_formats_package_exports_dataset_io_functions(self) -> None:
        exports = {
            'decode_detect': decode_detect,
            'decode_segment': decode_segment,
            'decode_pose': decode_pose,
            'write_labelimg': write_labelimg,
            'write_labelme': write_labelme,
            'read_coco': read_coco,
            'write_coco': write_coco,
        }

        for name, function in exports.items():
            with self.subTest(name=name):
                self.assertIs(function, getattr(formats, name))

    def _write_labelme(self, root: Path, shapes: list[dict[str, object]]) -> Path:
        path = root / 'annotations.json'
        path.write_text(
            json.dumps(
                {
                    'version': '5.0.0',
                    'flags': {},
                    'shapes': shapes,
                    'imagePath': 'sample.jpg',
                    'imageData': None,
                    'imageHeight': 80,
                    'imageWidth': 100,
                }
            ),
            encoding='utf-8',
        )
        return path

    def test_parses_labelimg_rectangle(self) -> None:
        path = FIXTURES_PATH / 'standard-detect' / 'src' / '20260620' / 'anns' / '0000.xml'

        annotations = read_labelimg(path, ImageInfo(width=1920, height=1080))

        self.assertEqual(1, len(annotations))
        self.assertIsInstance(annotations[0], Bbox)
        self.assertEqual('cc', annotations[0].label)
        self.assertEqual((911.0, 303.0, 1400.0, 735.0), annotations[0].bbox)
        self.assertEqual(((911.0, 303.0), (1400.0, 303.0), (1400.0, 735.0), (911.0, 735.0)), annotations[0].points)

    def test_parses_labelme_pose_point(self) -> None:
        path = FIXTURES_PATH / 'standard-pose' / 'src' / 'rw400' / 'anns_seg' / '0000.json'

        annotations = read_labelme(path, ImageInfo(width=4000, height=2250))

        self.assertEqual(1, len(annotations))
        self.assertIsInstance(annotations[0], Points)
        self.assertEqual('rw400', annotations[0].label)
        self.assertEqual(((1868.181818181818, 1406.8181818181818),), annotations[0].points)

    def test_parses_labelme_line_without_task_transform(self) -> None:
        path = FIXTURES_PATH / 'point' / 'src' / '20260620' / 'anns_seg' / '0000.json'

        annotations = read_labelme(path, ImageInfo(width=1920, height=1080))

        self.assertEqual(1, len(annotations))
        self.assertIsInstance(annotations[0], Polyline)
        self.assertEqual('1', annotations[0].label)
        self.assertEqual(
            ((1133.3333333333335, 516.0683760683761), (1047.008547008547, 334.87179487179486)), annotations[0].points
        )

    def test_parses_labelme_rotations_in_file_order(self) -> None:
        path = FIXTURES_PATH / 'knob' / 'src' / '251010' / 'anns_seg' / '0000.json'

        annotations = read_labelme(path, ImageInfo(width=1920, height=1080))

        self.assertEqual(4, len(annotations))
        for annotation in annotations:
            with self.subTest(annotation=annotation):
                self.assertIsInstance(annotation, Polygon)
                self.assertEqual('switch', annotation.label)
                self.assertEqual(4, len(annotation.points))

    def test_labelme_preserves_edge_shapes_order_and_groups(self) -> None:
        path = FIXTURES_PATH / 'annotation-formats' / 'edge-shapes.json'

        annotations = read_labelme(path, ImageInfo(width=100, height=80))

        self.assertEqual([Bbox, Circle, Polygon, Polyline, Polyline], [type(annotation) for annotation in annotations])
        self.assertEqual((2.0, 4.0, 20.0, 18.0), annotations[0].bbox)
        self.assertEqual((25.0, 25.0, 35.0, 35.0), annotations[1].bbox)
        self.assertIsNone(annotations[0].group)
        self.assertEqual(7, annotations[2].group)
        self.assertEqual(7, annotations[3].group)
        self.assertNotEqual(annotations[2].id, annotations[3].id)

    def test_labelme_rejects_invalid_line_and_rotation_boundaries(self) -> None:
        invalid_shapes = (
            {'label': 'line', 'points': [[1, 1], [2, 2], [3, 3]], 'group_id': None, 'shape_type': 'line', 'flags': {}},
            {
                'label': 'rotation',
                'points': [[1, 1], [3, 1], [3, 3]],
                'group_id': None,
                'shape_type': 'rotation',
                'flags': {},
            },
            {
                'label': 'rotation',
                'points': [[1, 1], [3, 1], [4, 2], [3, 3], [1, 3]],
                'group_id': None,
                'shape_type': 'rotation',
                'flags': {},
            },
            {
                'label': 'rotation',
                'points': [[1, 1], [4, 1], [3, 3], [1, 3]],
                'group_id': None,
                'shape_type': 'rotation',
                'flags': {},
            },
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for shape in invalid_shapes:
                with self.subTest(shape=shape), self.assertRaisesRegex(Exception, 'Failed to parse annotation'):
                    path = self._write_labelme(root, [shape])
                    read_labelme(path, ImageInfo(width=100, height=80))

    def test_labelme_rejects_boolean_group_ids(self) -> None:
        shape = {'label': 'box', 'points': [[1, 1], [2, 2]], 'group_id': True, 'shape_type': 'rectangle', 'flags': {}}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_labelme(Path(temp_dir), [shape])

            with self.assertRaisesRegex(Exception, 'Failed to parse annotation'):
                read_labelme(path, ImageInfo(width=100, height=80))

    def test_labelme_rejects_point_shapes_with_multiple_points(self) -> None:
        shape = {'label': 'marker', 'points': [[1, 1], [2, 2]], 'group_id': None, 'shape_type': 'point', 'flags': {}}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_labelme(Path(temp_dir), [shape])

            with self.assertRaisesRegex(Exception, 'Failed to parse annotation'):
                read_labelme(path, ImageInfo(width=100, height=80))

    def test_labelme_round_trips_supported_non_pose_shapes(self) -> None:
        annotations = (
            Bbox(label='box', group=0, x1=1, y1=2, x2=11, y2=12),
            Circle(label='circle', group='circles', center=(20, 20), edge=(23, 24)),
            Polygon(label='polygon', group=7, points=((30, 10), (40, 10), (35, 20))),
            Polyline(label='line', group=7, points=((31, 11), (39, 19))),
            Polyline(label='path', points=((50, 10), (55, 15), (60, 12))),
            Points(label='marker', group='markers', points=((70, 30),)),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'sample.json'

            write_labelme(annotations, path, ImageInfo(width=100, height=80))

            data = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(
                {'version', 'flags', 'shapes', 'imagePath', 'imageData', 'imageHeight', 'imageWidth'}, set(data)
            )
            self.assertIsNone(data['imageData'])
            self.assertEqual(
                ['rectangle', 'circle', 'polygon', 'line', 'linestrip', 'point'],
                [shape['shape_type'] for shape in data['shapes']],
            )
            self.assertTrue(
                all({'label', 'points', 'group_id', 'shape_type', 'flags'} == set(shape) for shape in data['shapes'])
            )
            output = read_labelme(path, ImageInfo(width=100, height=80))

        self.assertEqual([type(annotation) for annotation in annotations], [type(annotation) for annotation in output])
        for expected, actual in zip(annotations, output):
            with self.subTest(expected=expected):
                self.assertEqual(expected.label, actual.label)
                self.assertEqual(expected.group, actual.group)
                self.assertEqual(expected.points, actual.points)

    def test_labelme_writer_uses_requested_image_path_and_rotation_shapes_for_obb(self) -> None:
        valid_obb = Polygon(label='box', points=((10, 10), (30, 10), (30, 20), (10, 20)))
        triangle = Polygon(label='triangle', points=((10, 10), (30, 10), (20, 20)))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'annotations.json'

            write_labelme((valid_obb,), path, ImageInfo(width=100, height=80), image_path='../images/a.jpg', obb=True)

            payload = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual('../images/a.jpg', payload['imagePath'])
            self.assertEqual('rotation', payload['shapes'][0]['shape_type'])
            with self.assertRaisesRegex(ValueError, 'OBB'):
                write_labelme((triangle,), path, ImageInfo(width=100, height=80), obb=True)
            with self.assertRaisesRegex(ValueError, 'OBB'):
                write_labelme(
                    (Bbox(label='box', x1=10, y1=10, x2=30, y2=20),), path, ImageInfo(width=100, height=80), obb=True
                )

    def test_labelme_round_trips_pose_expansion_and_keypoint_order(self) -> None:
        annotations = (
            Pose(
                label='person-zero',
                group=0,
                x1=1,
                y1=2,
                x2=30,
                y2=40,
                keypoints=(Keypoint(label='nose', x=10, y=12), Keypoint(label='wrist', x=20, y=25)),
            ),
            Pose(label='person-id', x1=50, y1=10, x2=90, y2=70, keypoints=(Keypoint(label='eye', x=60, y=20),)),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'poses.json'

            write_labelme(annotations, path, ImageInfo(width=100, height=80))

            shapes = json.loads(path.read_text(encoding='utf-8'))['shapes']
            self.assertEqual(
                ['rectangle', 'point', 'point', 'rectangle', 'point'], [shape['shape_type'] for shape in shapes]
            )
            self.assertEqual([0, 0, 0], [shape['group_id'] for shape in shapes[:3]])
            self.assertEqual(
                [str(annotations[1].id), str(annotations[1].id)], [shape['group_id'] for shape in shapes[3:]]
            )
            output = read_labelme(path, ImageInfo(width=100, height=80))

        self.assertEqual([Pose, Pose], [type(annotation) for annotation in output])
        for index, (expected, actual) in enumerate(zip(annotations, output)):
            with self.subTest(expected=expected):
                self.assertEqual(expected.label, actual.label)
                self.assertEqual(expected.group if index == 0 else str(expected.id), actual.group)
                self.assertEqual(expected.bbox, actual.bbox)
                self.assertEqual(
                    [(keypoint.label, keypoint.x, keypoint.y, 2) for keypoint in expected.keypoints],
                    [(keypoint.label, keypoint.x, keypoint.y, keypoint.visibility) for keypoint in actual.keypoints],
                )

    def test_labelme_writer_rejects_lossy_or_unsupported_annotations_before_io(self) -> None:
        invalid_annotations = (
            (ValueError, Points(label='markers', points=((1, 1), (2, 2)))),
            (
                ValueError,
                Pose(
                    label='hidden',
                    x1=0,
                    y1=0,
                    x2=10,
                    y2=10,
                    keypoints=(Keypoint(label='nose', x=1, y=1, visibility=0),),
                ),
            ),
            (
                ValueError,
                Pose(
                    label='occluded',
                    x1=0,
                    y1=0,
                    x2=10,
                    y2=10,
                    keypoints=(Keypoint(label='nose', x=1, y=1, visibility=1),),
                ),
            ),
            (TypeError, UnsupportedAnnotation(label='unsupported')),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, (error_type, invalid) in enumerate(invalid_annotations):
                with self.subTest(invalid=invalid):
                    path = root / str(index) / 'annotations.json'
                    with self.assertRaises(error_type):
                        write_labelme(
                            (Bbox(label='valid', x1=0, y1=0, x2=10, y2=10), invalid),
                            path,
                            ImageInfo(width=100, height=80),
                        )
                    self.assertFalse(path.parent.exists())

    def test_labelme_writer_rejects_generic_groups_that_read_as_pose_before_io(self) -> None:
        annotations = (
            Bbox(label='box', group=0, x1=0, y1=0, x2=10, y2=10),
            Points(label='nose', group=0, points=((1, 1),)),
            Points(label='wrist', group=0, points=((2, 2),)),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'missing' / 'annotations.json'

            with self.assertRaises(ValueError):
                write_labelme(annotations, path, ImageInfo(width=100, height=80))

            self.assertFalse(path.parent.exists())

    def test_labelme_writer_allows_generic_groups_that_remain_generic(self) -> None:
        annotations = (
            Bbox(label='box', group=0, x1=0, y1=0, x2=10, y2=10),
            Points(label='nose', group=0, points=((1, 1),)),
            Polygon(label='mask', group=0, points=((0, 0), (2, 0), (1, 2))),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'annotations.json'

            write_labelme(annotations, path, ImageInfo(width=100, height=80))

            output = read_labelme(path, ImageInfo(width=100, height=80))

        self.assertEqual([Bbox, Points, Polygon], [type(annotation) for annotation in output])

    def test_labelme_writer_rejects_nonexclusive_pose_groups_before_io(self) -> None:
        shared_uuid = uuid4()
        first_pose = Pose(
            label='first', group=shared_uuid, x1=0, y1=0, x2=10, y2=10, keypoints=(Keypoint(label='nose', x=1, y=1),)
        )
        second_pose = Pose(
            label='second',
            group=str(shared_uuid),
            x1=20,
            y1=20,
            x2=30,
            y2=30,
            keypoints=(Keypoint(label='nose', x=21, y=21),),
        )
        ungrouped_pose = Pose(
            label='generated', x1=40, y1=40, x2=50, y2=50, keypoints=(Keypoint(label='nose', x=41, y=41),)
        )
        invalid_sequences = (
            (first_pose, second_pose),
            (
                ungrouped_pose,
                Polygon(label='alias', group=str(ungrouped_pose.id), points=((60, 60), (70, 60), (65, 70))),
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, annotations in enumerate(invalid_sequences):
                with self.subTest(annotations=annotations):
                    path = root / str(index) / 'annotations.json'
                    with self.assertRaises(ValueError):
                        write_labelme(annotations, path, ImageInfo(width=100, height=80))
                    self.assertFalse(path.parent.exists())

    def test_labelme_writer_rejects_non_integral_dimensions_before_io(self) -> None:
        annotation = Bbox(label='box', x1=1, y1=1, x2=2, y2=2)
        dimensions = (ImageInfo(width=100.5, height=80), ImageInfo(width=100, height=80.5))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, image_info in enumerate(dimensions):
                with self.subTest(image_info=image_info, path='missing'):
                    path = root / f'missing-{index}' / 'annotations.json'
                    with self.assertRaises(ValueError):
                        write_labelme((annotation,), path, image_info)
                    self.assertFalse(path.parent.exists())

                with self.subTest(image_info=image_info, path='existing'):
                    path = root / f'existing-{index}.json'
                    path.write_bytes(b'sentinel')
                    with self.assertRaises(ValueError):
                        write_labelme((annotation,), path, image_info)
                    self.assertEqual(b'sentinel', path.read_bytes())

    def test_labelme_writer_encodes_before_io_and_preserves_valid_unicode(self) -> None:
        image_info = ImageInfo(width=100, height=80)
        invalid = Bbox(label='bad\ud800', x1=1, y1=1, x2=2, y2=2)
        valid = Bbox(label='仪表😀', x1=1, y1=1, x2=2, y2=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = root / 'missing' / 'annotations.json'
            with self.assertRaises(UnicodeError):
                write_labelme((invalid,), missing, image_info)
            self.assertFalse(missing.parent.exists())

            existing = root / 'existing.json'
            existing.write_bytes(b'sentinel')
            with self.assertRaises(UnicodeError):
                write_labelme((invalid,), existing, image_info)
            self.assertEqual(b'sentinel', existing.read_bytes())

            valid_path = root / 'valid.json'
            write_labelme((valid,), valid_path, image_info)
            payload = json.loads(valid_path.read_text(encoding='utf-8'))
            restored = read_labelme(valid_path, image_info)

        self.assertEqual('仪表😀', restored[0].label)
        self.assertIs(type(payload['imageWidth']), int)
        self.assertIs(type(payload['imageHeight']), int)


if __name__ == '__main__':
    unittest.main()
