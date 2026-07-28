import unittest
from pathlib import Path

from xxtrain.data import Bbox, Circle, ImageInfo, Points, Polygon, Polyline
from xxtrain.data.formats import read_labelimg, read_labelme

FIXTURES_PATH = Path(__file__).resolve().parent / 'fixtures'


class AnnotationFormatsTest(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
