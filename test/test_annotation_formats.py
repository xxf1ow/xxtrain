import unittest
from pathlib import Path

import numpy as np

from test.support.current_api import ShapeType, TaskType, parse_labelimg, parse_labelme

FIXTURES_PATH = Path(__file__).resolve().parent / 'fixtures'


class AnnotationFormatsTest(unittest.TestCase):
    def test_parses_labelimg_rectangle(self) -> None:
        annotation_path = FIXTURES_PATH / 'standard-detect' / 'src' / '20260620' / 'anns' / '0000.xml'

        annotations = parse_labelimg(str(annotation_path), 1920, 1080)

        self.assertEqual(1, len(annotations))
        annotation = next(iter(annotations.values()))
        self.assertEqual('cc', annotation.label)
        self.assertEqual(ShapeType.RECTANGLE, annotation.type)
        np.testing.assert_array_equal([911.0, 303.0, 1400.0, 735.0], annotation.bbox)
        np.testing.assert_array_equal(
            [[911, 303], [1400, 303], [1400, 735], [911, 735]],
            annotation.points,
        )

    def test_parses_labelme_pose_point(self) -> None:
        annotation_path = FIXTURES_PATH / 'standard-pose' / 'src' / 'rw400' / 'anns_seg' / '0000.json'

        annotations = parse_labelme(str(annotation_path), 4000, 2250, TaskType.POSE)

        self.assertEqual(1, len(annotations))
        annotation = next(iter(annotations.values()))
        self.assertEqual('rw400', annotation.label)
        self.assertEqual(ShapeType.POINT, annotation.type)
        np.testing.assert_allclose(
            [[1868.181818181818, 1406.8181818181818]],
            annotation.points,
        )

    def test_parses_labelme_segment_line(self) -> None:
        annotation_path = FIXTURES_PATH / 'point' / 'src' / '20260620' / 'anns_seg' / '0000.json'

        annotations = parse_labelme(str(annotation_path), 1920, 1080, TaskType.SEGMENT)

        self.assertEqual(1, len(annotations))
        annotation = next(iter(annotations.values()))
        self.assertEqual('1', annotation.label)
        self.assertEqual(ShapeType.LINE, annotation.type)
        np.testing.assert_allclose(
            [[1133.3333333333335, 516.0683760683761], [1047.008547008547, 334.87179487179486]],
            annotation.points,
        )

    def test_parses_labelme_segment_rotations(self) -> None:
        annotation_path = FIXTURES_PATH / 'knob' / 'src' / '251010' / 'anns_seg' / '0000.json'

        annotations = parse_labelme(str(annotation_path), 1920, 1080, TaskType.SEGMENT)

        self.assertEqual(4, len(annotations))
        for annotation in annotations.values():
            with self.subTest(annotation=annotation):
                self.assertEqual('switch', annotation.label)
                self.assertEqual(ShapeType.ROTATION, annotation.type)
                self.assertEqual((4, 2), annotation.points.shape)


if __name__ == '__main__':
    unittest.main()
