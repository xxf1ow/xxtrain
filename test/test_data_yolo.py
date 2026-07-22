import unittest

from xxtrain.data import Bbox, ImageInfo, LabelCatalog, Line, Points, Polygon
from xxtrain.data.formats import encode_detect, encode_pose, encode_segment


class YoloEncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image = ImageInfo(width=200, height=100)

    def test_encodes_detect_line_with_catalog_index(self) -> None:
        annotation = Bbox(label='dial', x1=20, y1=10, x2=100, y2=50)
        labels = LabelCatalog(names=('other', 'dial'))

        self.assertEqual('1 0.300000 0.300000 0.400000 0.400000', encode_detect(annotation, self.image, labels))

    def test_encodes_segment_line_in_input_point_order(self) -> None:
        annotation = Polygon(label='mask', points=[[20, 10], [100, 10], [60, 50]])

        self.assertEqual(
            '0 0.100000 0.100000 0.500000 0.100000 0.300000 0.500000',
            encode_segment(annotation, self.image, LabelCatalog(names=('mask',))),
        )

    def test_segment_rejects_shapes_with_fewer_than_three_points(self) -> None:
        with self.assertRaisesRegex(ValueError, '至少三点'):
            encode_segment(
                Line(label='line', points=[[0, 0], [1, 1]]),
                self.image,
                LabelCatalog(names=('line',)),
            )

    def test_encodes_pose_in_catalog_order_with_fixed_class_zero(self) -> None:
        bbox = Bbox(label='object', x1=20, y1=10, x2=100, y2=50)
        keypoints = {
            'end': Points(label='end', points=[[100, 50]]),
            'start': Points(label='start', points=[[20, 10]]),
        }

        self.assertEqual(
            '0 0.300000 0.300000 0.400000 0.400000 0.100000 0.100000 2 0.500000 0.500000 2',
            encode_pose(
                bbox,
                keypoints,
                self.image,
                LabelCatalog(names=('start', 'end')),
            ),
        )

    def test_pose_requires_exact_single_point_catalog_mapping(self) -> None:
        bbox = Bbox(label='object', x1=20, y1=10, x2=100, y2=50)
        catalog = LabelCatalog(names=('start', 'end'))
        with self.assertRaisesRegex(ValueError, '关键点标签与目录不匹配'):
            encode_pose(
                bbox,
                {'start': Points(label='start', points=[[20, 10]])},
                self.image,
                catalog,
            )
        with self.assertRaisesRegex(ValueError, '单点'):
            encode_pose(
                bbox,
                {
                    'start': Points(label='start', points=[[20, 10], [21, 11]]),
                    'end': Points(label='end', points=[[100, 50]]),
                },
                self.image,
                catalog,
            )

    def test_detect_rejects_bbox_corners_outside_image(self) -> None:
        labels = LabelCatalog(names=('x',))
        annotations = (
            Bbox(label='x', x1=-10, y1=10, x2=10, y2=50),
            Bbox(label='x', x1=190, y1=10, x2=210, y2=50),
        )
        for annotation in annotations:
            with self.subTest(annotation=annotation):
                with self.assertRaises(AssertionError):
                    encode_detect(annotation, self.image, labels)

    def test_pose_rejects_bbox_corners_outside_image(self) -> None:
        keypoints = {'point': Points(label='point', points=[[100, 50]])}
        keypoint_labels = LabelCatalog(names=('point',))
        bboxes = (
            Bbox(label='object', x1=-10, y1=10, x2=10, y2=50),
            Bbox(label='object', x1=190, y1=10, x2=210, y2=50),
        )
        for bbox in bboxes:
            with self.subTest(bbox=bbox):
                with self.assertRaises(AssertionError):
                    encode_pose(bbox, keypoints, self.image, keypoint_labels)

    def test_segment_rejects_points_outside_image(self) -> None:
        labels = LabelCatalog(names=('mask',))
        annotations = (
            Polygon(label='mask', points=[[-1, 10], [100, 10], [60, 50]]),
            Polygon(label='mask', points=[[20, 10], [201, 10], [60, 50]]),
        )
        for annotation in annotations:
            with self.subTest(annotation=annotation):
                with self.assertRaises(AssertionError):
                    encode_segment(annotation, self.image, labels)


if __name__ == '__main__':
    unittest.main()
