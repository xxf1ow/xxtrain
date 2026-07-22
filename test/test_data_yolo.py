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

    def test_all_encoders_reject_out_of_image_coordinates(self) -> None:
        with self.assertRaises(AssertionError):
            encode_detect(
                Bbox(label='x', x1=0, y1=0, x2=201, y2=50),
                self.image,
                LabelCatalog(names=('x',)),
            )


if __name__ == '__main__':
    unittest.main()
