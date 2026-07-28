import unittest

from xxtrain.data import Bbox, ImageInfo, Keypoint, LabelCatalog, Polygon, Polyline, Pose
from xxtrain.data.formats import decode_detect, decode_pose, decode_segment, encode_detect, encode_pose, encode_segment


class YoloFormatTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image = ImageInfo(width=200, height=100)

    def test_detect_round_trip_preserves_geometry_and_label(self) -> None:
        annotation = Bbox(label='dial', x1=20, y1=10, x2=100, y2=50)
        labels = LabelCatalog(names=('other', 'dial'))

        decoded = decode_detect(encode_detect(annotation, self.image, labels), self.image, labels)

        self.assertEqual(annotation, decoded.wrap(id=annotation.id))

    def test_segment_round_trip_preserves_geometry_order_and_label(self) -> None:
        annotation = Polygon(label='mask', points=((20, 10), (100, 10), (60, 50)))
        labels = LabelCatalog(names=('mask',))

        decoded = decode_segment(encode_segment(annotation, self.image, labels), self.image, labels)

        self.assertEqual(annotation, decoded.wrap(id=annotation.id))

    def test_pose_round_trip_preserves_geometry_visibility_order_and_label(self) -> None:
        pose = Pose(
            label='nose',
            x1=20,
            y1=10,
            x2=100,
            y2=50,
            keypoints=(
                Keypoint(label='nose', x=20, y=10, visibility=2),
                Keypoint(label='wrist', x=100, y=50, visibility=1),
            ),
        )
        labels = LabelCatalog(names=('nose', 'wrist'))

        decoded = decode_pose(encode_pose(pose, self.image, labels), self.image, labels)

        self.assertEqual(pose, decoded.wrap(id=pose.id))

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

    def test_decoders_require_valid_token_counts(self) -> None:
        labels = LabelCatalog(names=('label',))
        invalid = (
            lambda: decode_detect('0 0.5 0.5 0.5', self.image, labels),
            lambda: decode_detect('0 0.5 0.5 0.5 0.5 0.5', self.image, labels),
            lambda: decode_pose('0 0.5 0.5 0.5 0.5 0.1 0.1', self.image, labels),
            lambda: decode_pose('0 0.5 0.5 0.5 0.5 0.1 0.1 2 3', self.image, labels),
        )
        for decode in invalid:
            with self.subTest(decode=decode):
                with self.assertRaises(ValueError):
                    decode()

    def test_decoders_reject_non_integer_and_out_of_range_class_tokens(self) -> None:
        labels = LabelCatalog(names=('first', 'second'))
        invalid = (
            lambda: decode_detect('x 0.5 0.5 0.5 0.5', self.image, labels),
            lambda: decode_detect('2 0.5 0.5 0.5 0.5', self.image, labels),
            lambda: decode_segment('x 0.1 0.1 0.5 0.1 0.3 0.5', self.image, labels),
            lambda: decode_segment('2 0.1 0.1 0.5 0.1 0.3 0.5', self.image, labels),
            lambda: decode_pose('x 0.5 0.5 0.5 0.5 0.1 0.1 2 0.2 0.2 1', self.image, labels),
            lambda: decode_pose('2 0.5 0.5 0.5 0.5 0.1 0.1 2 0.2 0.2 1', self.image, labels),
        )
        for decode in invalid:
            with self.subTest(decode=decode):
                with self.assertRaises(ValueError):
                    decode()

    def test_decoders_reject_non_finite_numbers(self) -> None:
        labels = LabelCatalog(names=('label',))
        invalid = (
            lambda: decode_detect('0 nan 0.5 0.5 0.5', self.image, labels),
            lambda: decode_segment('0 0.1 0.1 inf 0.1 0.3 0.5', self.image, labels),
            lambda: decode_pose('0 0.5 0.5 0.5 0.5 0.1 -inf 2', self.image, labels),
        )
        for decode in invalid:
            with self.subTest(decode=decode):
                with self.assertRaises(ValueError):
                    decode()

    def test_segment_decoder_requires_three_vertices_and_coordinate_pairs(self) -> None:
        labels = LabelCatalog(names=('mask',))
        invalid = ('0 0.1 0.1 0.5 0.5', '0 0.1 0.1 0.5 0.1 0.3 0.5 0.7')
        for line in invalid:
            with self.subTest(line=line):
                with self.assertRaises(ValueError):
                    decode_segment(line, self.image, labels)

    def test_pose_decoder_rejects_visibility_outside_zero_one_two(self) -> None:
        labels = LabelCatalog(names=('point',))
        for visibility in ('-1', '1.5', '3'):
            with self.subTest(visibility=visibility):
                with self.assertRaises(ValueError):
                    decode_pose(f'0 0.5 0.5 0.5 0.5 0.1 0.1 {visibility}', self.image, labels)

    def test_pose_decoder_requires_class_zero_even_when_other_class_is_in_range(self) -> None:
        labels = LabelCatalog(names=('first', 'second'))

        with self.assertRaises(ValueError):
            decode_pose('1 0.5 0.5 0.5 0.5 0.1 0.1 2 0.2 0.2 1', self.image, labels)

    def test_segment_rejects_shapes_with_fewer_than_three_points(self) -> None:
        with self.assertRaisesRegex(ValueError, '至少三点'):
            encode_segment(Polyline(label='line', points=[[0, 0], [1, 1]]), self.image, LabelCatalog(names=('line',)))

    def test_encodes_pose_in_catalog_order_with_fixed_class_zero(self) -> None:
        pose = Pose(
            label='start',
            x1=20,
            y1=10,
            x2=100,
            y2=50,
            keypoints=(Keypoint(label='end', x=100, y=50), Keypoint(label='start', x=20, y=10)),
        )

        self.assertEqual(
            '0 0.300000 0.300000 0.400000 0.400000 0.100000 0.100000 2 0.500000 0.500000 2',
            encode_pose(pose, self.image, LabelCatalog(names=('start', 'end'))),
        )

    def test_pose_encoder_requires_single_class_and_exact_keypoint_catalog(self) -> None:
        catalog = LabelCatalog(names=('start', 'end'))
        invalid = (
            Pose(label='start', x1=20, y1=10, x2=100, y2=50, keypoints=(Keypoint(label='start', x=20, y=10),)),
            Pose(
                label='other',
                x1=20,
                y1=10,
                x2=100,
                y2=50,
                keypoints=(Keypoint(label='start', x=20, y=10), Keypoint(label='end', x=100, y=50)),
            ),
        )
        for pose in invalid:
            with self.subTest(pose=pose):
                with self.assertRaises(ValueError):
                    encode_pose(pose, self.image, catalog)

    def test_detect_preserves_legacy_out_of_image_bbox_encoding(self) -> None:
        labels = LabelCatalog(names=('x',))
        cases = (
            (Bbox(label='x', x1=-30, y1=10, x2=-10, y2=50), '0 -0.100000 0.300000 0.100000 0.400000'),
            (Bbox(label='x', x1=190, y1=10, x2=210, y2=50), '0 1.000000 0.300000 0.100000 0.400000'),
        )
        for annotation, expected in cases:
            with self.subTest(annotation=annotation):
                self.assertEqual(expected, encode_detect(annotation, self.image, labels))

    def test_pose_allows_outside_corners_when_normalized_bbox_values_are_valid(self) -> None:
        keypoint_labels = LabelCatalog(names=('point',))
        cases = (
            (
                Pose(label='point', x1=-10, y1=10, x2=10, y2=50, keypoints=(Keypoint(label='point', x=100, y=50),)),
                '0 0.000000 0.300000 0.100000 0.400000 0.500000 0.500000 2',
            ),
            (
                Pose(label='point', x1=190, y1=10, x2=210, y2=50, keypoints=(Keypoint(label='point', x=100, y=50),)),
                '0 1.000000 0.300000 0.100000 0.400000 0.500000 0.500000 2',
            ),
        )
        for pose, expected in cases:
            with self.subTest(pose=pose):
                self.assertEqual(expected, encode_pose(pose, self.image, keypoint_labels))

    def test_pose_rejects_normalized_bbox_values_outside_unit_range(self) -> None:
        keypoint_labels = LabelCatalog(names=('point',))
        for pose in (
            Pose(label='point', x1=-30, y1=10, x2=-10, y2=50, keypoints=(Keypoint(label='point', x=100, y=50),)),
            Pose(label='point', x1=210, y1=10, x2=230, y2=50, keypoints=(Keypoint(label='point', x=100, y=50),)),
        ):
            with self.subTest(pose=pose):
                with self.assertRaises(ValueError):
                    encode_pose(pose, self.image, keypoint_labels)

    def test_segment_rejects_points_outside_image(self) -> None:
        labels = LabelCatalog(names=('mask',))
        annotations = (
            Polygon(label='mask', points=[[-1, 10], [100, 10], [60, 50]]),
            Polygon(label='mask', points=[[20, 10], [201, 10], [60, 50]]),
        )
        for annotation in annotations:
            with self.subTest(annotation=annotation):
                with self.assertRaises(ValueError):
                    encode_segment(annotation, self.image, labels)


if __name__ == '__main__':
    unittest.main()
