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
            label='person',
            x1=20,
            y1=10,
            x2=100,
            y2=50,
            keypoints=(
                Keypoint(label='nose', x=40, y=20, visibility=2),
                Keypoint(label='wrist', x=80, y=40, visibility=1),
            ),
        )
        labels = LabelCatalog(names=('person', 'nose', 'wrist'))

        decoded = decode_pose(encode_pose(pose, self.image, labels), self.image, labels)

        self.assertEqual(pose, decoded.wrap(id=pose.id))

    def test_pose_requires_object_label_and_keypoint_label(self) -> None:
        with self.assertRaisesRegex(ValueError, 'at least one object label and one keypoint'):
            decode_pose('0 0.5 0.5 0.5 0.5', self.image, LabelCatalog(('person',)))

    def test_pose_label_must_match_first_catalog_label(self) -> None:
        pose = Pose(
            label='person',
            x1=20,
            y1=10,
            x2=100,
            y2=50,
            keypoints=(Keypoint(label='nose', x=40, y=20, visibility=2), Keypoint(label='wrist', x=80, y=40)),
        )
        labels = LabelCatalog(names=('person', 'nose', 'wrist'))

        with self.assertRaisesRegex(ValueError, 'Pose label must match'):
            encode_pose(pose.wrap(label='face'), self.image, labels)

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
        labels = LabelCatalog(names=('person', 'label'))
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
        labels = LabelCatalog(names=('person', 'point'))
        invalid = (
            lambda: decode_detect('x 0.5 0.5 0.5 0.5', self.image, labels),
            lambda: decode_detect('2 0.5 0.5 0.5 0.5', self.image, labels),
            lambda: decode_segment('x 0.1 0.1 0.5 0.1 0.3 0.5', self.image, labels),
            lambda: decode_segment('2 0.1 0.1 0.5 0.1 0.3 0.5', self.image, labels),
        )
        for decode in invalid:
            with self.subTest(decode=decode):
                with self.assertRaises(ValueError):
                    decode()
        for line in ('x 0.5 0.5 0.5 0.5 0.1 0.1 2', '2 0.5 0.5 0.5 0.5 0.1 0.1 2'):
            with self.subTest(line=line), self.assertRaisesRegex(ValueError, 'YOLO class id'):
                decode_pose(line, self.image, labels)

    def test_class_tokens_reject_python_integer_conveniences(self) -> None:
        labels = LabelCatalog(names=('person', 'label'))
        for token in ('0_0', '+0', '-0', '0.0', '0e0', '+', ''):
            with self.subTest(token=token):
                with self.assertRaises(ValueError):
                    decode_detect(f'{token} 0.5 0.5 0.5 0.5', self.image, labels)

    def test_decoders_reject_non_finite_numbers(self) -> None:
        labels = LabelCatalog(names=('person', 'label'))
        invalid = (
            lambda: decode_detect('0 nan 0.5 0.5 0.5', self.image, labels),
            lambda: decode_segment('0 0.1 0.1 inf 0.1 0.3 0.5', self.image, labels),
        )
        for decode in invalid:
            with self.subTest(decode=decode):
                with self.assertRaises(ValueError):
                    decode()
        with self.assertRaisesRegex(ValueError, 'YOLO values must be finite'):
            decode_pose('0 0.5 0.5 0.5 0.5 0.1 -inf 2', self.image, labels)

    def test_segment_decoder_requires_three_vertices_and_coordinate_pairs(self) -> None:
        labels = LabelCatalog(names=('mask',))
        invalid = ('0 0.1 0.1 0.5 0.5', '0 0.1 0.1 0.5 0.1 0.3 0.5 0.7')
        for line in invalid:
            with self.subTest(line=line):
                with self.assertRaises(ValueError):
                    decode_segment(line, self.image, labels)

    def test_pose_decoder_rejects_visibility_outside_zero_one_two(self) -> None:
        labels = LabelCatalog(names=('person', 'point'))
        for visibility in ('-1', '1.5', '3'):
            with self.subTest(visibility=visibility):
                with self.assertRaises(ValueError):
                    decode_pose(f'0 0.5 0.5 0.5 0.5 0.1 0.1 {visibility}', self.image, labels)

    def test_pose_decoder_requires_class_zero_even_when_other_class_is_in_range(self) -> None:
        labels = LabelCatalog(names=('first', 'second'))

        with self.assertRaises(ValueError):
            decode_pose('1 0.5 0.5 0.5 0.5 0.1 0.1 2', self.image, labels)

    def test_decoded_boxes_require_positive_width_and_height(self) -> None:
        labels = LabelCatalog(names=('person', 'point'))
        invalid = (
            lambda: decode_detect('0 0.5 0.5 0 0.5', self.image, labels),
            lambda: decode_detect('0 0.5 0.5 0.5 -0.5', self.image, labels),
            lambda: decode_pose('0 0.5 0.5 0 0.5 0.1 0.1 2', self.image, labels),
            lambda: decode_pose('0 0.5 0.5 0.5 -0.5 0.1 0.1 2', self.image, labels),
        )
        for decode in invalid:
            with self.subTest(decode=decode):
                with self.assertRaises(ValueError):
                    decode()

    def test_segment_decoder_rejects_normalized_coordinates_outside_unit_range(self) -> None:
        labels = LabelCatalog(names=('mask',))
        invalid = (
            '0 -0.1 0.1 0.5 0.1 0.3 0.5',
            '0 0.1 -0.1 0.5 0.1 0.3 0.5',
            '0 0.1 0.1 1.1 0.1 0.3 0.5',
            '0 0.1 0.1 0.5 0.1 0.3 1.1',
        )
        for line in invalid:
            with self.subTest(line=line), self.assertRaises(ValueError):
                decode_segment(line, self.image, labels)

    def test_segment_decoder_accepts_encoder_compatible_boundary_coordinates(self) -> None:
        labels = LabelCatalog(names=('mask',))
        line = '0 0.000000 0.000000 1.000000 0.000000 0.000000 1.000000'

        decoded = decode_segment(line, self.image, labels)

        self.assertEqual(line, encode_segment(decoded, self.image, labels))

    def test_pose_decoder_rejects_bbox_tokens_outside_unit_range(self) -> None:
        labels = LabelCatalog(names=('person', 'point'))
        invalid = (
            '0 -0.1 0.5 0.2 0.2 0.5 0.5 2',
            '0 0.5 -0.1 0.2 0.2 0.5 0.5 2',
            '0 1.1 0.5 0.2 0.2 0.5 0.5 2',
            '0 0.5 1.1 0.2 0.2 0.5 0.5 2',
            '0 0.5 0.5 1.1 0.2 0.5 0.5 2',
            '0 0.5 0.5 0.2 1.1 0.5 0.5 2',
        )
        for line in invalid:
            with self.subTest(line=line), self.assertRaises(ValueError):
                decode_pose(line, self.image, labels)

    def test_pose_decoder_rejects_keypoint_coordinates_outside_unit_range(self) -> None:
        labels = LabelCatalog(names=('person', 'point'))
        invalid = (
            '0 0.5 0.5 0.2 0.2 -0.1 0.5 2',
            '0 0.5 0.5 0.2 0.2 0.5 -0.1 2',
            '0 0.5 0.5 0.2 0.2 1.1 0.5 2',
            '0 0.5 0.5 0.2 0.2 0.5 1.1 2',
        )
        for line in invalid:
            with self.subTest(line=line), self.assertRaises(ValueError):
                decode_pose(line, self.image, labels)

    def test_pose_decoder_rejects_in_range_tokens_whose_bbox_corners_escape_image(self) -> None:
        labels = LabelCatalog(names=('person', 'point'))
        invalid = (
            '0 0 0.5 0.1 0.2 0.5 0.5 2',
            '0 1 0.5 0.1 0.2 0.5 0.5 2',
            '0 0.5 0 0.2 0.1 0.5 0.5 2',
            '0 0.5 1 0.2 0.1 0.5 0.5 2',
        )
        for line in invalid:
            with self.subTest(line=line), self.assertRaises(ValueError):
                decode_pose(line, self.image, labels)

    def test_pose_decoder_accepts_encoder_compatible_boundary_values(self) -> None:
        labels = LabelCatalog(names=('person', 'point'))
        line = '0 0.500000 0.500000 1.000000 1.000000 0.000000 1.000000 2'

        decoded = decode_pose(line, self.image, labels)

        self.assertEqual((0.0, 0.0, 200.0, 100.0), decoded.bbox)
        self.assertEqual(line, encode_pose(decoded, self.image, labels))

    def test_segment_rejects_shapes_with_fewer_than_three_points(self) -> None:
        with self.assertRaisesRegex(ValueError, '至少三点'):
            encode_segment(Polyline(label='line', points=[[0, 0], [1, 1]]), self.image, LabelCatalog(names=('line',)))

    def test_encodes_pose_in_catalog_order_with_fixed_class_zero(self) -> None:
        pose = Pose(
            label='person',
            x1=20,
            y1=10,
            x2=100,
            y2=50,
            keypoints=(Keypoint(label='end', x=100, y=50), Keypoint(label='start', x=20, y=10)),
        )

        self.assertEqual(
            '0 0.300000 0.300000 0.400000 0.400000 0.100000 0.100000 2 0.500000 0.500000 2',
            encode_pose(pose, self.image, LabelCatalog(names=('person', 'start', 'end'))),
        )

    def test_pose_encoder_requires_single_class_and_exact_keypoint_catalog(self) -> None:
        catalog = LabelCatalog(names=('person', 'start', 'end'))
        invalid = (
            Pose(label='person', x1=20, y1=10, x2=100, y2=50, keypoints=(Keypoint(label='start', x=20, y=10),)),
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

    def test_pose_encoder_rejects_empty_keypoint_catalog_explicitly(self) -> None:
        pose = Pose(label='person', x1=20, y1=10, x2=100, y2=50, keypoints=(Keypoint(label='point', x=20, y=10),))
        empty_catalog = object.__new__(LabelCatalog)
        object.__setattr__(empty_catalog, 'names', ())

        with self.assertRaises(ValueError):
            encode_pose(pose, self.image, empty_catalog)

    def test_detect_encoder_rejects_bbox_corners_outside_image(self) -> None:
        labels = LabelCatalog(names=('x',))
        annotations = (Bbox(label='x', x1=-1, y1=10, x2=20, y2=50), Bbox(label='x', x1=190, y1=10, x2=210, y2=50))
        for annotation in annotations:
            with self.subTest(annotation=annotation):
                with self.assertRaisesRegex(ValueError, 'YOLO detect bbox must be inside image bounds'):
                    encode_detect(annotation, self.image, labels)

    def test_detect_decoder_rejects_out_of_image_bboxes(self) -> None:
        labels = LabelCatalog(names=('x',))
        for line in ('0 -0.1 0.5 0.1 0.4', '0 0.0 0.5 0.1 0.4'):
            with self.subTest(line=line):
                with self.assertRaises(ValueError):
                    decode_detect(line, self.image, labels)

    def test_detect_accepts_exact_image_boundary(self) -> None:
        labels = LabelCatalog(names=('x',))
        annotation = Bbox(label='x', x1=0, y1=0, x2=200, y2=100)

        decoded = decode_detect('0 0.5 0.5 1.0 1.0', self.image, labels)

        self.assertEqual(annotation, decoded.wrap(id=annotation.id))
        self.assertEqual('0 0.500000 0.500000 1.000000 1.000000', encode_detect(annotation, self.image, labels))

    def test_detect_accepts_boundary_after_coordinate_quantization(self) -> None:
        labels = LabelCatalog(names=('x',))
        image = ImageInfo(width=1920, height=1080)
        annotation = Bbox(label='x', x1=929, y1=904, x2=1144, y2=1080)

        self.assertEqual('0 0.539844 0.918519 0.111979 0.162963', encode_detect(annotation, image, labels))

    def test_detect_encoder_rejects_bboxes_that_escape_after_six_decimal_formatting(self) -> None:
        labels = LabelCatalog(names=('x',))
        annotation = Bbox(label='x', x1=0, y1=10, x2=0.0001, y2=50)

        with self.assertRaisesRegex(ValueError, 'YOLO detect bbox must be inside image bounds'):
            encode_detect(annotation, self.image, labels)

    def test_detect_encoder_rejects_non_finite_derived_values(self) -> None:
        labels = LabelCatalog(names=('x',))
        annotation = Bbox(label='x', x1=1e308, y1=0, x2=1.7e308, y2=1)
        image_info = ImageInfo(width=1.7e308, height=100)

        with self.assertRaisesRegex(ValueError, 'YOLO bbox values must be finite'):
            encode_detect(annotation, image_info, labels)

    def test_pose_encoder_rejects_bbox_corners_outside_image(self) -> None:
        keypoint_labels = LabelCatalog(names=('person', 'point'))
        poses = (
            Pose(label='person', x1=-10, y1=10, x2=10, y2=50, keypoints=(Keypoint(label='point', x=100, y=50),)),
            Pose(label='person', x1=10, y1=-10, x2=50, y2=10, keypoints=(Keypoint(label='point', x=100, y=50),)),
            Pose(label='person', x1=190, y1=10, x2=210, y2=50, keypoints=(Keypoint(label='point', x=100, y=50),)),
            Pose(label='person', x1=10, y1=90, x2=50, y2=110, keypoints=(Keypoint(label='point', x=100, y=50),)),
        )
        for pose in poses:
            with self.subTest(pose=pose):
                with self.assertRaisesRegex(ValueError, 'YOLO pose bbox must be inside image bounds'):
                    encode_pose(pose, self.image, keypoint_labels)

    def test_pose_rejects_normalized_bbox_values_outside_unit_range(self) -> None:
        keypoint_labels = LabelCatalog(names=('person', 'point'))
        for pose in (
            Pose(label='person', x1=-30, y1=10, x2=-10, y2=50, keypoints=(Keypoint(label='point', x=100, y=50),)),
            Pose(label='person', x1=210, y1=10, x2=230, y2=50, keypoints=(Keypoint(label='point', x=100, y=50),)),
        ):
            with self.subTest(pose=pose):
                with self.assertRaisesRegex(ValueError, 'YOLO pose bbox must be inside image bounds'):
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
