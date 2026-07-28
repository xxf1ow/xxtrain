import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from xxtrain.data import AnnotationType, Bbox, ImageInfo, Keypoint, Polygon, Polyline, Pose
from xxtrain.data.formats import read_labelimg, write_labelimg

FIXTURES_PATH = Path(__file__).resolve().parent / 'fixtures'


class LabelImgReaderTest(unittest.TestCase):
    def test_reads_bbox_in_file_order(self) -> None:
        path = FIXTURES_PATH / 'standard-detect' / 'src' / '20260620' / 'anns' / '0000.xml'

        annotations = read_labelimg(path, ImageInfo(width=1920, height=1080))

        self.assertEqual(1, len(annotations))
        self.assertIsInstance(annotations[0], Bbox)
        self.assertEqual('cc', annotations[0].label)
        self.assertEqual(AnnotationType.BBOX, annotations[0].type)
        self.assertEqual((911.0, 303.0, 1400.0, 735.0), annotations[0].bbox)

    def test_missing_file_returns_empty_list(self) -> None:
        self.assertEqual([], read_labelimg('missing.xml', ImageInfo(width=1, height=1)))

    def test_size_mismatch_preserves_wrapped_message(self) -> None:
        path = FIXTURES_PATH / 'standard-detect' / 'src' / '20260620' / 'anns' / '0000.xml'
        with self.assertRaisesRegex(Exception, '图片与标签不对应'):
            read_labelimg(path, ImageInfo(width=1919, height=1080))

    def test_invalid_bbox_is_reported_as_parse_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'invalid.xml'
            path.write_text(
                '<annotation><size><width>10</width><height>10</height></size>'
                '<object><name>x</name><bndbox><xmin>5</xmin><ymin>1</ymin>'
                '<xmax>5</xmax><ymax>2</ymax></bndbox></object></annotation>',
                encoding='utf-8',
            )
            with self.assertRaisesRegex(Exception, 'Failed to parse annotation'):
                read_labelimg(path, ImageInfo(width=10, height=10))


class LabelImgWriterTest(unittest.TestCase):
    def test_round_trips_two_bboxes_with_decimal_coordinates(self) -> None:
        annotations = [
            Bbox(label='first', group='ignored', x1=1.25, y1=2.5, x2=30.75, y2=40.125),
            Bbox(label='second', group=7, x1=50.5, y1=60.25, x2=70.75, y2=80.5),
        ]
        image_info = ImageInfo(width=1920.0, height=1080.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'nested' / 'annotations.xml'
            write_labelimg(annotations, path, image_info)

            root = ET.parse(path).getroot()
            self.assertEqual('1920', root.findtext('size/width'))
            self.assertEqual('1080', root.findtext('size/height'))
            self.assertEqual('1.25', root.findtext('object/bndbox/xmin'))
            self.assertEqual('40.125', root.findtext('object/bndbox/ymax'))
            restored = read_labelimg(path, image_info)

        self.assertEqual(
            [(annotation.label, annotation.bbox) for annotation in annotations],
            [(annotation.label, annotation.bbox) for annotation in restored],
        )

    def test_writes_empty_annotations(self) -> None:
        image_info = ImageInfo(width=10, height=20)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'empty.xml'
            write_labelimg([], path, image_info)

            self.assertEqual([], read_labelimg(path, image_info))

    def test_rejects_unsupported_annotations_without_path_side_effects(self) -> None:
        unsupported = [
            ('polygon', Polygon(label='shape', points=((1, 1), (2, 1), (1, 2)))),
            ('polyline', Polyline(label='line', points=((1, 1), (2, 2)))),
            ('pose', Pose(label='person', x1=1, y1=1, x2=2, y2=2, keypoints=(Keypoint(label='head', x=1.5, y=1.5),))),
        ]
        image_info = ImageInfo(width=10, height=10)
        bbox = Bbox(label='valid', x1=1, y1=1, x2=2, y2=2)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name, annotation in unsupported:
                with self.subTest(name=name, path='existing'):
                    path = root / f'{name}-existing.xml'
                    path.write_text('sentinel', encoding='utf-8')

                    with self.assertRaises(TypeError):
                        write_labelimg([bbox, annotation], path, image_info)

                    self.assertEqual('sentinel', path.read_text(encoding='utf-8'))
                with self.subTest(name=name, path='missing'):
                    path = root / name / 'missing.xml'

                    with self.assertRaises(TypeError):
                        write_labelimg([annotation], path, image_info)

                    self.assertFalse(path.exists())
                    self.assertFalse(path.parent.exists())

    def test_rejects_non_integral_image_dimensions_without_creating_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'missing' / 'annotations.xml'

            with self.assertRaises(ValueError):
                write_labelimg([], path, ImageInfo(width=10.5, height=20))

            self.assertFalse(path.exists())
            self.assertFalse(path.parent.exists())


if __name__ == '__main__':
    unittest.main()
