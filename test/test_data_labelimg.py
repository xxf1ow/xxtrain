import tempfile
import unittest
from pathlib import Path

from xxtrain.data import AnnotationType, Bbox, ImageInfo
from xxtrain.data.formats import read_labelimg

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


if __name__ == '__main__':
    unittest.main()
