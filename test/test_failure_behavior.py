import json
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from test.support.output_manifest import collect_output_manifest
from test.support.scenarios import load_case_scenario
from xxtrain.data import ImageInfo
from xxtrain.data.formats import read_labelimg, read_labelme
from xxtrain.pipeline import convert_dataset

FIXTURES_PATH = Path(__file__).resolve().parent / 'fixtures'
EXPECTED_PATH = Path(__file__).resolve().parent / 'expected' / 'conversions'
SYMLINK_FALLBACK_CASES = [('detect', 'standard-detect'), ('classify', 'standard-classify')]


@contextmanager
def unavailable_symlinks():
    with patch('xxtrain.pipeline.sinks.os.symlink', side_effect=OSError('symlink privilege unavailable')):
        yield


class FailureBehaviorTest(unittest.TestCase):
    def copy_fixture(self, fixture_name: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory(prefix='xxtrain-failures-')
        self.addCleanup(temp_dir.cleanup)
        root_path = Path(temp_dir.name) / fixture_name
        shutil.copytree(FIXTURES_PATH / fixture_name, root_path)
        return root_path

    def test_labelimg_rejects_mismatched_image_width(self) -> None:
        root_path = self.copy_fixture('standard-detect')
        annotation_path = root_path / 'src' / '20260620' / 'anns' / '0000.xml'

        with self.assertRaisesRegex(Exception, '图片与标签不对应'):
            read_labelimg(annotation_path, ImageInfo(width=1919, height=1080))

    def test_labelme_rejects_mismatched_image_width(self) -> None:
        root_path = self.copy_fixture('standard-segment')
        annotation_path = root_path / 'src' / '251010' / 'anns_seg' / '0000.json'

        with self.assertRaisesRegex(Exception, '图片与标签不对应'):
            read_labelme(annotation_path, ImageInfo(width=1919, height=1080))

    def test_invalid_annotation_reports_parse_failure(self) -> None:
        root_path = self.copy_fixture('standard-segment')
        annotation_path = root_path / 'src' / '251010' / 'anns_seg' / 'invalid.json'
        annotation_path.write_text('{', encoding='utf-8')

        with self.assertRaisesRegex(Exception, 'Failed to parse annotation'):
            read_labelme(annotation_path, ImageInfo(width=1920, height=1080))

    def test_classification_rejects_extra_class_directory(self) -> None:
        root_path = self.copy_fixture('standard-classify')
        (root_path / 'src' / 'extra' / 'imgs').mkdir(parents=True)
        scenario = load_case_scenario('classify')

        with self.assertRaisesRegex(ValueError, 'Classification labels mismatch'):
            convert_dataset(
                scenario.dataset, root_path, split=scenario.split, reserve_no_label=scenario.reserve_no_label
            )

    def test_classification_rejects_split_without_train_samples(self) -> None:
        root_path = self.copy_fixture('standard-classify')
        for class_path in (root_path / 'src').iterdir():
            images_path = class_path / 'imgs'
            if not images_path.is_dir():
                continue
            for image_path in sorted(images_path.iterdir())[1:]:
                image_path.unlink()
        scenario = load_case_scenario('classify')

        with self.assertRaisesRegex(ValueError, 'without train or val samples'):
            convert_dataset(
                scenario.dataset, root_path, split=scenario.split, reserve_no_label=scenario.reserve_no_label
            )

    def test_symlink_failure_falls_back_to_copied_images(self) -> None:
        for task_type, fixture_name in SYMLINK_FALLBACK_CASES:
            with self.subTest(task_type=task_type):
                root_path = self.copy_fixture(fixture_name)
                scenario = load_case_scenario(task_type)
                with unavailable_symlinks():
                    convert_dataset(
                        scenario.dataset, root_path, split=scenario.split, reserve_no_label=scenario.reserve_no_label
                    )

                actual = collect_output_manifest(root_path, task_type)
                expected = json.loads((EXPECTED_PATH / f'{task_type}.json').read_text(encoding='utf-8'))
                self.assertEqual(expected, actual)
                for relative_path in actual['images']:
                    image_path = root_path / task_type / relative_path
                    self.assertTrue(image_path.is_file())
                    self.assertFalse(image_path.is_symlink())


if __name__ == '__main__':
    unittest.main()
