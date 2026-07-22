import gc
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path

from test.support.current_api import convert_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_PATH = PROJECT_ROOT / 'test' / 'fixtures'


BASELINE_CASES = [
    ('detect', 'standard-detect'),
    ('segment', 'standard-segment'),
    ('pose', 'standard-pose'),
    ('classify', 'standard-classify'),
    ('point-detect', 'point'),
    ('point-classify', 'point'),
    ('point-segment', 'point'),
    ('knob-detect', 'knob'),
    ('knob-segment', 'knob'),
    ('light1-detect', 'light'),
    ('light2-detect', 'light'),
]


class ConversionBaselineTest(unittest.TestCase):
    def assert_conversion(self, task_type: str, fixture_name: str) -> None:
        with tempfile.TemporaryDirectory(prefix='xxtrain-conversion-') as temp_dir:
            root_path = Path(temp_dir) / fixture_name
            shutil.copytree(FIXTURES_PATH / fixture_name, root_path)

            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter('always', ResourceWarning)
                convert_dataset(task_type, str(root_path), split=10, reserve_no_label=False)
                gc.collect()
            resource_warnings = [item for item in caught_warnings if issubclass(item.category, ResourceWarning)]
            self.assertEqual([], resource_warnings)

            output_path = root_path / task_type
            dataset_path = output_path / 'dataset.yaml'
            train_path = output_path / 'train.txt'
            val_path = output_path / 'val.txt'
            self.assertTrue(dataset_path.is_file())
            self.assertTrue(train_path.is_file())
            self.assertTrue(val_path.is_file())

            train_images = [Path(line) for line in train_path.read_text(encoding='utf-8').splitlines() if line]
            val_images = [Path(line) for line in val_path.read_text(encoding='utf-8').splitlines() if line]
            self.assertTrue(train_images)
            self.assertTrue(val_images)
            for image_path in train_images + val_images:
                self.assertTrue(image_path.is_file(), image_path)

    def test_baseline_conversions(self) -> None:
        for task_type, fixture_name in BASELINE_CASES:
            with self.subTest(task_type=task_type):
                self.assert_conversion(task_type, fixture_name)


if __name__ == '__main__':
    unittest.main()
