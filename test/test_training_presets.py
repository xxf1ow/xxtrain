import shutil
import tempfile
import unittest
from pathlib import Path

from test.support.scenarios import SCENARIO_PATHS, load_case_scenario
from xxtrain.pipeline import convert_dataset

FIXTURES_PATH = Path(__file__).resolve().parent / 'fixtures'


class TrainingPresetTest(unittest.TestCase):
    def test_all_presets_load(self) -> None:
        for name in SCENARIO_PATHS:
            with self.subTest(name=name):
                self.assertIsNotNone(load_case_scenario(name))

    def test_tuned_classify_converts_standard_classify_fixture(self) -> None:
        with tempfile.TemporaryDirectory(prefix='xxtrain-tuned-classify-') as temp_dir:
            root_path = Path(temp_dir) / 'tuned-classify'
            shutil.copytree(FIXTURES_PATH / 'standard-classify', root_path)
            scenario = load_case_scenario('tuned-classify')

            report = convert_dataset(
                scenario.dataset, root_path, split=scenario.split, reserve_no_label=scenario.reserve_no_label
            )

            self.assertEqual(6, report.train_image_count + report.val_image_count)
            self.assertTrue((root_path / 'classify' / 'train').is_dir())


if __name__ == '__main__':
    unittest.main()
