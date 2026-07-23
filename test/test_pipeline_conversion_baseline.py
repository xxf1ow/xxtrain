import gc
import json
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path

from test.support.output_manifest import collect_output_manifest
from test.test_conversion_baseline import EXPECTED_PATH, FIXTURES_PATH
from xxtrain.pipeline.workflow import convert_dataset

STANDARD_CASES = [
    ('detect', 'standard-detect'),
    ('segment', 'standard-segment'),
    ('pose', 'standard-pose'),
    ('classify', 'standard-classify'),
]

CUSTOM_CASES = [
    ('point-detect', 'point'),
    ('point-classify', 'point'),
    ('knob-detect', 'knob'),
    ('light1-detect', 'light'),
]


class NewPipelineConversionBaselineTest(unittest.TestCase):
    cases = STANDARD_CASES + CUSTOM_CASES

    def assert_conversion(self, task_name: str, fixture_name: str) -> None:
        with tempfile.TemporaryDirectory(prefix='xxtrain-new-pipeline-') as temp_dir:
            root_path = Path(temp_dir) / fixture_name
            shutil.copytree(FIXTURES_PATH / fixture_name, root_path)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ResourceWarning)
                convert_dataset(task_name, root_path, split=10, reserve_no_label=False)
                gc.collect()
            self.assertEqual(
                [],
                [item for item in caught if issubclass(item.category, ResourceWarning)],
            )
            actual = collect_output_manifest(root_path, task_name)
            expected = json.loads((EXPECTED_PATH / f'{task_name}.json').read_text(encoding='utf-8'))
            self.assertEqual(expected, actual)

    def test_new_pipeline_conversions(self) -> None:
        for task_name, fixture_name in self.cases:
            with self.subTest(task_name=task_name):
                self.assert_conversion(task_name, fixture_name)


if __name__ == '__main__':
    unittest.main()
