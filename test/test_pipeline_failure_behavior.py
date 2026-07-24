import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test.support.output_manifest import collect_output_manifest
from test.support.scenarios import load_case_scenario
from test.test_conversion_baseline import EXPECTED_PATH, FIXTURES_PATH
from xxtrain.pipeline.sinks import YoloDatasetSink
from xxtrain.pipeline.workflow import convert_dataset


class NewPipelineFailureBehaviorTest(unittest.TestCase):
    def copy_fixture(self, fixture_name: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory(prefix='xxtrain-new-failures-')
        self.addCleanup(temp_dir.cleanup)
        root_path = Path(temp_dir.name) / fixture_name
        shutil.copytree(FIXTURES_PATH / fixture_name, root_path)
        return root_path

    def test_classification_rejects_extra_class_directory(self) -> None:
        root_path = self.copy_fixture('standard-classify')
        (root_path / 'src' / 'extra' / 'imgs').mkdir(parents=True)
        scenario = load_case_scenario('classify')
        with self.assertRaisesRegex(ValueError, 'Classification labels mismatch'):
            convert_dataset(
                scenario.dataset, root_path, split=scenario.split, reserve_no_label=scenario.reserve_no_label
            )

    def test_classification_rejects_unusable_split(self) -> None:
        root_path = self.copy_fixture('standard-classify')
        for class_path in (root_path / 'src').iterdir():
            images_path = class_path / 'imgs'
            if images_path.is_dir():
                for image_path in sorted(images_path.iterdir())[1:]:
                    image_path.unlink()
        scenario = load_case_scenario('classify')
        with self.assertRaisesRegex(ValueError, 'without train or val samples'):
            convert_dataset(
                scenario.dataset, root_path, split=scenario.split, reserve_no_label=scenario.reserve_no_label
            )

    def test_symlink_failure_falls_back_to_copy(self) -> None:
        for task_name, fixture_name in (('detect', 'standard-detect'), ('classify', 'standard-classify')):
            with self.subTest(task_name=task_name):
                root_path = self.copy_fixture(fixture_name)
                scenario = load_case_scenario(task_name)
                with patch('xxtrain.pipeline.sinks.os.symlink', side_effect=OSError('unavailable')):
                    convert_dataset(
                        scenario.dataset, root_path, split=scenario.split, reserve_no_label=scenario.reserve_no_label
                    )
                actual = collect_output_manifest(root_path, task_name)
                expected = json.loads((EXPECTED_PATH / f'{task_name}.json').read_text(encoding='utf-8'))
                self.assertEqual(expected, actual)
                for relative_path in actual['images']:
                    output = root_path / task_name / relative_path
                    self.assertTrue(output.is_file())
                    self.assertFalse(output.is_symlink())

    def test_sink_failure_keeps_outputs_written_before_failure(self) -> None:
        root_path = self.copy_fixture('standard-detect')
        original_write = YoloDatasetSink.write
        call_count = 0

        def fail_on_second(sink, item, context):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError('second output failed')
            return original_write(sink, item, context)

        with patch.object(YoloDatasetSink, 'write', autospec=True, side_effect=fail_on_second):
            with self.assertRaisesRegex(OSError, 'second output failed'):
                scenario = load_case_scenario('detect')
                convert_dataset(
                    scenario.dataset, root_path, split=scenario.split, reserve_no_label=scenario.reserve_no_label
                )
        self.assertTrue((root_path / 'detect' / '20260620' / '0000.jpg').exists())
        self.assertTrue((root_path / 'detect' / '20260620' / '0000.txt').exists())
        self.assertFalse((root_path / 'detect' / 'train.txt').exists())


if __name__ == '__main__':
    unittest.main()
