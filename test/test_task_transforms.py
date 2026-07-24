import shutil
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from test.support.scenarios import load_case_scenario
from xxtrain.pipeline import convert_dataset

FIXTURES_PATH = Path(__file__).resolve().parent / 'fixtures'


class TaskTransformsTest(unittest.TestCase):
    def copy_fixture(self, fixture_name: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory(prefix='xxtrain-transforms-')
        self.addCleanup(temp_dir.cleanup)
        root_path = Path(temp_dir.name) / fixture_name
        shutil.copytree(FIXTURES_PATH / fixture_name, root_path)
        return root_path

    def test_point_task_transforms(self) -> None:
        root_path = self.copy_fixture('point')

        for task_type in ('point-detect', 'point-classify', 'point-segment'):
            scenario = load_case_scenario(task_type)
            convert_dataset(
                scenario.dataset, root_path, split=scenario.split, reserve_no_label=scenario.reserve_no_label
            )

        dataset = yaml.safe_load((root_path / 'point-detect' / 'dataset.yaml').read_text(encoding='utf-8'))
        self.assertEqual({0: 'Point'}, dataset['names'])
        self.assertEqual(
            '0 0.601823 0.480556 0.254688 0.400000',
            (root_path / 'point-detect' / '20260620' / '0000.txt').read_text(encoding='utf-8'),
        )

        with Image.open(root_path / 'point-classify' / 'val' / '03-cc' / '20260620_0_0.jpg') as crop:
            self.assertEqual((489, 489), crop.size)

        self.assertEqual(
            '0 0.460208 0.490227 0.449131 0.496201 0.278136 0.073777',
            (root_path / 'point-segment' / '20260620' / '0000_0.txt').read_text(encoding='utf-8'),
        )

    def test_knob_segment_transforms(self) -> None:
        root_path = self.copy_fixture('knob')
        scenario = load_case_scenario('knob-segment')
        convert_dataset(scenario.dataset, root_path, split=scenario.split, reserve_no_label=scenario.reserve_no_label)

        output_path = root_path / 'knob-segment' / '251010'
        self.assertEqual(6, len(list(output_path.glob('*.txt'))))
        self.assertEqual(
            '0 0.296058 0.161550 1.000000 0.883818 0.896863 1.000000 0.178418 0.303400',
            (output_path / '0000_2.txt').read_text(encoding='utf-8'),
        )

    def test_light_stage_transforms(self) -> None:
        root_path = self.copy_fixture('light')
        for task_type in ('light1-detect', 'light2-detect'):
            scenario = load_case_scenario(task_type)
            convert_dataset(
                scenario.dataset, root_path, split=scenario.split, reserve_no_label=scenario.reserve_no_label
            )

        light1_lines = (root_path / 'light1-detect' / 'set1' / '000000.txt').read_text(encoding='utf-8').splitlines()
        self.assertEqual(4, len(light1_lines))
        self.assertTrue(all(line.split()[0] == '0' for line in light1_lines))

        light2_path = root_path / 'light2-detect' / 'set1'
        self.assertEqual(8, len(list(light2_path.glob('*.jpg'))))
        light2_lines = (light2_path / '000000_0.txt').read_text(encoding='utf-8').splitlines()
        self.assertEqual(5, len(light2_lines))
        self.assertTrue(all(line.split()[0] == '0' for line in light2_lines))

    def test_standard_pose_transform(self) -> None:
        root_path = self.copy_fixture('standard-pose')
        scenario = load_case_scenario('pose')
        convert_dataset(scenario.dataset, root_path, split=scenario.split, reserve_no_label=scenario.reserve_no_label)

        self.assertEqual(
            '0 0.476750 0.539333 0.143000 0.747111 0.467045 0.625253 2',
            (root_path / 'pose' / 'rw400' / '0000.txt').read_text(encoding='utf-8'),
        )


if __name__ == '__main__':
    unittest.main()
