import tempfile
import textwrap
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from xxtrain.training import export
from xxtrain.training.exporting import export_model_to_onnx


class TrainingExportingTest(unittest.TestCase):
    def write_scenario(self, root: Path, task: str) -> Path:
        path = root / 'scenario.py'
        path.write_text(
            textwrap.dedent(
                f'''
                from xxtrain.pipeline import standard_recipe
                from xxtrain.task import TaskType
                from xxtrain.training import TrainingScenario

                SCENARIO = TrainingScenario(dataset=standard_recipe(TaskType.{task}))
                '''
            ),
            encoding='utf-8',
        )
        return path

    def export_at_fixed_time(self, scenario_path: Path, weights: Path, model: MagicMock) -> Path:
        with (
            patch('xxtrain.training.exporting.YOLO', return_value=model) as yolo,
            patch('xxtrain.training.exporting.datetime') as clock,
        ):
            clock.now.return_value = datetime(2026, 7, 24, 12, 34, 56)
            result = export(scenario_path, weights)

        yolo.assert_called_once_with(weights)
        model.export.assert_called_once_with(format='onnx', simplify=True)
        return result

    def test_classification_export_moves_onnx_and_copies_sorted_class_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scenario_path = self.write_scenario(root, 'CLASSIFY')
            weights = root / 'best.pt'
            temporary_onnx = root / 'temporary.onnx'
            temporary_onnx.touch()
            alpha_image = root / 'classify' / 'train' / 'alpha' / 'alpha.PNG'
            beta_image = root / 'classify' / 'train' / 'beta' / 'beta.jpeg'
            alpha_image.parent.mkdir(parents=True)
            beta_image.parent.mkdir(parents=True)
            alpha_image.touch()
            beta_image.touch()
            model = MagicMock()
            model.export.return_value = temporary_onnx
            model.names = {1: 'beta', 0: 'alpha'}

            result = self.export_at_fixed_time(scenario_path, weights, model)

            expected = root / 'weights' / 'yolov8n-cls_2026-07-24_12-34-56.onnx'
            references = root / 'weights' / 'yolov8n-cls_2026-07-24_12-34-56_references'
            self.assertEqual(expected, result)
            self.assertTrue(expected.is_file())
            self.assertEqual(['0_alpha.PNG', '1_beta.jpeg'], sorted(path.name for path in references.iterdir()))

    def test_non_classification_export_does_not_create_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scenario_path = self.write_scenario(root, 'DETECT')
            weights = root / 'best.pt'
            temporary_onnx = root / 'temporary.onnx'
            temporary_onnx.touch()
            model = MagicMock()
            model.export.return_value = temporary_onnx

            result = self.export_at_fixed_time(scenario_path, weights, model)

            expected = root / 'weights' / 'yolov8n_2026-07-24_12-34-56.onnx'
            self.assertEqual(expected, result)
            self.assertTrue(expected.is_file())
            self.assertFalse(expected.with_name(f'{expected.stem}_references').exists())

    def test_export_propagates_model_export_failure(self) -> None:
        model = MagicMock()
        model.export.side_effect = RuntimeError('export failed')
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(RuntimeError, 'export failed'):
                export_model_to_onnx(model, root, 'yolov8n')


if __name__ == '__main__':
    unittest.main()
