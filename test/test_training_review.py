import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from xxtrain.training import review
from xxtrain.training.review import review_model


class FakeProbs:
    def __init__(self, top1: int):
        self.top1 = top1


class TrainingReviewTest(unittest.TestCase):
    def classification_model(self, save_dir: Path, predicted_index: int = 0) -> MagicMock:
        model = MagicMock()
        model.task = 'classify'
        model.names = {0: 'cat', 1: 'dog'}
        model.predict.return_value = [SimpleNamespace(probs=FakeProbs(predicted_index))]
        model.predictor = SimpleNamespace(save_dir=save_dir)
        return model

    def detection_model(self, save_dir: Path) -> MagicMock:
        model = MagicMock()
        model.task = 'detect'
        model.predictor = SimpleNamespace(save_dir=save_dir)
        return model

    def write_scenario(self, root: Path) -> Path:
        path = root / 'scenario.py'
        path.write_text(
            textwrap.dedent(
                '''
                from xxtrain.pipeline import standard_recipe
                from xxtrain.task import TaskType
                from xxtrain.training import TrainingScenario

                SCENARIO = TrainingScenario(dataset=standard_recipe(TaskType.DETECT))
                '''
            ),
            encoding='utf-8',
        )
        return path

    def test_classification_review_uses_the_passed_model_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.image_root = root / 'images'
            image = self.image_root / 'cat' / 'sample.jpg'
            image.parent.mkdir(parents=True)
            image.touch()
            model = self.classification_model(root / 'predictions')

            with patch('xxtrain.training.review.Probs', FakeProbs):
                review_model(model, self.image_root)

            model.predict.assert_called()

    def test_standard_review_consumes_stream_and_reports_predictor_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.image_root = root / 'images'
            consumed = []

            def predictions():
                consumed.append(True)
                yield object()

            model = self.detection_model(root / 'predictions')
            model.predict.return_value = predictions()
            with patch('builtins.print') as printed:
                review_model(model, self.image_root)

            model.predict.assert_called_once_with(source=self.image_root, verbose=False, save=True, stream=True)
            self.assertEqual([True], consumed)
            self.assertIn(
                call(f'✅ Inference completed. Results are saved in {model.predictor.save_dir}'),
                printed.call_args_list,
            )

    def test_classification_review_uses_parent_label_and_writes_mismatch_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_root = root / 'images'
            image = image_root / 'cat' / 'sample.PNG'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'image')
            save_dir = root / 'predictions'
            model = self.classification_model(save_dir, predicted_index=1)

            with patch('xxtrain.training.review.Probs', FakeProbs):
                review_model(model, image_root)

            model.predict.assert_called_once_with(source=str(image), verbose=False, save=False)
            self.assertEqual(b'image', (save_dir / 'cat - dog' / image.name).read_bytes())
            report = (save_dir / 'mismatched_samples.txt').read_text(encoding='utf-8')
            self.assertEqual(
                '# Mismatched Samples Report\n'
                '# Total mismatched: 1 / 1\n'
                f'expect: 0-cat, actual: 1-dog ==> {image}',
                report,
            )

    def test_review_loads_scenario_and_checkpoint_without_running_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scenario_path = self.write_scenario(root)
            weights = root / 'best.pt'
            directory = root / 'images'
            model = self.detection_model(root / 'predictions')

            with (
                patch('xxtrain.training.review.load_scenario') as load_scenario,
                patch('xxtrain.training.review.YOLO', return_value=model) as yolo,
                patch('xxtrain.training.review.review_model') as inspect_predictions,
            ):
                result = review(scenario_path, weights, directory)

            self.assertIsNone(result)
            load_scenario.assert_called_once_with(scenario_path)
            yolo.assert_called_once_with(weights)
            inspect_predictions.assert_called_once_with(model, directory)
            model.val.assert_not_called()


if __name__ == '__main__':
    unittest.main()
