import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from PIL import Image

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
                """
                from xxtrain.pipeline import standard_recipe
                from xxtrain.task import TaskType
                from xxtrain.training import TrainingScenario

                SCENARIO = TrainingScenario(dataset=standard_recipe(TaskType.DETECT))
                """
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
                call(f'✅ Inference completed. Results are saved in {model.predictor.save_dir}'), printed.call_args_list
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
                f'# Mismatched Samples Report\n# Total mismatched: 1 / 1\nexpect: 0-cat, actual: 1-dog ==> {image}',
                report,
            )

    def test_unlabeled_classification_review_copies_images_to_predicted_class(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_root = root / 'images'
            first = image_root / 'first.jpg'
            second = image_root / 'incoming' / 'second.PNG'
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            Image.new('RGB', (10, 5), (10, 20, 30)).save(first)
            Image.new('RGB', (5, 10), (40, 50, 60)).save(second)
            first_bytes = first.read_bytes()
            second_bytes = second.read_bytes()
            save_dir = root / 'predictions'
            model = self.classification_model(save_dir)
            model.predict.side_effect = lambda *, source, **_arguments: [
                SimpleNamespace(probs=FakeProbs(0 if tuple(source[112, 112]) == (30, 20, 10) else 1))
            ]

            with patch('xxtrain.training.review.Probs', FakeProbs):
                review_model(model, image_root, unlabeled=True)

            self.assertEqual(2, model.predict.call_count)
            self.assertEqual(first_bytes, (save_dir / 'cat' / first.name).read_bytes())
            self.assertEqual(second_bytes, (save_dir / 'dog' / second.name).read_bytes())
            self.assertFalse((save_dir / 'mismatched_samples.txt').exists())

    def test_unlabeled_classification_review_letterboxes_raw_image_before_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_root = root / 'images'
            image = image_root / 'wide.png'
            image.parent.mkdir(parents=True)
            Image.new('RGB', (10, 5), (10, 20, 30)).save(image)
            model = self.classification_model(root / 'predictions')

            with patch('xxtrain.training.review.Probs', FakeProbs):
                review_model(model, image_root, unlabeled=True)

            source = model.predict.call_args.kwargs['source']
            self.assertEqual((224, 224, 3), source.shape)
            self.assertEqual((114, 114, 114), tuple(source[0, 0]))
            self.assertEqual((30, 20, 10), tuple(source[112, 112]))

    def test_unlabeled_review_rejects_non_classification_model(self) -> None:
        model = self.detection_model(Path('predictions'))

        with self.assertRaisesRegex(ValueError, 'only supported for classification models'):
            review_model(model, Path('images'), unlabeled=True)

        model.predict.assert_not_called()

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
            inspect_predictions.assert_called_once_with(model, directory, False)
            model.val.assert_not_called()

    def test_review_forwards_unlabeled_mode(self) -> None:
        scenario_path = Path('scenario.py')
        weights = Path('best.pt')
        directory = Path('images')
        model = self.classification_model(Path('predictions'))

        with (
            patch('xxtrain.training.review.load_scenario'),
            patch('xxtrain.training.review.YOLO', return_value=model),
            patch('xxtrain.training.review.review_model') as inspect_predictions,
        ):
            review(scenario_path, weights, directory, unlabeled=True)

        inspect_predictions.assert_called_once_with(model, directory, True)


if __name__ == '__main__':
    unittest.main()
