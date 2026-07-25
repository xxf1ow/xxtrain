import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from xxtrain.pipeline import standard_recipe
from xxtrain.task import TaskType
from xxtrain.training import TrainingScenario, train
from xxtrain.training.workflow import merged_train_args, standard_train_args


class TrainingWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.scenario_path = self.root / 'scenario.py'
        self.scenario_path.write_text('# original Scenario\n', encoding='utf-8')

    def run_workflow(
        self,
        scenario: TrainingScenario,
        *,
        best_exists: bool = False,
        best_present: bool = True,
        calls: list[str] | None = None,
        export_error: Exception | None = None,
    ) -> SimpleNamespace:
        model_name = 'yolov8n-cls' if scenario.dataset.task_type is TaskType.CLASSIFY else 'yolov8n'
        model_yaml_path = self.root / scenario.dataset.name / f'{model_name}.yaml'
        pretrained_path = self.root / '.weights' / f'{model_name}.pt'
        save_dir = self.root / 'runs' / 'train'
        save_dir.mkdir(parents=True, exist_ok=True)
        best_path = self.root / 'best.pt'
        if best_exists:
            best_path.touch()

        model = MagicMock()
        trainer = SimpleNamespace(save_dir=save_dir)
        if best_present:
            trainer.best = best_path
        model.trainer = trainer
        model.names = {0: 'alpha'}
        best_model = MagicMock()
        best_model.names = {0: 'alpha'}
        onnx_path = self.root / 'weights' / f'{model_name}.onnx'
        recorded_calls = calls if calls is not None else []
        copy_scenario = shutil.copy

        def load_scenario(path: Path) -> TrainingScenario:
            recorded_calls.append('load_scenario')
            return scenario

        def generate_model_yaml(loaded_scenario: TrainingScenario, root: Path) -> tuple[str, Path]:
            recorded_calls.append('generate_model_yaml')
            return model_name, model_yaml_path

        def prepare_pretrained_weights(name: str) -> Path:
            recorded_calls.append('prepare_pretrained_weights')
            return pretrained_path

        yolo_models = iter((model, best_model))

        def construct_yolo(path: str | Path) -> MagicMock:
            recorded_calls.append('YOLO' if path == model_yaml_path else 'reload_best')
            return next(yolo_models)

        model.load.side_effect = lambda path: recorded_calls.append('load')
        model.train.side_effect = lambda **kwargs: recorded_calls.append('train')

        with (
            patch('xxtrain.training.workflow.load_scenario', side_effect=load_scenario) as load_scenario_mock,
            patch('xxtrain.training.workflow.convert_dataset') as convert_dataset_mock,
            patch(
                'xxtrain.training.workflow.generate_model_yaml', side_effect=generate_model_yaml
            ) as generate_model_yaml_mock,
            patch(
                'xxtrain.training.workflow.prepare_pretrained_weights', side_effect=prepare_pretrained_weights
            ) as prepare_pretrained_weights_mock,
            patch('xxtrain.training.workflow.YOLO', side_effect=construct_yolo) as yolo_mock,
            patch('xxtrain.training.workflow.export_model_to_onnx') as export_model_to_onnx_mock,
            patch('xxtrain.training.workflow.copy_class_reference_images') as copy_class_reference_images_mock,
            patch(
                'xxtrain.training.workflow.shutil.copy',
                side_effect=lambda source, target: (
                    recorded_calls.append('copy_scenario'),
                    copy_scenario(source, target),
                )[1],
            ),
        ):
            convert_dataset_mock.side_effect = lambda *args, **kwargs: recorded_calls.append('convert_dataset')
            if export_error is None:
                export_model_to_onnx_mock.side_effect = lambda *args: (
                    recorded_calls.append('export_model_to_onnx'),
                    onnx_path,
                )[1]
            else:
                export_model_to_onnx_mock.side_effect = export_error
            copy_class_reference_images_mock.side_effect = lambda *args: recorded_calls.append(
                'copy_class_reference_images'
            )

            if export_error is None:
                train(self.scenario_path)
            else:
                with self.assertRaisesRegex(type(export_error), str(export_error)):
                    train(self.scenario_path)

        return SimpleNamespace(
            best_model=best_model,
            best_path=best_path,
            copy_class_reference_images=copy_class_reference_images_mock,
            convert_dataset=convert_dataset_mock,
            export_model_to_onnx=export_model_to_onnx_mock,
            generate_model_yaml=generate_model_yaml_mock,
            load_scenario=load_scenario_mock,
            model=model,
            model_name=model_name,
            model_yaml_path=model_yaml_path,
            onnx_path=onnx_path,
            prepare_pretrained_weights=prepare_pretrained_weights_mock,
            pretrained_path=pretrained_path,
            save_dir=save_dir,
            yolo=yolo_mock,
        )

    def test_standard_training_args_returns_independent_mappings(self) -> None:
        first = standard_train_args(TaskType.CLASSIFY)
        second = standard_train_args(TaskType.CLASSIFY)

        first['test-only'] = True

        self.assertNotIn('test-only', second)

    def test_scenario_arguments_override_standard_arguments(self) -> None:
        scenario = TrainingScenario(
            dataset=standard_recipe(TaskType.CLASSIFY), train_args={'epochs': 80, 'optimizer': 'AdamW'}
        )
        with patch('xxtrain.training.workflow.standard_train_args', return_value={'epochs': 1, 'test-default': True}):
            result = merged_train_args(scenario)

        self.assertEqual({'epochs': 80, 'test-default': True, 'optimizer': 'AdamW'}, result)

    def test_classification_workflow_uses_fixed_order_and_reloads_present_best(self) -> None:
        calls: list[str] = []
        scenario = TrainingScenario(dataset=standard_recipe(TaskType.CLASSIFY))

        result = self.run_workflow(scenario, best_exists=True, calls=calls)

        self.assertEqual(
            [
                'load_scenario',
                'convert_dataset',
                'generate_model_yaml',
                'prepare_pretrained_weights',
                'YOLO',
                'load',
                'train',
                'copy_scenario',
                'reload_best',
                'export_model_to_onnx',
                'copy_class_reference_images',
            ],
            calls,
        )
        result.load_scenario.assert_called_once_with(self.scenario_path.resolve())
        self.assertEqual(
            [unittest.mock.call(result.model_yaml_path), unittest.mock.call(result.best_path)],
            result.yolo.call_args_list,
        )
        result.model.load.assert_called_once_with(result.pretrained_path)
        result.export_model_to_onnx.assert_called_once_with(result.best_model, self.root, result.model_name)

    def test_partial_classification_output_without_dataset_yaml_reconverts(self) -> None:
        scenario = TrainingScenario(dataset=standard_recipe(TaskType.CLASSIFY))
        (self.root / scenario.dataset.name / 'train' / 'partial-class').mkdir(parents=True)

        result = self.run_workflow(scenario)

        result.convert_dataset.assert_called_once_with(
            scenario.dataset, self.root, split=scenario.split, reserve_no_label=scenario.reserve_no_label
        )

    def test_existing_classification_dataset_yaml_skips_conversion(self) -> None:
        scenario = TrainingScenario(dataset=standard_recipe(TaskType.CLASSIFY))
        dataset_yaml = self.root / scenario.dataset.name / 'dataset.yaml'
        dataset_yaml.parent.mkdir()
        dataset_yaml.touch()

        result = self.run_workflow(scenario)

        result.convert_dataset.assert_not_called()

    def test_existing_non_classification_dataset_yaml_skips_conversion(self) -> None:
        scenario = TrainingScenario(dataset=standard_recipe(TaskType.DETECT))
        dataset_yaml = self.root / scenario.dataset.name / 'dataset.yaml'
        dataset_yaml.parent.mkdir()
        dataset_yaml.touch()

        result = self.run_workflow(scenario)

        result.convert_dataset.assert_not_called()

    def test_missing_output_converts_with_scenario_split_and_reserve_no_label(self) -> None:
        scenario = TrainingScenario(dataset=standard_recipe(TaskType.DETECT), split=7, reserve_no_label=True)

        result = self.run_workflow(scenario)

        result.convert_dataset.assert_called_once_with(scenario.dataset, self.root, split=7, reserve_no_label=True)

    def test_classification_trains_with_output_directory_and_scenario_arguments(self) -> None:
        scenario = TrainingScenario(
            dataset=standard_recipe(TaskType.CLASSIFY),
            train_args={'epochs': 3, 'batch': 2, 'imgsz': 64, 'optimizer': 'AdamW'},
        )

        result = self.run_workflow(scenario)

        result.model.train.assert_called_once()
        train_arguments = result.model.train.call_args.kwargs
        self.assertEqual(self.root / 'classify', train_arguments['data'])
        self.assertEqual(3, train_arguments['epochs'])
        self.assertEqual(2, train_arguments['batch'])
        self.assertEqual(64, train_arguments['imgsz'])
        self.assertEqual('AdamW', train_arguments['optimizer'])

    def test_non_classification_trains_with_dataset_yaml_and_scenario_arguments(self) -> None:
        scenario = TrainingScenario(dataset=standard_recipe(TaskType.DETECT), train_args={'test-argument': True})

        result = self.run_workflow(scenario)

        train_arguments = result.model.train.call_args.kwargs
        self.assertEqual(self.root / 'detect' / 'dataset.yaml', train_arguments['data'])
        self.assertTrue(train_arguments['test-argument'])

    def test_non_file_best_keeps_trained_model_for_export(self) -> None:
        scenario = TrainingScenario(dataset=standard_recipe(TaskType.DETECT))

        with patch('builtins.print') as print_mock:
            result = self.run_workflow(scenario)

        result.yolo.assert_called_once_with(result.model_yaml_path)
        result.export_model_to_onnx.assert_called_once_with(result.model, self.root, result.model_name)
        print_mock.assert_any_call(
            f'❌ Training completed! But the best model checkpoint not found at {result.best_path} ...'
        )

    def test_absent_best_keeps_trained_model_for_export(self) -> None:
        scenario = TrainingScenario(dataset=standard_recipe(TaskType.DETECT))

        with patch('builtins.print') as print_mock:
            result = self.run_workflow(scenario, best_present=False)

        result.yolo.assert_called_once_with(result.model_yaml_path)
        result.export_model_to_onnx.assert_called_once_with(result.model, self.root, result.model_name)
        print_mock.assert_any_call('❌ Training completed! But the best model checkpoint not found at  ...')

    def test_original_scenario_is_copied_into_trainer_save_dir(self) -> None:
        scenario = TrainingScenario(dataset=standard_recipe(TaskType.DETECT))

        result = self.run_workflow(scenario)

        copied_scenario = result.save_dir / self.scenario_path.name
        self.assertEqual('# original Scenario\n', copied_scenario.read_text(encoding='utf-8'))

    def test_only_classification_copies_reference_images(self) -> None:
        classify = TrainingScenario(dataset=standard_recipe(TaskType.CLASSIFY))
        classify_result = self.run_workflow(classify)
        classify_result.copy_class_reference_images.assert_called_once_with(
            self.root, 'classify', classify_result.onnx_path, classify_result.model.names
        )

        detect = TrainingScenario(dataset=standard_recipe(TaskType.DETECT))
        detect_result = self.run_workflow(detect)
        detect_result.copy_class_reference_images.assert_not_called()

    def test_onnx_export_failure_propagates(self) -> None:
        scenario = TrainingScenario(dataset=standard_recipe(TaskType.DETECT))

        result = self.run_workflow(scenario, export_error=RuntimeError('export failed'))

        result.copy_class_reference_images.assert_not_called()


if __name__ == '__main__':
    unittest.main()
