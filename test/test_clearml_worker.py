import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from xxtrain.integrations.clearml.worker import main
from xxtrain.training.settings import TrainingProgress, TrainingResult


class ClearMLWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'shared' / 'detect' / 'cache').mkdir(parents=True)
        self.task = Mock()
        self.task.get_logger.return_value = Mock()

    def argv(self, cache_relative_path: str = 'detect/cache') -> list[str]:
        return [
            '--task',
            'point',
            '--target',
            'detect',
            '--cache-relative-path',
            cache_relative_path,
            '--shared-root',
            str(self.root / 'shared'),
            '--run-id',
            '8f245d4e-0ab1-4eb4-98db-6415249cadf7',
            '--run-root',
            str(self.root / 'runs'),
        ]

    def test_task_initialization_matches_installed_clearml_sdk_signature(self):
        from clearml import Task

        with patch.object(Task, 'init', autospec=True, return_value=self.task) as task_init:
            from xxtrain.integrations.clearml.worker import _task_init

            self.assertIs(_task_init(), self.task)
            task_init.assert_called_once_with(auto_connect_frameworks=False)

    @patch('xxtrain.integrations.clearml.worker._task_init')
    @patch('xxtrain.integrations.clearml.worker.build_delivery')
    @patch('xxtrain.integrations.clearml.worker.train_prepared')
    def test_worker_trains_reports_and_publishes_manifest_before_completion(self, train, build, task_init):
        task_init.return_value = self.task
        onnx = self.root / 'model.onnx'
        onnx.write_bytes(b'onnx')
        delivery = self.root / 'delivery.zip'
        delivery.write_bytes(b'zip')
        train.return_value = TrainingResult(onnx, {0: 'Point'}, {'metrics/mAP50-95(B)': 0.75})
        build.return_value = delivery

        main(self.argv())

        progress = train.call_args.kwargs['on_progress']
        progress(TrainingProgress(3, 80))
        logger = self.task.get_logger.return_value
        logger.report_scalar.assert_any_call(title='training', series='epoch', value=3, iteration=3)
        logger.report_scalar.assert_any_call(title='training', series='total_epochs', value=80, iteration=3)
        logger.report_scalar.assert_any_call(title='validation', series='检测效果：mAP50-95', value=0.75, iteration=80)
        upload = self.task.upload_artifact.call_args
        self.assertEqual(upload.kwargs['name'], 'deployment')
        self.assertTrue(upload.kwargs['wait_on_upload'])
        manifest = json.loads(self.task.upload_artifact.call_args_list[0].kwargs['metadata']['manifest'])
        self.assertEqual(manifest['filename'], 'delivery.zip')
        self.assertEqual(manifest['target'], 'detect')
        self.assertEqual(manifest['metric_name'], '检测效果：mAP50-95')
        self.task.mark_completed.assert_called_once_with(ignore_errors=False)

    @patch('xxtrain.integrations.clearml.worker._task_init')
    @patch('xxtrain.integrations.clearml.worker.build_delivery')
    @patch('xxtrain.integrations.clearml.worker.train_prepared')
    def test_upload_failure_marks_failed_not_completed(self, train, build, task_init):
        task_init.return_value = self.task
        train.return_value = TrainingResult(self.root / 'model.onnx', {}, {'metrics/mAP50-95(B)': 0.5})
        build.return_value = self.root / 'model.onnx'
        self.task.upload_artifact.side_effect = RuntimeError('upload failed')

        with self.assertRaises(RuntimeError):
            main(self.argv())

        self.task.mark_failed.assert_called_once()
        self.task.mark_completed.assert_not_called()

    @patch('xxtrain.integrations.clearml.worker._task_init')
    @patch('xxtrain.integrations.clearml.worker.build_delivery')
    @patch('xxtrain.integrations.clearml.worker.train_prepared')
    def test_unsuccessful_upload_result_marks_failed_not_completed(self, train, build, task_init):
        task_init.return_value = self.task
        train.return_value = TrainingResult(self.root / 'model.onnx', {}, {'metrics/mAP50-95(B)': 0.5})
        build.return_value = self.root / 'model.onnx'
        self.task.upload_artifact.return_value = False

        with self.assertRaises(RuntimeError):
            main(self.argv())

        self.task.mark_failed.assert_called_once()
        self.task.mark_completed.assert_not_called()

    @patch('xxtrain.integrations.clearml.worker._task_init')
    @patch('xxtrain.integrations.clearml.worker.train_prepared', side_effect=RuntimeError('training failed'))
    def test_training_failure_marks_failed(self, train, task_init):
        task_init.return_value = self.task
        with self.assertRaises(RuntimeError):
            main(self.argv())
        self.task.mark_failed.assert_called_once()
        self.task.mark_completed.assert_not_called()

    @patch('xxtrain.integrations.clearml.worker._task_init')
    @patch('xxtrain.integrations.clearml.worker.build_delivery', side_effect=RuntimeError('delivery failed'))
    @patch('xxtrain.integrations.clearml.worker.train_prepared')
    def test_delivery_failure_marks_failed_not_completed(self, train, build, task_init):
        task_init.return_value = self.task
        train.return_value = TrainingResult(self.root / 'model.onnx', {}, {'metrics/mAP50-95(B)': 0.5})

        with self.assertRaises(RuntimeError):
            main(self.argv())

        self.task.mark_failed.assert_called_once()
        self.task.mark_completed.assert_not_called()

    @patch('xxtrain.integrations.clearml.worker._task_init')
    @patch('xxtrain.integrations.clearml.worker.train_prepared')
    def test_ultralytics_clearml_callback_is_disabled_before_training(self, train, task_init):
        from ultralytics import settings

        task_init.return_value = self.task

        def observe_setting(*args, **kwargs):
            self.assertFalse(settings['clearml'])
            raise RuntimeError('stop after observation')

        train.side_effect = observe_setting
        with self.assertRaises(RuntimeError):
            main(self.argv())

    @patch('xxtrain.integrations.clearml.worker._task_init')
    def test_rejects_cache_path_outside_shared_root(self, task_init):
        task_init.return_value = self.task
        with self.assertRaises(ValueError):
            main(self.argv('../escape'))
        self.task.mark_failed.assert_called_once()

    @patch('xxtrain.integrations.clearml.worker._task_init')
    @patch('xxtrain.integrations.clearml.worker.train_prepared')
    def test_training_receives_business_settings_not_clearml_output_model(self, train, task_init):
        task_init.return_value = self.task
        train.side_effect = RuntimeError('stop after capture')
        self.task.output_models = [SimpleNamespace(id='forbidden')]

        with self.assertRaises(RuntimeError):
            main(self.argv())

        settings = train.call_args.args[0]
        self.assertEqual(settings.model_version, 'v8')
        self.assertEqual(settings.model_scale, 'n')


if __name__ == '__main__':
    unittest.main()
