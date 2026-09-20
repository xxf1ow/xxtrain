import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from xxtrain.integrations.clearml.worker import main
from xxtrain.training.settings import TrainingProgress, TrainingResult


class ClearMLWorkerTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
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
            task_init.assert_called_once_with(auto_connect_arg_parser=False, auto_connect_frameworks=False)

    @patch('xxtrain.integrations.clearml.worker.build_delivery')
    @patch('xxtrain.integrations.clearml.worker.train_prepared')
    def test_agent_no_flags_consumes_installed_clearml_parameters_for_all_targets(self, train, build):
        from clearml.backend_interface.task.args import _Arguments

        delivery = self.root / 'delivery.zip'
        delivery.write_bytes(b'zip')
        build.return_value = delivery
        train.return_value = TrainingResult(self.root / 'model.onnx', {}, {})
        expected_train_args = {
            'detect': {'epochs': 80, 'batch': 32, 'imgsz': 640},
            'classify': {
                'epochs': 72,
                'batch': 64,
                'imgsz': 224,
                'scale': 0.0,
                'fliplr': 0.0,
                'flipud': 0.0,
                'degrees': 0.0,
                'auto_augment': None,
            },
            'segment': {'epochs': 80, 'batch': 32, 'imgsz': 640},
        }

        for target in ('detect', 'classify', 'segment'):
            with self.subTest(target=target):
                run_id = str(uuid4())
                cache_relative_path = f'{target}/cache'
                (self.root / 'shared' / target / 'cache').mkdir(parents=True, exist_ok=True)
                task = Mock()
                task.get_logger.return_value = Mock()
                task.upload_artifact.return_value = True
                task.get_parameters.return_value = {
                    'Args/task': 'point',
                    'Args/target': target,
                    'Args/cache_relative_path': cache_relative_path,
                    'Args/shared_root': str(self.root / 'shared'),
                    'Args/run_id': run_id,
                    'Args/run_root': str(self.root / 'runs'),
                }

                def connect(parser, *, current=task):
                    with patch('clearml.backend_interface.task.args.Session.check_min_api_version', return_value=True):
                        _Arguments(current).copy_to_parser(parser, None)
                    return parser

                task.connect.side_effect = connect
                with (
                    patch('xxtrain.integrations.clearml.worker._task_init', return_value=task),
                    patch.object(sys, 'argv', ['xxtrain-worker']),
                ):
                    main()

                settings, dataset_dir, run_dir = train.call_args.args
                self.assertEqual(target, settings.task_type.value)
                self.assertEqual('v8', settings.model_version)
                self.assertEqual('n', settings.model_scale)
                self.assertEqual(expected_train_args[target], dict(settings.train_args))
                self.assertEqual(self.root / 'shared' / target / 'cache', dataset_dir)
                self.assertEqual(self.root / 'runs' / run_id, run_dir)

    def test_help_does_not_initialize_clearml(self):
        with (
            patch('xxtrain.integrations.clearml.worker._task_init') as task_init,
            self.assertRaises(SystemExit) as exit,
        ):
            main(['--help'])

        self.assertEqual(0, exit.exception.code)
        task_init.assert_not_called()

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

    def test_worker_reports_fresh_validation_every_five_epochs_and_final_best_result(self):
        for interval in (1, 10):
            with self.subTest(validation_interval=interval):
                task = Mock()
                logger = task.get_logger.return_value

                def train(_settings, _dataset, _run, *, on_progress, on_validation):
                    for epoch in range(1, 81):
                        progress = TrainingProgress(epoch, 80)
                        on_progress(progress)
                        if epoch % interval == 0:
                            on_validation(progress, {'metrics/mAP50-95(B)': epoch / 100})
                    return TrainingResult(self.root / 'model.onnx', {0: 'Point'}, {'metrics/mAP50-95(B)': 0.9})

                argv = self.argv()
                argv[argv.index('--run-id') + 1] = str(uuid4())
                with (
                    patch('xxtrain.integrations.clearml.worker._task_init', return_value=task),
                    patch('xxtrain.integrations.clearml.worker.train_prepared', side_effect=train),
                    patch('xxtrain.integrations.clearml.worker.build_delivery', return_value=self.root / 'model.onnx'),
                ):
                    main(argv)

                actual = [
                    (call.kwargs['iteration'], call.kwargs['value'])
                    for call in logger.report_scalar.call_args_list
                    if call.kwargs['title'] == 'validation'
                ]
                cadence = 5 if interval == 1 else 10
                self.assertEqual([(epoch, epoch / 100) for epoch in range(cadence, 81, cadence)] + [(80, 0.9)], actual)
                self.assertEqual(('xxtrain/metric', 0.9), task.set_parameter.call_args.args)

    def test_missing_final_metric_does_not_leave_an_intermediate_score_as_final(self):
        def train(_settings, _dataset, _run, *, on_progress, on_validation):
            on_validation(TrainingProgress(5, 80), {'metrics/mAP50-95(B)': 0.0})
            on_validation(TrainingProgress(10, 80), {})
            return TrainingResult(self.root / 'model.onnx', {0: 'Point'}, {})

        with (
            patch('xxtrain.integrations.clearml.worker._task_init', return_value=self.task),
            patch('xxtrain.integrations.clearml.worker.train_prepared', side_effect=train),
            patch('xxtrain.integrations.clearml.worker.build_delivery', return_value=self.root / 'model.onnx'),
        ):
            main(self.argv())

        reports = self.task.get_logger.return_value.report_scalar.call_args_list
        self.assertEqual([(5, 0.0)], [(call.kwargs['iteration'], call.kwargs['value']) for call in reports])
        self.assertEqual(('xxtrain/metric', None), self.task.set_parameter.call_args.args)

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
