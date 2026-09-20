import unittest
from pathlib import Path

from xxtrain.business_tasks.point import point_task_definition
from xxtrain.training import load_scenario
from xxtrain.training.model import model_name
from xxtrain.training.workflow import merged_train_args


class PointTrainingDefinitionTest(unittest.TestCase):
    def test_point_targets_match_scenario_training_settings_and_deliverables(self) -> None:
        expected = {
            'detect': {
                'scenario': 'point_detect.py',
                'metric_key': 'metrics/mAP50-95(B)',
                'metric_name': '检测效果：mAP50-95',
                'model_name': 'yolov8n',
                'delivery': (False, False),
            },
            'classify': {
                'scenario': 'point_classify.py',
                'metric_key': 'metrics/accuracy_top1',
                'metric_name': '分类准确率：Top-1',
                'model_name': 'yolov8n-cls',
                'delivery': (True, True),
            },
            'segment': {
                'scenario': 'point_segment.py',
                'metric_key': 'metrics/mAP50-95(M)',
                'metric_name': '指针分割效果：Mask mAP50-95',
                'model_name': 'yolov8n-seg',
                'delivery': (False, False),
            },
        }
        task_definition = point_task_definition()

        for target, values in expected.items():
            with self.subTest(target=target):
                scenario = load_scenario(Path('data/point') / values['scenario'])
                step = task_definition.step(target)
                self.assertTrue(hasattr(step, 'training'))
                training = step.training

                self.assertIsNotNone(training)
                assert training is not None
                self.assertEqual(merged_train_args(scenario), training.settings.train_args)
                self.assertEqual(scenario.model_version, training.settings.model_version)
                self.assertEqual(scenario.model_scale, training.settings.model_scale)
                self.assertEqual(values['model_name'], model_name(scenario))
                self.assertEqual(values['metric_key'], training.metric_key)
                self.assertEqual(values['metric_name'], training.metric_name)
                self.assertEqual(values['delivery'], (training.delivery.labels, training.delivery.reference_images))
                if target == 'classify':
                    self.assertEqual(0.0, training.settings.train_args['fliplr'])
                    self.assertEqual(0.0, training.settings.train_args['flipud'])
                    self.assertEqual(0.0, training.settings.train_args['degrees'])
                    self.assertIsNone(training.settings.train_args['auto_augment'])


if __name__ == '__main__':
    unittest.main()
