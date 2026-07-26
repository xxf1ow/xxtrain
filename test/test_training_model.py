import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ruamel.yaml import YAML

from xxtrain.pipeline import standard_recipe
from xxtrain.task import TaskType
from xxtrain.training import TrainingScenario
from xxtrain.training.model import generate_model_yaml, model_name, prepare_pretrained_weights


class TrainingModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.yaml = YAML()

    def scenario(self, task_type: TaskType) -> TrainingScenario:
        return TrainingScenario(dataset=standard_recipe(task_type))

    def write_yaml(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as stream:
            self.yaml.dump(value, stream)

    def read_yaml(self, path: Path) -> dict:
        with path.open(encoding='utf-8') as stream:
            return dict(self.yaml.load(stream))

    def fake_package(self, root: Path, template_name: str, template: dict) -> Path:
        package_path = root / 'ultralytics'
        package_file = package_path / '__init__.py'
        self.write_yaml(package_path / 'cfg' / 'models' / 'v8' / template_name, template)
        package_file.touch()
        return package_file

    def assert_only_owned_keys_changed(self, source: dict, target: dict) -> None:
        source_without_owned = dict(source)
        target_without_owned = dict(target)
        for key in ('nc', 'kpt_shape'):
            source_without_owned.pop(key, None)
            target_without_owned.pop(key, None)
        self.assertEqual(source_without_owned, target_without_owned)

    def test_detect_model_name_and_template_are_standard_ultralytics_names(self) -> None:
        scenario = self.scenario(TaskType.DETECT)
        self.assertEqual('yolov8n', model_name(scenario))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package_file = self.fake_package(root, 'yolov8.yaml', {'nc': 80, 'backbone': ['detect']})
            self.write_yaml(root / 'detect' / 'dataset.yaml', {'names': ['widget']})

            with patch('xxtrain.training.model.ultralytics.__file__', str(package_file)):
                generated_name, target_path = generate_model_yaml(scenario, root)

            self.assertEqual('yolov8n', generated_name)
            self.assertEqual(root / 'detect' / 'yolov8n.yaml', target_path)
            self.assertTrue(target_path.is_file())

    def test_task_specific_model_names_use_ultralytics_suffixes(self) -> None:
        expected = {TaskType.CLASSIFY: 'yolov8n-cls', TaskType.SEGMENT: 'yolov8n-seg', TaskType.POSE: 'yolov8n-pose'}
        for task_type, expected_name in expected.items():
            with self.subTest(task_type=task_type):
                self.assertEqual(expected_name, model_name(self.scenario(task_type)))

    def test_generated_detect_yaml_changes_only_class_count(self) -> None:
        scenario = self.scenario(TaskType.DETECT)
        source = {'nc': 80, 'scales': {'n': [0.33, 0.25, 1024]}, 'backbone': [[-1, 1, 'Conv', [64, 3, 2]]]}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package_file = self.fake_package(root, 'yolov8.yaml', source)
            self.write_yaml(root / 'detect' / 'dataset.yaml', {'names': {0: 'one', 1: 'two'}})

            with patch('xxtrain.training.model.ultralytics.__file__', str(package_file)):
                _, target_path = generate_model_yaml(scenario, root)

            target = self.read_yaml(target_path)
            self.assertEqual(2, target['nc'])
            self.assert_only_owned_keys_changed(source, target)

    def test_generated_pose_yaml_changes_only_class_count_and_keypoint_shape(self) -> None:
        scenario = self.scenario(TaskType.POSE)
        source = {'nc': 80, 'kpt_shape': [17, 3], 'head': [[-1, 1, 'Pose', ['nc', 'kpt_shape']]]}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package_file = self.fake_package(root, 'yolov8-pose.yaml', source)
            self.write_yaml(root / 'pose' / 'dataset.yaml', {'names': ['person'], 'kpt_shape': [5, 3]})

            with patch('xxtrain.training.model.ultralytics.__file__', str(package_file)):
                _, target_path = generate_model_yaml(scenario, root)

            target = self.read_yaml(target_path)
            self.assertEqual(1, target['nc'])
            self.assertEqual([5, 3], target['kpt_shape'])
            self.assert_only_owned_keys_changed(source, target)

    def test_missing_template_preserves_current_failure(self) -> None:
        scenario = self.scenario(TaskType.DETECT)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package_file = root / 'ultralytics' / '__init__.py'
            package_file.parent.mkdir()
            package_file.touch()
            self.write_yaml(root / 'detect' / 'dataset.yaml', {'names': ['one']})

            with (
                patch('xxtrain.training.model.ultralytics.__file__', str(package_file)),
                self.assertRaisesRegex(ValueError, 'Template model configuration file not found'),
            ):
                generate_model_yaml(scenario, root)

    def test_empty_class_list_preserves_current_failure(self) -> None:
        scenario = self.scenario(TaskType.DETECT)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package_file = self.fake_package(root, 'yolov8.yaml', {'nc': 80})
            self.write_yaml(root / 'detect' / 'dataset.yaml', {'names': []})

            with (
                patch('xxtrain.training.model.ultralytics.__file__', str(package_file)),
                self.assertRaisesRegex(ValueError, 'No classes found in dataset.yaml'),
            ):
                generate_model_yaml(scenario, root)

    def test_pose_dataset_without_keypoint_shape_preserves_current_failure(self) -> None:
        scenario = self.scenario(TaskType.POSE)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package_file = self.fake_package(root, 'yolov8-pose.yaml', {'nc': 80, 'kpt_shape': [17, 3]})
            self.write_yaml(root / 'pose' / 'dataset.yaml', {'names': ['person']})

            with (
                patch('xxtrain.training.model.ultralytics.__file__', str(package_file)),
                self.assertRaisesRegex(ValueError, "'kpt_shape' missing in dataset.yaml for pose task"),
            ):
                generate_model_yaml(scenario, root)

    def test_cached_pretrained_weights_are_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir)
            cache_path = cache_root / 'weights' / 'yolov8n.pt'
            cache_path.parent.mkdir(parents=True)
            cache_path.touch()

            with (
                patch('xxtrain.training.model.user_cache_path', return_value=cache_root),
                patch('xxtrain.training.model.YOLO') as yolo,
            ):
                result = prepare_pretrained_weights('yolov8n')

            self.assertEqual(cache_path, result)
            yolo.assert_not_called()

    def test_missing_pretrained_weights_are_downloaded_to_user_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir)
            cache_path = cache_root / 'weights' / 'yolov8n.pt'

            with (
                patch('xxtrain.training.model.user_cache_path', return_value=cache_root),
                patch('xxtrain.training.model.YOLO') as yolo,
            ):
                result = prepare_pretrained_weights('yolov8n')

            self.assertEqual(cache_path, result)
            yolo.assert_called_once_with('yolov8n.pt')
            yolo.return_value.save.assert_called_once_with(cache_path)
            self.assertTrue(cache_path.parent.is_dir())


if __name__ == '__main__':
    unittest.main()
