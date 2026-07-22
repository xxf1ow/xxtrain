import tempfile
import unittest
from pathlib import Path

import yaml

from xxtrain.data import LabelCatalog
from xxtrain.data.dataset import (
    dataset_yaml_path,
    output_path,
    split_membership,
    train_list_path,
    val_list_path,
    write_dataset_yaml,
    write_split_lists,
)
from xxtrain.task import TaskType


class DatasetArtifactTest(unittest.TestCase):
    def test_calculates_paths_without_parsing_output_name(self) -> None:
        root = Path('root')
        self.assertEqual(root / 'scale-pose', output_path(root, 'scale-pose'))
        self.assertEqual(root / 'scale-pose' / 'train.txt', train_list_path(root, 'scale-pose'))
        self.assertEqual(root / 'scale-pose' / 'val.txt', val_list_path(root, 'scale-pose'))
        self.assertEqual(root / 'scale-pose' / 'dataset.yaml', dataset_yaml_path(root, 'scale-pose'))

    def test_split_membership_preserves_existing_rules(self) -> None:
        self.assertEqual((True, True), split_membership(index=0, split=0))
        self.assertEqual((True, True), split_membership(index=3, split=-1))
        self.assertEqual((False, True), split_membership(index=0, split=10))
        self.assertEqual((True, False), split_membership(index=1, split=10))

    def test_writes_lists_in_caller_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_path(root, 'detect').mkdir()
            write_split_lists(root, 'detect', ['b.jpg', 'a.jpg'], ['v.jpg'])

            self.assertEqual('b.jpg\na.jpg', train_list_path(root, 'detect').read_text(encoding='utf-8'))
            self.assertEqual('v.jpg', val_list_path(root, 'detect').read_text(encoding='utf-8'))

    def test_writes_pose_yaml_and_preserves_unquoted_numeric_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_path(root, 'scale-pose').mkdir()
            labels = LabelCatalog(names=('1008', 'point'))

            write_dataset_yaml(root, 'scale-pose', TaskType.POSE, labels)

            text = dataset_yaml_path(root, 'scale-pose').read_text(encoding='utf-8')
            self.assertIn('  0: 1008\n', text)
            parsed = yaml.safe_load(text)
            self.assertEqual({0: 1008, 1: 'point'}, parsed['names'])
            self.assertEqual([2, 3], parsed['kpt_shape'])
            self.assertEqual({0: [1008, 'point']}, parsed['kpt_names'])

    def test_non_pose_yaml_has_no_keypoint_section(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_path(root, 'detect').mkdir()
            write_dataset_yaml(root, 'detect', TaskType.DETECT, LabelCatalog(names=('dial',)))
            parsed = yaml.safe_load(dataset_yaml_path(root, 'detect').read_text(encoding='utf-8'))
            self.assertNotIn('kpt_shape', parsed)


if __name__ == '__main__':
    unittest.main()
