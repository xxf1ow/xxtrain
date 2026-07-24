import os
from collections.abc import Sequence
from pathlib import Path

from xxtrain.task import TaskType

from .labels import LabelCatalog


def output_path(root_path: str | Path, output_name: str) -> Path:
    return Path(root_path) / output_name


def train_list_path(root_path: str | Path, output_name: str) -> Path:
    return output_path(root_path, output_name) / 'train.txt'


def val_list_path(root_path: str | Path, output_name: str) -> Path:
    return output_path(root_path, output_name) / 'val.txt'


def dataset_yaml_path(root_path: str | Path, output_name: str) -> Path:
    return output_path(root_path, output_name) / 'dataset.yaml'


def split_membership(index: int, split: int) -> tuple[bool, bool]:
    in_train = split <= 0 or index % split != 0
    in_val = split <= 0 or index % split == 0
    return in_train, in_val


def write_split_lists(
    root_path: str | Path, output_name: str, train_items: Sequence[str], val_items: Sequence[str]
) -> None:
    train_list_path(root_path, output_name).write_text('\n'.join(train_items), encoding='utf-8')
    val_list_path(root_path, output_name).write_text('\n'.join(val_items), encoding='utf-8')


def write_dataset_yaml(root_path: str | Path, output_name: str, task_type: TaskType, labels: LabelCatalog) -> None:
    content = f'path: {os.path.abspath(root_path)}\n'
    content += f'train: {output_name}/train.txt\n'
    content += f'val: {output_name}/val.txt\n'
    content += 'names:\n'
    for index, name in enumerate(labels):
        content += f'  {index}: {name}\n'
    if task_type is TaskType.POSE:
        content += f'\nkpt_shape: [{len(labels)}, 3]\nkpt_names:\n  0:\n'
        for name in labels:
            content += f'    - {name}\n'
    dataset_yaml_path(root_path, output_name).write_text(content, encoding='utf-8')
