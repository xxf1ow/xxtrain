import shutil
from pathlib import Path

from ultralytics.models import YOLO

from xxtrain.pipeline import convert_dataset
from xxtrain.task import TaskType

from .exporting import copy_class_reference_images, export_model_to_onnx
from .model import generate_model_yaml, prepare_pretrained_weights
from .scenario import TrainingScenario, load_scenario

_CLASSIFY_TRAIN_ARGS = {'epochs': 72, 'batch': 128, 'imgsz': 224}
_OTHER_TRAIN_ARGS = {'epochs': 100, 'batch': 32, 'imgsz': 640}


def standard_train_args(task_type: TaskType) -> dict[str, object]:
    source = _CLASSIFY_TRAIN_ARGS if task_type is TaskType.CLASSIFY else _OTHER_TRAIN_ARGS
    return dict(source)


def merged_train_args(scenario: TrainingScenario) -> dict[str, object]:
    return standard_train_args(scenario.dataset.task_type) | dict(scenario.train_args)


def train(scenario_path: str | Path) -> None:
    resolved_scenario_path = Path(scenario_path).resolve()
    scenario = load_scenario(resolved_scenario_path)
    root = resolved_scenario_path.parent
    dataset_root = root / scenario.dataset.name
    dataset_yaml = dataset_root / 'dataset.yaml'

    dataset_exists = (
        dataset_root.exists()
        if scenario.dataset.task_type is TaskType.CLASSIFY
        else dataset_yaml.exists()
    )
    if not dataset_exists:
        convert_dataset(
            scenario.dataset,
            root,
            split=scenario.split,
            reserve_no_label=scenario.reserve_no_label,
        )

    name, model_yaml_path = generate_model_yaml(scenario, root)
    pretrained_path = prepare_pretrained_weights(name)
    model = YOLO(model_yaml_path)
    model.load(pretrained_path)
    data_path = dataset_root if scenario.dataset.task_type is TaskType.CLASSIFY else dataset_yaml
    model.train(data=data_path, **merged_train_args(scenario))

    trainer = model.trainer
    save_dir = getattr(trainer, 'save_dir', None)
    if save_dir is not None:
        shutil.copy(resolved_scenario_path, Path(save_dir) / resolved_scenario_path.name)

    best_model = model
    best_path = getattr(trainer, 'best', None)
    if best_path is not None and Path(best_path).is_file():
        best_model = YOLO(best_path)

    onnx_path = export_model_to_onnx(best_model, root, name)
    if onnx_path is not None and scenario.dataset.task_type is TaskType.CLASSIFY:
        copy_class_reference_images(root, scenario.dataset.name, onnx_path, best_model.names)
