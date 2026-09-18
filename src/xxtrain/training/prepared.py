import shutil
from collections.abc import Callable, Mapping
from pathlib import Path

import yaml
from ultralytics.models import YOLO

from xxtrain.task import TaskType

from .exporting import export_model_to_path
from .model import configured_model_name, generate_configured_model_yaml, prepare_pretrained_weights
from .settings import TrainingProgress, TrainingResult, TrainingSettings, training_arguments


def train_prepared(
    settings: TrainingSettings,
    dataset_dir: Path,
    run_dir: Path,
    *,
    on_progress: Callable[[TrainingProgress], None] | None = None,
) -> TrainingResult:
    dataset_dir = Path(dataset_dir).resolve()
    run_dir = Path(run_dir).resolve()
    _validate_separate_trees(dataset_dir, run_dir)
    source_yaml = dataset_dir / 'dataset.yaml'
    if not source_yaml.is_file():
        raise FileNotFoundError(f'Prepared dataset metadata not found: {source_yaml}')

    run_dir.mkdir(parents=True, exist_ok=True)
    local_data = _prepare_local_dataset(settings.task_type, dataset_dir, run_dir / 'dataset')
    model_yaml = run_dir / f'{configured_model_name(settings)}.yaml'
    name, model_yaml = generate_configured_model_yaml(settings, local_data / 'dataset.yaml', model_yaml)
    model = YOLO(model_yaml)
    model.load(prepare_pretrained_weights(name))
    if on_progress is not None:
        model.add_callback(
            'on_train_epoch_end', lambda trainer: on_progress(TrainingProgress(trainer.epoch + 1, trainer.epochs))
        )

    arguments = training_arguments(settings)
    arguments.update(project=run_dir, name='training')
    data_argument = local_data if settings.task_type is TaskType.CLASSIFY else local_data / 'dataset.yaml'
    model.train(data=data_argument, **arguments)

    trained_model = model
    best_path = Path(getattr(model.trainer, 'best', ''))
    if best_path.is_file():
        trained_model = YOLO(best_path)
    validation = trained_model.val(data=data_argument)
    metrics = _numeric_metrics(getattr(validation, 'results_dict', {}))
    exported = export_model_to_path(trained_model, run_dir / 'model.onnx')
    return TrainingResult(exported, dict(trained_model.names), metrics)


def _validate_separate_trees(dataset_dir: Path, run_dir: Path) -> None:
    if dataset_dir == run_dir or dataset_dir in run_dir.parents or run_dir in dataset_dir.parents:
        raise ValueError('Prepared dataset and run directory must not overlap')


def _prepare_local_dataset(task_type: TaskType, source: Path, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    metadata = yaml.safe_load((source / 'dataset.yaml').read_text(encoding='utf-8'))
    if task_type is TaskType.CLASSIFY:
        class_names = _link_classification_splits(source, target)
        metadata['path'] = str(target)
        metadata['train'] = 'train'
        metadata['val'] = 'val'
        metadata['names'] = {index: name for index, name in enumerate(class_names)}
    else:
        metadata['path'] = str(target)
        for split in ('train', 'val'):
            source_list = _resolve_metadata_path(source, metadata[split])
            local_images = []
            for index, image in enumerate(_read_paths(source_list)):
                suffix = image.suffix
                local_image = target / 'images' / split / f'{index:08d}{suffix}'
                local_label = target / 'labels' / split / f'{index:08d}.txt'
                _copy_image(image, local_image)
                local_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image.with_suffix('.txt'), local_label)
                local_images.append(str(local_image))
            list_path = target / f'{split}.txt'
            list_path.write_text('\n'.join(local_images), encoding='utf-8')
            metadata[split] = list_path.name
    (target / 'dataset.yaml').write_text(yaml.safe_dump(metadata, sort_keys=False), encoding='utf-8')
    return target


def _link_classification_splits(source: Path, target: Path) -> list[str]:
    classes: set[str] = set()
    for split in ('train', 'val'):
        split_root = source / split
        if not split_root.is_dir():
            continue
        for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            classes.add(class_dir.name)
            for image in sorted(path for path in class_dir.iterdir() if path.is_file()):
                _copy_image(image, target / split / class_dir.name / image.name)
    if not classes:
        raise ValueError(f'Prepared classification dataset has no classes: {source}')
    return sorted(classes)


def _resolve_metadata_path(dataset_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    direct = dataset_dir / path
    if direct.exists():
        return direct
    return dataset_dir.parent / path


def _read_paths(path: Path) -> list[Path]:
    return [Path(line.strip()).resolve() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _copy_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _numeric_metrics(values: Mapping[str, object]) -> dict[str, float]:
    return {key: float(value) for key, value in values.items() if isinstance(value, (int, float))}
