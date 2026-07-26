from pathlib import Path

import ultralytics
from platformdirs import user_cache_path
from ruamel.yaml import YAML
from ultralytics.models import YOLO

from xxtrain.task import TaskType

from .scenario import TrainingScenario

_TASK_SUFFIXES = {
    TaskType.CLASSIFY: '-cls',
    TaskType.DETECT: '',
    TaskType.OBB: '-obb',
    TaskType.POSE: '-pose',
    TaskType.SEGMENT: '-seg',
}


def model_name(scenario: TrainingScenario) -> str:
    suffix = _TASK_SUFFIXES[scenario.dataset.task_type]
    return f'yolo{scenario.model_version}{scenario.model_scale}{suffix}'


def generate_model_yaml(scenario: TrainingScenario, root: Path) -> tuple[str, Path]:
    name = model_name(scenario)
    target_path = root / scenario.dataset.name / f'{name}.yaml'
    try:
        yaml_handler = YAML()
        yaml_handler.preserve_quotes = True

        suffix = _TASK_SUFFIXES[scenario.dataset.task_type]
        template_name = f'yolo{scenario.model_version}{suffix}.yaml'
        package_path = Path(ultralytics.__file__).resolve().parent
        source_path = package_path / 'cfg' / 'models' / scenario.model_version / template_name
        if not source_path.is_file():
            raise FileNotFoundError(f'❌ Template model configuration file not found: {source_path}')

        dataset_yaml_path = root / scenario.dataset.name / 'dataset.yaml'
        with dataset_yaml_path.open(encoding='utf-8') as stream:
            dataset = yaml_handler.load(stream)
        with source_path.open(encoding='utf-8') as stream:
            model = yaml_handler.load(stream)

        num_classes = len(dataset['names'])
        if num_classes <= 0:
            raise ValueError(f'❌ No classes found in dataset.yaml: {dataset_yaml_path}')
        model['nc'] = num_classes

        if scenario.dataset.task_type is TaskType.POSE:
            if 'kpt_shape' not in dataset:
                raise KeyError("❌ 'kpt_shape' missing in dataset.yaml for pose task")
            model['kpt_shape'] = dataset['kpt_shape']

        with target_path.open('w', encoding='utf-8') as stream:
            yaml_handler.dump(model, stream)
    except Exception as error:
        raise ValueError(f'❌ Failed to generate model configuration file: {error}') from error

    return name, target_path


def prepare_pretrained_weights(model_name: str) -> Path:
    cache_path = user_cache_path('xxtrain') / 'weights' / f'{model_name}.pt'
    if cache_path.is_file():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = YOLO(f'{model_name}.pt')
    downloaded.save(cache_path)
    return cache_path
