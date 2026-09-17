from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from xxtrain.task import TaskType

DEFAULT_MODEL_VERSION = 'v8'
DEFAULT_MODEL_SCALE = 'n'
SUPPORTED_MODEL_SCALES = frozenset({'n', 's', 'm', 'l', 'x'})

_CLASSIFY_TRAIN_ARGS = {'epochs': 72, 'batch': 64, 'imgsz': 224, 'scale': 0.0}
_OTHER_TRAIN_ARGS = {'epochs': 80, 'batch': 32, 'imgsz': 640}


def standard_train_args(task_type: TaskType) -> dict[str, object]:
    source = _CLASSIFY_TRAIN_ARGS if task_type is TaskType.CLASSIFY else _OTHER_TRAIN_ARGS
    return dict(source)


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    task_type: TaskType
    model_version: str = DEFAULT_MODEL_VERSION
    model_scale: str = DEFAULT_MODEL_SCALE
    train_args: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_scale not in SUPPORTED_MODEL_SCALES:
            raise ValueError(f'Unsupported model scale: {self.model_scale}')
        object.__setattr__(self, 'train_args', MappingProxyType(dict(self.train_args)))


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    epoch: int
    total_epochs: int


@dataclass(frozen=True, slots=True)
class TrainingResult:
    onnx_path: Path
    class_names: Mapping[int, str]
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'class_names', MappingProxyType(dict(self.class_names)))
        object.__setattr__(self, 'metrics', MappingProxyType(dict(self.metrics)))
