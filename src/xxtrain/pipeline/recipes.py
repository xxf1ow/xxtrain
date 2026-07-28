from dataclasses import dataclass
from pathlib import Path

from xxtrain.data import LabelCatalog
from xxtrain.task import TaskType

from .annotation_io import ReadAnnotations, ValidateObb
from .core import Pipeline
from .discovery import DirectorySource
from .processors import (
    EncodeDetection,
    EncodePose,
    EncodeSegment,
    FilterLabels,
    PrepareClassification,
    PrepareSegmentShapes,
    ReadImageInfo,
)
from .sinks import ClassificationDatasetSink, DatasetSink, YoloDatasetSink


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetRecipe:
    name: str
    task_type: TaskType
    labels: LabelCatalog | None
    pipeline: Pipeline
    sink: DatasetSink

    def __post_init__(self) -> None:
        self.pipeline.validate_boundaries(DirectorySource.output_type, self.sink.input_type)


class _LegacyLabelCatalog(LabelCatalog):
    def __post_init__(self) -> None:
        if type(self.names) is not tuple or not self.names or any(not isinstance(name, str) for name in self.names):
            raise ValueError('Legacy label catalog names must be a non-empty tuple of strings')


def _read_labels(root_path: Path, *, strict: bool) -> LabelCatalog:
    labels_path = root_path / 'src' / 'labels.txt'
    assert labels_path.is_file(), f'标签列表不存在: {labels_path}'
    labels = tuple(line.strip() for line in labels_path.read_text(encoding='utf-8').splitlines())
    assert labels, f'标签列表为空: {labels_path}'
    return LabelCatalog(labels) if strict else _LegacyLabelCatalog(labels)


def standard_recipe(task_type: TaskType) -> DatasetRecipe:
    if task_type is TaskType.DETECT:
        return DatasetRecipe(
            name='detect',
            task_type=task_type,
            labels=None,
            pipeline=Pipeline((ReadImageInfo(), ReadAnnotations(), FilterLabels(), EncodeDetection())),
            sink=YoloDatasetSink(),
        )
    if task_type is TaskType.SEGMENT:
        return DatasetRecipe(
            name='segment',
            task_type=task_type,
            labels=None,
            pipeline=Pipeline(
                (ReadImageInfo(), ReadAnnotations(), FilterLabels(), PrepareSegmentShapes(), EncodeSegment())
            ),
            sink=YoloDatasetSink(),
        )
    if task_type is TaskType.POSE:
        return DatasetRecipe(
            name='pose',
            task_type=task_type,
            labels=None,
            pipeline=Pipeline((ReadImageInfo(), ReadAnnotations(), FilterLabels(), EncodePose())),
            sink=YoloDatasetSink(),
        )
    if task_type is TaskType.OBB:
        return DatasetRecipe(
            name='obb',
            task_type=task_type,
            labels=None,
            pipeline=Pipeline((ReadImageInfo(), ReadAnnotations(), FilterLabels(), ValidateObb(), EncodeSegment())),
            sink=YoloDatasetSink(),
        )
    if task_type is TaskType.CLASSIFY:
        return DatasetRecipe(
            name='classify',
            task_type=task_type,
            labels=None,
            pipeline=Pipeline((PrepareClassification(),)),
            sink=ClassificationDatasetSink(),
        )
    raise ValueError(f'Unsupported standard task type: {task_type.value}')
