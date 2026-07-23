from dataclasses import dataclass
from pathlib import Path

from xxtrain.data import LabelCatalog
from xxtrain.task import TaskType

from .core import Context, ConversionConfig, ConversionReport, Pipeline
from .discovery import DirectorySource, SampleSource, validate_classification_source
from .processors import (
    EncodeDetection,
    EncodePose,
    EncodeSegment,
    FilterLabels,
    FilterMatchingAnnotations,
    MatchAnnotations,
    PrepareClassification,
    PrepareMatchChildren,
    PrepareSegmentShapes,
    ReadImageInfo,
    ReadLabelImg,
    ReadLabelMe,
    ReadMatchingAnnotations,
)
from .sinks import ClassificationDatasetSink, DatasetSink, YoloDatasetSink

_STANDARD_TASK_TYPES = {
    'detect': TaskType.DETECT,
    'segment': TaskType.SEGMENT,
    'pose': TaskType.POSE,
    'classify': TaskType.CLASSIFY,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class Recipe:
    name: str
    task_type: TaskType
    labels: LabelCatalog
    source: SampleSource
    pipeline: Pipeline
    sink: DatasetSink

    def __post_init__(self) -> None:
        self.pipeline.validate_boundaries(self.source.output_type, self.sink.input_type)


def _read_labels(root_path: Path) -> LabelCatalog:
    labels_path = root_path / 'src' / 'labels.txt'
    assert labels_path.is_file(), f'标签列表不存在: {labels_path}'
    labels = tuple(line.strip() for line in labels_path.read_text(encoding='utf-8').splitlines())
    assert labels, f'标签列表为空: {labels_path}'
    return LabelCatalog(labels)


def build_recipe(
    task_name: str,
    root_path: str | Path,
    *,
    split: int = 10,
    reserve_no_label: bool = True,
) -> tuple[Recipe, Context]:
    try:
        task_type = _STANDARD_TASK_TYPES[task_name]
    except KeyError:
        raise ValueError(f'Unsupported standard task: {task_name}') from None

    root = Path(root_path)
    labels = _read_labels(root)
    if task_type is TaskType.CLASSIFY:
        labels = validate_classification_source(root, labels, split)

    context = Context(
        config=ConversionConfig(
            task_name=task_name,
            task_type=task_type,
            root_path=root,
            split=split,
            labels=labels,
            reserve_no_label=reserve_no_label,
        ),
        report=ConversionReport(),
    )
    source = DirectorySource()

    if task_type is TaskType.DETECT:
        pipeline = Pipeline((ReadImageInfo(), ReadLabelImg(), FilterLabels(), EncodeDetection()))
        sink = YoloDatasetSink()
    elif task_type is TaskType.SEGMENT:
        pipeline = Pipeline(
            (ReadImageInfo(), ReadLabelMe(), FilterLabels(), PrepareSegmentShapes(), EncodeSegment())
        )
        sink = YoloDatasetSink()
    elif task_type is TaskType.POSE:
        pipeline = Pipeline(
            (
                ReadImageInfo(),
                ReadMatchingAnnotations(),
                FilterMatchingAnnotations(),
                PrepareMatchChildren(TaskType.POSE),
                MatchAnnotations(),
                EncodePose(),
            )
        )
        sink = YoloDatasetSink()
    else:
        pipeline = Pipeline((PrepareClassification(),))
        sink = ClassificationDatasetSink()

    recipe = Recipe(
        name=task_name,
        task_type=task_type,
        labels=labels,
        source=source,
        pipeline=pipeline,
        sink=sink,
    )
    return recipe, context
