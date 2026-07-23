from dataclasses import dataclass
from pathlib import Path

from xxtrain.data import LabelCatalog
from xxtrain.task import TaskType

from .core import Context, ConversionConfig, ConversionReport, Pipeline
from .discovery import DirectorySource, SampleSource, validate_classification_source
from .processors import (
    CropDetectionBoxes,
    CropMatches,
    EncodeCropDetection,
    EncodeDetection,
    EncodeKnobSegment,
    EncodePointSegment,
    EncodePose,
    EncodeSegment,
    FilterLabels,
    FilterMatchingAnnotations,
    MatchAnnotations,
    PartitionAnnotations,
    PrepareClassification,
    PrepareMatchChildren,
    PrepareSegmentShapes,
    ReadImageInfo,
    ReadLabelImg,
    ReadLabelMe,
    ReadMatchingAnnotations,
    RelabelAnnotations,
    RelabelCropAnnotations,
)
from .sinks import ClassificationDatasetSink, DatasetSink, YoloDatasetSink

POINT_LABELS = ('tl', 'tc', 'cl', 'cc')
KNOB_LABELS = ('switch',)
LIGHT1_LABELS = ('1008',)

_STANDARD_TASK_TYPES = {
    'detect': TaskType.DETECT,
    'segment': TaskType.SEGMENT,
    'pose': TaskType.POSE,
    'classify': TaskType.CLASSIFY,
}

_CUSTOM_TASK_TYPES = {
    'point-detect': TaskType.DETECT,
    'point-classify': TaskType.CLASSIFY,
    'point-segment': TaskType.SEGMENT,
    'knob-detect': TaskType.DETECT,
    'knob-segment': TaskType.SEGMENT,
    'light1-detect': TaskType.DETECT,
    'light2-detect': TaskType.DETECT,
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
        try:
            task_type = _CUSTOM_TASK_TYPES[task_name]
        except KeyError:
            raise ValueError(f'Unsupported task type: {task_name}') from None

    root = Path(root_path)
    if task_name == 'point-detect':
        labels = LabelCatalog(('Point',))
    elif task_name == 'point-classify':
        labels = LabelCatalog(POINT_LABELS)
    elif task_name == 'point-segment':
        labels = LabelCatalog(('Point',))
    elif task_name in ('knob-detect', 'knob-segment'):
        labels = LabelCatalog(KNOB_LABELS)
    elif task_name == 'light1-detect':
        labels = LabelCatalog(LIGHT1_LABELS)
    elif task_name == 'light2-detect':
        labels = LabelCatalog(('0',))
    else:
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

    if task_name == 'point-detect':
        pipeline = Pipeline(
            (
                ReadImageInfo(),
                ReadLabelImg(),
                FilterLabels(POINT_LABELS),
                RelabelAnnotations('Point'),
                EncodeDetection(),
            )
        )
        sink = YoloDatasetSink()
    elif task_name == 'point-classify':
        pipeline = Pipeline(
            (ReadImageInfo(), ReadLabelImg(), FilterLabels(POINT_LABELS), CropDetectionBoxes())
        )
        sink = ClassificationDatasetSink(indexed_class_directories=True)
    elif task_name == 'point-segment':
        pipeline = Pipeline(
            (
                ReadImageInfo(),
                ReadMatchingAnnotations(),
                FilterMatchingAnnotations(
                    parent_labels=('tl', 'tc', 'cl', 'cc'),
                    child_labels=('1',),
                    strict=True,
                ),
                PrepareMatchChildren(TaskType.SEGMENT),
                MatchAnnotations(wide=0, strict=False),
                CropMatches(),
                EncodePointSegment(),
            )
        )
        sink = YoloDatasetSink()
    elif task_name == 'knob-detect':
        pipeline = Pipeline(
            (ReadImageInfo(), ReadLabelImg(), FilterLabels(KNOB_LABELS), EncodeDetection())
        )
        sink = YoloDatasetSink()
    elif task_name == 'knob-segment':
        pipeline = Pipeline(
            (
                ReadImageInfo(),
                ReadMatchingAnnotations(),
                FilterMatchingAnnotations(
                    parent_labels=('switch',),
                    child_labels=('switch',),
                    strict=True,
                ),
                PrepareMatchChildren(TaskType.SEGMENT),
                MatchAnnotations(wide=0.15, strict=False),
                CropMatches(),
                EncodeKnobSegment(),
            )
        )
        sink = YoloDatasetSink()
    elif task_name == 'light1-detect':
        pipeline = Pipeline(
            (
                ReadImageInfo(),
                ReadLabelImg(),
                FilterLabels(LIGHT1_LABELS, strict=False),
                EncodeDetection(),
            )
        )
        sink = YoloDatasetSink()
    elif task_name == 'light2-detect':
        pipeline = Pipeline(
            (
                ReadImageInfo(),
                ReadLabelImg(),
                PartitionAnnotations(
                    parent_labels=('1008',),
                    child_labels=('0', '1', '2'),
                ),
                MatchAnnotations(wide=0.1, strict=False),
                CropMatches(),
                RelabelCropAnnotations('0'),
                EncodeCropDetection(),
            )
        )
        sink = YoloDatasetSink()
    elif task_type is TaskType.DETECT:
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
