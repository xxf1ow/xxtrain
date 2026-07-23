from pathlib import Path

import numpy as np
import PIL.Image

from xxtrain.data import (
    Annotation,
    Bbox,
    Circle,
    ImageInfo,
    Line,
    Points,
    Polygon,
    Polyline,
    RotatedBbox,
    Shape,
)
from xxtrain.data.formats import read_labelimg, read_labelme
from xxtrain.pipeline.core import ClassifyOutput, Context, ItemProcessor, MatchInput, Sample
from xxtrain.task import TaskType


def _annotation_path(sample: Sample, directory: str, suffix: str) -> Path:
    return sample.image.path.parent.parent / directory / f'{sample.image.path.stem}{suffix}'


def _allowed(labels: tuple[str, ...] | None, context: Context) -> set[str]:
    return set(context.config.labels.names if labels is None else labels)


def _filter(
    annotations: tuple[Annotation, ...],
    labels: set[str],
    *,
    strict: bool,
    sample: Sample,
    context: Context,
) -> tuple[Annotation, ...]:
    invalid = tuple(annotation for annotation in annotations if annotation.label not in labels)
    if strict and invalid:
        context.report.record_skipped_file(sample.image.path)
        for annotation in invalid:
            context.report.record_skipped_label(annotation.label)
    return tuple(annotation for annotation in annotations if annotation.label in labels)


def _prepare_segment_shape(shape: Shape) -> Shape:
    common = {'label': shape.label, 'id': shape.id, 'group': shape.group}
    if isinstance(shape, Bbox):
        return Polygon(points=shape.points, **common)
    if isinstance(shape, Circle):
        radius = shape.radius
        count = max(int(np.pi / np.arccos(1 - 1 / radius)), 12)
        cx, cy = shape.center
        points = tuple(
            (
                cx + radius * np.sin(2 * np.pi / count * index),
                cy + radius * np.cos(2 * np.pi / count * index),
            )
            for index in range(count)
        )
        return Polygon(points=points, **common)
    if isinstance(shape, (Polygon, RotatedBbox, Line)):
        return shape
    raise Exception(f"[Error] Task segment usually doesn't use {shape.type}")


class ReadImageInfo(ItemProcessor[Sample, Sample]):
    input_type = Sample
    output_type = Sample

    def transform(self, item: Sample, context: Context) -> Sample:
        with PIL.Image.open(item.image.path) as image:
            width, height = image.size
        return item.wrap(image=item.image.wrap(info=ImageInfo(width=width, height=height)))


class ReadLabelImg(ItemProcessor[Sample, Sample]):
    input_type = Sample
    output_type = Sample

    def transform(self, item: Sample, context: Context) -> Sample:
        annotations = read_labelimg(_annotation_path(item, 'anns', '.xml'), item.image.require_info())
        return item.wrap(annotations=tuple(annotations))


class ReadLabelMe(ItemProcessor[Sample, Sample]):
    input_type = Sample
    output_type = Sample

    def transform(self, item: Sample, context: Context) -> Sample:
        annotations = read_labelme(_annotation_path(item, 'anns_seg', '.json'), item.image.require_info())
        return item.wrap(annotations=tuple(annotations))


class ReadMatchingAnnotations(ItemProcessor[Sample, MatchInput]):
    input_type = Sample
    output_type = MatchInput

    def transform(self, item: Sample, context: Context) -> MatchInput:
        image_info = item.image.require_info()
        parents = read_labelimg(_annotation_path(item, 'anns', '.xml'), image_info)
        children = read_labelme(_annotation_path(item, 'anns_seg', '.json'), image_info)
        return MatchInput(sample=item, parents=tuple(parents), children=tuple(children))


class FilterLabels(ItemProcessor[Sample, Sample]):
    input_type = Sample
    output_type = Sample

    def __init__(self, labels: tuple[str, ...] | None = None, strict: bool = True):
        self.labels = labels
        self.strict = strict

    def transform(self, item: Sample, context: Context) -> Sample:
        annotations = _filter(
            item.annotations,
            _allowed(self.labels, context),
            strict=self.strict,
            sample=item,
            context=context,
        )
        return item.wrap(annotations=annotations)


class FilterMatchingAnnotations(ItemProcessor[MatchInput, MatchInput]):
    input_type = MatchInput
    output_type = MatchInput

    def __init__(
        self,
        parent_labels: tuple[str, ...] | None = None,
        child_labels: tuple[str, ...] | None = None,
        strict: bool = True,
    ):
        self.parent_labels = parent_labels
        self.child_labels = child_labels
        self.strict = strict

    def transform(self, item: MatchInput, context: Context) -> MatchInput:
        parents = _filter(
            item.parents,
            _allowed(self.parent_labels, context),
            strict=self.strict,
            sample=item.sample,
            context=context,
        )
        children = _filter(
            item.children,
            _allowed(self.child_labels, context),
            strict=self.strict,
            sample=item.sample,
            context=context,
        )
        return MatchInput(sample=item.sample, parents=tuple(parents), children=tuple(children))


class RelabelAnnotations(ItemProcessor[Sample, Sample]):
    input_type = Sample
    output_type = Sample

    def __init__(self, label: str):
        self.label = label

    def transform(self, item: Sample, context: Context) -> Sample:
        return item.wrap(annotations=tuple(annotation.wrap(label=self.label) for annotation in item.annotations))


class PrepareSegmentShapes(ItemProcessor[Sample, Sample]):
    input_type = Sample
    output_type = Sample

    def transform(self, item: Sample, context: Context) -> Sample:
        return item.wrap(annotations=tuple(_prepare_segment_shape(annotation) for annotation in item.annotations))


class PrepareMatchChildren(ItemProcessor[MatchInput, MatchInput]):
    input_type = MatchInput
    output_type = MatchInput

    def __init__(self, task_type: TaskType):
        self.task_type = task_type

    def transform(self, item: MatchInput, context: Context) -> MatchInput:
        if self.task_type is TaskType.SEGMENT:
            children = tuple(_prepare_segment_shape(child) for child in item.children)
            return MatchInput(sample=item.sample, parents=item.parents, children=children)
        if self.task_type is TaskType.POSE:
            accepted = (Polygon, RotatedBbox, Line, Polyline, Points)
            for child in item.children:
                if not isinstance(child, accepted):
                    raise Exception(f"[Error] Task pose usually doesn't use {child.type}")
            return item
        raise ValueError(f'Unsupported matching task type: {self.task_type.value}')


class PartitionAnnotations(ItemProcessor[Sample, MatchInput]):
    input_type = Sample
    output_type = MatchInput

    def __init__(self, parent_labels: tuple[str, ...], child_labels: tuple[str, ...]):
        self.parent_labels = set(parent_labels)
        self.child_labels = set(child_labels)

    def transform(self, item: Sample, context: Context) -> MatchInput:
        parents = tuple(
            annotation
            for annotation in item.annotations
            if isinstance(annotation, Bbox) and annotation.label in self.parent_labels
        )
        children = tuple(
            annotation
            for annotation in item.annotations
            if isinstance(annotation, Shape) and annotation.label in self.child_labels
        )
        return MatchInput(sample=item, parents=parents, children=children)


class PrepareClassification(ItemProcessor[Sample, ClassifyOutput]):
    input_type = Sample
    output_type = ClassifyOutput

    def transform(self, item: Sample, context: Context) -> ClassifyOutput:
        return ClassifyOutput(sample=item, class_name=item.source_group, output_name=item.image.path.name)
