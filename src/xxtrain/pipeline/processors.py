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
from xxtrain.data.geometry import match_parent_children
from xxtrain.pipeline.core import (
    AnnotationMatch,
    ClassifyOutput,
    Context,
    CropOutput,
    ExpandProcessor,
    ItemProcessor,
    MatchInput,
    MatchOutput,
    Sample,
)
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


class MatchAnnotations(ItemProcessor[MatchInput, MatchOutput]):
    input_type = MatchInput
    output_type = MatchOutput

    def __init__(self, wide: float = 0, strict: bool = True):
        self.wide = wide
        self.strict = strict

    def transform(self, item: MatchInput, context: Context) -> MatchOutput:
        mapping = match_parent_children(
            item.parents,
            item.children,
            image_path=str(item.sample.image.path),
            wide=self.wide,
            strict=self.strict,
        )
        parents = {parent.id: parent for parent in item.parents}
        children = {child.id: child for child in item.children}
        matches = tuple(
            AnnotationMatch(
                parent=parents[parent_id],
                children=tuple(children[child_id] for child_id in child_ids),
            )
            for parent_id, child_ids in mapping.items()
        )
        return MatchOutput(sample=item.sample, matches=matches)


class CropMatches(ExpandProcessor[MatchOutput, CropOutput]):
    input_type = MatchOutput
    output_type = CropOutput

    def expand(self, item: MatchOutput, context: Context):
        for crop_index, match in enumerate(item.matches):
            x1, y1, x2, y2 = match.parent.bbox
            image = item.sample.image.wrap(
                crop_box=match.parent.bbox,
                info=ImageInfo(width=int(x2 - x1), height=int(y2 - y1)),
            )
            sample = item.sample.wrap(
                id=f'{item.sample.id}_{crop_index}',
                image=image,
                annotations=tuple(child.translate(-x1, -y1) for child in match.children),
            )
            yield CropOutput(sample=sample, parent=match.parent)


class CropDetectionBoxes(ExpandProcessor[Sample, ClassifyOutput]):
    input_type = Sample
    output_type = ClassifyOutput

    def expand(self, item: Sample, context: Context):
        for crop_index, annotation in enumerate(item.annotations):
            if not isinstance(annotation, Bbox):
                raise TypeError(
                    f'CropDetectionBoxes requires Bbox, got {type(annotation).__name__}'
                )
            x1, y1, x2, y2 = map(int, annotation.bbox)
            sample = item.wrap(
                id=f'{item.id}_{crop_index}',
                image=item.image.wrap(
                    crop_box=annotation.bbox,
                    info=ImageInfo(width=x2 - x1, height=y2 - y1),
                ),
                annotations=(),
            )
            yield ClassifyOutput(
                sample=sample,
                class_name=annotation.label,
                output_name=f'{item.source_group}_{item.source_index}_{crop_index}.jpg',
            )


class RelabelCropAnnotations(ItemProcessor[CropOutput, CropOutput]):
    input_type = CropOutput
    output_type = CropOutput

    def __init__(self, label: str):
        self.label = label

    def transform(self, item: CropOutput, context: Context) -> CropOutput:
        sample = item.sample.wrap(
            annotations=tuple(
                annotation.wrap(label=self.label) for annotation in item.sample.annotations
            )
        )
        return CropOutput(sample=sample, parent=item.parent)
