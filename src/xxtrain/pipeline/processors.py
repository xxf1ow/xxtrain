from pathlib import Path

import numpy as np
import PIL.Image

from xxtrain.data import Annotation, Bbox, Circle, ImageInfo, Points, Polygon, Polyline, Shape
from xxtrain.data.formats import encode_detect, encode_pose, encode_segment, read_labelimg, read_labelme
from xxtrain.data.geometry import match_parent_children
from xxtrain.pipeline.core import (
    AnnotationMatch,
    ClassifyOutput,
    Context,
    CropOutput,
    EncodeOutput,
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
    annotations: tuple[Annotation, ...], labels: set[str], *, strict: bool, sample: Sample, context: Context
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
            (cx + radius * np.sin(2 * np.pi / count * index), cy + radius * np.cos(2 * np.pi / count * index))
            for index in range(count)
        )
        return Polygon(points=points, **common)
    if isinstance(shape, Polygon):
        return shape
    if isinstance(shape, Polyline) and len(shape.points) == 2:
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
            item.annotations, _allowed(self.labels, context), strict=self.strict, sample=item, context=context
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
            item.parents, _allowed(self.parent_labels, context), strict=self.strict, sample=item.sample, context=context
        )
        children = _filter(
            item.children, _allowed(self.child_labels, context), strict=self.strict, sample=item.sample, context=context
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
            accepted = (Polygon, Polyline, Points)
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
            item.parents, item.children, image_path=str(item.sample.image.path), wide=self.wide, strict=self.strict
        )
        parents = {parent.id: parent for parent in item.parents}
        children = {child.id: child for child in item.children}
        matches = tuple(
            AnnotationMatch(parent=parents[parent_id], children=tuple(children[child_id] for child_id in child_ids))
            for parent_id, child_ids in mapping.items()
        )
        return MatchOutput(sample=item.sample, matches=matches)


class CropMatches(ExpandProcessor[MatchOutput, CropOutput]):
    input_type = MatchOutput
    output_type = CropOutput

    def expand(self, item: MatchOutput, context: Context):
        for crop_index, match in enumerate(item.matches):
            x1, y1, x2, y2 = match.parent.bbox
            image = item.sample.image.wrap(crop_box=match.parent.bbox, info=ImageInfo(width=x2 - x1, height=y2 - y1))
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
                raise TypeError(f'CropDetectionBoxes requires Bbox, got {type(annotation).__name__}')
            x1, y1, x2, y2 = annotation.bbox
            sample = item.wrap(
                id=f'{item.id}_{crop_index}',
                image=item.image.wrap(crop_box=annotation.bbox, info=ImageInfo(width=x2 - x1, height=y2 - y1)),
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
            annotations=tuple(annotation.wrap(label=self.label) for annotation in item.sample.annotations)
        )
        return CropOutput(sample=sample, parent=item.parent)


class EncodeDetection(ItemProcessor[Sample, EncodeOutput]):
    input_type = Sample
    output_type = EncodeOutput

    def transform(self, item: Sample, context: Context) -> EncodeOutput:
        info = item.image.require_info()
        boxes = tuple(annotation for annotation in item.annotations if isinstance(annotation, Bbox))
        if len(boxes) != len(item.annotations):
            raise TypeError('EncodeDetection requires only Bbox annotations')
        return EncodeOutput(sample=item, lines=tuple(encode_detect(box, info, context.config.labels) for box in boxes))


class EncodeCropDetection(ItemProcessor[CropOutput, EncodeOutput]):
    input_type = CropOutput
    output_type = EncodeOutput

    def transform(self, item: CropOutput, context: Context) -> EncodeOutput:
        return EncodeDetection().transform(item.sample, context)


class EncodeSegment(ItemProcessor[Sample, EncodeOutput]):
    input_type = Sample
    output_type = EncodeOutput

    def transform(self, item: Sample, context: Context) -> EncodeOutput:
        info = item.image.require_info()
        shapes = tuple(annotation for annotation in item.annotations if isinstance(annotation, Shape))
        return EncodeOutput(
            sample=item, lines=tuple(encode_segment(shape, info, context.config.labels) for shape in shapes)
        )


class EncodePose(ItemProcessor[MatchOutput, EncodeOutput]):
    input_type = MatchOutput
    output_type = EncodeOutput

    def transform(self, item: MatchOutput, context: Context) -> EncodeOutput:
        info = item.sample.image.require_info()
        lines = []
        for match in item.matches:
            keypoints = {child.label: child for child in match.children if isinstance(child, Points)}
            if len(keypoints) != len(match.children):
                raise ValueError(f'骨骼标注必须是点类型: {item.sample.image.path}')
            lines.append(encode_pose(match.parent, keypoints, info, context.config.labels))
        return EncodeOutput(sample=item.sample, lines=tuple(lines))


class EncodePointSegment(ItemProcessor[CropOutput, EncodeOutput]):
    input_type = CropOutput
    output_type = EncodeOutput

    def transform(self, item: CropOutput, context: Context) -> EncodeOutput:
        info = item.sample.image.require_info()
        line_thickness = 6.0
        lines = []
        for annotation in item.sample.annotations:
            if not (isinstance(annotation, Polyline) and len(annotation.points) == 2):
                raise ValueError(f'标注类型错误: {item.sample.image.path} 中的 {annotation} 不是两点 Polyline 类型')
            p1 = np.asarray(annotation.points[0])
            p2 = np.asarray(annotation.points[1])
            vector = p2 - p1
            length = np.linalg.norm(vector)
            unit_vector = np.array([1.0, 0.0]) if length == 0 else vector / length
            normal_vector = np.array([-unit_vector[1], unit_vector[0]])
            half_thickness = line_thickness / 2.0
            corners = (p1 + normal_vector * half_thickness, p1 - normal_vector * half_thickness, p2)
            values = ['0']
            for x, y in corners:
                norm_x = x / info.width
                norm_y = y / info.height
                assert 0 <= norm_x <= 1 and 0 <= norm_y <= 1
                values.extend((f'{norm_x:.6f}', f'{norm_y:.6f}'))
            lines.append(' '.join(values))
        return EncodeOutput(sample=item.sample, lines=tuple(lines))


class EncodeKnobSegment(ItemProcessor[CropOutput, EncodeOutput]):
    input_type = CropOutput
    output_type = EncodeOutput

    def transform(self, item: CropOutput, context: Context) -> EncodeOutput:
        info = item.sample.image.require_info()
        label_id = context.config.labels.index(item.parent.label)
        lines = []
        for annotation in item.sample.annotations:
            if not isinstance(annotation, Shape):
                raise TypeError(f'EncodeKnobSegment requires Shape, got {type(annotation).__name__}')
            values = [str(label_id)]
            for x, y in annotation.points:
                norm_x = max(0.0, min(1.0, x / info.width))
                norm_y = max(0.0, min(1.0, y / info.height))
                values.extend((f'{norm_x:.6f}', f'{norm_y:.6f}'))
            lines.append(' '.join(values))
        return EncodeOutput(sample=item.sample, lines=tuple(lines))
