from dataclasses import replace
from pathlib import Path

from xxtrain.data import Annotation, Bbox, Circle, Polygon, Pose, assemble_poses, validate_obb
from xxtrain.data.formats import decode_detect, decode_pose, decode_segment, read_labelimg, read_labelme
from xxtrain.pipeline.core import Context, CropOutput, ItemProcessor, Sample
from xxtrain.task import TaskType


def _annotation_path(sample: Sample, directory: str, suffix: str) -> Path:
    return sample.image.path.parent.parent / directory / f'{sample.image.path.stem}{suffix}'


def _read_yolo(path: Path, sample: Sample, context: Context) -> tuple[Annotation, ...]:
    if not path.is_file():
        return ()
    info = sample.image.require_info()
    lines = tuple(line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
    decoder = {
        TaskType.DETECT: decode_detect,
        TaskType.SEGMENT: decode_segment,
        TaskType.OBB: decode_segment,
        TaskType.POSE: decode_pose,
    }.get(context.config.task_type)
    if decoder is None:
        raise ValueError(f'Unsupported annotation task: {context.config.task_type.value}')
    return tuple(decoder(line, info, context.config.labels) for line in lines)


def _normalize_pose(annotation: Annotation, pose_label: str, keypoint_labels: tuple[str, ...]) -> Annotation:
    if not isinstance(annotation, Pose):
        return annotation
    if annotation.label != pose_label:
        raise ValueError(f'Pose bbox label must be {pose_label}')
    keypoints = {keypoint.label: keypoint for keypoint in annotation.keypoints}
    if len(keypoints) != len(annotation.keypoints) or set(keypoints) != set(keypoint_labels):
        raise ValueError('Pose keypoint labels do not match the configured schema')
    return replace(annotation, keypoints=tuple(keypoints[label] for label in keypoint_labels))


def validate_annotations_for_task(
    annotations: tuple[Annotation, ...], task_type: TaskType, labels: tuple[str, ...]
) -> tuple[Annotation, ...]:
    if task_type is TaskType.DETECT:
        expected = Bbox
    elif task_type is TaskType.SEGMENT:
        expected = (Bbox, Circle, Polygon)
    elif task_type is TaskType.OBB:
        expected = Polygon
    elif task_type is TaskType.POSE:
        if len(labels) < 2:
            raise ValueError('Pose label catalog requires an object label and keypoint labels')
        expected = Pose
    else:
        raise ValueError(f'Unsupported annotation task: {task_type.value}')

    for annotation in annotations:
        if not isinstance(annotation, expected):
            raise TypeError(f'{task_type.value} does not support {type(annotation).__name__}')
        if annotation.label not in labels:
            raise ValueError(f'Unknown {task_type.value} label: {annotation.label}')
        if task_type is TaskType.POSE:
            expected_keypoints = labels[1:]
            actual_keypoints = tuple(keypoint.label for keypoint in annotation.keypoints)
            if annotation.label != labels[0] or actual_keypoints != expected_keypoints:
                raise ValueError('Pose annotation does not match the configured schema')
        if task_type is TaskType.OBB:
            validate_obb(annotation)
    return annotations


class ReadAnnotations(ItemProcessor[Sample, Sample]):
    input_type = Sample
    output_type = Sample

    def transform(self, item: Sample, context: Context) -> Sample:
        info = item.image.require_info()
        yolo = _read_yolo(_annotation_path(item, 'labels', '.txt'), item, context)
        xml_path = _annotation_path(item, 'anns', '.xml')
        xml = tuple(read_labelimg(xml_path, info)) if xml_path.is_file() else ()
        json_path = _annotation_path(item, 'anns_seg', '.json')
        labelme = tuple(read_labelme(json_path, info)) if json_path.is_file() else ()
        source_values = (yolo, xml, labelme)

        if context.config.task_type is TaskType.POSE:
            labels = context.config.labels.names
            if len(labels) < 2:
                raise ValueError('Pose label catalog requires an object label and keypoint labels')
            pose_label, keypoint_labels = labels[0], labels[1:]
            complete_sources = [values for values in source_values if any(isinstance(value, Pose) for value in values)]
            nonempty_sources = [values for values in source_values if values]
            if complete_sources and len(nonempty_sources) > 1:
                raise ValueError('duplicate pose representations from multiple formats')

            combined = tuple(annotation for values in source_values for annotation in values)
            assembled = (
                combined
                if complete_sources
                else assemble_poses(combined, keypoint_labels=keypoint_labels, match_ungrouped=True)
            )
            normalized = tuple(_normalize_pose(annotation, pose_label, keypoint_labels) for annotation in assembled)
            if any(not isinstance(annotation, Pose) for annotation in normalized):
                raise ValueError('pose fragments could not be assembled completely')
            annotations = normalized
        else:
            nonempty_sources = sum(bool(values) for values in source_values)
            if nonempty_sources > 1:
                raise ValueError('multiple non-empty annotation formats')
            annotations = tuple(annotation for values in source_values for annotation in values)

        validate_annotations_for_task(annotations, context.config.task_type, context.config.labels.names)
        return item.wrap(annotations=annotations)


class ValidateObb(ItemProcessor[Sample, Sample]):
    input_type = Sample
    output_type = Sample

    def transform(self, item: Sample, context: Context) -> Sample:
        validate_annotations_for_task(item.annotations, TaskType.OBB, context.config.labels.names)
        return item


class ExtractSample(ItemProcessor[CropOutput, Sample]):
    input_type = CropOutput
    output_type = Sample

    def transform(self, item: CropOutput, context: Context) -> Sample:
        return item.sample
