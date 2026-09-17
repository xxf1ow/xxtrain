import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from uuid import UUID, uuid4

from xxtrain.business_tasks import target_complete, validate_target_annotations
from xxtrain.business_tasks.definition import TaskDefinition
from xxtrain.platform.contracts import (
    AnnotationRecord,
    CvatBinding,
    EditAnnotation,
    EditFrame,
    EditFrameResult,
    EditJob,
    FrameMapping,
    ImageInput,
    JsonValue,
    TargetSummary,
    TargetSync,
)

from .changes import plan_changes
from .crops import _crop_bounds, crop_frames, to_local, to_original


def project_target_frames(
    target: str, images: tuple[ImageInput, ...], records: tuple[AnnotationRecord, ...], runtime_root: Path
) -> tuple[EditFrame, ...]:
    """Materialize current crop frames and project one target's records into crop coordinates."""
    _validate_target(target)
    by_parent = _target_records_by_parent(target, records)
    projected = []
    for frame in crop_frames(images, runtime_root):
        annotations = tuple(
            _to_edit_annotation(target, record, frame.mapping) for record in by_parent.get(frame.mapping.parent_id, ())
        )
        validate_target_annotations(target, annotations)
        projected.append(replace(frame, annotations=annotations))
    return tuple(projected)


def summarize_target(
    target: str, images: tuple[ImageInput, ...], records: tuple[AnnotationRecord, ...]
) -> TargetSummary:
    """Derive target crop and completed-crop counts without reading or creating crop files."""
    _validate_target(target)
    by_parent = _target_records_by_parent(target, records)
    parent_ids = tuple(box.geometry.id for image in images for box in image.boxes)
    completed = 0
    for parent_id in parent_ids:
        annotations = tuple(_to_edit_annotation(target, record) for record in by_parent.get(parent_id, ()))
        if target_complete(target, annotations):
            completed += 1
    return TargetSummary(len(parent_ids), completed)


def fingerprint_target(target: str, images: tuple[ImageInput, ...], records: tuple[AnnotationRecord, ...]) -> str:
    """Hash target input facts, stable identities, associations, and upstream requirements."""
    _validate_target(target)
    relevant_steps = {'detect', 'classify'} if target == 'classify' else {'detect', 'classify', 'segment'}
    annotations = [_record_fingerprint(record) for record in records if record.step_key in relevant_steps]
    annotations.sort(key=lambda value: json.dumps(value, sort_keys=True, separators=(',', ':')))
    image_facts = sorted(
        ({'id': image.sample_id, 'width': image.width, 'height': image.height} for image in images),
        key=lambda value: value['id'],
    )
    payload = json.dumps(
        {'target': target, 'images': image_facts, 'annotations': annotations},
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return sha256(payload).hexdigest()


def prepare_sync(
    target: str,
    job: EditJob,
    results: tuple[EditFrameResult, ...],
    images: tuple[ImageInput, ...],
    current: tuple[AnnotationRecord, ...],
    existing_bindings: tuple[CvatBinding, ...],
    task: TaskDefinition,
) -> TargetSync:
    """Validate an edit Job result and plan its target-scoped atomic database update."""
    _validate_target(target)
    expected_mappings = _current_mappings(images)
    if job.frames != expected_mappings:
        raise ValueError('Edit job frames do not match the current crop identities and bounds')
    expected_image_ids = tuple(dict.fromkeys(mapping.image_id for mapping in expected_mappings))
    if job.ref.sample_ids != expected_image_ids:
        raise ValueError('Edit job original image scope does not match its current frames')
    expected_frame_ids = tuple(mapping.frame_id for mapping in expected_mappings)
    received_frame_ids = tuple(result.frame_id for result in results)
    if received_frame_ids != expected_frame_ids or len(set(received_frame_ids)) != len(received_frame_ids):
        raise ValueError('Edit results must cover every current crop frame exactly once and in order')

    object_type = 'tag' if target == 'classify' else 'shape'
    if any(binding.object_type != object_type for binding in existing_bindings):
        raise ValueError(f'{target!r} jobs contain an incompatible CVAT object binding')
    bindings_by_key = {(binding.object_type, binding.object_id): binding for binding in existing_bindings}
    if len(bindings_by_key) != len(existing_bindings):
        raise ValueError('CVAT edit object bindings must be unique by object type and native ID')
    current_by_id = {record.id: record for record in current}
    current_target = tuple(record for record in current if record.step_key == target)
    incoming: list[AnnotationRecord] = []
    bindings: list[CvatBinding] = []
    seen_keys: set[tuple[str, int]] = set()

    for mapping, result in zip(expected_mappings, results, strict=True):
        validate_target_annotations(target, result.annotations)
        for annotation in result.annotations:
            if annotation.id is not None:
                raise ValueError('Edited annotations use native CVAT IDs, not initialization UUID tokens')
            object_id = annotation.cvat_id
            if isinstance(object_id, bool) or not isinstance(object_id, int) or object_id <= 0:
                raise ValueError('Persistent target synchronization requires a native CVAT object ID')
            key = object_type, object_id
            if key in seen_keys:
                raise ValueError('CVAT edit object IDs must be unique by object type')
            seen_keys.add(key)
            existing_binding = bindings_by_key.get(key)
            if existing_binding is None:
                annotation_id = uuid4()
            else:
                existing = current_by_id.get(existing_binding.annotation_id)
                if (
                    existing is None
                    or existing_binding.sample_id != mapping.image_id
                    or existing.image_id != mapping.image_id
                    or existing.step_key != target
                    or existing.parent_id != mapping.parent_id
                ):
                    raise ValueError('A known CVAT object ID cannot move to another crop frame')
                annotation_id = existing.id

            geometry = annotation.geometry
            if target == 'segment':
                _validate_local_geometry(mapping, geometry)
                geometry = to_original(mapping, geometry)
            incoming.append(
                AnnotationRecord(
                    annotation_id,
                    mapping.image_id,
                    target,
                    mapping.parent_id,
                    annotation.kind,
                    annotation.label,
                    geometry,
                )
            )
            bindings.append(CvatBinding(mapping.image_id, object_type, object_id, annotation_id))

    incoming_ids = {record.id for record in incoming}
    delete_ids = frozenset(record.id for record in current_target if record.id not in incoming_ids)
    changes = plan_changes(current, tuple(incoming), delete_ids, task)
    final = {record.id: record for record in current}
    for annotation_id in changes.delete_ids:
        final.pop(annotation_id, None)
    final.update((record.id, record) for record in changes.upserts)
    return TargetSync(changes, tuple(bindings), fingerprint_target(target, images, tuple(final.values())))


def _validate_target(target: str) -> None:
    validate_target_annotations(target, ())


def _target_records_by_parent(
    target: str, records: tuple[AnnotationRecord, ...]
) -> dict[UUID | None, list[AnnotationRecord]]:
    by_parent: dict[UUID | None, list[AnnotationRecord]] = {}
    for record in records:
        if record.step_key == target:
            by_parent.setdefault(record.parent_id, []).append(record)
    return by_parent


def _to_edit_annotation(target: str, record: AnnotationRecord, mapping: FrameMapping | None = None) -> EditAnnotation:
    geometry = record.geometry
    if target == 'segment' and mapping is not None:
        geometry = to_local(mapping, geometry)
    return EditAnnotation(record.id, record.kind, record.label or '', geometry)


def _current_mappings(images: tuple[ImageInput, ...]) -> tuple[FrameMapping, ...]:
    mappings = []
    frame_ids: set[str] = set()
    for image in images:
        for box in image.boxes:
            parent_id = box.geometry.id
            if not isinstance(parent_id, UUID):
                raise ValueError('Crop source annotations require UUID identities')
            frame_id = str(parent_id)
            if frame_id in frame_ids:
                raise ValueError(f'Duplicate crop frame ID {frame_id}')
            frame_ids.add(frame_id)
            bounds = _crop_bounds(box.geometry.bbox, image.width, image.height)
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                raise ValueError(f'Crop frame {frame_id} has no pixels after clamping')
            mappings.append(FrameMapping(frame_id, image.sample_id, parent_id, bounds))
    return tuple(mappings)


def _validate_local_geometry(mapping: FrameMapping, geometry: JsonValue) -> None:
    if not isinstance(geometry, list):
        raise ValueError(f'Frame {mapping.frame_id} geometry must be a point list')
    width = mapping.bounds[2] - mapping.bounds[0]
    height = mapping.bounds[3] - mapping.bounds[1]
    for point in geometry:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f'Frame {mapping.frame_id} geometry must contain coordinate pairs')
        if not 0 <= float(point[0]) <= width or not 0 <= float(point[1]) <= height:
            raise ValueError(f'Frame {mapping.frame_id} geometry lies outside its crop bounds')


def _record_fingerprint(record: AnnotationRecord) -> dict[str, object]:
    return {
        'id': str(record.id),
        'image_id': record.image_id,
        'step': record.step_key,
        'parent_id': str(record.parent_id) if record.parent_id is not None else None,
        'kind': record.kind,
        'label': record.label,
        'geometry': _canonical_json(record.geometry),
    }


def _canonical_json(value: JsonValue) -> object:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        return [_canonical_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonical_json(item) for key, item in sorted(value.items())}
    raise ValueError('Annotation geometry must be valid JSON')
