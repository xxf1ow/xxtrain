import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from uuid import UUID, uuid4

from xxtrain.business_tasks import step_complete, validate_step_annotations
from xxtrain.business_tasks.definition import InputAdapter, StepDefinition, TaskDefinition
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


def project_target_frames(
    target: str,
    images: tuple[ImageInput, ...],
    records: tuple[AnnotationRecord, ...],
    runtime_root: Path,
    task: TaskDefinition,
) -> tuple[EditFrame, ...]:
    """Materialize a task step's frames and project its records into frame coordinates."""
    step = task.step(target)
    adapter = _input_adapter(step)
    mappings = adapter.mappings(images, records, step)
    by_source = _target_records_by_source(target, records)
    projected = []
    for frame in adapter.materialize(images, mappings, runtime_root):
        annotations = tuple(
            _to_edit_annotation(record, frame.mapping, adapter)
            for record in by_source.get((frame.mapping.image_id, frame.mapping.parent_id), ())
        )
        validate_step_annotations(step, annotations)
        projected.append(replace(frame, annotations=annotations))
    return tuple(projected)


def summarize_target(
    target: str, images: tuple[ImageInput, ...], records: tuple[AnnotationRecord, ...], task: TaskDefinition
) -> TargetSummary:
    """Derive sample, completion, and positive counts without materializing frame files."""
    step = task.step(target)
    adapter = _input_adapter(step)
    mappings = adapter.mappings(images, records, step)
    by_source = _target_records_by_source(target, records)
    completed = 0
    positive = 0
    for mapping in mappings:
        annotations = tuple(
            _to_edit_annotation(record, mapping, adapter)
            for record in by_source.get((mapping.image_id, mapping.parent_id), ())
        )
        if step_complete(step, annotations):
            completed += 1
            if any(annotation.kind != 'negative' for annotation in annotations):
                positive += 1
    return TargetSummary(len(mappings), completed, positive)


def fingerprint_target(
    target: str, images: tuple[ImageInput, ...], records: tuple[AnnotationRecord, ...], task: TaskDefinition
) -> str:
    """Hash target input facts, stable identities, associations, and upstream requirements."""
    step = task.step(target)
    mappings = _input_adapter(step).mappings(images, records, step)
    relevant = _relevant_records(target, mappings, records, task)
    annotations = [_record_fingerprint(record) for record in relevant]
    annotations.sort(key=lambda value: json.dumps(value, sort_keys=True, separators=(',', ':')))
    used_images = {mapping.image_id for mapping in mappings}
    image_facts = sorted(
        (
            {'id': image.sample_id, 'width': image.width, 'height': image.height}
            for image in images
            if image.sample_id in used_images
        ),
        key=lambda value: value['id'],
    )
    mapping_facts = [
        {
            'frame_id': mapping.frame_id,
            'image_id': mapping.image_id,
            'parent_id': str(mapping.parent_id) if mapping.parent_id is not None else None,
            'bounds': list(mapping.bounds),
        }
        for mapping in mappings
    ]
    mapping_facts.sort(key=lambda value: json.dumps(value, sort_keys=True, separators=(',', ':')))
    payload = json.dumps(
        {'target': target, 'images': image_facts, 'mappings': mapping_facts, 'annotations': annotations},
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return sha256(payload).hexdigest()


def fingerprint_training(target_fingerprint: str, target: str, task: TaskDefinition) -> str:
    """Hash editable input identity with stable task-declared conversion semantics."""
    step = task.step(target)
    training = step.training
    if training is None:
        raise ValueError(f'Task step {target!r} does not define training conversion')
    assert step.annotation is not None
    payload = json.dumps(
        {
            'input': target_fingerprint,
            'task': task.key,
            'target': target,
            'conversion': {
                'key': training.conversion_key,
                'labels': list(training.labels),
                'task_type': training.settings.task_type.value,
                'reserve_no_label': step.annotation.negative_label is not None,
            },
        },
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
    step = task.step(target)
    adapter = _input_adapter(step)
    expected_mappings = adapter.mappings(images, current, step)
    if job.frames != expected_mappings:
        raise ValueError('Edit job frames do not match the current crop identities and bounds')
    expected_image_ids = tuple(dict.fromkeys(mapping.image_id for mapping in expected_mappings))
    if job.ref.sample_ids != expected_image_ids:
        raise ValueError('Edit job original image scope does not match its current frames')
    expected_frame_ids = tuple(mapping.frame_id for mapping in expected_mappings)
    received_frame_ids = tuple(result.frame_id for result in results)
    if received_frame_ids != expected_frame_ids or len(set(received_frame_ids)) != len(received_frame_ids):
        raise ValueError('Edit results must cover every current crop frame exactly once and in order')

    bindings_by_key = {(binding.object_type, binding.object_id): binding for binding in existing_bindings}
    if len(bindings_by_key) != len(existing_bindings):
        raise ValueError('CVAT edit object bindings must be unique by object type and native ID')
    current_by_id = {record.id: record for record in current}
    for binding in existing_bindings:
        bound = current_by_id.get(binding.annotation_id)
        if bound is None or binding.object_type != _object_type(bound.kind):
            raise ValueError(f'{target!r} jobs contain an incompatible CVAT object binding')
    current_target = tuple(record for record in current if record.step_key == target)
    incoming: list[AnnotationRecord] = []
    bindings: list[CvatBinding] = []
    seen_keys: set[tuple[str, int]] = set()

    for mapping, result in zip(expected_mappings, results, strict=True):
        validate_step_annotations(step, result.annotations)
        for annotation in result.annotations:
            if annotation.id is not None:
                raise ValueError('Edited annotations use native CVAT IDs, not initialization UUID tokens')
            object_id = annotation.cvat_id
            if isinstance(object_id, bool) or not isinstance(object_id, int) or object_id <= 0:
                raise ValueError('Persistent target synchronization requires a native CVAT object ID')
            object_type = _object_type(annotation.kind)
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
            if geometry is not None:
                _validate_local_geometry(mapping, geometry)
                geometry = adapter.to_original(mapping, geometry)
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
    return TargetSync(changes, tuple(bindings), fingerprint_target(target, images, tuple(final.values()), task))


def _target_records_by_source(
    target: str, records: tuple[AnnotationRecord, ...]
) -> dict[tuple[str, UUID | None], list[AnnotationRecord]]:
    by_source: dict[tuple[str, UUID | None], list[AnnotationRecord]] = {}
    for record in records:
        if record.step_key == target:
            by_source.setdefault((record.image_id, record.parent_id), []).append(record)
    return by_source


def _to_edit_annotation(record: AnnotationRecord, mapping: FrameMapping, adapter: InputAdapter) -> EditAnnotation:
    geometry = record.geometry
    if geometry is not None:
        geometry = adapter.to_local(mapping, geometry)
    return EditAnnotation(record.id, record.kind, record.label, geometry)


def _input_adapter(step: StepDefinition) -> InputAdapter:
    if step.input_adapter is None:
        raise ValueError(f'Task step {step.key!r} requires an input adapter')
    return step.input_adapter


def _object_type(kind: str) -> str:
    return 'tag' if kind in {'classification', 'negative'} else 'shape'


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


def _relevant_records(
    target: str, mappings: tuple[FrameMapping, ...], records: tuple[AnnotationRecord, ...], task: TaskDefinition
) -> tuple[AnnotationRecord, ...]:
    input_steps = task.input_steps(target)
    by_id = {record.id: record for record in records}
    by_source: dict[tuple[str, UUID | None, str], list[AnnotationRecord]] = {}
    for record in records:
        by_source.setdefault((record.image_id, record.parent_id, record.step_key), []).append(record)
    selected_ids: set[UUID] = set()
    pending: list[UUID] = []

    def select(record: AnnotationRecord) -> None:
        if record.id not in selected_ids:
            selected_ids.add(record.id)
            pending.append(record.id)

    target_step = task.step(target)
    for mapping in mappings:
        if mapping.parent_id is not None:
            parent = by_id.get(mapping.parent_id)
            if parent is not None:
                select(parent)
        for record in by_source.get((mapping.image_id, mapping.parent_id, target), ()):
            select(record)
        for dependency in target_step.depends_on:
            for record in by_source.get((mapping.image_id, mapping.parent_id, dependency), ()):
                select(record)

    while pending:
        record = by_id[pending.pop()]
        if record.parent_id is not None:
            parent = by_id.get(record.parent_id)
            if parent is not None:
                select(parent)
        for dependency in task.step(record.step_key).depends_on:
            for related in by_source.get((record.image_id, record.parent_id, dependency), ()):
                select(related)
    return tuple(record for record in records if record.id in selected_ids and record.step_key in input_steps)


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
