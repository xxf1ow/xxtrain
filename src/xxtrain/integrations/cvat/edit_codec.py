import json
from math import isfinite
from uuid import UUID

from xxtrain.business_tasks.definition import AnnotationPolicy
from xxtrain.platform.contracts import CvatBinding, EditAnnotation, EditFrame, EditFrameResult, EditJob, FrameMapping

from .codec import _ANNOTATION_ID, _decode_extra, _label_catalog


def encode_edit_annotations(
    frames: tuple[EditFrame, ...], labels: list[dict], policy: AnnotationPolicy | None = None, *, mapped: bool = False
) -> dict:
    """Encode task-policy annotations in exact edit-frame order.

    ``mapped`` adds each existing platform UUID to the reserved CVAT attribute for one-time initialization
    correlation. It never changes the input annotations or emits a native CVAT object ID.
    """

    by_name, _ = _edit_label_catalog(labels, policy)
    _validate_frame_mappings(tuple(frame.mapping for frame in frames))
    tags = []
    shapes = []
    seen_tokens: set[str] = set()
    for frame_index, frame in enumerate(frames):
        for annotation in frame.annotations:
            label_name = (
                policy.negative_label if annotation.kind == 'negative' and policy is not None else annotation.label
            )
            label = by_name.get(label_name)
            if label is None:
                raise ValueError(f'Unknown CVAT edit label: {label_name!r}')
            label_id, attribute_id, label_type = label
            object_type = _object_type(annotation.kind)
            expected_label_type = _native_type(annotation.kind)
            if expected_label_type != label_type:
                raise ValueError(f'CVAT label {label_name!r} has the wrong type for {annotation.kind!r}')
            extra = {}
            if mapped:
                if not isinstance(annotation.id, UUID):
                    raise ValueError('Mapped edit annotations require platform UUIDs')
                token = str(annotation.id)
                if token in seen_tokens:
                    raise ValueError('CVAT initialization annotation tokens must be unique')
                seen_tokens.add(token)
                extra[_ANNOTATION_ID] = token
            attributes = [{'spec_id': attribute_id, 'value': json.dumps(extra)}]
            if object_type == 'tag':
                if annotation.geometry is not None:
                    raise ValueError('CVAT tags require null geometry')
                tags.append({'frame': frame_index, 'label_id': label_id, 'attributes': attributes})
            else:
                shapes.append(
                    {
                        'type': expected_label_type,
                        'frame': frame_index,
                        'label_id': label_id,
                        'points': _flatten_points(annotation.geometry),
                        'occluded': False,
                        'outside': False,
                        'rotation': 0,
                        'z_order': 0,
                        'attributes': attributes,
                    }
                )
    return {'version': 0, 'tags': tags, 'shapes': shapes, 'tracks': []}


def decode_edit_annotations(
    payload: dict, job: EditJob, labels: list[dict], policy: AnnotationPolicy | None = None
) -> tuple[EditFrameResult, ...]:
    """Decode persistent CVAT annotations using native IDs and exact edit-frame identities.

    Initialization UUID attributes are transport-only and never become edited annotation identity. Unsupported
    tracks, label types, geometry, frame references, attributes, and missing or duplicate native IDs raise
    ``ValueError``.
    """

    mappings = _validate_job(job)
    by_id = _edit_label_catalog(labels, policy)[1]
    tags, shapes = _annotation_lists(payload)
    annotations: list[list[EditAnnotation]] = [[] for _ in mappings]
    seen_object_ids: set[tuple[str, int]] = set()

    for tag in tags:
        frame_index, name, attribute_id = _decode_header(tag, 'tag', mappings, by_id)
        object_id = _object_id(tag, 'tag')
        _remember_object_id(seen_object_ids, 'tag', object_id)
        _decode_extra(tag.get('attributes', []), attribute_id)
        if policy is not None and name == policy.negative_label:
            annotations[frame_index].append(EditAnnotation(None, 'negative', None, None, object_id))
        else:
            annotations[frame_index].append(EditAnnotation(None, 'classification', name, None, object_id))

    for shape in shapes:
        if not isinstance(shape, dict):
            raise ValueError('CVAT shapes must be objects')
        shape_type = shape.get('type')
        if shape_type not in {'rectangle', 'polyline'}:
            raise ValueError('CVAT edit shapes must be rectangles or polylines')
        frame_index, name, attribute_id = _decode_header(shape, shape_type, mappings, by_id)
        object_id = _object_id(shape, 'shape')
        _remember_object_id(seen_object_ids, 'shape', object_id)
        _validate_shape_state(shape)
        geometry = _unflatten_points(shape.get('points'))
        _decode_extra(shape.get('attributes', []), attribute_id)
        annotations[frame_index].append(EditAnnotation(None, shape_type, name, geometry, object_id))

    return tuple(
        EditFrameResult(mapping.frame_id, tuple(frame_annotations))
        for mapping, frame_annotations in zip(mappings, annotations, strict=True)
    )


def decode_edit_bindings(
    payload: dict,
    frames: tuple[EditFrame, ...],
    job: EditJob,
    labels: list[dict],
    policy: AnnotationPolicy | None = None,
) -> tuple[CvatBinding, ...]:
    """Bind every initialization UUID token to its native CVAT tag or shape ID.

    Frame mappings must equal the verified Job order. Tokens must be complete, unique, on their original frame,
    and attached to the same CVAT object type as the platform annotation.
    """

    mappings = _validate_job(job)
    if tuple(frame.mapping for frame in frames) != mappings:
        raise ValueError('CVAT initialization frames do not match the edit job')
    decode_edit_annotations(payload, job, labels, policy)
    by_id = _edit_label_catalog(labels, policy)[1]
    expected: dict[str, tuple[int, str, str, UUID]] = {}
    for frame_index, frame in enumerate(frames):
        for annotation in frame.annotations:
            if not isinstance(annotation.id, UUID):
                raise ValueError('CVAT initialization annotations require platform UUIDs')
            token = str(annotation.id)
            if token in expected:
                raise ValueError('CVAT initialization annotation tokens must be unique')
            expected[token] = frame_index, frame.mapping.image_id, _object_type(annotation.kind), annotation.id

    seen_tokens: set[str] = set()
    bindings = []
    tags, shapes = _annotation_lists(payload)
    for object_type, objects in (('tag', tags), ('shape', shapes)):
        for value in objects:
            native_type = value.get('type') if object_type == 'shape' and isinstance(value, dict) else 'tag'
            frame_index, _, attribute_id = _decode_header(value, native_type, mappings, by_id)
            extra = _decode_extra(value.get('attributes', []), attribute_id)
            token = extra.get(_ANNOTATION_ID)
            if not isinstance(token, str) or token not in expected or token in seen_tokens:
                raise ValueError('CVAT initialization annotation token is missing, unknown, or duplicated')
            expected_frame, sample_id, expected_type, annotation_id = expected[token]
            if frame_index != expected_frame:
                raise ValueError('CVAT initialization annotation token is on the wrong frame')
            if object_type != expected_type:
                raise ValueError('CVAT initialization annotation token is on the wrong object type')
            seen_tokens.add(token)
            bindings.append(CvatBinding(sample_id, object_type, _object_id(value, object_type), annotation_id))

    if seen_tokens != set(expected):
        raise ValueError('CVAT initialization annotation token set does not match the encoded annotations')
    return tuple(bindings)


def _edit_label_catalog(
    labels: list[dict], policy: AnnotationPolicy | None = None
) -> tuple[dict[str, tuple[int, int, str]], dict[int, tuple[str, int, str]]]:
    by_name, by_id = _label_catalog(labels)
    types_by_id: dict[int, str] = {}
    for label in labels:
        name = label.get('name') if isinstance(label, dict) else None
        label_id = label.get('id') if isinstance(label, dict) else None
        label_type = label.get('type') if isinstance(label, dict) else None
        expected_type = None
        if policy is not None:
            expected_type = 'tag' if name == policy.negative_label else policy.cvat_type
        allowed_types = {'tag', 'polyline'} if policy is None else {expected_type}
        if label_type not in allowed_types:
            raise ValueError(f'CVAT edit label {name!r} has unsupported type {label_type!r}')
        types_by_id[label_id] = label_type
    return (
        {name: (label_id, attribute_id, types_by_id[label_id]) for name, (label_id, attribute_id) in by_name.items()},
        {label_id: (name, attribute_id, types_by_id[label_id]) for label_id, (name, attribute_id) in by_id.items()},
    )


def _validate_job(job: EditJob) -> tuple[FrameMapping, ...]:
    mappings = _validate_frame_mappings(job.frames)
    if len(job.ref.sample_ids) != len(set(job.ref.sample_ids)):
        raise ValueError('Edit Job original image IDs must be unique')
    ordered_image_ids = tuple(dict.fromkeys(mapping.image_id for mapping in mappings))
    if ordered_image_ids != job.ref.sample_ids:
        raise ValueError('Edit Job frames do not match its original image IDs')
    return mappings


def _validate_frame_mappings(mappings: tuple[FrameMapping, ...]) -> tuple[FrameMapping, ...]:
    frame_ids = [mapping.frame_id for mapping in mappings]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError('CVAT edit frame IDs must be unique')
    return mappings


def _annotation_lists(payload: dict) -> tuple[list, list]:
    if not isinstance(payload, dict):
        raise ValueError('CVAT annotation payload must be an object')
    tracks = payload.get('tracks', [])
    if not isinstance(tracks, list) or tracks:
        raise ValueError('CVAT tracks are not supported')
    tags = payload.get('tags', [])
    shapes = payload.get('shapes', [])
    if not isinstance(tags, list):
        raise ValueError('CVAT tags must be a list')
    if not isinstance(shapes, list):
        raise ValueError('CVAT shapes must be a list')
    return tags, shapes


def _decode_header(
    value: object, native_type: str, mappings: tuple[FrameMapping, ...], by_id: dict[int, tuple[str, int, str]]
) -> tuple[int, str, int]:
    if not isinstance(value, dict):
        raise ValueError(f'CVAT {native_type}s must be objects')
    frame_index = value.get('frame')
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or not 0 <= frame_index < len(mappings):
        raise ValueError(f'CVAT {native_type} frame is out of range: {frame_index!r}')
    label_id = value.get('label_id')
    if isinstance(label_id, bool) or label_id not in by_id:
        raise ValueError(f'Unknown CVAT label ID: {label_id!r}')
    name, attribute_id, label_type = by_id[label_id]
    if label_type != native_type:
        raise ValueError(f'CVAT label {name!r} has the wrong type for {native_type}')
    return frame_index, name, attribute_id


def _object_type(kind: str) -> str:
    if kind in {'classification', 'negative'}:
        return 'tag'
    if kind in {'rectangle', 'polyline'}:
        return 'shape'
    raise ValueError(f'Unsupported CVAT edit annotation kind: {kind!r}')


def _native_type(kind: str) -> str:
    if kind in {'classification', 'negative'}:
        return 'tag'
    if kind in {'rectangle', 'polyline'}:
        return kind
    raise ValueError(f'Unsupported CVAT edit annotation kind: {kind!r}')


def _object_id(value: dict, object_type: str) -> int:
    object_id = value.get('id')
    if isinstance(object_id, bool) or not isinstance(object_id, int) or object_id <= 0:
        raise ValueError(f'CVAT {object_type} ID must be a positive integer')
    return object_id


def _remember_object_id(seen: set[tuple[str, int]], object_type: str, object_id: int) -> None:
    key = object_type, object_id
    if key in seen:
        raise ValueError('CVAT edit object IDs must be unique by object type')
    seen.add(key)


def _flatten_points(geometry: object) -> list[float]:
    if not isinstance(geometry, list) or len(geometry) < 2:
        raise ValueError('CVAT polylines require at least two points')
    flattened = []
    for point in geometry:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError('CVAT polyline points require coordinate pairs')
        flattened.extend(_coordinate(coordinate) for coordinate in point)
    return flattened


def _unflatten_points(points: object) -> list[list[float]]:
    if not isinstance(points, list) or len(points) < 4 or len(points) % 2:
        raise ValueError('CVAT polylines require paired coordinates for at least two points')
    coordinates = [_coordinate(coordinate) for coordinate in points]
    return [coordinates[index : index + 2] for index in range(0, len(coordinates), 2)]


def _coordinate(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise ValueError('CVAT polyline coordinates must be finite numbers')
    return float(value)


def _validate_shape_state(shape: dict) -> None:
    unsupported = (
        shape.get('occluded', False) is not False
        or shape.get('outside', False) is not False
        or shape.get('rotation', 0) != 0
        or shape.get('z_order', 0) != 0
        or shape.get('group', 0) not in (0, None)
    )
    if unsupported:
        raise ValueError('CVAT polyline state cannot be preserved')
