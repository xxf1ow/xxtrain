import json
from uuid import UUID

from xxtrain.business_tasks import POINT_BOX_LABELS
from xxtrain.data import Bbox
from xxtrain.platform.contracts import CvatBinding, DetectionBox, FrameResult, ImageInput, JobRef

_EXTRA_ATTRIBUTE = 'xxtrain_labelme_extra'
_ANNOTATION_ID = 'xxtrain_annotation_id'


def _label_catalog(labels: list[dict]) -> tuple[dict[str, tuple[int, int]], dict[int, tuple[str, int]]]:
    by_name: dict[str, tuple[int, int]] = {}
    by_id: dict[int, tuple[str, int]] = {}
    for label in labels:
        if not isinstance(label, dict):
            raise ValueError('CVAT labels must be objects')
        label_id = label.get('id')
        name = label.get('name')
        if isinstance(label_id, bool) or not isinstance(label_id, int) or not isinstance(name, str):
            raise ValueError('CVAT labels require integer IDs and string names')
        attributes = label.get('attributes', [])
        if not isinstance(attributes, list):
            raise ValueError('CVAT label attributes must be a list')
        reserved = [
            attribute
            for attribute in attributes
            if isinstance(attribute, dict) and attribute.get('name') == _EXTRA_ATTRIBUTE
        ]
        if len(reserved) != 1:
            raise ValueError(f'CVAT label {name!r} must define one {_EXTRA_ATTRIBUTE} attribute')
        attribute_id = reserved[0].get('id')
        if isinstance(attribute_id, bool) or not isinstance(attribute_id, int):
            raise ValueError(f'CVAT label {name!r} has an invalid {_EXTRA_ATTRIBUTE} attribute ID')
        if name in by_name or label_id in by_id:
            raise ValueError('CVAT label IDs and names must be unique')
        by_name[name] = label_id, attribute_id
        by_id[label_id] = name, attribute_id
    return by_name, by_id


def encode_annotations(images: tuple[ImageInput, ...], labels: list[dict]) -> dict:
    """Encode image rectangles as a CVAT annotation payload using IDs from the supplied label schema."""
    return _encode_annotations(images, labels, include_annotation_id=False)


def encode_mapped_annotations(images: tuple[ImageInput, ...], labels: list[dict]) -> dict:
    """Encode rectangles with transient UUID tokens for initialization correlation.

    The token is stored inside the existing reserved extra attribute and does not mutate the input boxes. Invalid
    labels or non-JSON extras raise ``ValueError``.
    """
    return _encode_annotations(images, labels, include_annotation_id=True)


def _encode_annotations(images: tuple[ImageInput, ...], labels: list[dict], *, include_annotation_id: bool) -> dict:
    by_name, _ = _label_catalog(labels)
    shapes = []
    for frame, image in enumerate(images):
        for box in image.boxes:
            if box.geometry.label not in POINT_BOX_LABELS or box.geometry.label not in by_name:
                raise ValueError(f'Unknown Point box label: {box.geometry.label!r}')
            label_id, attribute_id = by_name[box.geometry.label]
            extra = dict(box.extra)
            if include_annotation_id:
                extra[_ANNOTATION_ID] = str(box.geometry.id)
            try:
                encoded_extra = json.dumps(extra, ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError('Detection box extra must be a JSON object') from error
            shapes.append(
                {
                    'type': 'rectangle',
                    'frame': frame,
                    'label_id': label_id,
                    'points': [box.geometry.x1, box.geometry.y1, box.geometry.x2, box.geometry.y2],
                    'occluded': False,
                    'outside': False,
                    'rotation': 0,
                    'z_order': 0,
                    'attributes': [{'spec_id': attribute_id, 'value': encoded_extra}],
                }
            )
    return {'version': 0, 'shapes': shapes, 'tracks': [], 'tags': []}


def decode_annotations(payload: dict, ref: JobRef, labels: list[dict]) -> tuple[FrameResult, ...]:
    """Decode CVAT rectangles in frame order and reject annotation data that this adapter cannot preserve."""
    if not isinstance(payload, dict):
        raise ValueError('CVAT annotation payload must be an object')
    for field in ('tracks', 'tags'):
        value = payload.get(field, [])
        if not isinstance(value, list) or value:
            raise ValueError(f'CVAT {field} are not supported')
    shapes = payload.get('shapes', [])
    if not isinstance(shapes, list):
        raise ValueError('CVAT shapes must be a list')

    _, by_id = _label_catalog(labels) if labels else ({}, {})
    frames: list[list[DetectionBox]] = [[] for _ in ref.sample_ids]
    for shape in shapes:
        if not isinstance(shape, dict):
            raise ValueError('CVAT shapes must be objects')
        if shape.get('type') != 'rectangle':
            raise ValueError('CVAT detection annotations must be rectangles')
        frame = shape.get('frame')
        if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < len(frames):
            raise ValueError(f'CVAT shape frame is out of range: {frame!r}')
        label_id = shape.get('label_id')
        if isinstance(label_id, bool) or label_id not in by_id:
            raise ValueError(f'Unknown CVAT label ID: {label_id!r}')
        name, attribute_id = by_id[label_id]
        if name not in POINT_BOX_LABELS:
            raise ValueError(f'Unsupported Point box label: {name!r}')
        _validate_rectangle_state(shape)
        points = shape.get('points')
        if not isinstance(points, list) or len(points) != 4:
            raise ValueError('CVAT rectangles require four coordinates')
        attributes = shape.get('attributes', [])
        if not isinstance(attributes, list):
            raise ValueError('CVAT shape attributes must be a list')
        extra = _decode_extra(attributes, attribute_id)
        extra.pop(_ANNOTATION_ID, None)
        cvat_id = _shape_id(shape, required=False)
        try:
            geometry = Bbox(label=name, x1=points[0], y1=points[1], x2=points[2], y2=points[3])
        except (TypeError, ValueError) as error:
            raise ValueError('CVAT rectangle coordinates are invalid') from error
        frames[frame].append(DetectionBox(geometry=geometry, extra=extra, cvat_id=cvat_id))

    return tuple(FrameResult(sample_id, tuple(boxes)) for sample_id, boxes in zip(ref.sample_ids, frames, strict=True))


def decode_initial_bindings(
    payload: dict, images: tuple[ImageInput, ...], ref: JobRef, labels: list[dict]
) -> tuple[CvatBinding, ...]:
    """Bind native CVAT IDs to encoded platform UUIDs without relying on response order.

    Unsupported annotation content and any missing, duplicate, unknown, or wrong-frame identity evidence raise
    ``ValueError``. An empty initialization returns no bindings.
    """
    sample_ids = tuple(image.sample_id for image in images)
    if sample_ids != ref.sample_ids:
        raise ValueError('CVAT initialization images do not match the job samples')

    decode_annotations(payload, ref, labels)
    _, by_id = _label_catalog(labels) if labels else ({}, {})
    expected: dict[str, tuple[int, str, UUID]] = {}
    for frame, image in enumerate(images):
        for box in image.boxes:
            token = str(box.geometry.id)
            if token in expected:
                raise ValueError('CVAT initialization annotation tokens must be unique')
            expected[token] = frame, image.sample_id, box.geometry.id

    seen_tokens: set[str] = set()
    seen_object_ids: set[int] = set()
    bindings = []
    for shape in payload.get('shapes', []):
        label_id = shape['label_id']
        _, attribute_id = by_id[label_id]
        extra = _decode_extra(shape.get('attributes', []), attribute_id)
        token = extra.get(_ANNOTATION_ID)
        if not isinstance(token, str) or token not in expected or token in seen_tokens:
            raise ValueError('CVAT initialization annotation token is missing, unknown, or duplicated')
        frame, sample_id, annotation_id = expected[token]
        if shape['frame'] != frame:
            raise ValueError('CVAT initialization annotation token is on the wrong frame')
        object_id = _shape_id(shape, required=True)
        if object_id in seen_object_ids:
            raise ValueError('CVAT initialization object IDs must be unique')
        seen_tokens.add(token)
        seen_object_ids.add(object_id)
        bindings.append(CvatBinding(sample_id, 'shape', object_id, annotation_id))

    if seen_tokens != set(expected):
        raise ValueError('CVAT initialization annotation token set does not match the encoded annotations')
    return tuple(bindings)


def _validate_rectangle_state(shape: dict) -> None:
    unsupported = (
        shape.get('occluded', False) is not False
        or shape.get('outside', False) is not False
        or shape.get('rotation', 0) != 0
        or shape.get('z_order', 0) != 0
        or shape.get('group', 0) not in (0, None)
    )
    if unsupported:
        raise ValueError('CVAT rectangle state cannot be preserved')


def _shape_id(shape: dict, *, required: bool) -> int | None:
    object_id = shape.get('id')
    if object_id is None and not required:
        return None
    if isinstance(object_id, bool) or not isinstance(object_id, int) or object_id <= 0:
        raise ValueError('CVAT shape ID must be a positive integer')
    return object_id


def _decode_extra(attributes: list, attribute_id: int) -> dict:
    values = []
    for attribute in attributes:
        if not isinstance(attribute, dict) or attribute.get('spec_id') != attribute_id:
            raise ValueError('CVAT shape contains an unsupported attribute')
        values.append(attribute.get('value'))
    if not values:
        return {}
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError(f'CVAT {_EXTRA_ATTRIBUTE} attribute must appear once as text')
    try:
        extra = json.loads(values[0])
    except json.JSONDecodeError as error:
        raise ValueError(f'CVAT {_EXTRA_ATTRIBUTE} attribute must contain valid JSON') from error
    if not isinstance(extra, dict):
        raise ValueError(f'CVAT {_EXTRA_ATTRIBUTE} attribute must contain a JSON object')
    return extra
