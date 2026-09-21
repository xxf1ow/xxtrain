import json
from hashlib import sha256

from xxtrain.platform.contracts import AnnotationRecord, ImageInput, JsonValue


def legacy_point_fingerprint(target: str, images: tuple[ImageInput, ...], records: tuple[AnnotationRecord, ...]) -> str:
    """Reproduce the frozen Point fingerprints used before task-defined input identity."""
    if target == 'detect':
        payload = _legacy_detection_payload(images, records)
    elif target in {'classify', 'segment'}:
        relevant_steps = {'detect', 'classify'} if target == 'classify' else {'detect', 'classify', 'segment'}
        annotations = [_record_fingerprint(record) for record in records if record.step_key in relevant_steps]
        annotations.sort(key=lambda value: json.dumps(value, sort_keys=True, separators=(',', ':')))
        image_facts = sorted(
            ({'id': image.sample_id, 'width': image.width, 'height': image.height} for image in images),
            key=lambda value: value['id'],
        )
        payload = {'target': target, 'images': image_facts, 'annotations': annotations}
    else:
        raise ValueError(f'Unknown legacy Point target: {target!r}')
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode(
        'utf-8'
    )
    return sha256(encoded).hexdigest()


def _legacy_detection_payload(
    images: tuple[ImageInput, ...], records: tuple[AnnotationRecord, ...]
) -> list[dict[str, object]]:
    by_image: dict[str, list[AnnotationRecord]] = {}
    for record in records:
        if record.step_key == 'detect':
            by_image.setdefault(record.image_id, []).append(record)
    payload = []
    for image in images:
        detection_records = by_image.get(image.sample_id, ())
        boxes = [
            {'label': record.label, 'points': [[float(value) for value in point] for point in record.geometry]}
            for record in detection_records
            if record.kind == 'rectangle'
        ]
        if boxes:
            detection: object = {
                'boxes': sorted(
                    boxes, key=lambda box: json.dumps(box, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
                )
            }
        elif any(record.kind == 'negative' for record in detection_records):
            detection = {'negative': True}
        else:
            detection = None
        payload.append({'sha256': image.sample_id, 'detection': detection})
    return sorted(payload, key=lambda item: str(item['sha256']))


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
