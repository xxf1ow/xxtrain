import hashlib
import math
import os
import tempfile
from pathlib import Path

from PIL import Image

from xxtrain.platform.contracts import EditFrame, FrameMapping, ImageInput, JsonValue

_ENCODING_POLICY = 'rgb-png-v1'


def crop_frames(images: tuple[ImageInput, ...], root: Path) -> tuple[EditFrame, ...]:
    """Materialize one lossless RGB crop per detection box in caller order.

    Empty detection images produce no frames. Crop files are disposable and reused by original image ID,
    clamped integer bounds, and encoding policy; each source box retains its UUID as an independent frame ID.
    Duplicate frame IDs, registered-dimension mismatches, and boxes with no clamped pixels raise ``ValueError``.
    """
    mappings: list[FrameMapping] = []
    frame_ids: set[str] = set()
    for image in images:
        for box in image.boxes:
            frame_id = str(box.geometry.id)
            if frame_id in frame_ids:
                raise ValueError(f'Duplicate crop frame ID {frame_id}')
            frame_ids.add(frame_id)
            bounds = _crop_bounds(box.geometry.bbox, image.width, image.height)
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                raise ValueError(f'Crop frame {frame_id} has no pixels after clamping')
            mappings.append(FrameMapping(frame_id, image.sample_id, box.geometry.id, bounds))
    return materialize_crops(images, tuple(mappings), root)


def materialize_crops(
    images: tuple[ImageInput, ...], mappings: tuple[FrameMapping, ...], root: Path
) -> tuple[EditFrame, ...]:
    """Materialize crop mappings from original images without deriving their source associations."""
    crop_root = Path(root) / 'crops'
    by_image: dict[str, list[FrameMapping]] = {}
    for mapping in mappings:
        by_image.setdefault(mapping.image_id, []).append(mapping)
    frames: list[EditFrame] = []
    for image in images:
        image_mappings = by_image.pop(image.sample_id, ())
        if not image_mappings:
            continue
        with Image.open(image.image_path) as source:
            if source.size != (image.width, image.height):
                raise ValueError(f'Image {image.sample_id!r} dimensions do not match its registered size')
            rgb = source.convert('RGB')
        for mapping in image_mappings:
            path = crop_root / f'{_content_key(image.sample_id, mapping.bounds)}.png'
            if not path.exists():
                _publish_crop(rgb.crop(mapping.bounds), path)
            frames.append(
                EditFrame(
                    mapping, path, mapping.bounds[2] - mapping.bounds[0], mapping.bounds[3] - mapping.bounds[1], ()
                )
            )
    if by_image:
        raise ValueError(f'Crop mappings reference unknown images: {sorted(by_image)}')
    return tuple(frames)


def to_local(mapping: FrameMapping, geometry: JsonValue) -> JsonValue:
    """Translate JSON point geometry to edit-frame coordinates, preserving null geometry.

    Malformed point lists raise ``ValueError``.
    """
    return _translate(mapping, geometry, -mapping.bounds[0], -mapping.bounds[1])


def to_original(mapping: FrameMapping, geometry: JsonValue) -> JsonValue:
    """Translate JSON point geometry to original-image coordinates, preserving null geometry.

    Malformed point lists raise ``ValueError``.
    """
    return _translate(mapping, geometry, mapping.bounds[0], mapping.bounds[1])


def _crop_bounds(bounds: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bounds
    return (
        max(0, min(width, math.floor(min(x1, x2)))),
        max(0, min(height, math.floor(min(y1, y2)))),
        max(0, min(width, math.ceil(max(x1, x2)))),
        max(0, min(height, math.ceil(max(y1, y2)))),
    )


def _content_key(image_id: str, bounds: tuple[int, int, int, int]) -> str:
    payload = '\0'.join((_ENCODING_POLICY, image_id, *(str(value) for value in bounds)))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _publish_crop(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        image.save(stream, format='PNG')
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _translate(mapping: FrameMapping, geometry: JsonValue, dx: int, dy: int) -> JsonValue:
    if geometry is None:
        return None
    if not isinstance(geometry, list):
        raise ValueError(f'Frame {mapping.frame_id} geometry must be a JSON point list or null')
    translated: list[JsonValue] = []
    for point in geometry:
        if (
            not isinstance(point, list)
            or len(point) != 2
            or isinstance(point[0], bool)
            or not isinstance(point[0], int | float)
            or isinstance(point[1], bool)
            or not isinstance(point[1], int | float)
        ):
            raise ValueError(f'Frame {mapping.frame_id} geometry must contain numeric two-coordinate points')
        translated.append([point[0] + dx, point[1] + dy])
    return translated
