import copy

from xxtrain.data import Bbox
from xxtrain.platform.contracts import DetectionBox, JsonObject

_GEOMETRY_FIELDS = {'label', 'points', 'shape_type'}


def merge_detection(document: JsonObject, boxes: tuple[DetectionBox, ...]) -> JsonObject:
    """Return a deep copy with rectangles replaced and all other LabelMe data preserved."""
    result = copy.deepcopy(document)
    shapes = result.get('shapes', [])
    if not isinstance(shapes, list):
        raise ValueError('LabelMe shapes must be a list')

    preserved = []
    for shape in shapes:
        if not isinstance(shape, dict):
            raise ValueError('LabelMe shapes must be objects')
        if shape.get('shape_type') != 'rectangle':
            preserved.append(shape)

    rectangles = []
    for box in boxes:
        shape = copy.deepcopy(box.extra)
        shape.update(
            {
                'label': box.geometry.label,
                'points': [[box.geometry.x1, box.geometry.y1], [box.geometry.x2, box.geometry.y2]],
                'shape_type': 'rectangle',
            }
        )
        rectangles.append(shape)

    result['shapes'] = [*preserved, *rectangles]
    return result


def _detection_boxes(document: JsonObject) -> tuple[DetectionBox, ...]:
    shapes = document.get('shapes', [])
    if not isinstance(shapes, list):
        raise ValueError('LabelMe shapes must be a list')

    boxes = []
    for shape in shapes:
        if not isinstance(shape, dict):
            raise ValueError('LabelMe shapes must be objects')
        if shape.get('shape_type') != 'rectangle':
            continue
        points = shape.get('points')
        if not isinstance(points, list) or len(points) != 2:
            raise ValueError('LabelMe rectangles require two points')
        try:
            first, second = points
            x1, y1 = first
            x2, y2 = second
        except (TypeError, ValueError) as error:
            raise ValueError('LabelMe rectangle points must be coordinate pairs') from error
        geometry = Bbox(label=shape.get('label'), x1=min(x1, x2), y1=min(y1, y2), x2=max(x1, x2), y2=max(y1, y2))
        extra = copy.deepcopy({key: value for key, value in shape.items() if key not in _GEOMETRY_FIELDS})
        boxes.append(DetectionBox(geometry=geometry, extra=extra))
    return tuple(boxes)


def _empty_document(*, image_path: str, width: int, height: int) -> JsonObject:
    return {
        'version': '5.0.0',
        'flags': {},
        'shapes': [],
        'imagePath': image_path,
        'imageData': None,
        'imageHeight': height,
        'imageWidth': width,
    }
