import json
import os
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

from ..annotation import Annotation, Bbox, Circle, ImageInfo, Points, Polygon, Polyline, Pose
from ..geometry import validate_obb
from ..pose import assemble_poses


def _shape(label: str, shape_type: str, points: object, group: object) -> Annotation:
    common = {'label': label, 'group': group}
    if shape_type == 'rectangle':
        if not isinstance(points, list) or len(points) not in (2, 4):
            raise ValueError('Rectangle must have two or four points')
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        return Bbox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys), **common)
    if shape_type == 'polygon':
        return Polygon(points=points, **common)
    if shape_type == 'rotation':
        polygon = Polygon(points=points, **common)
        validate_obb(polygon)
        return polygon
    if shape_type == 'line':
        polyline = Polyline(points=points, **common)
        if len(polyline.points) != 2:
            raise ValueError('Line requires exactly two points')
        return polyline
    if shape_type == 'linestrip':
        return Polyline(points=points, **common)
    if shape_type == 'point':
        return Points(points=points, **common)
    if shape_type == 'circle':
        if not isinstance(points, list) or len(points) != 2:
            raise ValueError('Circle must have a center and edge point')
        return Circle(center=points[0], edge=points[1], **common)
    raise ValueError(f'Unsupported LabelMe shape type: {shape_type}')


def read_labelme(path: str | Path, image_info: ImageInfo) -> list[Annotation]:
    annotation_path = os.fspath(path)
    try:
        if not os.path.isfile(annotation_path):
            return []
        with open(annotation_path, encoding='utf-8') as file:
            data = json.load(file)
        assert image_info.width == int(data['imageWidth']), f'图片与标签不对应: {annotation_path}'
        assert image_info.height == int(data['imageHeight']), f'图片与标签不对应: {annotation_path}'
        annotations = tuple(
            _shape(
                label=shape['label'],
                shape_type=shape['shape_type'],
                points=shape['points'],
                group=shape.get('group_id'),
            )
            for shape in data['shapes']
        )
        return list(assemble_poses(annotations))
    except Exception as error:
        raise Exception(f'Failed to parse annotation: {annotation_path}, {error}') from error


def _group_id(group: int | str | UUID | None) -> int | str | None:
    return str(group) if isinstance(group, UUID) else group


def _labelme_shape(*, label: str, points: object, group: int | str | UUID | None, shape_type: str) -> dict[str, object]:
    return {'label': label, 'points': points, 'group_id': _group_id(group), 'shape_type': shape_type, 'flags': {}}


def _shapes(annotation: Annotation) -> tuple[dict[str, object], ...]:
    common = {'label': annotation.label, 'group': annotation.group}
    if isinstance(annotation, Bbox):
        return (
            _labelme_shape(
                points=((annotation.x1, annotation.y1), (annotation.x2, annotation.y2)),
                shape_type='rectangle',
                **common,
            ),
        )
    if isinstance(annotation, Circle):
        return (_labelme_shape(points=annotation.points, shape_type='circle', **common),)
    if isinstance(annotation, Polygon):
        return (_labelme_shape(points=annotation.points, shape_type='polygon', **common),)
    if isinstance(annotation, Polyline):
        shape_type = 'line' if len(annotation.points) == 2 else 'linestrip'
        return (_labelme_shape(points=annotation.points, shape_type=shape_type, **common),)
    if isinstance(annotation, Points):
        if len(annotation.points) != 1:
            raise ValueError('LabelMe point annotations require exactly one point')
        return (_labelme_shape(points=annotation.points, shape_type='point', **common),)
    if isinstance(annotation, Pose):
        if any(keypoint.visibility != 2 for keypoint in annotation.keypoints):
            raise ValueError('LabelMe cannot preserve Pose keypoint visibility other than 2')
        group = annotation.group if annotation.group is not None else str(annotation.id)
        shapes = [
            _labelme_shape(
                label=annotation.label,
                points=((annotation.x1, annotation.y1), (annotation.x2, annotation.y2)),
                group=group,
                shape_type='rectangle',
            )
        ]
        shapes.extend(
            _labelme_shape(label=keypoint.label, points=((keypoint.x, keypoint.y),), group=group, shape_type='point')
            for keypoint in annotation.keypoints
        )
        return tuple(shapes)
    raise TypeError(f'Unsupported LabelMe annotation type: {type(annotation).__name__}')


def write_labelme(annotations: Sequence[Annotation], path: str | Path, image_info: ImageInfo) -> None:
    shapes = [shape for annotation in annotations for shape in _shapes(annotation)]
    annotation_path = Path(path)
    payload = {
        'version': '5.0.0',
        'flags': {},
        'shapes': shapes,
        'imagePath': annotation_path.with_suffix('.jpg').name,
        'imageData': None,
        'imageHeight': image_info.height,
        'imageWidth': image_info.width,
    }
    content = json.dumps(payload, ensure_ascii=False, indent=2)

    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(content, encoding='utf-8')
