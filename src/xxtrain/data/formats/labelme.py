import json
import os
from pathlib import Path

from ..annotation import Annotation, Bbox, Circle, ImageInfo, Line, Points, Polygon, Polyline


def _shape(label: str, shape_type: str, points: object, group: object) -> Annotation:
    common = {'label': label, 'group': group}
    if shape_type == 'rectangle':
        if not isinstance(points, list) or len(points) not in (2, 4):
            raise ValueError('Rectangle must have two or four points')
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        return Bbox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys), **common)
    if shape_type in ('polygon', 'rotation'):
        return Polygon(points=points, **common)
    if shape_type == 'line':
        return Line(points=points, **common)
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
        return [
            _shape(
                label=shape['label'],
                shape_type=shape['shape_type'],
                points=shape['points'],
                group=shape.get('group_id'),
            )
            for shape in data['shapes']
        ]
    except Exception as error:
        raise Exception(f'Failed to parse annotation: {annotation_path}, {error}') from error
