import math
from collections.abc import Sequence

from ..annotation import Bbox, ImageInfo, Keypoint, Polygon, Pose, Shape
from ..labels import LabelCatalog


def _tokens(line: str, *, count: int | None = None) -> tuple[str, ...]:
    if not isinstance(line, str):
        raise ValueError('YOLO line must be a string')
    values = tuple(line.split())
    if not values or (count is not None and len(values) != count):
        raise ValueError(f'Invalid YOLO token count: {len(values)}')
    return values


def _class_id(token: str, labels: LabelCatalog) -> int:
    if not token.isascii() or not token.isdecimal():
        raise ValueError(f'Invalid YOLO class id: {token}') from None
    value = int(token)
    if not 0 <= value < len(labels):
        raise ValueError(f'YOLO class id out of range: {value}')
    return value


def _finite_values(tokens: Sequence[str]) -> tuple[float, ...]:
    try:
        values = tuple(float(token) for token in tokens)
    except (TypeError, ValueError):
        raise ValueError('YOLO values must be numbers') from None
    if not all(math.isfinite(value) for value in values):
        raise ValueError('YOLO values must be finite')
    return values


def _bbox_values(annotation: Bbox | Pose, image_info: ImageInfo) -> tuple[float, float, float, float]:
    x_center = (annotation.x1 + annotation.x2) / 2.0 / image_info.width
    y_center = (annotation.y1 + annotation.y2) / 2.0 / image_info.height
    width = (annotation.x2 - annotation.x1) / image_info.width
    height = (annotation.y2 - annotation.y1) / image_info.height
    return x_center, y_center, width, height


def decode_detect(line: str, image_info: ImageInfo, labels: LabelCatalog) -> Bbox:
    tokens = _tokens(line, count=5)
    label = labels.names[_class_id(tokens[0], labels)]
    x_center, y_center, width, height = _finite_values(tokens[1:])
    x_center *= image_info.width
    y_center *= image_info.height
    width *= image_info.width
    height *= image_info.height
    return Bbox(
        label=label,
        x1=x_center - width / 2.0,
        y1=y_center - height / 2.0,
        x2=x_center + width / 2.0,
        y2=y_center + height / 2.0,
    )


def encode_detect(annotation: Bbox, image_info: ImageInfo, labels: LabelCatalog) -> str:
    label_id = labels.index(annotation.label)
    x_center, y_center, width, height = _bbox_values(annotation, image_info)
    return f'{label_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}'


def decode_segment(line: str, image_info: ImageInfo, labels: LabelCatalog) -> Polygon:
    tokens = _tokens(line)
    coordinate_count = len(tokens) - 1
    if coordinate_count < 6 or coordinate_count % 2:
        raise ValueError('YOLO segment requires at least three coordinate pairs')
    label = labels.names[_class_id(tokens[0], labels)]
    coordinates = _finite_values(tokens[1:])
    points = tuple(
        (coordinates[index] * image_info.width, coordinates[index + 1] * image_info.height)
        for index in range(0, len(coordinates), 2)
    )
    return Polygon(label=label, points=points)


def encode_segment(annotation: Shape, image_info: ImageInfo, labels: LabelCatalog) -> str:
    if len(annotation.points) < 3:
        raise ValueError(f'分割标注必须是至少三点的多边形: {annotation}')
    values = [str(labels.index(annotation.label))]
    for x, y in annotation.points:
        norm_x = x / image_info.width
        norm_y = y / image_info.height
        if not 0 <= norm_x <= 1 or not 0 <= norm_y <= 1:
            raise ValueError(f'分割标注点必须位于图像范围内: {annotation}')
        values.extend((f'{norm_x:.6f}', f'{norm_y:.6f}'))
    return ' '.join(values)


def decode_pose(line: str, image_info: ImageInfo, keypoint_labels: LabelCatalog) -> Pose:
    tokens = _tokens(line, count=5 + 3 * len(keypoint_labels))
    class_id = _class_id(tokens[0], keypoint_labels)
    if class_id != 0:
        raise ValueError(f'YOLO pose class id must be 0, got {class_id}')
    x_center, y_center, width, height = _finite_values(tokens[1:5])
    x_center *= image_info.width
    y_center *= image_info.height
    width *= image_info.width
    height *= image_info.height
    keypoint_values = _finite_values(tokens[5:])
    keypoints = []
    for index, label in enumerate(keypoint_labels):
        x, y, visibility = keypoint_values[index * 3 : index * 3 + 3]
        if visibility not in (0, 1, 2):
            raise ValueError(f'YOLO pose visibility must be 0, 1, or 2, got {visibility}')
        keypoints.append(
            Keypoint(label=label, x=x * image_info.width, y=y * image_info.height, visibility=int(visibility))
        )
    return Pose(
        label=keypoint_labels.names[0],
        x1=x_center - width / 2.0,
        y1=y_center - height / 2.0,
        x2=x_center + width / 2.0,
        y2=y_center + height / 2.0,
        keypoints=tuple(keypoints),
    )


def encode_pose(pose: Pose, image_info: ImageInfo, keypoint_labels: LabelCatalog) -> str:
    if not keypoint_labels.names:
        raise ValueError('关键点目录不能为空')
    if pose.label != keypoint_labels.names[0]:
        raise ValueError('YOLO pose label must match the first keypoint label')
    keypoints = {keypoint.label: keypoint for keypoint in pose.keypoints}
    if set(keypoints) != set(keypoint_labels.names):
        raise ValueError('关键点标签与目录不匹配')
    x_center, y_center, width, height = _bbox_values(pose, image_info)
    if not all(0 <= value <= 1 for value in (x_center, y_center, width, height)):
        raise ValueError('骨骼标注边界框必须位于 YOLO 归一化范围内')
    values = ['0', f'{x_center:.6f}', f'{y_center:.6f}', f'{width:.6f}', f'{height:.6f}']
    for label in keypoint_labels:
        keypoint = keypoints[label]
        norm_x = keypoint.x / image_info.width
        norm_y = keypoint.y / image_info.height
        if not 0 <= norm_x <= 1 or not 0 <= norm_y <= 1:
            raise ValueError(f'骨骼标注点必须位于图像范围内: {keypoint}')
        values.extend((f'{norm_x:.6f}', f'{norm_y:.6f}', str(keypoint.visibility)))
    return ' '.join(values)
