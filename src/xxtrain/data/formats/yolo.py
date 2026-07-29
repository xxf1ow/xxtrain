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


def _validate_normalized(values: Sequence[float], message: str) -> None:
    if not all(0 <= value <= 1 for value in values):
        raise ValueError(message)


def _validate_bbox_inside_image(
    x1: float, y1: float, x2: float, y2: float, image_info: ImageInfo, message: str
) -> None:
    if not 0 <= x1 < x2 <= image_info.width or not 0 <= y1 < y2 <= image_info.height:
        raise ValueError(message)


def _bbox_values(annotation: Bbox | Pose, image_info: ImageInfo) -> tuple[float, float, float, float]:
    x_center = (annotation.x1 + annotation.x2) / 2.0 / image_info.width
    y_center = (annotation.y1 + annotation.y2) / 2.0 / image_info.height
    width = (annotation.x2 - annotation.x1) / image_info.width
    height = (annotation.y2 - annotation.y1) / image_info.height
    values = x_center, y_center, width, height
    if not all(math.isfinite(value) for value in values):
        raise ValueError('YOLO bbox values must be finite')
    return values


def decode_detect(line: str, image_info: ImageInfo, labels: LabelCatalog) -> Bbox:
    tokens = _tokens(line, count=5)
    label = labels.names[_class_id(tokens[0], labels)]
    values = _finite_values(tokens[1:])
    _validate_normalized(values, 'YOLO detect values must be normalized')
    x_center, y_center, width, height = values
    x_center *= image_info.width
    y_center *= image_info.height
    width *= image_info.width
    height *= image_info.height
    x1, y1 = x_center - width / 2.0, y_center - height / 2.0
    x2, y2 = x_center + width / 2.0, y_center + height / 2.0
    _validate_bbox_inside_image(x1, y1, x2, y2, image_info, 'YOLO detect bbox must be inside image bounds')
    return Bbox(
        label=label,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def encode_detect(annotation: Bbox, image_info: ImageInfo, labels: LabelCatalog) -> str:
    label_id = labels.index(annotation.label)
    _validate_bbox_inside_image(
        annotation.x1, annotation.y1, annotation.x2, annotation.y2, image_info, 'YOLO detect bbox must be inside image bounds'
    )
    x_center, y_center, width, height = _bbox_values(annotation, image_info)
    _validate_normalized((x_center, y_center, width, height), 'YOLO detect values must be normalized')
    values = tuple(f'{value:.6f}' for value in (x_center, y_center, width, height))
    x_center, y_center, width, height = _finite_values(values)
    x_center *= image_info.width
    y_center *= image_info.height
    width *= image_info.width
    height *= image_info.height
    _validate_bbox_inside_image(
        x_center - width / 2.0,
        y_center - height / 2.0,
        x_center + width / 2.0,
        y_center + height / 2.0,
        image_info,
        'YOLO detect bbox must be inside image bounds',
    )
    return f'{label_id} {" ".join(values)}'


def decode_segment(line: str, image_info: ImageInfo, labels: LabelCatalog) -> Polygon:
    tokens = _tokens(line)
    coordinate_count = len(tokens) - 1
    if coordinate_count < 6 or coordinate_count % 2:
        raise ValueError('YOLO segment requires at least three coordinate pairs')
    label = labels.names[_class_id(tokens[0], labels)]
    coordinates = _finite_values(tokens[1:])
    _validate_normalized(coordinates, 'YOLO segment coordinates must be between 0 and 1')
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
        _validate_normalized((norm_x, norm_y), f'分割标注点必须位于图像范围内: {annotation}')
        values.extend((f'{norm_x:.6f}', f'{norm_y:.6f}'))
    return ' '.join(values)


def _pose_schema(labels: LabelCatalog) -> tuple[str, tuple[str, ...]]:
    if len(labels) < 2:
        raise ValueError('YOLO pose requires at least one object label and one keypoint label')
    return labels.names[0], labels.names[1:]


def decode_pose(line: str, image_info: ImageInfo, labels: LabelCatalog) -> Pose:
    pose_label, keypoint_labels = _pose_schema(labels)
    tokens = _tokens(line, count=5 + 3 * len(keypoint_labels))
    class_id = _class_id(tokens[0], labels)
    if class_id != 0:
        raise ValueError(f'YOLO pose class id must be 0, got {class_id}')
    x_center, y_center, width, height = _finite_values(tokens[1:5])
    _validate_normalized((x_center, y_center, width, height), 'YOLO pose bbox values must be between 0 and 1')
    x_center *= image_info.width
    y_center *= image_info.height
    width *= image_info.width
    height *= image_info.height
    x1, y1 = x_center - width / 2.0, y_center - height / 2.0
    x2, y2 = x_center + width / 2.0, y_center + height / 2.0
    _validate_bbox_inside_image(x1, y1, x2, y2, image_info, 'YOLO pose bbox must be inside image bounds')
    keypoint_values = _finite_values(tokens[5:])
    keypoints = tuple(
        Keypoint(
            label=label,
            x=keypoint_values[index * 3] * image_info.width,
            y=keypoint_values[index * 3 + 1] * image_info.height,
            visibility=int(keypoint_values[index * 3 + 2]),
        )
        for index, label in enumerate(keypoint_labels)
    )
    for index in range(len(keypoint_labels)):
        x, y, visibility = keypoint_values[index * 3 : index * 3 + 3]
        _validate_normalized((x, y), 'YOLO pose keypoint coordinates must be between 0 and 1')
        if visibility not in (0, 1, 2):
            raise ValueError(f'YOLO pose visibility must be 0, 1, or 2, got {visibility}')
    return Pose(label=pose_label, x1=x1, y1=y1, x2=x2, y2=y2, keypoints=keypoints)


def encode_pose(pose: Pose, image_info: ImageInfo, labels: LabelCatalog) -> str:
    pose_label, keypoint_labels = _pose_schema(labels)
    if pose.label != pose_label:
        raise ValueError('YOLO Pose label must match the first catalog label')
    keypoints = {keypoint.label: keypoint for keypoint in pose.keypoints}
    if set(keypoints) != set(keypoint_labels):
        raise ValueError('关键点标签与目录不匹配')
    _validate_bbox_inside_image(
        pose.x1, pose.y1, pose.x2, pose.y2, image_info, 'YOLO pose bbox must be inside image bounds'
    )
    x_center, y_center, width, height = _bbox_values(pose, image_info)
    _validate_normalized((x_center, y_center, width, height), '骨骼标注边界框必须位于 YOLO 归一化范围内')
    values = ['0', f'{x_center:.6f}', f'{y_center:.6f}', f'{width:.6f}', f'{height:.6f}']
    for label in keypoint_labels:
        keypoint = keypoints[label]
        norm_x = keypoint.x / image_info.width
        norm_y = keypoint.y / image_info.height
        _validate_normalized((norm_x, norm_y), f'骨骼标注点必须位于图像范围内: {keypoint}')
        values.extend((f'{norm_x:.6f}', f'{norm_y:.6f}', str(keypoint.visibility)))
    return ' '.join(values)
