from collections.abc import Mapping

from ..annotation import Bbox, ImageInfo, Points, Shape
from ..labels import LabelCatalog


def _bbox_values(annotation: Bbox, image_info: ImageInfo) -> tuple[float, float, float, float]:
    assert 0 <= annotation.x1 <= annotation.x2 <= image_info.width
    assert 0 <= annotation.y1 <= annotation.y2 <= image_info.height
    x_center = (annotation.x1 + annotation.x2) / 2.0 / image_info.width
    y_center = (annotation.y1 + annotation.y2) / 2.0 / image_info.height
    width = (annotation.x2 - annotation.x1) / image_info.width
    height = (annotation.y2 - annotation.y1) / image_info.height
    assert all(0 <= value <= 1 for value in (x_center, y_center, width, height))
    return x_center, y_center, width, height


def encode_detect(annotation: Bbox, image_info: ImageInfo, labels: LabelCatalog) -> str:
    label_id = labels.index(annotation.label)
    x_center, y_center, width, height = _bbox_values(annotation, image_info)
    return f'{label_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}'


def encode_segment(annotation: Shape, image_info: ImageInfo, labels: LabelCatalog) -> str:
    if len(annotation.points) < 3:
        raise ValueError(f'分割标注必须是至少三点的多边形: {annotation}')
    values = [str(labels.index(annotation.label))]
    for x, y in annotation.points:
        norm_x = x / image_info.width
        norm_y = y / image_info.height
        assert 0 <= norm_x <= 1 and 0 <= norm_y <= 1
        values.extend((f'{norm_x:.6f}', f'{norm_y:.6f}'))
    return ' '.join(values)


def encode_pose(
    bbox: Bbox,
    keypoints: Mapping[str, Points],
    image_info: ImageInfo,
    keypoint_labels: LabelCatalog,
) -> str:
    if set(keypoints) != set(keypoint_labels.names):
        raise ValueError('关键点标签与目录不匹配')
    x_center, y_center, width, height = _bbox_values(bbox, image_info)
    values = ['0', f'{x_center:.6f}', f'{y_center:.6f}', f'{width:.6f}', f'{height:.6f}']
    for label in keypoint_labels:
        annotation = keypoints[label]
        if len(annotation.points) != 1:
            raise ValueError(f'骨骼标注必须是单点 Points: {annotation}')
        x, y = annotation.points[0]
        norm_x = x / image_info.width
        norm_y = y / image_info.height
        assert 0 <= norm_x <= 1 and 0 <= norm_y <= 1
        values.extend((f'{norm_x:.6f}', f'{norm_y:.6f}', '2'))
    return ' '.join(values)
