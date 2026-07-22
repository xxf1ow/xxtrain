import os
import xml.etree.ElementTree as ET
from pathlib import Path

from ..annotation import Annotation, Bbox, ImageInfo


def _required(parent: ET.Element, tag: str) -> ET.Element:
    value = parent.find(tag)
    if value is None:
        raise ValueError(f'Missing XML element: {tag}')
    return value


def _text(parent: ET.Element, tag: str) -> str:
    value = _required(parent, tag).text
    if value is None:
        raise ValueError(f'Missing XML text: {tag}')
    return value


def _number(parent: ET.Element, tag: str) -> float:
    return float(_text(parent, tag))


def read_labelimg(path: str | Path, image_info: ImageInfo) -> list[Annotation]:
    annotation_path = os.fspath(path)
    try:
        if not os.path.isfile(annotation_path):
            return []
        root = ET.parse(annotation_path).getroot()
        size = _required(root, 'size')
        assert image_info.width == int(_number(size, 'width')), f'图片与标签不对应: {annotation_path}'
        assert image_info.height == int(_number(size, 'height')), f'图片与标签不对应: {annotation_path}'
        annotations: list[Annotation] = []
        for obj in root.findall('object'):
            box = _required(obj, 'bndbox')
            annotation = Bbox(
                label=_text(obj, 'name'),
                x1=_number(box, 'xmin'),
                y1=_number(box, 'ymin'),
                x2=_number(box, 'xmax'),
                y2=_number(box, 'ymax'),
            )
            assert annotation.x2 <= image_info.width and annotation.y2 <= image_info.height, annotation_path
            annotations.append(annotation)
        return annotations
    except Exception as error:
        raise Exception(f'Failed to parse annotation: {annotation_path}, {error}') from error
