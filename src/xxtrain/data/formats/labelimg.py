import os
import xml.etree.ElementTree as ET
from collections.abc import Sequence
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


def write_labelimg(annotations: Sequence[Annotation], path: str | Path, image_info: ImageInfo) -> None:
    boxes: list[Bbox] = []
    for annotation in annotations:
        if not isinstance(annotation, Bbox):
            raise TypeError('LabelImg only supports Bbox annotations')
        boxes.append(annotation)
    if not image_info.width.is_integer() or not image_info.height.is_integer():
        raise ValueError('LabelImg image dimensions must be integral pixels')

    root = ET.Element('annotation')
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(int(image_info.width))
    ET.SubElement(size, 'height').text = str(int(image_info.height))
    for box in boxes:
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = box.label
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(box.x1)
        ET.SubElement(bndbox, 'ymin').text = str(box.y1)
        ET.SubElement(bndbox, 'xmax').text = str(box.x2)
        ET.SubElement(bndbox, 'ymax').text = str(box.y2)

    annotation_path = Path(path)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(annotation_path, encoding='utf-8', xml_declaration=True)
