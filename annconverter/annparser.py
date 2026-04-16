import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Optional, Union
from uuid import UUID, uuid1
from xml.dom import minidom

import numpy as np
import PIL.Image
import PIL.ImageDraw


class ShapeType(Enum):
    RECTANGLE = 'rectangle'
    CIRCLE = 'circle'
    POLYGON = 'polygon'
    LINE = 'line'
    LINESTRIP = 'linestrip'
    POINT = 'point'
    ROTATION = 'rotation'  # OBB 或 旋转矩形


class TaskType(Enum):
    OBB = 'obb'
    POSE = 'pose'
    DETECT = 'detect'
    SEGMENT = 'segment'
    CLASSIFY = 'classify'


@dataclass
class Annotation:
    """单个标注实例的通用类"""

    label: str
    type: ShapeType
    parts: 'list[list[float]]' = field(default_factory=list)  # [x1, y1, x2, y2, ...] (可能多个形状是一个实例的不同部分)
    instance: Union[tuple, UUID] = field(default_factory=uuid1)  # noqa: UP007, UP045, group id 仅在语义分割标注中使用
    mask: Optional[np.ndarray] = None  # noqa: UP007, UP045, 掩码图像, 部分分割标注中使用

    @property
    def points(self) -> 'list[float]':
        """返回 parts[0], 不适用 group id 的任务使用"""
        return self.parts[0] if self.parts else []

    @property
    def bbox(self) -> 'list[float]':
        """根据 points 自动计算外接矩形 [xmin, ymin, xmax, ymax]"""
        if not self.points:
            return [0.0, 0.0, 0.0, 0.0]
        xs = self.points[0::2]
        ys = self.points[1::2]
        return [min(xs), min(ys), max(xs), max(ys)]

    def translate(self, dx: float, dy: float) -> 'Annotation':
        """返回平移后的新对象"""
        new_parts = [[(p + dx if i % 2 == 0 else p + dy) for i, p in enumerate(part)] for part in self.parts]
        return replace(self, parts=new_parts)


# 判断点 point 是否在矩形 rect 内部. rect: [xmin, ymin, xmax, ymax]
def rectangle_include_point(r: 'list[float]', p: 'list[float]') -> bool:
    return p[0] >= r[0] and p[0] <= r[2] and p[1] >= r[1] and p[1] <= r[3]


# 判断点 point 是否在矩形 rect 内部, 宽松版本. rect: [xmin, ymin, xmax, ymax]
def rectangle_include_point_wide(r: 'list[float]', p: 'list[float]', w: float) -> bool:
    return p[0] >= r[0] - w and p[0] <= r[2] + w and p[1] >= r[1] - w and p[1] <= r[3] + w


# 判断一个 shape 是否完全在矩形 rect 内部, 宽松版本. rect: [xmin, ymin, xmax, ymax], shape_points: [x1, y1, x2, y2, ...]
def rectangle_include_shape(rect: 'list[float]', shape_points: 'list[float]', shape_type=None) -> bool:
    w = 10  # 宽松版本的宽度, 用于容忍标注误差.
    if shape_type == 'circle':
        assert len(shape_points) == 4, 'Shape of shape_type=circle must have 4 points'
        (cx, cy), (px, py) = shape_points[:2], shape_points[2:]
        r = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        return rectangle_include_point_wide(rect, [cx, cy], r + w)
    else:
        for i in range(0, len(shape_points), 2):
            if not rectangle_include_point_wide(rect, [shape_points[i], shape_points[i + 1]], w):
                return False
        return True


# box (list): [xmin, ymin, xmax, ymax]
def calculate_iou(box1: 'list[float]', box2: 'list[float]') -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0


# nms
def calculate_nms(bboxes: dict, iou_threshold: float = 0.25):
    sorted_bboxes = sorted(bboxes.items(), key=lambda x: x[1][2] * x[1][3], reverse=True)
    selected_bboxes = {}
    while sorted_bboxes:
        instance, box = sorted_bboxes.pop(0)
        selected_bboxes[instance] = box
        remaining_bboxes = []
        for other_instance, other_box in sorted_bboxes:
            iou = calculate_iou(box, other_box)
            if iou < iou_threshold:
                remaining_bboxes.append((other_instance, other_box))
        sorted_bboxes = remaining_bboxes
    return selected_bboxes


def get_color_map(num_classes: int) -> 'list[tuple[int, int, int]]':
    color_map = []
    for i in range(num_classes):
        r, g, b, j, lab = 0, 0, 0, 0, i
        while lab:
            r |= ((lab >> 0) & 1) << (7 - j)
            g |= ((lab >> 1) & 1) << (7 - j)
            b |= ((lab >> 2) & 1) << (7 - j)
            j += 1
            lab >>= 3
        color_map.append((r, g, b))
    return color_map


def find_dir(path: str) -> 'list[str]':
    return [item.name for item in os.scandir(path) if item.is_dir()]


def find_img(path: str) -> 'list[str]':
    return [
        item.name
        for item in os.scandir(path)
        if item.is_file() and item.name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]


# 取出 xml 单项 (length 预期长度, 为 0 则不检查)
def get_xml_value(root: 'ET.Element[str]', name: str) -> ET.Element[str]:
    XmlValue = root.findall(name)
    if XmlValue is None:
        raise Exception(f'Cannot find {name} in XML file.')
    if len(XmlValue) != 1:
        raise Exception(f'The size of {name} is supposed to be 1, but is {len(XmlValue)}.')
    return XmlValue[0]


# 取出 xml 单项
def get_xml_str_value(root: 'ET.Element[str]', name: str) -> str:
    value = get_xml_value(root, name)
    return '' if value.text is None else value.text


# 取出 xml 单项
def get_xml_float_value(root: 'ET.Element[str]', name: str) -> float:
    value = get_xml_value(root, name)
    return 0 if value.text is None else float(value.text)


# 取出 xml 列表 (length 预期长度, 为 0 则不检查)
def get_xml_list(root: 'ET.Element[str]', name: str, length: int) -> 'list[ET.Element[str]]':
    XmlValue = root.findall(name)
    if XmlValue is None:
        raise Exception(f'Cannot find {name} in XML file.')
    if length > 0 and len(XmlValue) != length:
        raise Exception(f'The size of {name} is supposed to be {length}, but is {len(XmlValue)}.')
    return XmlValue


# 解析单个 labelimg 标注文件(xml)
def parse_det_anns_from_labelimg(
    det_path: str, img_width: int, img_height: int, overlap_check: bool = True
) -> 'dict[UUID, Annotation]':
    try:
        if not os.path.isfile(det_path):
            raise FileNotFoundError('file not found ...')
        tree = ET.parse(det_path)
        root = tree.getroot()
        # check image size
        imgsize = get_xml_value(root, 'size')
        assert img_width == int(get_xml_float_value(imgsize, 'width')), f'图片与标签不对应: {det_path}'
        assert img_height == int(get_xml_float_value(imgsize, 'height')), f'图片与标签不对应: {det_path}'
        # parse box info
        instances_map: dict[UUID, Annotation] = {}
        for obj in get_xml_list(root, 'object', 0):
            name = get_xml_str_value(obj, 'name')
            bndbox = get_xml_value(obj, 'bndbox')
            xmin = get_xml_float_value(bndbox, 'xmin')
            ymin = get_xml_float_value(bndbox, 'ymin')
            xmax = get_xml_float_value(bndbox, 'xmax')
            ymax = get_xml_float_value(bndbox, 'ymax')
            assert xmax > xmin and ymax > ymin and xmax <= img_width and ymax <= img_height, f'{det_path}'
            points = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]  # 矩形转多边形点集
            instances_map[uuid1()] = Annotation(label=name, parts=[points], type=ShapeType.RECTANGLE)

        # todo: fix it ...
        # if overlap_check:
        #     sorted_bboxes = calculate_nms(bbox)
        #     if len(bbox) != len(sorted_bboxes):
        #         print(f'\n labelimg 标注出现重叠框: {det_path}\n')

        return instances_map
    except Exception as e:
        raise Exception(f'Failed to parse annotation: {det_path}, {e}')


def create_labelimg(xml_path: str, image_name: str, width: int, height: int, bbox_dict: dict):
    # 创建根元素 <annotation>
    root = ET.Element('annotation')
    ET.SubElement(root, 'folder').text = 'imgs'
    ET.SubElement(root, 'filename').text = image_name
    ET.SubElement(root, 'path').text = image_name
    source = ET.SubElement(root, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = '3'  # 假设为彩色图像
    ET.SubElement(root, 'segmented').text = '0'
    # 从 bbox_dict 添加 <object> 元素
    for instance, box in bbox_dict.items():
        label = instance[0]
        xmin, ymin, xmax, ymax = box
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = label
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(xmin))
        ET.SubElement(bndbox, 'ymin').text = str(int(ymin))
        ET.SubElement(bndbox, 'xmax').text = str(int(xmax))
        ET.SubElement(bndbox, 'ymax').text = str(int(ymax))
    # 写入 XML 文件
    xml_str = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(xml_str)
    pretty_xml_str = reparsed.toprettyxml(indent='    ')
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml_str)


# shape_to_mask
def shape_to_mask(img_shape, points, shape_type=None, line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == 'circle':
        assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == 'rectangle':
        if (
            len(xy) == 4
            and xy[0][0] == xy[3][0]
            and xy[1][0] == xy[2][0]
            and xy[0][1] == xy[1][1]
            and xy[2][1] == xy[3][1]
        ):
            xy = [tuple(xy[0]), tuple(xy[2])]
        assert len(xy) == 2, 'Shape of shape_type=rectangle must have 2 points'
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == 'line':
        assert len(xy) == 2, 'Shape of shape_type=line must have 2 points'
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'linestrip':
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'point':
        assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, 'Polygon must have points more than 2'
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


class TaskProcessor:
    # 定义不同任务允许的形状
    RULES = {
        TaskType.DETECT: [ShapeType.RECTANGLE],
        TaskType.SEGMENT: [ShapeType.POLYGON, ShapeType.CIRCLE, ShapeType.RECTANGLE, ShapeType.ROTATION],
        TaskType.POSE: [ShapeType.POLYGON, ShapeType.POINT, ShapeType.LINE, ShapeType.LINESTRIP],
        TaskType.OBB: [
            ShapeType.POLYGON,
            ShapeType.CIRCLE,
            ShapeType.RECTANGLE,
            ShapeType.ROTATION,
            ShapeType.LINESTRIP,
        ],
    }

    @staticmethod
    def circle_to_polygon(points: 'list[float]') -> 'list[float]':
        assert len(points) == 4, 'Shape of shape_type=circle must have 2 points'
        x1, y1, x2, y2 = points
        r = np.linalg.norm([x2 - x1, y2 - y1])
        # r(1-cos(a/2))<x, a=2*pi/N => N>pi/arccos(1-x/r)
        # x: tolerance of the gap between the arc and the line segment
        n_points_circle = max(int(np.pi / np.arccos(1 - 1 / r)), 12)
        i = np.arange(n_points_circle)
        x = x1 + r * np.sin(2 * np.pi / n_points_circle * i)
        y = y1 + r * np.cos(2 * np.pi / n_points_circle * i)
        return np.stack((x, y), axis=1).flatten().tolist()

    @staticmethod
    def transform(task_type: TaskType, shape_type: ShapeType, points: 'list[float]') -> 'list[float]':
        if shape_type not in TaskProcessor.RULES[task_type]:
            raise Exception(f"[Warning] Task {task_type.value} usually doesn't use {shape_type}")
        if task_type == TaskType.SEGMENT:
            if shape_type == ShapeType.RECTANGLE:
                assert len(points) == 4 or len(points) == 8, 'Shape of rectangle must have 2 or 4 points'
                if len(points) == 2:
                    x1, y1, x2, y2 = points
                    x1, x2 = sorted([x1, x2])
                    y1, y2 = sorted([y1, y2])
                    return [x1, y1, x2, y1, x2, y2, x1, y2]
            elif shape_type == ShapeType.CIRCLE:
                return TaskProcessor.circle_to_polygon(points)
            return points
        elif task_type == TaskType.OBB:
            if shape_type == ShapeType.CIRCLE:
                points = TaskProcessor.circle_to_polygon(points)
            assert False, 'Not implemented yet: obb conversion from shape points'
            # todo: 获取 points 的最小外接矩形 (obb)
            return points
        return points


# 解析单个 labelme 标注文件(json)
def parse_seg_anns_from_labelme(
    seg_path: str, img_width: int, img_height: int, task: TaskType, need_mask: bool = False
) -> 'dict[Any, Annotation]':
    try:
        if not os.path.isfile(seg_path):
            raise FileNotFoundError('file not found ...')
        # load json label file
        with open(seg_path, encoding='utf-8') as file:
            data = json.load(file)
        # check image size
        assert img_width == int(data['imageWidth']), f'图片与标签不对应: {seg_path}'
        assert img_height == int(data['imageHeight']), f'图片与标签不对应: {seg_path}'
        # process shapes info
        instances_map: dict[Any, Annotation] = {}
        for shape in data['shapes']:
            # read shape info
            label = shape['label']
            group_id = shape.get('group_id')
            shape_type = shape['shape_type']
            raw_points = np.array(shape['points']).flatten().tolist()
            instance_key = uuid1() if group_id is None else (label, group_id)
            # create or get instance
            if instance_key not in instances_map:
                instances_map[instance_key] = Annotation(label=label, type=shape_type, instance=instance_key)
            # points convert and transform according to task
            processed_points = TaskProcessor.transform(task, shape_type, raw_points)
            # add points to instance
            instances_map[instance_key].parts.append(processed_points)
            # if need_mask, convert shape to mask and add to instance mask (only for segment task)
            if need_mask and task == TaskType.SEGMENT:
                new_mask = shape_to_mask([img_height, img_width], raw_points, shape_type)
                if instances_map[instance_key].mask is None:
                    instances_map[instance_key].mask = new_mask
                else:
                    instances_map[instance_key].mask |= new_mask
        # return result
        return instances_map
    except Exception as e:
        raise Exception(f'Failed to parse annotation: {seg_path}, {e}')


def map_parent_child_annotations(parents: 'dict[Any, Annotation]', children: 'dict[Any, Annotation]'):
    # 计算 parent 和 children 之间的匹配关系, 返回一个 dict
    # key 是 parent 的 instance, value 是一个 list 包含所有匹配的 cheren instance
    mapping = {pkey: [] for pkey, _ in parents.items()}
    # 遍历每个子标注，寻找其唯一的父标注
    for ckey, cval in children.items():
        # 找到所有包含 child 的 parent
        matched_parents = []
        for pkey, pval in parents.items():
            if rectangle_include_shape(pval.bbox, cval.points, cval.type):
                matched_parents.append(pkey)
        # 约束检查：不允许一个 child 没有 parent, 或一个 child 匹配多个 parent
        if not matched_parents:
            raise ValueError(f'Child annotation (id: {ckey}, bbox: {cval.bbox}) does not belong to any parent.')
        if len(matched_parents) > 1:
            pkeys = [pkey for pkey in matched_parents]
            raise ValueError(f'Child annotation (id: {ckey}) is ambiguous: it matches multiple parents: {pkeys}')
        # 记录匹配关系
        mapping[matched_parents[0]].append(ckey)
    # 约束检查：如果有 parent 没有匹配的 child，则抛出异常
    for pkey, linked_children in mapping.items():
        if not linked_children:
            raise ValueError(f'Parent annotation (id: {pkey}) has no matching children.')
    return mapping
