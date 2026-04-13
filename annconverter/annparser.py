import collections
import json
import math
import os
import uuid
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
import PIL.Image
import PIL.ImageDraw


# 判断点 point 是否在矩形 rect 内部. rect: [xmin, ymin, xmax, ymax]
def rectangle_include_point(r: 'list[float]', p: 'list[float]') -> bool:
    return p[0] >= r[0] and p[0] <= r[2] and p[1] >= r[1] and p[1] <= r[3]


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
def get_xml_value(root: 'ET.Element[str]', name: str) -> float:
    XmlValue = root.findall(name)
    if XmlValue is None:
        raise Exception(f'Cannot find {name} in XML file.')
    if len(XmlValue) != 1:
        raise Exception(f'The size of {name} is supposed to be 1, but is {len(XmlValue)}.')
    return 0 if XmlValue[0].text is None else float(XmlValue[0].text)


# 取出 xml 列表 (length 预期长度, 为 0 则不检查)
def get_xml_list(root: 'ET.Element[str]', name: str, length: int) -> 'list[ET.Element[str]]':
    XmlValue = root.findall(name)
    if XmlValue is None:
        raise Exception(f'Cannot find {name} in XML file.')
    if length > 0 and len(XmlValue) != length:
        raise Exception(f'The size of {name} is supposed to be {length}, but is {len(XmlValue)}.')
    return XmlValue


# 解析单个 labelimg 标注文件(xml)
def parse_labelimg(det_path: str, img_width: int, img_height: int, overlap_check: bool = True) -> dict:
    if not os.path.isfile(det_path):
        return {}
    try:
        tree = ET.parse(det_path)
        root = tree.getroot()
        # check image size
        imgsize = get_xml_list(root, 'size', 1)[0]
        assert img_width == int(get_xml_value(imgsize, 'width')), f'图片与标签不对应: {det_path}'
        assert img_height == int(get_xml_value(imgsize, 'height')), f'图片与标签不对应: {det_path}'
        # parse box info
        bbox = {}
        for obj in get_xml_list(root, 'object', 0):
            name = get_xml_list(obj, 'name', 1)[0].text
            bndbox = get_xml_list(obj, 'bndbox', 1)[0]
            xmin = round(get_xml_value(bndbox, 'xmin'))
            ymin = round(get_xml_value(bndbox, 'ymin'))
            xmax = round(get_xml_value(bndbox, 'xmax'))
            ymax = round(get_xml_value(bndbox, 'ymax'))
            assert xmax > xmin and ymax > ymin and xmax <= img_width and ymax <= img_height, f'{det_path}'
            bbox[(name, uuid.uuid1())] = [xmin, ymin, xmax, ymax]
        if overlap_check:
            sorted_bboxes = calculate_nms(bbox)
            if len(bbox) != len(sorted_bboxes):
                print(f'\n labelimg 标注出现重叠框: {det_path}\n')
    except Exception as e:
        raise Exception(f'Failed to parse XML file: {det_path}, {e}')
    return bbox


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


# 解析单个 labelme 标注文件(json)
def parse_labelme(
    seg_path,
    img_width,
    img_height,
    allow_shape_type=['circle', 'rectangle', 'line', 'linestrip', 'point', 'polygon', 'rotation'],
    need_shape_type=False,
):
    if not os.path.isfile(seg_path):
        return {}, {}
    # load json label file
    with open(seg_path, encoding='utf-8') as file:
        data = json.load(file)
    # check image size
    assert img_width == int(data['imageWidth']), f'图片与标签不对应: {seg_path}'
    assert img_height == int(data['imageHeight']), f'图片与标签不对应: {seg_path}'
    # parse shapes info
    masks = {}
    shapes = collections.defaultdict(list)  # 如果你访问一个不存在的键, defaultdict 会自动为这个键创建一个默认值
    for shape in data['shapes']:
        # check shape type (rotation == polygon)
        shape_type = shape['shape_type']
        if shape_type not in allow_shape_type:
            raise Exception(f'Unsupported shape types: {shape_type}, check: {seg_path}')
        # get instance (唯一实例 flag 值)
        label = shape['label']
        group_id = uuid.uuid1() if shape['group_id'] is None else shape['group_id']
        instance = (label, group_id, shape_type) if need_shape_type else (label, group_id)
        # generate mask (如果存在同一 group_id 的 mask , 就合并它们)
        points = shape['points']
        mask = shape_to_mask([img_height, img_width], points, shape_type)
        masks[instance] = masks[instance] | mask if instance in masks else mask
        # points convert
        if shape_type == 'rectangle':  # 矩形将两个对角点转换为四个顶点
            assert len(points) == 2 or len(points) == 4, f'{seg_path}: Shape of rectangle must have 2 or 4 points'
            if len(points) == 2:
                (x1, y1), (x2, y2) = points
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                points = [x1, y1, x2, y1, x2, y2, x1, y2]
            elif len(points) == 4:
                assert (
                    points[0][0] == points[3][0]
                    and points[1][0] == points[2][0]
                    and points[0][1] == points[1][1]
                    and points[2][1] == points[3][1]
                ), f'{seg_path}: Shape of shape_type=rectangle is invalid box'
        elif shape_type == 'circle':  # 圆形根据圆心和半径，生成一个多边形的点坐标。
            (x1, y1), (x2, y2) = points
            r = np.linalg.norm([x2 - x1, y2 - y1])
            # r(1-cos(a/2))<x, a=2*pi/N => N>pi/arccos(1-x/r)
            # x: tolerance of the gap between the arc and the line segment
            n_points_circle = max(int(np.pi / np.arccos(1 - 1 / r)), 12)
            i = np.arange(n_points_circle)
            x = x1 + r * np.sin(2 * np.pi / n_points_circle * i)
            y = y1 + r * np.cos(2 * np.pi / n_points_circle * i)
            points = np.stack((x, y), axis=1).flatten()
        else:
            points = np.asarray(points).flatten().tolist()
        # points round to int
        shapes[instance].append(points)
    # shapes convert to normal dict
    shapes = dict(shapes)

    return masks, shapes


def query_labelme_flags(seg_path, flag):
    with open(seg_path, encoding='utf-8') as file:
        data = json.load(file)
    flags = data.get('flags', {})
    return flags.get(flag, False)


def set_labelme_flags(seg_path, flag):
    with open(seg_path, encoding='utf-8') as file:
        data = json.load(file)
    data.setdefault('flags', {})
    data['flags'][flag] = True
    with open(seg_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


def get_matching_pairs(seg_path, bbox, shapes, check_no_rotation=False):
    # [Warning] no rotation
    if check_no_rotation:
        for _, xy in shapes.items():
            if (
                len(xy[0]) == 8
                and xy[0][0] == xy[0][6]
                and xy[0][2] == xy[0][4]
                and xy[0][1] == xy[0][3]
                and xy[0][5] == xy[0][7]
                and not query_labelme_flags(seg_path, 'Ignoring_no_rotation_warning')
            ):
                print(f'\n\033[1;33m[Warning] no rotation: {seg_path}\033[0m')
                print('\nEnter [Y/N] to choose to keep/discard the annotations for this file: ')
                user_input = input().lower()
                if user_input != 'y':
                    return {}
                set_labelme_flags(seg_path, 'Ignoring_no_rotation_warning')
    # get_matching_pairs
    pairs = {}
    selected_shapes = set()
    centers = {instance: np.asarray(shape).reshape(-1, 2).mean(axis=0) for instance, shape in shapes.items()}
    for box_instance, box in bbox.items():
        matching_shapes = []
        for shape_instance, _ in shapes.items():
            if shape_instance in selected_shapes or not rectangle_include_point(box, centers[shape_instance]):
                continue
            selected_shapes.add(shape_instance)
            matching_shapes.append(shape_instance)
        if len(matching_shapes) > 0:
            pairs[box_instance] = matching_shapes
    # [Error] matching pairs
    if (len(bbox) != len(pairs) or len(shapes) != len(selected_shapes)) and not query_labelme_flags(
        seg_path, 'Ignoring_matching_errors'
    ):
        print(
            f'\n\033[1;31m[Error] matching pairs: {seg_path}\nlen(bbox): {len(bbox)}, len(pairs): {len(pairs)}, '
            f'len(shapes): {len(shapes)}, len(selected_shapes): {len(selected_shapes)}\033[0m'
        )
        for box_instance, box in bbox.items():
            if box_instance not in pairs:
                print(f'box: {box}')
        for shape_instance, shape in shapes.items():
            if shape_instance not in selected_shapes:
                print(f'shape: {centers[shape_instance]}, {np.rint(shape).astype(int).tolist()}')
        print('\nEnter [Y/N] to choose to keep/discard the annotations for this file: ')
        user_input = input().lower()
        if user_input != 'y':
            return {}
        set_labelme_flags(seg_path, 'Ignoring_matching_errors')
    return pairs
