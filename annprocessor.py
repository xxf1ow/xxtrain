import os

import numpy as np
import PIL.Image
from tqdm import tqdm

from annparser import (
    ShapeType,
    TaskType,
    find_dir,
    find_img,
    map_parent_child_annotations,
    parse_det_anns_from_labelimg,
    parse_seg_anns_from_labelme,
)


class GlobalContext:
    def __init__(
        self, task_type: str, root_path: str, split: int, label_list: 'list[str]' = [], reserve_no_label: bool = True
    ):
        self.task_type = task_type
        self.root_path = root_path
        self.split = split
        self.label_list = label_list
        self.reserve_no_label = reserve_no_label

        # ---------------------- 全局统计与状态 ----------------------
        self.train_list = []
        self.val_list = []
        self.images_count = [0, 0]  # [train, val]
        self.labels_count = [0, 0]  # [train, val] todo: 统计每个类别的标签数量
        self.skip_label_list = set()
        self.skip_files = set()
        self.not_ann_cnt_dict = {}  # 记录每个目录无标注图片的数量

        # ---------------------- 初始化流程 ----------------------
        # 数据集目录检查
        work_path = self.get_images_path()
        assert os.path.isdir(work_path), f'数据集不存在: {work_path}'
        # 读取标签列表
        if len(label_list) == 0:
            label_path = self.get_labels_list_path()
            assert os.path.isfile(label_path), f'标签列表不存在: {label_path}'
            with open(label_path, encoding='utf-8') as f:
                self.label_list = [line.strip() for line in f.readlines()]
            assert len(self.label_list) > 0, f'标签列表为空: {label_path}'

    def get_images_path(self):
        return os.path.join(self.root_path, 'images')

    def get_labels_list_path(self):
        return os.path.join(self.get_images_path(), 'labels.txt')

    def get_sub_labels_list_path(self):
        return os.path.join(self.get_images_path(), 'sub_labels.txt')

    def get_labels_path(self):
        return os.path.join(self.root_path, 'labels')

    def get_cropped_images_path(self):
        return os.path.join(self.get_labels_path(), 'cropped_images')

    def get_train_list_path(self):
        return os.path.join(self.root_path, 'train.txt')

    def get_val_list_path(self):
        return os.path.join(self.root_path, 'val.txt')

    def get_dataset_yaml_path(self):
        return os.path.join(self.root_path, 'dataset.yaml')

    def dataset_finalize(self):
        # 写入 txt 和 yaml
        with open(self.get_train_list_path(), 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.train_list))
        with open(self.get_val_list_path(), 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.val_list))
        # detect
        content = f'path: {self.root_path}\ntrain: train.txt\nval: val.txt\n\nnames:\n'
        for i, name in enumerate(self.label_list):
            content += f'  {i}: {name}\n'
        # pose
        if self.task_type.endswith('pose'):
            content += f'\nkpt_shape: [{len(self.label_list)}, 3]\nkpt_names:\n  0:\n'
            for i, name in enumerate(self.label_list):
                content += f'    - {name}\n'
        # 写入 dataset.yaml
        with open(self.get_dataset_yaml_path(), 'w', encoding='utf-8') as f:
            f.write(content)

    def print_summary(self):
        print('\n\033[1;32m[Convert Summary]\033[0m')
        print(f'训练集图片总数: {self.images_count[0]}, 标注总数: {self.labels_count[0]}')
        print(f'验证集图片总数: {self.images_count[1]}, 标注总数: {self.labels_count[1]}')
        print(f'类别列表: {self.label_list}\n')
        if self.not_ann_cnt_dict:
            print('\033[1;31m[Warning] 以下目录包含没有标注的图片\033[0m')
            for dir_name, not_ann_cnt in self.not_ann_cnt_dict.items():
                print(f'  - {dir_name}: {not_ann_cnt}张图片')
        if self.skip_label_list:
            print('\033[1;33m[Warning] 以下类别在标签列表中未定义\033[0m')
            for cat in self.skip_label_list:
                print(f'  - {cat}')
        if self.skip_files:
            print('\033[1;33m[Warning] 以下图片因包含未定义类别而被跳过:\033[0m')
            for file in sorted(self.skip_files):
                print(f'  - {file}')


class TaskPayload:
    """单张图片流转的载荷"""

    def __init__(self):
        self._data = {}
        self._trace = {}  # 核心:记录变量是由哪个算子产生的!

    def set(self, key: str, value, source_processor: str):
        if key in self._data:  # 打印覆盖警告
            print(
                f"\033[1;33m[Warning]\033[0m Key '{key}' overwritten by [{source_processor}]. "
                f'Previous setter: [{self._trace[key]}]'
            )
        self._data[key] = value
        self._trace[key] = source_processor

    def get(self, key: str):
        if key not in self._data:
            raise KeyError(f"Variable '{key}' is missing. No processor has provided it yet.")
        return self._data[key]

    def has(self, key: str) -> bool:
        return key in self._data


class BaseProcessor:
    def set(self, payload: TaskPayload, key: str, value):
        """向 Payload 中写入数据的包装方法, 自动记录来源算子"""
        payload.set(key, value, self.__class__.__name__)

    def required_inputs(self) -> list:
        """声明该算子需要从 Payload 中读取哪些键"""
        return []

    def __call__(self, ctx: GlobalContext, payload: TaskPayload):
        """校验自己需要的输入是否存在, 然后执行处理逻辑"""
        for key in self.required_inputs():
            if not payload.has(key):
                raise ValueError(f"[{self.__class__.__name__}] Failed: Missing required input '{key}' in payload.")
        self.process(ctx, payload)

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        raise NotImplementedError


class Pipeline(BaseProcessor):
    """流水线引擎(组合模式):本身也是一个处理器,可以嵌套"""

    def __init__(self, processors: 'list[BaseProcessor]'):
        self.processors = processors

    def required_inputs(self) -> list:
        return self.processors[0].required_inputs() if self.processors else []

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        for processor in self.processors:
            processor.process(ctx, payload)


class DirectoryIterator(BaseProcessor):
    """遍历所有子目录和图片, 并执行单图处理流水线. 子流程使用新的 Payload 包裹"""

    def __init__(self, sub_pipeline: Pipeline):
        self.sub_pipeline = sub_pipeline

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        work_path = ctx.get_images_path()
        save_path = ctx.get_labels_path()
        os.makedirs(save_path, exist_ok=True)
        for dir_name in find_dir(work_path):
            imgs_dir_path = os.path.join(work_path, dir_name, 'imgs')
            if not os.path.isdir(imgs_dir_path):
                continue
            img_list = find_img(imgs_dir_path)
            os.makedirs(os.path.join(save_path, dir_name, 'imgs'), exist_ok=True)
            for num, file in enumerate(tqdm(img_list, desc=f'{dir_name}\t', leave=True, ncols=100, colour='CYAN')):
                sub_payload = TaskPayload()  # 每张图片流转前,创建全新的包裹,杜绝脏数据累积
                raw_name, extension = os.path.splitext(file)
                self.set(sub_payload, 'current_dir', dir_name)
                self.set(sub_payload, 'current_idx', num)
                self.set(sub_payload, 'in_img_path', f'{work_path}/{dir_name}/imgs/{raw_name}{extension}')
                self.set(sub_payload, 'in_det_path', f'{work_path}/{dir_name}/anns/{raw_name}.xml')
                self.set(sub_payload, 'in_seg_path', f'{work_path}/{dir_name}/anns_seg/{raw_name}.json')
                self.set(sub_payload, 'output_path', f'{save_path}/{dir_name}/imgs/{raw_name}')
                self.sub_pipeline.process(ctx, sub_payload)  # 执行单图处理流水线


class ImageSizeParser(BaseProcessor):
    """读取图片宽高信息"""

    def required_inputs(self) -> list:
        return ['in_img_path']

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        # outputs: img_size
        in_img_path = payload.get('in_img_path')
        assert os.path.isfile(in_img_path), f'图片文件不存在: {in_img_path}'
        img = PIL.Image.open(in_img_path)
        width, height = img.size
        assert width > 0 and height > 0
        self.set(payload, 'img_size', (width, height))


class DetectAnnsParser(BaseProcessor):
    def __init__(self, label_list: 'list[str]' = []):
        self.label_list = label_list

    def required_inputs(self) -> list:
        return ['in_img_path', 'in_det_path', 'img_size']

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        # outputs: det_anns
        in_det_path = payload.get('in_det_path')
        width, height = payload.get('img_size')
        det_anns = parse_det_anns_from_labelimg(in_det_path, width, height)
        # remove anns whose labels are not in label_list, and mark the file to be skipped
        labelset = set(ctx.label_list) if len(self.label_list) == 0 else set(self.label_list)
        invalid_instance = [key for key, val in det_anns.items() if val.label not in labelset]
        if invalid_instance:
            ctx.skip_files.add(payload.get('in_img_path'))
            for key in invalid_instance:
                ctx.skip_label_list.add(key)
                del det_anns[key]
        # set to payload
        self.set(payload, 'det_anns', det_anns)


class SegmentAnnsParser(BaseProcessor):
    def __init__(self, label_list: 'list[str]' = []):
        self.label_list = label_list

    def required_inputs(self) -> list:
        return ['in_img_path', 'in_seg_path', 'img_size']

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        # outputs: seg_anns
        in_seg_path = payload.get('in_seg_path')
        width, height = payload.get('img_size')
        seg_anns = parse_seg_anns_from_labelme(in_seg_path, width, height, TaskType.SEGMENT)
        # remove anns whose labels are not in label_list, and mark the file to be skipped
        labelset = set(ctx.label_list) if len(self.label_list) == 0 else set(self.label_list)
        invalid_instance = [key for key, val in seg_anns.items() if val.label not in labelset]
        if invalid_instance:
            ctx.skip_files.add(payload.get('in_img_path'))
            for key in invalid_instance:
                ctx.skip_label_list.add(key)
                del seg_anns[key]
        # set to payload
        self.set(payload, 'seg_anns', seg_anns)


class PoseAnnsParser(BaseProcessor):
    def __init__(self, label_list: 'list[str]' = []):
        self.label_list = label_list

    def required_inputs(self) -> list:
        return ['in_img_path', 'in_pose_path', 'img_size']

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        # outputs: pose_anns
        in_pose_path = payload.get('in_pose_path')
        width, height = payload.get('img_size')
        seg_anns = parse_seg_anns_from_labelme(in_pose_path, width, height, TaskType.POSE)
        # remove anns whose labels are not in label_list, and mark the file to be skipped
        labelset = set(ctx.label_list) if len(self.label_list) == 0 else set(self.label_list)
        invalid_instance = [key for key, val in seg_anns.items() if val.label not in labelset]
        if invalid_instance:
            ctx.skip_files.add(payload.get('in_img_path'))
            for key in invalid_instance:
                ctx.skip_label_list.add(key)
                del seg_anns[key]
        # set to payload
        self.set(payload, 'seg_anns', seg_anns)


class DetectAnnsGenerator(BaseProcessor):
    """将 det_anns 转为 YOLO 格式并保存 Txt"""

    def required_inputs(self) -> list:
        return ['in_img_path', 'output_path', 'img_size', 'det_anns']

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        # outputs: ann_count, out_img_path
        in_img_path = payload.get('in_img_path')
        width, height = payload.get('img_size')
        det_anns = payload.get('det_anns')
        out_txt_path = f'{payload.get("output_path")}.txt'
        boxes = []
        for key, val in det_anns.items():
            # val.bbox: [xmin, ymin, xmax, ymax]
            label_id = ctx.label_list.index(val.label)
            x_center = (val.bbox[0] + val.bbox[2]) / 2.0 / width
            y_center = (val.bbox[1] + val.bbox[3]) / 2.0 / height
            w_norm = (val.bbox[2] - val.bbox[0]) / width
            h_norm = (val.bbox[3] - val.bbox[1]) / height
            # detect 标注格式: 类别ID + det框信息
            # <class-index> <x_center> <y_center> <width> <height>
            boxes.append(f'{label_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}')
        self.set(payload, 'ann_count', len(boxes))
        self.set(payload, 'out_img_path', in_img_path)
        with open(out_txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(boxes))


class DetectAndSegAnnsMatcher(BaseProcessor):
    def required_inputs(self) -> list:
        return ['det_anns', 'seg_anns']

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        det_anns = payload.get('det_anns')
        seg_anns = payload.get('seg_anns')
        matched_map = map_parent_child_annotations(det_anns, seg_anns)
        self.set(payload, 'matched_map', matched_map)


class PoseAnnsGenerator(BaseProcessor):
    """将 det_anns & seg_anns 转为 YOLO 格式并保存 Txt"""

    def required_inputs(self) -> list:
        return ['in_img_path', 'output_path', 'img_size', 'det_anns', 'seg_anns', 'matched_map']

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        # outputs: ann_count, out_img_path
        in_img_path = payload.get('in_img_path')
        width, height = payload.get('img_size')
        det_anns = payload.get('det_anns')
        seg_anns = payload.get('seg_anns')
        matched_map = payload.get('matched_map')
        out_txt_path = f'{payload.get("output_path")}.txt'
        boxes = []
        for dkey, skeys in matched_map.items():
            # 得到骨骼点坐标列表, 骨骼点必须是点类型, 且数量必须与标签列表一致
            points = dict()
            for skey in skeys:
                sval = seg_anns[skey]
                if sval.type != ShapeType.POINT or sval.points.shape[0] != 1:
                    raise ValueError(f'骨骼标注必须是点类型: {in_img_path}, {sval}')
                points[sval.label] = sval.points  # 骨骼点坐标, {label: [x, y]}
            if len(points) != len(ctx.label_list):
                raise ValueError(f'标注点数与标签数量不匹配: {in_img_path}, points: {points}, labels: {ctx.label_list}')
            # 计算 det 框的中心点坐标和宽高, 并归一化到 [0, 1]
            bbox = det_anns[dkey].bbox
            label_id = 0  # pose 任务只有一个类别, 可以直接设置为 0
            x_center = (bbox[0] + bbox[2]) / 2.0 / width
            y_center = (bbox[1] + bbox[3]) / 2.0 / height
            w_norm = (bbox[2] - bbox[0]) / width
            h_norm = (bbox[3] - bbox[1]) / height
            # pose 标注格式: 类别ID + det框信息 + 关键点坐标 (带可见性 2-可见不遮挡 1-遮挡 0-没有点)
            # <class-index> <x> <y> <width> <height> <px1> <py1> <p1-visibility> ... <pxn> <pyn> <pn-visibility>
            assert 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= w_norm <= 1 and 0 <= h_norm <= 1
            result = f'{label_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}'
            for cls in ctx.label_list:
                if cls not in points:
                    px = py = visibility = 0
                else:
                    px = points[cls][0][0] / width
                    py = points[cls][0][1] / height
                    visibility = 2
                assert 0 <= px <= 1 and 0 <= py <= 1
                result += f' {px:.6f} {py:.6f} {visibility}'
            boxes.append(result)
        self.set(payload, 'ann_count', len(boxes))
        self.set(payload, 'out_img_path', in_img_path)
        with open(out_txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(boxes))


class SegmentAnnsGenerator(BaseProcessor):
    """将 det_anns & seg_anns 转为 YOLO 格式并保存 Txt"""

    def required_inputs(self) -> list:
        return ['in_img_path', 'output_path', 'img_size', 'seg_anns']

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        # outputs: ann_count, out_img_path
        in_img_path = payload.get('in_img_path')
        width, height = payload.get('img_size')
        seg_anns = payload.get('seg_anns')
        out_txt_path = f'{payload.get("output_path")}.txt'
        boxes = []
        for key, val in seg_anns.items():
            if val.type != ShapeType.POLYGON or val.points.shape[0] < 3:
                raise ValueError(f'分割标注必须是至少三点的多边形: {in_img_path}, {val}')
            # seg 标注格式: 类别ID + mask shape 每个点的坐标, 不需要 det 框信息, 训练时自动取 bounding box
            # <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
            label_id = ctx.label_list.index(val.label)
            result = f'{label_id}'
            for pt in val.points:
                norm_x = pt[0] / width
                norm_y = pt[1] / height
                assert 0 <= norm_x <= 1 and 0 <= norm_y <= 1
                result += f' {norm_x:.6f} {norm_y:.6f}'
            boxes.append(result)
        self.set(payload, 'ann_count', len(boxes))
        self.set(payload, 'out_img_path', in_img_path)
        with open(out_txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(boxes))


class DatasetSplitter(BaseProcessor):
    """[处理算子]负责依据规则将数据划分到训练集或验证集"""

    def required_inputs(self) -> list:
        return ['ann_count', 'current_dir', 'current_idx', 'out_img_path']

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        # outputs: None, but updates ctx 中的统计信息和划分结果
        ann_count = payload.get('ann_count')
        if ann_count == 0:
            current_dir = payload.get('current_dir')
            ctx.not_ann_cnt_dict[current_dir] = ctx.not_ann_cnt_dict.get(current_dir, 0) + 1
            if not ctx.reserve_no_label:
                return
        out_img_path = payload.get('out_img_path')
        current_idx = payload.get('current_idx')
        if ctx.split <= 0 or current_idx % ctx.split != 0:  # 训练集
            ctx.images_count[0] += 1
            ctx.labels_count[0] += ann_count
            ctx.train_list.append(out_img_path)
        if ctx.split <= 0 or current_idx % ctx.split == 0:  # 验证集
            ctx.images_count[1] += 1
            ctx.labels_count[1] += ann_count
            ctx.val_list.append(out_img_path)


class DirectoryIteratorForPointTask(BaseProcessor):
    """根据 det_anns 裁剪出目标图像, 保存并执行单图处理流水线. 子流程使用新的 Payload 包裹"""

    def __init__(self, sub_pipeline: Pipeline):
        self.sub_pipeline = sub_pipeline

    def required_inputs(self) -> list:
        return ['in_img_path', 'output_path', 'current_dir', 'current_idx', 'det_anns', 'seg_anns', 'matched_map']

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        in_img_path = payload.get('in_img_path')
        det_anns = payload.get('det_anns')
        seg_anns = payload.get('seg_anns')
        matched_map = payload.get('matched_map')
        output_path = payload.get('output_path')
        current_dir = payload.get('current_dir')
        current_idx = payload.get('current_idx')
        # crop images and generate anns
        for idx, (dkey, skeys) in enumerate(matched_map.items()):
            sub_output_path = f'{output_path}_{idx}'
            sub_in_img_path = f'{output_path}_{idx}.jpg'
            #
            parent = det_anns[dkey]
            bbox = parent.bbox
            label = parent.label
            # crop image and save
            img = PIL.Image.open(in_img_path)
            cimg = img.crop(bbox)
            if cimg.mode != 'RGB':
                cimg = cimg.convert('RGB')
            cimg.save(sub_in_img_path)
            #
            sub_seg_anns = dict()
            for skey in skeys:  # 将坐标转换为相对于裁剪后图像的坐标
                sub_seg_anns[skey] = seg_anns[skey].translate(-bbox[0], -bbox[1])
            # create new payload for sub pipeline
            sub_payload = TaskPayload()  # 每张图片流转前,创建全新的包裹,杜绝脏数据累积
            self.set(sub_payload, 'current_dir', current_dir)
            self.set(sub_payload, 'current_idx', current_idx)
            self.set(sub_payload, 'img_size', (bbox[2] - bbox[0], bbox[3] - bbox[1]))
            self.set(sub_payload, 'in_img_path', sub_in_img_path)
            self.set(sub_payload, 'output_path', sub_output_path)
            self.set(sub_payload, 'seg_anns', sub_seg_anns)
            self.set(sub_payload, 'parent_label', label)
            self.sub_pipeline.process(ctx, sub_payload)  # 执行单图处理流水线


class ClassifyAnnsGeneratorForPointTask(BaseProcessor):
    """根据 det_anns 裁剪出目标图像, 保存并生成分类标注 (不支持复合类别)"""

    def required_inputs(self) -> list:
        return ['in_img_path', 'current_dir', 'current_idx', 'det_anns']

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        in_img_path = payload.get('in_img_path')
        det_anns = payload.get('det_anns')
        current_dir = payload.get('current_dir')
        current_idx = payload.get('current_idx')
        save_path = ctx.get_labels_path()
        for idx, (_, val) in enumerate(det_anns.items()):
            label = val.label
            out_img_path = ''
            if ctx.split <= 0 or current_idx % ctx.split != 0:  # 训练集
                out_img_path = f'{save_path}/train/{label}/{current_dir}_{current_idx}_{idx}.jpg'
                ctx.images_count[0] += 1
                ctx.labels_count[0] += 1
                ctx.train_list.append(out_img_path)
            if ctx.split <= 0 or current_idx % ctx.split == 0:  # 验证集
                out_img_path = f'{save_path}/test/{label}/{current_dir}_{current_idx}_{idx}.jpg'
                ctx.images_count[1] += 1
                ctx.labels_count[1] += 1
                ctx.val_list.append(out_img_path)
            assert out_img_path != '', '输出路径不能为空'
            # create output directory if not exists
            out_dirname = os.path.dirname(out_img_path)
            if not os.path.exists(out_dirname):
                os.makedirs(out_dirname, exist_ok=True)
            # crop image and save
            img = PIL.Image.open(in_img_path)
            cimg = img.crop(val.bbox)
            if cimg.mode != 'RGB':
                cimg = cimg.convert('RGB')
            cimg.save(out_img_path)


class DetectAnnsConverterForPointTask(BaseProcessor):
    """将 det_anns 标签全部转为同一个类别, 以适配 PointTask 任务"""

    def __init__(self, target_label: str):
        self.target_label = target_label

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        det_anns = payload.get('det_anns')
        for key, val in det_anns.items():
            val.label = self.target_label  # 将所有检测标签转换为同一个类别


class PoseAnnsGeneratorForPointTask(BaseProcessor):
    """从 line 标注生成 det + pose 标注"""

    def required_inputs(self) -> list:
        return ['in_img_path', 'output_path', 'img_size', 'seg_anns', 'parent_label']

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        assert len(ctx.label_list) == 2, '子标签列表, 应当有且仅有两个标签, 分别代表线的起点和终点'
        # outputs: ann_count, out_img_path
        in_img_path = payload.get('in_img_path')
        width, height = payload.get('img_size')
        seg_anns = payload.get('seg_anns')
        out_txt_path = f'{payload.get("output_path")}.txt'
        parent_label_id = ctx.label_list.index(payload.get('parent_label'))
        boxes = []
        for key, val in seg_anns.items():
            if val.type != ShapeType.LINE or val.points.shape[0] != 2:
                raise ValueError(f'标注类型错误: {in_img_path} 中的 {val} 不是 line 类型')
            # box: [xmin, ymin, xmax, ymax]
            x = (val.bbox[0] + val.bbox[2]) / 2.0 / width
            y = (val.bbox[1] + val.bbox[3]) / 2.0 / height
            w = (val.bbox[2] - val.bbox[0]) / width
            h = (val.bbox[3] - val.bbox[1]) / height
            # 如果父标签是第一个类别, 则 p[0] 是起点, p[1] 是终点
            x0 = val.points[0][0] / width
            y0 = val.points[0][1] / height
            x1 = val.points[1][0] / width
            y1 = val.points[1][1] / height
            if parent_label_id != 0:
                x0, y0, x1, y1 = x1, y1, x0, y0
            # pose 标注格式: 类别ID + det框信息 + 关键点坐标 (带可见性 2-可见不遮挡 1-遮挡 0-没有点)
            # <class-index> <x> <y> <width> <height> <px1> <py1> <p1-visibility> ... <pxn> <pyn> <pn-visibility>
            assert 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1
            assert 0 <= x0 <= 1 and 0 <= y0 <= 1 and 0 <= x1 <= 1 and 0 <= y1 <= 1
            boxes.append(f'{parent_label_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {x0:.6f} {y0:.6f} 2 {x1:.6f} {y1:.6f} 2')
        self.set(payload, 'ann_count', len(boxes))
        self.set(payload, 'out_img_path', in_img_path)
        with open(out_txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(boxes))


class SegmentAnnsGeneratorForPointTask(BaseProcessor):
    """从 line 标注生成 det + seg 标注"""

    def required_inputs(self) -> list:
        return ['in_img_path', 'output_path', 'img_size', 'seg_anns', 'parent_label']

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        # outputs: ann_count, out_img_path
        in_img_path = payload.get('in_img_path')
        width, height = payload.get('img_size')
        seg_anns = payload.get('seg_anns')
        out_txt_path = f'{payload.get("output_path")}.txt'
        parent_label_id = 0  # PointTask 任务只有一个类别, 直接设置为 0
        line_thickness = 6.0  # 定义线段加粗的宽度(像素单位)
        boxes = []
        for key, val in seg_anns.items():
            if val.type != ShapeType.LINE or val.points.shape[0] != 2:
                raise ValueError(f'标注类型错误: {in_img_path} 中的 {val} 不是 line 类型')
            # line to roatated box
            p1 = val.points[0]  # 起点
            p2 = val.points[1]  # 终点
            vec = p2 - p1  # 线段方向向量
            length = np.linalg.norm(vec)  # 线段长度
            unit_vec = np.array([1, 0]) if length == 0 else vec / length  # 线段单位方向向量
            normal_vec = np.array([-unit_vec[1], unit_vec[0]])  # 线段的法向量 (垂直于线段的方向)
            half_t = line_thickness / 2.0  # 线段加粗后, 法向量的长度为半个线宽
            # 将线段表示为一个矩形的四个顶点坐标, 这里直接生成一个三角形(起点的两个顶点 + 终点), 以保持尖头效果
            corners = [
                p1 + half_t * normal_vec,  # 左上
                p1 - half_t * normal_vec,  # 右上
                # p2 - half_t * normal_vec,  # 右下
                # p2 + half_t * normal_vec,  # 左下
                p2,  # 线段终点 (不加法向量, 保持尖头效果)
            ]
            # seg 标注格式: 类别ID + mask shape 每个点的坐标, 不需要 det 框信息, 训练时自动取 bounding box
            # <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
            result = f'{parent_label_id}'
            for pt in corners:
                norm_x = pt[0] / width
                norm_y = pt[1] / height
                assert 0 <= norm_x <= 1 and 0 <= norm_y <= 1
                result += f' {norm_x:.6f} {norm_y:.6f}'
            boxes.append(result)
        self.set(payload, 'ann_count', len(boxes))
        self.set(payload, 'out_img_path', in_img_path)
        with open(out_txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(boxes))
