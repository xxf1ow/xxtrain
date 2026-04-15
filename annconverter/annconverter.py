#  Repositories : https://github.com/ultralytics/ultralytics
#     Task type : Common Object Detection
# Preprocessing : object_detection_voc2coco.py
#  Model Define : ultralytics/cfg/models/v8/yolov8.yaml
#     Reference : https://docs.ultralytics.com/zh/guides/model-yaml-config/
#                 https://docs.ultralytics.com/zh/datasets/detect/

# 数据集目录约定:
#   root_path/
#   ├── images/
#   │   ├── dir1/
#   │   │   ├── imgs/       # 存放图片文件
#   │   │   ├── anns/       # 存放检测标注 (labelimg, xml 格式)
#   │   │   ├── anns_seg/   # 存放分割标注 (json 格式, 实例分割任务可选)
#   │   │   ├── anns_obb/   # 存放旋转标注 (json 格式, 旋转目标检测可选)
#   │   │   └── anns_pose/  # 存放骨骼标注 (json 格式, 骨骼关键点检测可选)
#   │   ├── dir2/
#   │   │   ├── imgs/
#   │   │   ├── anns/
#   │   │   └── ...
#   │   └── ...
#   ├── labels.txt
#   ├── train.txt
#   ├── val.txt
#   └── dataset.yaml


import os

import PIL.Image
from tqdm import tqdm

from .annparser import (
    find_dir,
    find_img,
    parse_det_anns_from_labelimg,
    parse_pose_anns_from_labelme,
    parse_seg_anns_from_labelme,
    rectangle_include_shape,
)


class GlobalContext:
    def __init__(self, task_type: str, root_path: str, split: int, reserve_no_label: bool):
        self.task_type = task_type
        self.root_path = root_path
        self.split = split
        self.reserve_no_label = reserve_no_label
        self.label_list = []
        self.sub_label_list = []

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
        label_path = self.get_labels_list_path()
        assert os.path.isfile(label_path), f'标签列表不存在: {label_path}'
        with open(label_path, encoding='utf-8') as f:
            self.label_list = [line.strip() for line in f.readlines()]
        assert len(self.label_list) > 0, f'标签列表为空: {label_path}'
        # 读取子标签列表 (如果存在)
        sub_label_path = self.get_sub_labels_list_path()
        if os.path.isfile(sub_label_path):
            with open(sub_label_path, encoding='utf-8') as f:
                self.sub_label_list = [line.strip() for line in f.readlines()]
            assert len(self.sub_label_list) > 0, f'标签列表为空: {sub_label_path}'

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


class DirectoryIteratorProcessor(BaseProcessor):
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
                self.set(sub_payload, 'in_obb_path', f'{work_path}/{dir_name}/anns_obb/{raw_name}.json')
                self.set(sub_payload, 'in_pose_path', f'{work_path}/{dir_name}/anns_pose/{raw_name}.json')
                self.set(sub_payload, 'output_path', f'{save_path}/{dir_name}/imgs/{raw_name}')
                self.sub_pipeline.process(ctx, sub_payload)  # 执行单图处理流水线


class ReadImageMetaProcessor(BaseProcessor):
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


class ParseDetectAnnsProcessor(BaseProcessor):
    def required_inputs(self) -> list:
        return ['in_det_path', 'img_size']

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        # outputs: det_anns
        in_det_path = payload.get('in_det_path')
        width, height = payload.get('img_size')
        self.set(payload, 'det_anns', parse_det_anns_from_labelimg(in_det_path, width, height))


class ParseSegmentAnnsProcessor(BaseProcessor):
    def required_inputs(self) -> list:
        return ['in_seg_path', 'img_size']

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        # outputs: seg_anns
        in_seg_path = payload.get('in_seg_path')
        width, height = payload.get('img_size')
        self.set(payload, 'seg_anns', parse_seg_anns_from_labelme(in_seg_path, width, height))


class ParsePoseAnnsProcessor(BaseProcessor):
    def required_inputs(self) -> list:
        return ['in_pose_path', 'img_size']

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        # outputs: pose_anns
        in_pose_path = payload.get('in_pose_path')
        width, height = payload.get('img_size')
        self.set(payload, 'pose_anns', parse_pose_anns_from_labelme(in_pose_path, width, height))


class GenerateDetectYoloAnnsProcessor(BaseProcessor):
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
        for instance, box in det_anns.items():
            label = instance[0]
            if label not in ctx.label_list:
                ctx.skip_files.add(in_img_path)
                ctx.skip_label_list.add(label)
                continue
            # box: [xmin, ymin, xmax, ymax]
            label_id = ctx.label_list.index(label)
            x_center = (box[0] + box[2]) / 2.0 / width
            y_center = (box[1] + box[3]) / 2.0 / height
            w_norm = (box[2] - box[0]) / width
            h_norm = (box[3] - box[1]) / height
            boxes.append(f'{label_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}')
        self.set(payload, 'ann_count', len(boxes))
        self.set(payload, 'out_img_path', in_img_path)
        with open(out_txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(boxes))


class GeneratePoseYoloAnnsProcessor(BaseProcessor):
    """将 det_anns & pose_anns 转为 YOLO 格式并保存 Txt"""

    def required_inputs(self) -> list:
        return ['in_img_path', 'output_path', 'img_size', 'det_anns', 'pose_anns']

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        assert len(ctx.sub_label_list) > 0, '子标签列表不能为空, 请提供 sub_labels.txt'
        # outputs: ann_count, out_img_path
        in_img_path = payload.get('in_img_path')
        width, height = payload.get('img_size')
        det_anns = payload.get('det_anns')
        pose_anns = payload.get('pose_anns')
        out_txt_path = f'{payload.get("output_path")}.txt'
        # remove pose anns whose labels are not in sub_label_list, and mark the file to be skipped if any
        pose_remove = []
        for instance, _ in pose_anns.items():
            if instance[0] not in ctx.sub_label_list:
                pose_remove.append(instance)
                ctx.skip_files.add(in_img_path)
                ctx.skip_label_list.add(instance[0])
        for instance in pose_remove:
            del pose_anns[instance]
        # generate anns
        boxes = []
        for instance, box in det_anns.items():
            label = instance[0]
            if label not in ctx.label_list:
                ctx.skip_files.add(in_img_path)
                ctx.skip_label_list.add(label)
                continue
            # 处理框内的形状 (同类别会覆盖, 宽松版本, 容忍点在框外一点的情况)
            points = {
                instance[0]: shape[0]  # 为了支持 group id, shape 是个 list, 这里直接取 [0] 即可
                for instance, shape in pose_anns.items()
                if instance[1] == 'point' and rectangle_include_shape(box, shape[0], instance[1])
            }
            if len(points) != len(ctx.sub_label_list):
                print('\n\n', box, '\n', points, '\n', pose_anns, '\n')
                print(f'\n\033[1;31m[Error] 标注点数错误: {in_img_path}\033[0m\n')
                continue
            # 组成一个框的标签
            label_id = ctx.label_list.index(label)
            x_center = (box[0] + box[2]) / 2.0 / width
            y_center = (box[1] + box[3]) / 2.0 / height
            w_norm = (box[2] - box[0]) / width
            h_norm = (box[3] - box[1]) / height
            result = f'{label_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}'
            # 将这些点按类别排序后, 添加到框的标签后面
            for cls in ctx.sub_label_list:
                px = points[cls][0] / width if cls in points else 0
                py = points[cls][1] / height if cls in points else 0
                visibility = 2 if cls in points else 0  # 2-可见不遮挡 1-遮挡 0-没有点 (这里简化为只有可见和没有)
                result += f' {px:.6f} {py:.6f} {visibility}'
            boxes.append(result)
        self.set(payload, 'ann_count', len(boxes))
        self.set(payload, 'out_img_path', in_img_path)
        with open(out_txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(boxes))


class CropImageByDetectByPoseAnnsIteratorProcessor(BaseProcessor):
    """根据 det_anns 裁剪出目标图像, 保存并执行单图处理流水线. 子流程使用新的 Payload 包裹"""

    def __init__(self, sub_pipeline: Pipeline):
        self.sub_pipeline = sub_pipeline

    def required_inputs(self) -> list:
        return ['in_img_path', 'output_path', 'current_dir', 'current_idx', 'det_anns', 'pose_anns']

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        in_img_path = payload.get('in_img_path')
        det_anns = payload.get('det_anns')
        pose_anns = payload.get('pose_anns')
        output_path = payload.get('output_path')
        current_dir = payload.get('current_dir')
        current_idx = payload.get('current_idx')
        # crop images and generate anns
        for idx, (instance, box) in enumerate(det_anns.items()):
            label = instance[0]
            if label not in ctx.label_list:
                ctx.skip_files.add(in_img_path)
                ctx.skip_label_list.add(label)
                continue
            sub_output_path = f'{output_path}_{idx}'
            sub_in_img_path = f'{output_path}_{idx}.jpg'
            sub_pose_anns = {  # 处理框内的形状 (同类别会覆盖, 宽松版本, 容忍点在框外一点的情况)
                instance: shape[0]  # 为了支持 group id, shape 是个 list, 这里直接取 [0] 即可
                for instance, shape in pose_anns.items()
                if rectangle_include_shape(box, shape[0], instance[1])
            }
            # 将坐标转换为相对于裁剪后图像的坐标
            for instance, points in sub_pose_anns.items():
                for i in range(len(points) // 2):
                    points[2 * i] -= box[0]  # x - xmin
                    points[2 * i + 1] -= box[1]  # y - ymin
            # crop image and save
            img = PIL.Image.open(in_img_path)
            cimg = img.crop(box)
            cimg.save(sub_in_img_path)
            # create new payload for sub pipeline
            sub_payload = TaskPayload()  # 每张图片流转前,创建全新的包裹,杜绝脏数据累积
            self.set(sub_payload, 'current_dir', current_dir)
            self.set(sub_payload, 'current_idx', current_idx)
            self.set(sub_payload, 'img_size', (box[2] - box[0], box[3] - box[1]))
            self.set(sub_payload, 'in_img_path', sub_in_img_path)
            self.set(sub_payload, 'output_path', sub_output_path)
            self.set(sub_payload, 'pose_anns', sub_pose_anns)
            self.set(sub_payload, 'parent_label', label)
            self.sub_pipeline.process(ctx, sub_payload)  # 执行单图处理流水线


class GenerateLinePoseYoloAnnsByCropProcessor(BaseProcessor):
    """根据 pose_anns line 标注生成 YOLO 格式的标注并保存 Txt, 其中检测标注由 line 标注生成"""

    def required_inputs(self) -> list:
        return ['in_img_path', 'output_path', 'img_size', 'pose_anns', 'parent_label']

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        assert len(ctx.sub_label_list) == 2, '子标签列表, 应当有且仅有两个标签, 分别代表线的起点和终点'
        # outputs: ann_count, out_img_path
        in_img_path = payload.get('in_img_path')
        width, height = payload.get('img_size')
        pose_anns = payload.get('pose_anns')
        out_txt_path = f'{payload.get("output_path")}.txt'
        parent_label_id = ctx.label_list.index(payload.get('parent_label'))
        boxes = []
        for instance, line in pose_anns.items():
            shape_type = instance[1]
            if shape_type != 'line' or len(line) != 4:
                raise ValueError(f'标注类型错误: {in_img_path} 中的 {instance} 不是 line 类型')
            # box: [xmin, ymin, xmax, ymax]
            xmin = min(line[0], line[2])
            xmax = max(line[0], line[2])
            ymin = min(line[1], line[3])
            ymax = max(line[1], line[3])
            x = (xmin + xmax) / 2.0 / width
            y = (ymin + ymax) / 2.0 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            # 如果父标签是第一个类别, 则 p[0] 是起点, p[1] 是终点
            x0 = line[0] / width
            y0 = line[1] / height
            x1 = line[2] / width
            y1 = line[3] / height
            if parent_label_id != 0:
                x0, y0, x1, y1 = x1, y1, x0, y0
            boxes.append(f'0 {x:.6f} {y:.6f} {w:.6f} {h:.6f} {x0:.6f} {y0:.6f} 2 {x1:.6f} {y1:.6f} 2')
        self.set(payload, 'ann_count', len(boxes))
        self.set(payload, 'out_img_path', in_img_path)
        with open(out_txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(boxes))


class DatasetSplitProcessor(BaseProcessor):
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


class FinalizeDatasetProcessor(BaseProcessor):
    """[后置算子]收尾工作:写出 yaml 配置文件和汇总报告"""

    def process(self, ctx: GlobalContext, payload: TaskPayload):
        # 写入 txt 和 yaml
        with open(ctx.get_train_list_path(), 'w', encoding='utf-8') as f:
            f.write('\n'.join(ctx.train_list))
        with open(ctx.get_val_list_path(), 'w', encoding='utf-8') as f:
            f.write('\n'.join(ctx.val_list))

        # detect
        content = f'path: {ctx.root_path}\ntrain: train.txt\nval: val.txt\n\nnames:\n'
        for i, name in enumerate(ctx.label_list):
            content += f'  {i}: {name}\n'
        # pose
        if ctx.task_type == 'pose':
            content += f'\nkpt_shape: [{len(ctx.sub_label_list)}, 3]\nkpt_names:\n  0:\n'
            for i, name in enumerate(ctx.sub_label_list):
                content += f'    - {name}\n'
        # 写入 dataset.yaml
        with open(ctx.get_dataset_yaml_path(), 'w', encoding='utf-8') as f:
            f.write(content)


def process(task_type: str, root_path: str, split: int, reserve_no_label: bool):
    if task_type not in ['detect', 'pose']:
        raise ValueError(f'Unsupported task type: {task_type}')

    # 单步 det 标签, 训练 detect 模型
    det_pipeline = Pipeline(
        [
            ReadImageMetaProcessor(),  # 1. 读图片获取宽高
            ParseDetectAnnsProcessor(),  # 2. 解析 XML
            GenerateDetectYoloAnnsProcessor(),  # 3. 转换坐标并保存 TXT
            DatasetSplitProcessor(),  # 4. 统计并决定划入Train还是Val
        ]
    )

    # 两步 det + pose 标签, 训练 pose 模型
    pose_pipeline = Pipeline(
        [
            ReadImageMetaProcessor(),  # 1. 读图片获取宽高
            ParseDetectAnnsProcessor(),  # 2. 解析 XML
            ParsePoseAnnsProcessor(),  # 3. 解析 Pose JSON
            GeneratePoseYoloAnnsProcessor(),  # 4. 转换坐标并保存 TXT
            DatasetSplitProcessor(),  # 5. 统计并决定划入Train还是Val
        ]
    )

    # 两步 det + pose 标签, 在 det 裁剪图上训练 pose 模型, 其中 pose 的 bbox 由关键点标注生成
    det_pose_sub_pipeline = Pipeline(
        [
            GenerateLinePoseYoloAnnsByCropProcessor(),  # 4. 转换坐标并保存 TXT
            DatasetSplitProcessor(),  # 5. 统计并决定划入Train还是Val
        ]
    )
    det_pose_pipeline = Pipeline(
        [
            ReadImageMetaProcessor(),  # 1. 读图片获取宽高
            ParseDetectAnnsProcessor(),  # 2. 解析 XML
            ParsePoseAnnsProcessor(),  # 3. 解析 Pose JSON
            CropImageByDetectByPoseAnnsIteratorProcessor(det_pose_sub_pipeline),
        ]
    )

    # 主流水线
    sub_pipeline = pose_pipeline if task_type == 'pose' else det_pipeline
    main_pipeline = Pipeline(
        [
            DirectoryIteratorProcessor(sub_pipeline),  # 2. 遍历图片,丢进子流水线
            FinalizeDatasetProcessor(),  # 3. 收尾生成 yaml 文件
        ]
    )

    # 赋予上下文,启动
    ctx = GlobalContext(task_type, root_path, split, reserve_no_label)
    main_pipeline.process(ctx, TaskPayload())
    ctx.print_summary()
