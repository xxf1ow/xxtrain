#  Repositories : https://github.com/ultralytics/ultralytics
#     Task type : Common Object Detection
# Preprocessing : object_detection_voc2coco.py
#  Model Define : ultralytics/cfg/models/v8/yolov8.yaml
#     Reference : https://docs.ultralytics.com/zh/guides/model-yaml-config/
#                 https://docs.ultralytics.com/zh/datasets/detect/

# 数据集目录约定:
#   root_path/
#   └── src/
#       ├── dir1/
#       │   ├── imgs/       # 存放图片文件
#       │   ├── anns/       # 存放检测标注 (labelimg, xml 格式)
#       │   ├── anns_seg/   # 存放分割/旋转/骨骼标注 (json 格式)
#       ├── dir2/
#       │   ├── imgs/
#       │   ├── anns/
#       │   └── ...
#       ├── ...
#       └── labels.txt      # 存放标签列表 (可选)

# 执行完生成后的目录状态:
#   root_path/
#   ├── src/
#   │   ├── dir1/
#   │   ├── dir2/
#   │   ├── ...
#   │   └── labels.txt      # 与执行前完全相同, 没有任何增删
#   └── detect/
#       ├── dir1/
#       │   ├── xx.jpg      # 原始图片的软链接 (对应 src/dir1/xx.jpg)
#       │   └── xx.txt      # 生成的 yolo 格式标签文件
#       ├── dir2/
#       │   └── ...
#       ├── train.txt
#       ├── val.txt
#       └── dataset.yaml

import os

from annparser import find_dir, find_img
from annprocessor import (
    AnnotationsConverter,
    ClassifyAnnsGenerator,
    ClassifyAnnsGeneratorForPointTask,
    DatasetSplitter,
    DetectAndSegAnnsMatcher,
    DetectAnnsGenerator,
    DetectAnnsParser,
    DetectBboxCropIterator,
    DirectoryIterator,
    GlobalContext,
    ImageSizeParser,
    Pipeline,
    PoseAnnsGenerator,
    PoseAnnsGeneratorForPointTask,  # noqa: F401
    PoseAnnsParser,
    SegmentAnnsGenerator,
    SegmentAnnsGeneratorForKnobTask,
    SegmentAnnsGeneratorForPointTask,
    SegmentAnnsParser,
    TaskPayload,
)

# 标准检测子流水线
standard_detect_pipe = [ImageSizeParser(), DetectAnnsParser(), DetectAnnsGenerator(), DatasetSplitter()]
# 标准分割子流水线
standard_segment_pipe = [ImageSizeParser(), SegmentAnnsParser(), SegmentAnnsGenerator(), DatasetSplitter()]
# 标准姿态子流水线
standard_pose_pipe = [
    ImageSizeParser(),
    DetectAnnsParser(),
    PoseAnnsParser(),
    DetectAndSegAnnsMatcher(),
    PoseAnnsGenerator(),
    DatasetSplitter(),
]


def validate_standard_classify_input(ctx: GlobalContext) -> None:
    labels = ctx.label_list
    if not labels or any(not label for label in labels):
        raise ValueError('Classification labels must not be empty')
    if len(labels) != len(set(labels)):
        raise ValueError('Classification labels must not contain duplicates')

    work_path = ctx.get_images_path()
    class_dirs = find_dir(work_path)
    missing = sorted(set(labels) - set(class_dirs))
    extra = sorted(set(class_dirs) - set(labels))
    if missing or extra:
        raise ValueError(f'Classification labels mismatch: missing={missing}, extra={extra}')

    for class_name in class_dirs:
        imgs_path = os.path.join(work_path, class_name, 'imgs')
        if not os.path.isdir(imgs_path):
            raise ValueError(f'Classification image directory does not exist: {imgs_path}')
        images = find_img(imgs_path)
        if not images:
            raise ValueError(f'Classification class has no images: {class_name}')
        if ctx.split > 0:
            train_count = sum(index % ctx.split != 0 for index in range(len(images)))
            val_count = sum(index % ctx.split == 0 for index in range(len(images)))
            if train_count == 0 or val_count == 0:
                raise ValueError(f"Classification split leaves class '{class_name}' without train or val samples")

    ctx.label_list = sorted(labels)


def standard_process(task_type: str):
    # python3 src/train.py --task_type type --root_path data/path
    if task_type == 'detect':
        pipeline = Pipeline([DirectoryIterator(Pipeline(standard_detect_pipe))])
    elif task_type == 'segment':
        pipeline = Pipeline([DirectoryIterator(Pipeline(standard_segment_pipe))])
    elif task_type == 'pose':
        pipeline = Pipeline([DirectoryIterator(Pipeline(standard_pose_pipe))])
    elif task_type == 'classify':
        pipeline = Pipeline([DirectoryIterator(Pipeline([ClassifyAnnsGenerator()]), True)])
    else:
        raise ValueError(f'Unsupported task type: {task_type}')
    return pipeline, []


def task_point_process(task_type: str):
    # 指针仪表识别任务 = detect + segment + classify, 表盘检测 + 指针分割定位 + 位置分类(仅子项生成时使用)
    # 数据集特殊约定:
    #    1. detect 标注的任何标签都被视为同一种类别, 训练时不区分不同标签的 detect 框, 只关注框的位置和大小
    #    2. classify 使用 detect 的框类别标签, 不专门做标注
    #    3. segment 的标注为 line 类型, 且每个 detect 框内必须有至少一个 line, 分别代表指针的起点和终点
    det_labels = ['tl', 'tc', 'cl', 'cc']
    seg_labels = ['1']
    if task_type == 'point-detect':
        # python3 src/train.py --task_type point-detect --root_path data/point (约半个小时)
        label_list = ['Point']
        pipe = [
            ImageSizeParser(),
            DetectAnnsParser(det_labels),
            AnnotationsConverter(out_labels=label_list[0]),
            DetectAnnsGenerator(),
            DatasetSplitter(),
        ]
        pipeline = Pipeline([DirectoryIterator(Pipeline(pipe))])
    elif task_type == 'point-classify':
        # python3 src/train.py --task_type point-classify --root_path data/point (约十五分钟)
        label_list = det_labels
        pipe = [ImageSizeParser(), DetectAnnsParser(det_labels), ClassifyAnnsGeneratorForPointTask()]
        pipeline = Pipeline([DirectoryIterator(Pipeline(pipe), True)])
    elif task_type == 'point-segment':
        # python3 src/train.py --task_type point-segment --root_path data/point (约一个小时)
        label_list = ['Point']
        subpipe = [SegmentAnnsGeneratorForPointTask(), DatasetSplitter()]
        pipe = [
            ImageSizeParser(),
            DetectAnnsParser(det_labels),
            SegmentAnnsParser(seg_labels),
            DetectAndSegAnnsMatcher(strict=False),
            DetectBboxCropIterator(Pipeline(subpipe)),
        ]
        pipeline = Pipeline([DirectoryIterator(Pipeline(pipe), True)])
    else:
        raise ValueError(f'Unsupported task type: {task_type}')
    return pipeline, label_list


def task_knob_process(task_type: str):
    # 旋钮仪表识别任务 = detect + segment + classify, 表盘检测 + 旋钮分割判断角度 + 方向分类
    # 数据集特殊约定:
    #    classify 是将整理好了的, 全部指向上的旋钮裁剪图片, 生成其它三种 flip 方向的图片, 然后做分类
    labels = ['switch']
    if task_type == 'knob-detect':
        # python3 src/train.py --task_type knob-detect --root_path data/knob (约四十五分钟)
        label_list = labels
        pipe = standard_detect_pipe
        pipeline = Pipeline([DirectoryIterator(Pipeline(pipe))])
    elif task_type == 'knob-segment':
        # python3 src/train.py --task_type knob-segment --root_path data/knob (约两个半小时)
        label_list = labels
        subpipe = [SegmentAnnsGeneratorForKnobTask(), DatasetSplitter()]
        pipe = [
            ImageSizeParser(),
            DetectAnnsParser(labels),
            SegmentAnnsParser(labels),
            DetectAndSegAnnsMatcher(0.15, False),  # 这个数据集标注质量较差, 不严格要求 detect 和 segment 的匹配
            DetectBboxCropIterator(Pipeline(subpipe)),
        ]
        pipeline = Pipeline([DirectoryIterator(Pipeline(pipe), True)])
    # elif task_type == 'knob-classify':
    #     # python3 src/train.py --task_type knob-classify --root_path data/knob
    #     label_list = ['up', 'down']
    #     pipe = [ImageSizeParser(), DetectAnnsParser(labels), ClassifyAnnsGeneratorForSwitchTask()]
    else:
        raise ValueError(f'Unsupported task type: {task_type}')
    return pipeline, label_list


def task_scale_process(task_type: str):
    # 刻度仪表识别任务 = pose, 表盘检测 + 关键点得到刻度长度 + 刻度位置
    # 数据集特殊约定: pose 标注有些是传统的五点标注, 有些是旋转框 + 一点标注, 需要将旋转框转为四个点
    det_labels = ['Scale']
    seg_labels = ['beg_tl', 'beg_br', 'end_tl', 'end_br', 'point']
    if task_type.endswith('pose'):
        # python3 src/train.py --task_type scale-pose --root_path data/scale
        label_list = seg_labels
        pipe = [
            ImageSizeParser(),
            DetectAnnsParser(det_labels),
            PoseAnnsParser(seg_labels),
            DetectAndSegAnnsMatcher(),
            PoseAnnsGenerator(),
            DatasetSplitter(),
        ]
    else:
        raise ValueError(f'Unsupported task type: {task_type}')
    pipeline = Pipeline([DirectoryIterator(Pipeline(pipe))])
    return pipeline, label_list


def task_light_process(task_type: str):
    # 小灯识别任务 = detect + classify
    # 数据集特殊约定:
    #    1. detect 标注的任何标签都被视为同一种类别, 训练时不区分不同标签的 detect 框, 只关注框的位置和大小
    #    2. classify 使用 detect 的框类别标签, 不专门做标注
    if task_type == 'light1-detect':
        # python3 src/train.py --task_type light1-detect --root_path data/light2 (约二十分钟)
        label_list = ['1008']
        pipe = [ImageSizeParser(), DetectAnnsParser(label_list, False), DetectAnnsGenerator(), DatasetSplitter()]
        pipeline = Pipeline([DirectoryIterator(Pipeline(pipe))])
    elif task_type == 'light2-detect':
        # python3 src/train.py --task_type light2-detect --root_path data/light2 (约二十分钟)
        label_list = ['0']
        subpipe = [AnnotationsConverter('seg_anns', 'det_anns', [], '0'), DetectAnnsGenerator(), DatasetSplitter()]
        pipe = [
            ImageSizeParser(),
            DetectAnnsParser(['1008'], False),
            DetectAnnsParser(['0', '1', '2'], False, 'seg_anns'),
            DetectAndSegAnnsMatcher(0.1, strict=False),
            DetectBboxCropIterator(Pipeline(subpipe)),
        ]
        pipeline = Pipeline([DirectoryIterator(Pipeline(pipe), True)])
    else:
        raise ValueError(f'Unsupported task type: {task_type}')
    return pipeline, label_list


def process(task_type: str, root_path: str, split: int, reserve_no_label: bool):
    if task_type.find('-') == -1:
        pipeline, label_list = standard_process(task_type)
    elif task_type.startswith('point'):
        pipeline, label_list = task_point_process(task_type)
    elif task_type.startswith('knob'):
        pipeline, label_list = task_knob_process(task_type)
    elif task_type.startswith('scale'):
        pipeline, label_list = task_scale_process(task_type)
    elif task_type.startswith('light'):
        pipeline, label_list = task_light_process(task_type)
    else:
        raise ValueError(f'Unsupported task type: {task_type}')

    # 初始化上下文并启动流水线
    ctx = GlobalContext(task_type, root_path, split, label_list, reserve_no_label)
    if task_type == 'classify':
        validate_standard_classify_input(ctx)
    pipeline.process(ctx, TaskPayload())
    ctx.dataset_finalize()
    ctx.print_summary()
