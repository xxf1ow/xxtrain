# Training 迁移设计

## 目标

将 `src/train.py` 中的数据集转换编排、模型配置生成、预训练权重准备、训练、导出和预测结果检查迁入 `xxtrain.training`，并用与具体数据集同目录的 Python Scenario 文件描述真实会变化的训练场景。

本阶段保持 Ultralytics 为唯一训练后端，不建立通用实验配置框架。标准行为继续由源码提供，Scenario 只覆盖现有数据集 Recipe、模型版本与规模、数据划分参数和 `YOLO.train()` 参数。

## 已确认的原则

- 启动完整训练时只传入一个 Python Scenario 文件。
- Scenario 文件父目录就是数据集根目录。
- 原始数据固定放在 `<scenario_dir>/src/`。
- 数据集产物固定写入 `<scenario_dir>/<recipe.name>/`。
- ONNX 和分类参考图固定写入 `<scenario_dir>/weights/`。
- Scenario 内显式出现的相对 `Path` 以 Scenario 文件目录为基准解析。
- `reserve_no_label` 的新默认值统一为 `False`。
- 数据集是否强制重建不是训练场景参数；本阶段继续在产物存在时跳过转换。
- 模型 YAML 继续从当前安装的 Ultralytics 包复制，只修改 `nc` 和 pose 的 `kpt_shape`。
- Python Scenario 仅供内部使用，不考虑执行任意代码的安全风险。
- 不引入 YAML、TOML、Hydra、MMEngine、配置继承、注册表或命令行参数覆盖。

## 入口

训练、导出和预测结果检查拆成三个入口：

```bash
python src/train.py data/standard-detect/standard_detect.py

python src/export.py data/standard-detect/standard_detect.py \
    --weights path/to/best.pt

python src/review.py data/standard-detect/standard_detect.py \
    --weights path/to/best.pt \
    --directory path/to/images
```

`train.py` 只要求 Scenario 文件参数。`export.py` 和 `review.py` 的 checkpoint、待检查目录属于本次运行输入，继续由命令行提供，不固化进 Scenario。

`review.py` 保留当前 `standard_validate()` 和 `classify_validate()` 的实际语义：

- detect、segment、pose 和 obb 保存预测可视化结果；
- classify 根据图片父目录得到期望类别，整理错分图片并生成报告。

它不是 `model.val()` 指标评估入口。Ultralytics 默认在训练过程中验证并将指标、曲线和样例写入 run 目录；单独重新计算指标不属于本阶段。

## 目标源码结构

```text
src/
├── train.py
├── export.py
├── review.py
└── xxtrain/
    └── training/
        ├── __init__.py
        ├── scenario.py
        ├── model.py
        ├── workflow.py
        ├── exporting.py
        └── review.py
```

职责如下：

- `src/train.py`：解析 Scenario 路径并调用完整训练工作流；
- `src/export.py`：解析 Scenario、weights 并调用独立导出；
- `src/review.py`：解析 Scenario、weights、directory 并调用预测结果检查；
- `scenario.py`：Scenario 类型、Python 文件加载、相对 `Path` 解析和校验；
- `model.py`：模型名和模板名计算、Ultralytics 模板复制、`nc/kpt_shape` 修改及预训练权重准备；
- `workflow.py`：固定的转换、建模、训练和训练后导出顺序；
- `exporting.py`：ONNX 导出和分类参考图复制；
- `review.py`：标准任务预测结果保存和分类错分整理。

`xxtrain.training` 的包级受支持接口限制为：

```python
TrainingScenario
load_scenario
train
export
review
```

模型模板处理、预训练下载、分类错分整理和其他工作流细节保持内部实现。

## 最小配置模型

第一版只新增两个类型：

```python
from collections.abc import Mapping
from dataclasses import dataclass, field

from xxtrain.data import LabelCatalog
from xxtrain.pipeline import Pipeline
from xxtrain.pipeline.sinks import DatasetSink
from xxtrain.task import TaskType


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetRecipe:
    name: str
    task_type: TaskType
    labels: LabelCatalog | None
    pipeline: Pipeline
    sink: DatasetSink


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingScenario:
    dataset: DatasetRecipe
    model_version: str = 'v8'
    model_scale: str = 'n'
    split: int = 10
    reserve_no_label: bool = False
    train_args: Mapping[str, object] = field(default_factory=dict)
```

具体语义：

- `DatasetRecipe.name` 同时是数据集产物目录名，例如 `detect`、`point-classify`；
- `DatasetRecipe.task_type` 只使用基础 `TaskType`，不再从 Recipe 名称后缀推断；
- `labels=None` 表示运行时读取 `<scenario_dir>/src/labels.txt`；
- 特殊 Recipe 使用固定 `LabelCatalog`；
- Source 固定为当前 `DirectorySource`，因为现有 11 条 Recipe 没有 Source 差异；
- `Pipeline` 和 Sink 由 Recipe 明确提供；
- `train_args` 是对当前 xxtrain 标准训练参数的覆盖，合并后原样传给 `YOLO.train()`；
- `train_args` 内显式使用的 `Path` 会在 dict、list 和 tuple 中递归解析，普通字符串保持原样。

类名使用大驼峰，函数和变量使用下划线命名，并通过当前 Ruff 配置检查。

## 标准训练参数

源码保留两组当前 xxtrain 标准参数：

```python
STANDARD_TRAIN_ARGS = {
    TaskType.CLASSIFY: {
        'epochs': 72,
        'batch': 128,
        'imgsz': 224,
    },
    'non_classification': {
        'epochs': 100,
        'batch': 32,
        'imgsz': 640,
    },
}
```

执行训练时先选择基础任务对应的标准参数，再用 `scenario.train_args` 覆盖：

```python
train_args = standard_train_args(scenario.dataset.task_type) | dict(scenario.train_args)
```

第一组 11 个预置 Scenario 的 `train_args` 均为空，只在 Dataset Recipe 上存在差异。当前 `point-classify` 和 `knob` 相关分类分支中的翻转、旋转、自动增强覆盖不进入新标准配置；这是已确认的有意调整。

`digit-cls` 是唯一预置的非标准训练参数参考：

```python
train_args={
    'epochs': 80,
    'batch': 16,
    'imgsz': 320,
    'patience': 15,
    'optimizer': 'AdamW',
    'lr0': 0.0005,
    'lrf': 0.05,
    'weight_decay': 0.001,
    'warmup_epochs': 3.0,
    'cos_lr': True,
    'dropout': 0.15,
    'fliplr': 0.0,
    'flipud': 0.0,
    'auto_augment': None,
    'erasing': 0.0,
}
```

本阶段不迁移保留分支中的自定义 Classification Trainer 和 torchvision transforms。`digit-cls` 只演示如何覆盖 `YOLO.train()` 参数。

分类与检测的参数覆盖机制相同。检测特有的 `box`、`dfl`、`mosaic`、`mixup` 等参数可以直接加入 `train_args`，但没有经过真实实验的非标准检测配置不作为预置参考。

## Dataset Recipe

`xxtrain.pipeline.recipes` 只保留 detect、segment、pose、classify 四个标准 Recipe。新增：

```python
def standard_recipe(task_type: TaskType) -> DatasetRecipe:
    ...
```

`standard_recipe()` 对不受支持的 `TaskType` 明确失败。标准 obb 转换当前不存在，不在本阶段补齐。

point、knob、light 的七个特殊 Recipe 从 `xxtrain.pipeline.recipes` 移到对应 Scenario 文件。Scenario 可以从 `xxtrain.pipeline.processors` 和 `xxtrain.pipeline.sinks` 导入当前构件；本阶段只承诺预置 Scenario 与同版本源码共同工作，不把所有具体 Processor 承诺为永久第三方插件 API。

`convert_dataset()` 改为接收 `DatasetRecipe`：

```python
def convert_dataset(
    recipe: DatasetRecipe,
    root_path: str | Path,
    *,
    split: int = 10,
    reserve_no_label: bool = False,
) -> ConversionReport:
    ...
```

它使用固定的 `DirectorySource`，按 Recipe 和运行参数创建 `Context`，再执行 `Source -> Pipeline -> Sink`。不保留旧的任务名称注册表或新旧签名兼容层。

`xxtrain.pipeline` 新增两个包级受支持符号：

- `DatasetRecipe`：类，使用大驼峰；
- `standard_recipe`：函数，使用下划线。

## 预置 Scenario

随迁移使用 `git add -f` 提交以下 12 个文件：

```text
data/
├── standard-detect/standard_detect.py
├── standard-segment/standard_segment.py
├── standard-pose/standard_pose.py
├── standard-classify/standard_classify.py
├── point/
│   ├── point_detect.py
│   ├── point_classify.py
│   └── point_segment.py
├── knob/
│   ├── knob_detect.py
│   └── knob_segment.py
├── light/
│   ├── light1_detect.py
│   └── light2_detect.py
└── digit-cls/digit_cls.py
```

这些文件可以在没有真实 `data/` 内容的源码副本中独立存在。加载 Scenario 不读取数据文件；只有执行转换或训练时才验证数据输入。

前 11 个 Scenario 使用当前 xxtrain 标准训练参数，仅 Dataset Recipe 不同。`digit_cls.py` 使用标准 classify Recipe 和上节列出的覆盖参数，用于与 `standard_classify.py` 对比。

不增加 `digit-cls-6`，因为它与 `digit-cls` 是相同场景、不同数据集，不提供新的配置方式。

新增 Scenario 时必须显式执行：

```bash
git add -f data/<dataset>/<scenario>.py
```

`.gitignore` 继续忽略其余 `data/` 内容。

## 模型与权重行为

模型处理保持当前路径：

1. 根据 `model_version`、`model_scale` 和基础 `TaskType` 计算 Ultralytics 模板名与模型名；
2. 从 `ultralytics/cfg/models/<version>/` 复制对应模板；
3. 从生成后的 `dataset.yaml` 读取类别数；
4. 只修改模板中的 `nc`；
5. pose 额外复制 `kpt_shape`；
6. 将模型 YAML 写到 `<scenario_dir>/<recipe.name>/<model_name>.yaml`；
7. 使用 `<model_name>.pt` 触发 Ultralytics 下载；
8. 将预训练权重继续缓存到当前 `src/.weights/`。

预训练缓存路径留到打包阶段重新设计，本阶段不因源码移动而改变。

训练数据参数保持当前行为：

- classify 使用 `<scenario_dir>/<recipe.name>/`；
- 其他任务使用 `<scenario_dir>/<recipe.name>/dataset.yaml`。

ONNX 继续导出到 `<scenario_dir>/weights/<model_name>_<timestamp>.onnx`。分类导出继续在相邻 `<onnx_stem>_references/` 目录复制每类一张参考图。

## 固定训练工作流

`train()` 的顺序固定为：

```text
load_scenario()
→ 检查数据集产物
→ convert_dataset()
→ 复制并修改 Ultralytics 模型模板
→ 准备预训练权重
→ YOLO.train()
→ best.pt 存在时重新加载
→ 导出 ONNX
→ classify 时复制参考图
```

数据集产物存在时继续跳过转换。强制重建和源文件失效检测不进入 Scenario，也不在本阶段实现。

找不到 `best.pt` 时保留当前回退行为：继续使用训练后的当前模型对象导出。

训练完成后，将原始 Scenario 文件复制到 Ultralytics trainer 的 run 目录。Ultralytics 已保存 `args.yaml`、`results.csv`、指标图和 checkpoint，本阶段不重复建立环境快照系统。

## 错误处理

Scenario 加载在写文件前检查：

- 配置文件存在；
- 模块导出 `SCENARIO`；
- `SCENARIO` 是 `TrainingScenario`；
- Recipe 的 Pipeline 和 Sink 边界兼容；
- model scale 属于 `n/s/m/l/x`；
- 标准 Recipe 支持对应基础任务。

迁移中确认修复两个既有问题：

1. 分类预测检查使用函数参数 `best_model.task`，不再引用入口模块的全局 `model`；
2. ONNX 导出不再捕获所有异常后返回 `None`，失败应传播到 `train.py` 或 `export.py` 并产生非零退出码。

第二项使用独立提交，和纯源码移动分开。

其他错误行为保持当前语义，不在迁移中顺带收紧。

## 测试策略

不运行真实训练、不下载真实权重、不要求 `data/` 下存在完整数据集。使用临时目录、现有 fixtures 和 mock 覆盖：

- Scenario 文件不存在、缺少 `SCENARIO`、类型错误；
- Scenario 文件父目录成为数据集根目录；
- `Path` 参数递归按 Scenario 目录解析；
- `reserve_no_label` 默认是 `False`；
- 标准训练参数按基础任务选择；
- Scenario 参数覆盖标准训练参数并完整传入 `YOLO.train()`；
- `digit_cls.py` 与 `standard_classify.py` 使用相同 Recipe，但训练覆盖参数不同；
- 模型模板只修改 `nc/kpt_shape`；
- 预训练权重存在时不下载，不存在时调用 Ultralytics 并保存；
- 完整训练工作流调用顺序；
- `train.py` 只要求 Scenario 参数；
- `export.py` 的失败产生非零退出；
- `review.py` 使用传入模型，不依赖全局变量；
- 训练完成后复制 Scenario 到 run 目录；
- 12 个预置 Scenario 都能加载；
- 前 11 个 Scenario 分别用 `test/fixtures/` 的对应数据冒烟转换；
- `digit_cls.py` 复制到 `standard-classify` fixture 的临时副本后进行冒烟转换；
- 七条特殊 Recipe 移出源码后，原有 11 条转换快照保持不变；
- 完整 unittest、Ruff、compileall 和 `git diff --check` 通过。

## 文档更新

- `MIGRATION.md` 单独增加“第二阶段 C：training 迁移”，修复 training 被写进打包阶段的章节不一致；
- `ARCHITECTURE.md` 记录 Scenario 是具体数据集转换与训练的组合根；
- `CLAUDE.md` 更新训练、导出、预测结果检查命令；
- 文档明确 `reserve_no_label=False`；
- 文档明确 `review.py` 不是 `model.val()`；
- 文档明确 `data/` 下 Scenario 使用 `git add -f`。

## 非目标

- 不建立通用多后端训练接口；
- 不新增真正的 `model.val()` 指标评估入口；
- 不实现数据集强制重建或源变化检测；
- 不保留旧 `src/train.py` 参数兼容层；
- 不支持任意第三方 Scenario 插件；
- 不迁移自定义 Classification Trainer 或 torchvision transforms；
- 不增加 `digit-cls-6` 或未经实验的非标准检测配置；
- 不改变 Ultralytics 模板复制和最小修改行为；
- 不处理最终包安装入口和 `xxtrain` CLI；
- 不重新设计预训练缓存目录。

## 完成标准

- 三个入口分别承担训练、导出和预测结果检查；
- 完整训练只需要一个 Scenario 文件参数；
- 标准与特殊 Dataset Recipe 均由 Scenario 驱动；
- 中央 Recipe 源码只保留四条标准 Pipeline；
- 12 个预置 Scenario 强制纳入 Git，且不依赖真实 `data/` 内容完成测试；
- 当前 11 条转换快照和标准训练参数得到自动化保护；
- 已确认的两个既有 bug 分别修复并有回归测试；
- `MIGRATION.md`、`ARCHITECTURE.md` 和 `CLAUDE.md` 与实现一致；
- 全部规范验证命令通过。
