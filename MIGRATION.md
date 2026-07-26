# xxtrain 基础架构迁移

## 目标

在保持现有训练与数据转换行为稳定的前提下，整理源码边界，并将项目建设为可安装、可测试的 Python 包。

迁移只调整架构和组织方式。既有问题修复、校验收紧和新增能力原则上单独处理，避免与迁移混在一起。

## 当前状态

| 阶段 | 状态 | 说明 |
| --- | --- | --- |
| 第一阶段：测试基线 | 完成 | 固定输入、语义快照、失败行为和可复现性测试已建立 |
| 第二阶段 A：data 迁移 | 完成 | 不可变标注、格式读取、几何、编码和数据集产物已进入 `xxtrain.data` |
| 第二阶段 B：pipeline 迁移 | 完成 | typed pipeline、全部 11 条任务基线、执行入口切换和小数裁剪范围回归测试均已完成 |
| 第二阶段 C：training 迁移 | 完成 | Python Scenario、训练、独立导出和预测结果检查已进入 `xxtrain.training` |
| 第三阶段：打包与 CLI | 完成 | setuptools 元数据、editable/普通安装、统一 `xxtrain` CLI 和用户级权重缓存均已建立 |

## 第一阶段：建立测试基线

- 用固定输入和期望输出记录可观察行为，不绑定内部函数名和文件位置。
- 覆盖标注解析、几何处理、11 种任务转换、数据集产物、失败行为和可复现性。
- 迁移期间测试直接使用当前公共入口，不保留新旧模块适配层。

完成标准：核心数据转换行为可以自动验证，迁移前基线全部通过。

状态：已完成。规范命令为：

```bash
python -m unittest discover -s test -t . -p 'test_*.py' -v
ruff check src test
python -m compileall -q src test
git diff --check
```

## 第二阶段：迁移源码

### 已确认的目标结构

```text
src/
└── xxtrain/
    ├── __init__.py
    ├── cli.py                      # 安装后的 train/export/review 子命令
    ├── task.py                      # 仅包含基础 TaskType
    ├── data/
    │   ├── __init__.py
    │   ├── annotation.py            # 不可变 Annotation、Shape、ImageInfo
    │   ├── geometry.py              # 几何计算与匹配
    │   ├── labels.py                # 有序 LabelCatalog
    │   ├── dataset.py               # 数据划分和数据集描述文件
    │   └── formats/
    │       ├── __init__.py
    │       ├── base.py
    │       ├── labelimg.py          # LabelImg -> Annotation
    │       ├── labelme.py           # LabelMe -> Annotation
    │       └── yolo.py              # Annotation -> YOLO 文本
    ├── pipeline/
        ├── __init__.py              # 十一个受支持的公共符号
        ├── core.py                  # typed records、Processor、Pipeline、Context
        ├── discovery.py             # 有序 Source
        ├── processors.py            # 纯转换步骤和延迟裁剪描述
        ├── recipes.py               # DatasetRecipe 与四个标准任务组合
        ├── sinks.py                 # 唯一的数据集写入边界
        └── workflow.py              # convert_dataset
    └── training/
        ├── __init__.py              # 五个受支持的公共符号
        ├── scenario.py              # TrainingScenario 与 Python Scenario 加载
        ├── model.py                 # 模型 YAML 与预训练权重准备
        ├── workflow.py              # 固定训练工作流
        ├── exporting.py             # ONNX 导出与分类参考图
        └── review.py                # 预测结果检查
```

`xxtrain.training` 已完成源码边界迁移。是否新增 `xxtrain.cli`、如何安装入口以及预训练缓存的安装后路径，留到打包阶段确定。仅供多个内部模块复用的工具函数可以进入内部模块，但不得扩大包级公共 API。

### Data 层迁移结果

- `Annotation` 是抽象基类，具体几何由 `Bbox`、`Polygon`、`Line`、`Polyline`、`Points`、`Circle` 和 `RotatedBbox` 表示。
- 标注 `id` 表示对象身份；`label` 表示类别；`group` 表示外部格式中的分组关系，三者语义不重复。
- 几何值不可变，`wrap()` 和 `translate()` 返回新对象。
- 标签目录保持声明顺序，数字形式的标签名仍按字符串处理。
- LabelImg/LabelMe 读取保持文件顺序；YOLO 编码器保持纯函数边界。
- `task.py` 只定义基础任务类型。任务名称解析、任务配方和稳定参数不进入公共基础头文件。

### 坐标契约

标注坐标仍为浮点数，没有迁移为整数：

- `Point = tuple[float, float]`；
- `Bbox.x1/y1/x2/y2` 为 `float`；
- LabelImg 和 LabelMe 坐标读取后统一转为 `float`；
- 平移、外接框、匹配、圆半径和 YOLO 归一化都按浮点数计算。

`ImageInfo.width/height` 使用统一的 `float` 内存表示。原图的整数栅格尺寸可以作为构造输入，但进入模型后立即规范化为浮点数；由浮点裁剪框产生的局部坐标范围保留小数。整数化只能发生在明确的栅格操作边界，例如 OpenCV 数组切片；Source、Processor、匹配和编码都必须保留原始浮点几何。

#### 已解决的迁移一致性问题：裁剪尺寸截断

问题代码曾使用：

```python
ImageInfo(width=int(x2 - x1), height=int(y2 - y1))
```

它会截断小数父框的局部坐标范围，改变 `point-segment`、`knob-segment` 和 `light2-detect` 的归一化结果。现已删除 `CropMatches` 和 `CropDetectionBoxes` 处理器中的提前整数化，局部宽高直接使用 `x2 - x1`、`y2 - y1`。真正的像素索引转换仍保留在 Classification Sink 的 OpenCV 切片边界。

回归测试使用带小数的父框，锁定裁剪框、局部坐标和浮点宽高不会被截断。

### Pipeline 层迁移结果

- 用不可变 `Sample`、`ImageRef` 和阶段专用的 `*Input` / `*Output` 记录替代无类型 `Payload`。
- 处理顺序为 `Source -> Pipeline -> Sink`；处理器只转换数据，Sink 负责全部文件写入。
- `ItemProcessor` 表示一对零或一，`ExpandProcessor` 表示一对多；所有输出保持输入顺序。
- 裁剪处理器只生成延迟裁剪描述，不在处理器阶段写图像。
- 标注匹配通过 UUID 关联父子对象，保持父组与组内子项顺序。
- `scale-pose` 按已确认范围跳过；`standard_recipe()` 对没有标准转换的基础类型明确失败。
- 新架构完成后一次性删除旧 `annparser.py`、`annprocessor.py`、`annconverter.py` 和测试适配层；训练工作流通过 `xxtrain.pipeline.convert_dataset` 执行 Scenario 提供的 Recipe。
- `xxtrain.pipeline` 包级受支持 API 固定为十一个符号：`Context`、`ConversionConfig`、`ConversionReport`、`DatasetRecipe`、`ExpandProcessor`、`ImageRef`、`ItemProcessor`、`Pipeline`、`Sample`、`convert_dataset`、`standard_recipe`。
- `convert_dataset()` 直接接收 `DatasetRecipe` 和数据集根目录；旧任务名注册表、`Recipe`、`build_recipe()` 和旧转换签名均已删除。

### Training 层迁移结果

- Python Scenario 是具体数据集转换与训练的组合根：它提供 `DatasetRecipe`、模型版本与规模、数据划分参数和传给 `YOLO.train()` 的覆盖参数。
- Scenario 文件父目录是数据集根目录。原始输入位于 `<scenario_dir>/src/`，数据集产物位于 `<scenario_dir>/<recipe.name>/`，ONNX 与分类参考图位于 `<scenario_dir>/weights/`；Scenario 中显式使用的相对 `Path` 也以该目录解析。
- `reserve_no_label` 在 `TrainingScenario` 和 `convert_dataset()` 的新公共路径上都默认为 `False`。
- `standard_recipe()` 只拥有 detect、segment、pose、classify 四个标准 Recipe；point、knob、light 七个特殊 Recipe 由各自的 Scenario 文件拥有，不再保留中央任务名注册表。
- 训练、独立导出和预测结果检查分别由 `xxtrain train`、`xxtrain export` 和 `xxtrain review` 调用 `xxtrain.training`。`xxtrain review` 检查预测结果，不调用 `model.val()` 重新计算验证指标。
- `xxtrain.training` 包级受支持 API 固定为：`TrainingScenario`、`load_scenario`、`train`、`export`、`review`。
- 13 个预置 Scenario 已用强制添加方式纳入 Git。`standard_classify.py` 保留默认训练参数，`direction_sensitive_classify.py` 提供标准 Pipeline 的方向敏感参考，`tuned_classify.py` 继承标准 epochs 和输入尺寸，同时保留其优化器与增强覆盖；`point_classify.py` 同样显式禁用水平/垂直翻转、旋转和自动增强。由于 `data/` 默认被忽略，新增 Scenario 仍需执行 `git add -f data/<dataset>/<scenario>.py`。
- 模型 YAML 继续复制当前 Ultralytics 模板，只修改 `nc`，pose 额外修改 `kpt_shape`；ONNX 导出失败直接向入口传播。
- 所有任务统一以 `<scenario_dir>/<recipe.name>/dataset.yaml` 作为转换完成信号。分类 Sink 在 finalize 最后生成该文件；仅存在 `train/<class>/` 或 `val/<class>/` 等部分目录时必须重新转换，不能把中断后的部分数据集误判为完成。
- 分类数据转换统一在 `ClassificationDatasetSink` 中生成 `224×224`、padding 114 的居中 Letterbox 图片；whole-image 和延迟裁剪输出使用相同几何契约。
- 分类标准训练参数使用 `imgsz=224` 和 `scale=0.0`，避免 Ultralytics 再次随机裁掉已经规范化的方图；Scenario 仍可显式覆盖默认参数。
- 旧分类生成目录不兼容该契约，允许整体删除后从同级 `src/` 重建；重建不得删除或修改原始 `src/` 数据集。

## 第三阶段：建立打包配置

- setuptools 按 `src` 布局发现并安装 `xxtrain`，项目版本为 `0.1.0`，Python 下限为 3.11。
- 运行依赖在 `pyproject.toml` 中直接声明但不固定版本；当前不维护 lock 文件或包仓库发布配置。
- 安装后统一使用 `xxtrain train`、`xxtrain export` 和 `xxtrain review`，不保留 checkout 脚本入口。
- 预训练权重位于 `platformdirs.user_cache_path('xxtrain') / 'weights'`，不再写入源码或安装目录。
- 测试在 editable install 后运行，`test/__init__.py` 不再注入 `src`。
- wheel 普通安装及仓库外 CLI 帮助命令已经过验证。

完成标准：项目可以通过标准 Python 包方式安装、测试和执行完整训练工作流。

状态：已完成。

## 已确认的迁移边界

- LabelImg、LabelMe、YOLO 的统一双向导入导出接口属于迁移后的功能补齐。当前只承接原有转换方向。
- 当前工作流仅在目标数据集的 `dataset.yaml` 已存在时跳过转换。该文件表示 Sink 已执行 finalize，但不承担源数据变化检测或完整性校验；失效判断和强制重建机制留待后续设计。
- 不保留新旧管线兼容层。开发期间允许整体功能暂时不可用，但入口切换只能在新架构承接全部既有配方后进行。
- 发现的算法缺陷和校验策略变化单独处理；迁移提交只承担结构变化与经确认的行为等价转换。

## 下一步

基础架构迁移已经完成。后续工作按 `ARCHITECTURE.md` 进入离线伪标签迭代训练设计，不属于本次迁移。
