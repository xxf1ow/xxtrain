# xxtrain 基础架构迁移

## 目标

在保持现有训练与数据转换行为稳定的前提下，整理源码边界，并将项目建设为可安装、可测试的 Python 包。

迁移只调整架构和组织方式。既有问题修复、校验收紧和新增能力原则上单独处理，避免与迁移混在一起。

## 当前状态

| 阶段 | 状态 | 说明 |
| --- | --- | --- |
| 第一阶段：测试基线 | 完成 | 固定输入、语义快照、失败行为和可复现性测试已建立 |
| 第二阶段 A：data 迁移 | 完成 | 不可变标注、格式读取、几何、编码和数据集产物已进入 `xxtrain.data` |
| 第二阶段 B：pipeline 迁移 | 实现完成，验收未关闭 | typed pipeline、全部 11 条任务基线和执行入口切换已完成；小数父框的裁剪尺寸仍有一项迁移一致性问题 |
| 第二阶段 C：training 迁移 | 未开始 | 训练、验证、导出仍在 `src/train.py` |
| 第三阶段：打包与 CLI | 未开始 | 尚未建立完整项目元数据、安装入口和 `xxtrain` CLI |

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
├── train.py                         # 当前训练入口；待 training/CLI 阶段迁移
└── xxtrain/
    ├── __init__.py
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
    └── pipeline/
        ├── __init__.py              # 九个稳定公共符号
        ├── core.py                  # typed records、Processor、Pipeline、Context
        ├── discovery.py             # 有序 Source
        ├── processors.py            # 纯转换步骤和延迟裁剪描述
        ├── recipes.py               # 标准与定制任务组合
        ├── sinks.py                 # 唯一的数据集写入边界
        └── workflow.py              # convert_dataset
```

后续 training 迁移仍计划引入 `xxtrain.training`；是否新增 `xxtrain.cli` 在打包阶段确定。仅供多个内部模块复用的工具函数可以进入内部模块，但不得扩大包级公共 API。

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

`ImageInfo.width/height` 为正整数，因为它描述实际栅格尺寸，不代表标注坐标类型。整数化只能发生在明确的栅格操作边界，例如 OpenCV 数组切片；匹配和编码必须保留原始浮点几何。

#### 已发现的迁移一致性问题：裁剪尺寸截断

`CropMatches` 当前使用：

```python
ImageInfo(width=int(x2 - x1), height=int(y2 - y1))
```

旧管线传递的是浮点几何尺寸 `(x2 - x1, y2 - y1)`。因此父框坐标含小数时，新管线会截断归一化分母。

影响范围仅限经过 `CropMatches` 的三条配方：

- `point-segment`
- `knob-segment`
- `light2-detect`

影响评估：

- 当前回归夹具中的 LabelImg 父框均为整数，因此 122 项基线不能暴露此差异，现有夹具输出不受影响。
- 对宽度 `w`，截断带来的归一化相对放大约为 `w / floor(w) - 1`，上界接近 `1 / floor(w)`；框越小，影响越大。
- 例如父框宽度为 `10.6`，位于右边界的局部坐标按旧尺寸归一化为 `1.0`，按当前整数尺寸归一化为 `1.06`。
- `light2-detect` 可能输出大于 `1` 的检测坐标；`point-segment` 可能触发范围断言；`knob-segment` 会把越界值夹到 `1`，从而静默改变轮廓。
- 当宽或高小于 `1` 时，`int()` 得到 `0`，`ImageInfo` 会直接拒绝该裁剪。

定性：这是中等范围、样本级影响可能较大的迁移回归。常见整数 LabelImg 数据风险较低，但格式和旧实现都接受小数框，所以在解决或明确放弃这项兼容性之前，pipeline 迁移验收不能视为完全关闭。修复应单独设计，不在本次文档更新中顺带改代码。

### Pipeline 层迁移结果

- 用不可变 `Sample`、`ImageRef` 和阶段专用的 `*Input` / `*Output` 记录替代无类型 `Payload`。
- 处理顺序为 `Source -> Pipeline -> Sink`；处理器只转换数据，Sink 负责全部文件写入。
- `ItemProcessor` 表示一对零或一，`ExpandProcessor` 表示一对多；所有输出保持输入顺序。
- 裁剪处理器只生成延迟裁剪描述，不在处理器阶段写图像。
- 标注匹配通过 UUID 关联父子对象，保持父组与组内子项顺序。
- `scale-pose` 按已确认范围跳过；未知任务在访问文件系统前失败。
- 新架构完成后一次性删除旧 `annparser.py`、`annprocessor.py`、`annconverter.py` 和测试适配层，并将 `src/train.py` 的转换调用切换到 `xxtrain.pipeline.convert_dataset`。
- `xxtrain.pipeline` 包级公共 API 固定为九个符号：`Context`、`ConversionConfig`、`ConversionReport`、`ExpandProcessor`、`ImageRef`、`ItemProcessor`、`Pipeline`、`Sample`、`convert_dataset`。

## 第三阶段：建立打包配置

- 完善 `pyproject.toml` 的项目元数据、依赖声明和包发现。
- 迁移训练、验证、导出编排到 `xxtrain.training`。
- 建立安装后的命令行入口，处理资源文件和缓存路径。
- 确保 editable install、标准安装、CLI 和测试都不依赖 `sys.path` 偶然行为。

完成标准：项目可以通过标准 Python 包方式安装、测试和执行完整训练工作流。

## 已确认的迁移边界

- LabelImg、LabelMe、YOLO 的统一双向导入导出接口属于迁移后的功能补齐。当前只承接原有转换方向。
- 当前工作流在目标数据集产物已存在时跳过转换。源数据变化检测、失效判断和强制重建机制留待后续设计。
- 不保留新旧管线兼容层。开发期间允许整体功能暂时不可用，但入口切换只能在新架构承接全部既有配方后进行。
- 发现的算法缺陷和校验策略变化单独处理；迁移提交只承担结构变化与经确认的行为等价转换。

## 下一步

1. 单独设计并修复小数父框的裁剪尺寸契约，补充至少一组带小数父框的三配方回归测试。
2. 重新执行 pipeline 迁移验收门禁。
3. 进入 training 迁移设计。
4. 最后建立打包配置和 CLI。
