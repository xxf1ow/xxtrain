# xxtrain 架构

## Scope

xxtrain 是围绕 Ultralytics YOLO 的已安装 Python 包和训练工具。当前系统读取外部标注，把样本转换为任务数据集，生成模型配置，准备预训练权重，执行训练与预测检查，并导出 ONNX 模型。仓库尚未实现用户平台、SQLite 元数据、CVAT/ClearML 连接器、工作区或训练快照。

项目采用 `src` 布局，全部包源码位于 `src/xxtrain/`。安装后只有一个命令入口 `xxtrain`，由 `xxtrain.cli` 分派 `train`、`export` 和 `review`。

## Runtime flow

```text
Python Scenario
  → DatasetRecipe
  → Source → Pipeline → Sink
  → generated dataset + model YAML
  → Ultralytics training
  → prediction review / ONNX export
```

Scenario 是一次数据集转换和训练的组合根。`DatasetRecipe` 组合来源、不可变转换记录、处理器和唯一写入边界；训练层只编排数据准备、模型配置、预训练权重、Ultralytics 调用和导出，不承载标注解析或几何变换。

## Package boundaries

- `xxtrain.task` 只定义 detect、segment、pose、classify 和 OBB 五种基础 `TaskType`；数据集名称、标签和转换行为属于 Recipe 与 Scenario；
- [`xxtrain.data`](subsystems/annotation-data.md) 拥有不可变标注、几何、标签目录、格式 I/O 和数据集产物辅助函数；
- [`xxtrain.pipeline`](subsystems/dataset-pipeline.md) 拥有样本发现、typed records、Processor 组合、转换报告和数据集写入边界；
- [`xxtrain.training`](subsystems/training-workflow.md) 拥有 Scenario 加载、模型配置、训练、ONNX 导出和预测检查；
- `xxtrain.cli` 只把命令参数传给训练包 API，不重新实现数据或训练逻辑。

三个子系统页面完整描述各自契约；本文只维护它们之间的运行关系和所有权边界。

## Authoritative and generated data

Scenario 目录的 `src/` 是当前训练流程的权威输入。数据集目录、模型 YAML、临时裁剪和训练列表都可重新生成；具体完成信号和重建规则由 [训练工作流](subsystems/training-workflow.md) 所有。

预训练权重位于 checkout、安装包和 Scenario 之外的共享用户缓存。训练得到的 ONNX 和分类参考图片写入 Scenario 的 `weights/`；具体路径与失败行为由训练子系统所有。

## Extension points

常规任务通过组合 Source、Processor、Sink 和 `DatasetRecipe` 扩展；通用差异进入 Scenario 配置，特殊几何或业务转换进入代码。配置不承担任意程序逻辑，具体 Processor 和 Sink 也不是通用第三方插件 API。

Ultralytics YOLO 是唯一训练后端。只有第二个真实后端形成共同边界后，才引入后端抽象。

## Future direction

内部自助训练平台仍处于提案阶段，计划在现有训练核心之外增加权威数据与用户流程控制层。其边界、路线和验收标准由 [内部自助训练平台 Agent Note](agent-notes/proposed/feature/2026-07-30-self-service-training-platform.md) 所有，未实现内容不属于当前架构。
