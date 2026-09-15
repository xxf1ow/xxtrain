# xxtrain 架构

## Scope

xxtrain 是围绕 Ultralytics YOLO 的已安装 Python 包和训练工具。当前系统读取外部标注，把样本转换为任务数据集，生成模型配置，准备预训练权重，执行训练与预测检查，并导出 ONNX 模型。平台增量包含预置现场的数据存储边界、Point 任务定义、CVAT 矩形标注转换、CVAT HTTP 适配、可恢复的检测标注业务流程，以及供现场人员使用的检测标注入口；仓库尚未实现完整自助训练平台、SQLite 元数据、ClearML 服务连接或训练快照。

项目采用 `src` 布局，全部包源码位于 `src/xxtrain/`。安装后的 `xxtrain` 由 `xxtrain.cli` 分派 `train`、`export` 和 `review`；安装 `platform` 可选依赖后，`xxtrain-platform` 运行单工作进程的 Point 检测标注入口。

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
- `xxtrain.platform.contracts` 定义平台组件共享的数据类型、状态和错误，包根不导入服务或页面实现；`xxtrain.platform.config` 从严格字段的 JSON 对象加载单个预置工作区并相对配置文件解析路径，`xxtrain.platform.state` 以单文件原子替换保存流程私有状态，`xxtrain.platform.service.AnnotationService` 串行协调数据存储与 CVAT 的开始、继续和取回保存；
- `xxtrain.platform.app` 提供同源页面与窄 HTTP 接口；浏览器会话由 CVAT 认证，所有写请求检查来源和页面 CSRF 令牌，认证失败、工作区归属失败和操作失败分别返回 401、403 和 502；
- `xxtrain.workspace_data` 以工作区的 `images/` 和 `annotations/` 为权威输入，按 SHA-256 文件名接纳去重后的图片，并从当前图片及 LabelMe 检测框或显式负样本派生检测摘要和输入指纹；保存只替换检测矩形，保留其他标注和未知字段，并以逐文件原子替换落盘；
- `xxtrain.business_tasks` 定义 Point 的五种框标签和检测、分类、分割目标开放状态；
- `xxtrain.integrations.cvat.codec` 在共享平台类型与 CVAT 标注字典之间转换矩形，按 CVAT 标签定义解析真实 ID，并拒绝无法无损映射的标注类型；
- `xxtrain.integrations.cvat.CvatClient` 通过受限同源 HTTP 请求创建、准备、分配和读取 CVAT 标注任务；浏览器会话与服务令牌隔离，准备进度由调用方通过不依赖 HTTPX 的公开检查点持久化；同版本 CVAT UI 加载的返回插件只负责保存、完成状态确认和返回平台，不写平台文件；
- `xxtrain.cli` 只把命令参数传给训练包 API，不重新实现数据或训练逻辑。

三个子系统页面完整描述各自契约；本文只维护它们之间的运行关系和所有权边界。

## Authoritative and generated data

Scenario 目录的 `src/` 是当前训练流程的权威输入。数据集目录、模型 YAML、临时裁剪和训练列表都可重新生成；具体完成信号和重建规则由 [训练工作流](subsystems/training-workflow.md) 所有。

预训练权重位于 checkout、安装包和 Scenario 之外的共享用户缓存。训练得到的 ONNX 和分类参考图片写入 Scenario 的 `weights/`；具体路径与失败行为由训练子系统所有。

## Extension points

常规任务通过组合 Source、Processor、Sink 和 `DatasetRecipe` 扩展；通用差异进入 Scenario 配置，特殊几何或业务转换进入代码。配置不承担任意程序逻辑，具体 Processor 和 Sink 也不是通用第三方插件 API。

Ultralytics YOLO 是唯一训练后端。只有第二个真实后端形成共同边界后，才引入后端抽象。

## Future direction

内部自助训练平台仍处于提案阶段；当前实现到达预置 Point 现场的登录、检测编辑和平台文件保存闭环，上传、现场管理、分类、分割、训练提交和 ClearML 服务连接尚未实现。平台边界、路线和验收标准由 [内部自助训练平台 Agent Note](agent-notes/proposed/feature/2026-07-30-self-service-training-platform.md) 所有，未实现内容不属于当前架构。
