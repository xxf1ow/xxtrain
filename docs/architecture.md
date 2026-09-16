# xxtrain 架构

## Scope

xxtrain 是围绕 Ultralytics YOLO 的已安装 Python 包和训练工具。当前系统读取外部标注，把样本转换为任务数据集，生成模型配置，准备预训练权重，执行训练与预测检查，并导出 ONNX 模型。平台增量包含预置现场的数据存储边界、Point 任务定义、SQLite 权威标注、CVAT 对象身份映射、可恢复的检测标注业务流程，以及供现场人员使用的检测标注入口。完整自助训练平台、ClearML 服务连接及训练快照尚未实现。

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
- `xxtrain.platform.contracts` 定义平台组件共享的数据类型和错误；`xxtrain.platform.config` 从严格 JSON 配置加载单个工作区及独立运行目录；`xxtrain.platform.service.AnnotationService` 从数据库事实派生计数与缓存资格，以一个非阻塞进程锁协调上传接纳、CVAT 创建和同步、检测缓存生成；
- `xxtrain.platform.app` 提供同源图片上传、检测标注和缓存生成页面；浏览器会话由 CVAT 认证，所有写请求检查来源和页面 CSRF 令牌，认证失败、工作区归属失败和操作失败分别返回 401、403 和 502；上传文件在独立运行目录暂存，页面计数和缓存按钮由服务返回的文件派生结果驱动；
- `xxtrain.workspace_data` 以工作区的 `images/` 原图和 `annotations.db` 为权威输入。图片接纳保存 SHA-256、尺寸及无损 64 位感知哈希；摘要、输入指纹和 CVAT 输入只读取已登记图片及数据库标注。对象级同步通过稳定 UUID 和当前 Job 的 CVAT 原生 ID 保留未修改对象，按任务依赖清除受影响的下游对象，并在同一事务内提交标注和映射。检测缓存从数据库生成可丢弃的 LabelMe 输入，不回写权威 JSON；
- `xxtrain.business_tasks` 定义 Point 的五种框标签、检测、分类和分割步骤规则及目标开放状态；
- `xxtrain.platform.runtime` 保存可丢弃的目标与输入指纹到 CVAT Job 引用映射，`xxtrain.platform.cache` 从完成的 Point 工作区标注构建并原子发布检测数据集缓存；
- `xxtrain.integrations.cvat.codec` 在共享平台类型与 CVAT 标注字典之间转换矩形，初始化时用临时 UUID 令牌关联 CVAT 返回的原生 ID，常规回收只传递原生 ID 并移除该令牌；
- `xxtrain.integrations.cvat.CvatClient` 通过受限同源 HTTP 请求创建、准备、分配和读取 CVAT 标注任务，只有初始化响应或已初始化任务的恢复读取建立完整对象映射后才返回可发布的 Job；浏览器会话与服务令牌隔离。同版本 CVAT UI 加载的返回插件负责保存、完成状态确认和返回平台，不写平台文件；
- `xxtrain.cli` 只把命令参数传给训练包 API，不重新实现数据或训练逻辑。

三个子系统页面完整描述各自契约；本文只维护它们之间的运行关系和所有权边界。

## Authoritative and generated data

Scenario 目录的 `src/` 是当前训练流程的权威输入。数据集目录、模型 YAML、临时裁剪和训练列表都可重新生成；具体完成信号和重建规则由 [训练工作流](subsystems/training-workflow.md) 所有。

预训练权重位于 checkout、安装包和 Scenario 之外的共享用户缓存。训练得到的 ONNX 和分类参考图片写入 Scenario 的 `weights/`；具体路径与失败行为由训练子系统所有。

## Extension points

常规任务通过组合 Source、Processor、Sink 和 `DatasetRecipe` 扩展；通用差异进入 Scenario 配置，特殊几何或业务转换进入代码。配置不承担任意程序逻辑，具体 Processor 和 Sink 也不是通用第三方插件 API。

Ultralytics YOLO 是唯一训练后端。只有第二个真实后端形成共同边界后，才引入后端抽象。

## Future direction

内部自助训练平台仍处于提案阶段；Point 页面在业务任务与现场下共享图片上传区，按检测、分类、分割顺序逐行展示各模型的图片标注进度与标注、训练入口。当前支持批量上传、检测标注同步和至少 50 张有框图片的检测缓存生成；“开始训练”仅生成缓存。现场管理、分类、分割、训练提交和 ClearML 服务连接尚未实现，对应模型入口禁用。平台路线和验收标准由 [内部自助训练平台 Agent Note](agent-notes/proposed/feature/2026-07-30-self-service-training-platform.md) 所有。
