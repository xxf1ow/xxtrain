# xxtrain 架构

## Scope

xxtrain 是围绕 Ultralytics YOLO 的已安装 Python 包和训练工具。当前系统读取外部标注，把样本转换为任务数据集，生成模型配置，准备预训练权重，执行训练与预测检查，并导出 ONNX 模型。平台增量包含预置现场的数据存储边界、Point 任务定义、SQLite 权威标注、CVAT 对象身份映射，以及供现场人员完成检测、分类、指针分割标注和自助训练的可恢复业务流程。[SQLite 标注存储与对象身份](agent-notes/implemented/architecture/2026-09-16-sqlite-annotation-storage.md)和 [Point 三模型数据准备](agent-notes/implemented/feature/2026-09-16-point-classification-segmentation.md)均已通过离线回归和真实人工验收。训练提交协调、工作区编辑保护和独立用户任务页面已有离线实现；真实 ClearML 服务验收尚未完成。

项目采用 `src` 布局，全部包源码位于 `src/xxtrain/`。安装后的 `xxtrain` 由 `xxtrain.cli` 分派 `train`、`export` 和 `review`；安装 `platform` 可选依赖后，`xxtrain-platform` 运行单工作进程的 Point 检测、分类和指针分割标注入口。

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
- [`xxtrain.training`](subsystems/training-workflow.md) 拥有 Scenario 加载、共用模型与训练默认值、模型配置、训练、ONNX 导出和预测检查；
- `xxtrain.platform.contracts` 定义平台组件共享的数据类型和错误；`xxtrain.platform.config` 从严格 JSON 配置加载单个工作区及独立运行目录；`xxtrain.platform.service.AnnotationService` 从数据库事实派生三个目标的计数、前置条件和缓存资格，以一个非阻塞进程锁协调上传、CVAT Job 创建与同步及缓存生成；
- `xxtrain.platform.app` 提供同源图片上传、三个 Point 目标的标注及缓存生成页面，以及可选训练服务的提交、查询、取消和部署产物下载 API；浏览器会话由 CVAT 认证，所有写请求检查来源和页面 CSRF 令牌。训练提交只接受空对象，运行身份由服务端按用户、工作区、目标和输入指纹确定；训练 API 只返回用户运行、现场、目标、时间、取消请求及执行摘要，不返回持久意图、缓存路径、ClearML 任务身份或后端异常。启用训练时，应用生命周期拥有一个 `training_coordinator` 线程，启动立即协调并在每轮结束后等待 5 秒，关闭等待在途协调结束；仅标注模式不创建该线程。上传文件在独立运行目录暂存，页面计数和按钮由服务返回的数据库派生结果驱动；
- `xxtrain.workspace_data` 以工作区的 `images/` 原图和 `annotations.db` 为权威输入。图片接纳保存 SHA-256、尺寸及无损 64 位感知哈希；摘要、输入指纹和 CVAT 输入只读取已登记图片及数据库标注。分类和分割投影共用检测框裁剪与实际边界，数据层从数据库派生目标计数和包含上游关联的输入指纹。对象级同步通过稳定 UUID 和当前 Job 的 CVAT 原生 ID 保留未修改对象，按任务依赖清除受影响的下游对象，并在同一事务内提交标注和映射。检测缓存从数据库生成可丢弃的 LabelMe 输入，不回写权威 JSON；
- `xxtrain.business_tasks` 定义 Point 的五种框标签、检测、分类和分割步骤规则、目标开放状态、训练设置、主要指标与交付内容；
- `xxtrain.platform.training_contracts` 定义训练运行关联事实；`xxtrain.platform.training_store` 在调用者指定的平台元数据路径保存独立 SQLite 账本，按用户限制读取，并以可空的 `desired_action` 保存 `execute` 或 `cancel` 意图，但不存执行状态、计数或标注。旧账本原样保留既有事实并增加该可空列，NULL 不推断为任何意图；`xxtrain.platform.runtime` 保存可丢弃的目标与输入指纹到 CVAT Job 引用映射，并只为包含数据集元数据、训练和验证输入的完整已发布缓存返回目标目录；`xxtrain.platform.cache` 从完成的 Point 工作区标注构建并原子发布检测数据集缓存，`xxtrain.platform.target_cache` 使用共享裁剪图生成分类和指针分割训练缓存；
- `xxtrain.platform.training_service` 在标注写互斥内计算当前输入指纹，返回该用户和目标已有的输入运行，或在资格与缓存准备完成后创建带执行意图的服务端运行身份；提交和取消先持久保存意图，再由同一协调入口根据原 ClearML 任务事实创建、绑定、入队或停止。列表、详情和工作区投影只观察本地与远端事实，不修复关联或触发 ClearML 写入；工作区编辑保护检查配置工作区的全部运行，按钮只投影当前用户、当前工作区和当前输入，用户历史仍可显示其其他工作区运行。未尝试的执行意图显示为待处理并锁定编辑，未尝试的取消意图可由本地事实确认未执行，旧 NULL 意图只供观察。终态执行与部署产物资格分别派生，远端缺失、歧义或执行查询失败继续锁定编辑。提供训练配置时，平台组合根在独立元数据目录构造运行账本和使用有限请求边界的 ClearML 适配器，并把实时编辑保护及运行引用缓存的重建保护绑定到 `AnnotationService`；省略训练配置时保留仅标注启动方式；
- `xxtrain.integrations.clearml` 通过可选 ClearML SDK 创建、入队、观察和取消训练任务，并提供普通 Agent 执行入口；任务使用运行 UUID 作为初始名称恢复关联，Agent 从共享发布缓存调用训练核心，只上传固定名称的部署产物；
- `xxtrain.integrations.cvat.codec` 在共享平台类型与 CVAT 标注字典之间转换检测矩形与图片级负样本 Tag；负样本按原图映射回收，删除 Tag 撤销确认，框与负样本冲突时返回修正链接且不写入。`xxtrain.integrations.cvat.edit_codec` 按 `EditJob.frames` 的精确顺序转换分类 tag 和指针 polyline。需要对象身份的标注只在初始化时用临时 UUID 令牌关联 CVAT 原生 ID，常规回收只返回原生 ID；
- `xxtrain.integrations.cvat.CvatClient` 通过受限同源 HTTP 请求创建类型化任务，并以共享流程完成检测或编辑图片的上传、frame 核对、Job 分配、初始化和读取；只有完整对象映射建立后才返回可发布的 Job，浏览器会话与服务令牌隔离。平台依据运行目录中当前输入指纹对应的 Job 和 frame 来源映射回收分类及指针结果；同版本 CVAT UI 加载的返回插件负责保存、完成状态确认和返回平台，不写平台文件；
- `xxtrain.cli` 只把命令参数传给训练包 API，不重新实现数据或训练逻辑。

三个子系统页面完整描述各自契约；本文只维护它们之间的运行关系和所有权边界。

## Authoritative and generated data

Scenario 目录的 `src/` 是当前训练流程的权威输入。数据集目录、模型 YAML、临时裁剪和训练列表都可重新生成；具体完成信号和重建规则由 [训练工作流](subsystems/training-workflow.md) 所有。

预训练权重位于 checkout、安装包和 Scenario 之外的共享用户缓存。训练得到的 ONNX 和分类参考图片写入 Scenario 的 `weights/`；具体路径与失败行为由训练子系统所有。

## Extension points

常规任务通过组合 Source、Processor、Sink 和 `DatasetRecipe` 扩展；通用差异进入 Scenario 配置，特殊几何或业务转换进入代码。配置不承担任意程序逻辑，具体 Processor 和 Sink 也不是通用第三方插件 API。

Ultralytics YOLO 是唯一训练后端。只有第二个真实后端形成共同边界后，才引入后端抽象。

## Future direction

Point 工作区在业务任务与现场下共享图片上传区，按检测、分类、指针分割顺序逐行展示原图或裁剪图进度及标注、训练入口。检测满足全部原图已标注且至少 50 张有框原图后开放分类，全部裁剪图各有一个分类后开放指针分割；进入后续步骤不依赖前一步缓存。配置训练服务后，“开始训练”提交当前目标并留在工作区，独立任务页面显示历史运行、实时状态和产物操作；仅标注启动仍生成所选目标的缓存。现场管理、管理员数据治理、模型推理和真实 ClearML 服务验收尚未完成。完整平台路线由 [内部自助训练平台 Agent Note](agent-notes/proposed/feature/2026-07-30-self-service-training-platform.md) 所有。
