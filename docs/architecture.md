# xxtrain 架构

## Scope

xxtrain 是围绕 Ultralytics YOLO 的已安装 Python 包和训练工具。当前系统读取外部标注，把样本转换为任务数据集，生成模型配置，准备预训练权重，执行训练与预测检查，并导出 ONNX 模型。平台增量包含预置现场的数据存储边界、Point 任务定义、SQLite 权威标注、CVAT 对象身份映射，以及供现场人员完成检测、分类、指针分割标注和自助训练的可恢复业务流程。[SQLite 标注存储与对象身份](agent-notes/implemented/architecture/2026-09-16-sqlite-annotation-storage.md)和 [Point 三模型数据准备](agent-notes/implemented/feature/2026-09-16-point-classification-segmentation.md)均已通过离线回归和真实人工验收。训练提交协调、工作区编辑保护和独立用户任务页面通过离线回归，三个模型完成真实 ClearML、Agent、GPU 训练及人工验收；证据与限制见 [训练闭环记录](agent-notes/implemented/feature/2026-09-17-point-training-loop.md#live-acceptance-and-limitations)。

项目采用 `src` 布局，全部包源码位于 `src/xxtrain/`。安装后的 `xxtrain` 由 `xxtrain.cli` 分派 `train`、`export` 和 `review`；安装 `platform` 可选依赖后，`xxtrain-platform` 运行单工作进程的 Point 检测、分类和指针分割标注入口。

服务端从一个 Git checkout 的锁定 uv 环境运行 `xxtrain-platform`，由 `xxtrain-server.service` 管理平台进程与单个 Compose 项目。单元先准备私有配置、启动 Compose、等待 CVAT 迁移并同步管理员和服务令牌，再验证 ClearML 身份；前台平台进程读取引导后的环境文件，初始化失败则不启动平台。CVAT、ClearML Server 和 nginx 运行在 Compose 容器中；nginx 是对外入口，平台与 CVAT 后端仅在宿主机 loopback 监听。CVAT 服务和 worker 共用 supervisor 所需的空 `SMOKESCREEN_OPTS`；OPA 使用 `cvat-server` 网络别名访问 CVAT。ClearML API 通过 `/debug.ping` 健康检查后，fileserver 才启动；Web 等待 API 健康和 fileserver 启动后再运行。API 预填充期间 fileserver 保持等待。ClearML Redis 同时提供上游 fileserver 要求的 `redis` 网络别名。ClearML SDK 的 API、Web 和 files 主机必须显式配置为本机 Compose 映射，防止 SDK 默认连接 ClearML Cloud。`.deployment/` 保存配置、凭据和持久数据，系统注册的 unit 是目录边界的例外。生命周期、版本切换和验证步骤见[服务端部署指南](cookbook/server-deployment.md)。

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

Point 工作区在业务任务与现场下共享图片上传区，并按任务定义的顺序逐行展示检测原图、分类裁剪图和指针分割裁剪图的进度与操作。全部原图已标注且至少 50 张有框原图后，分类和指针分割同时开放；两个下游只要求自身标注完整即可生成缓存，不依赖兄弟步骤或上游训练缓存。配置训练服务后，提交停留在工作区，历史运行、实时状态和产物操作由独立任务页面承载；仅标注启动生成所选目标的缓存。

## Package boundaries

- `xxtrain.task` 只定义 detect、segment、pose、classify 和 OBB 五种基础 `TaskType`；数据集名称、标签和转换行为属于 Recipe 与 Scenario；
- [`xxtrain.data`](subsystems/annotation-data.md) 拥有不可变标注、几何、标签目录、格式 I/O 和数据集产物辅助函数；
- [`xxtrain.pipeline`](subsystems/dataset-pipeline.md) 拥有样本发现、typed records、Processor 组合、转换报告和数据集写入边界；
- [`xxtrain.training`](subsystems/training-workflow.md) 拥有 Scenario 加载、共用模型与训练默认值、模型配置、训练、ONNX 导出和预测检查；
- `xxtrain.platform.contracts` 定义平台组件共享的数据类型和错误；`xxtrain.platform.config` 从严格 JSON 配置加载单个工作区及独立运行目录；`xxtrain.platform.service.AnnotationService` 遍历任务步骤，从数据库事实、传递输入依赖和步骤门槛派生计数与操作资格，并以一个非阻塞进程锁协调上传、CVAT Job 创建与同步及缓存生成；
- `xxtrain.platform.app` 按组合根注入的任务定义提供同源图片上传、有序目标标注与缓存生成页面，以及可选训练服务的提交、查询、取消和部署产物下载 API。HTTP 只接受当前定义声明的目标；工作区载荷返回任务、目标元数据、计数、服务端派生的操作权限、缓存状态和当前训练摘要，页面据此动态创建目标行。浏览器会话由 CVAT 认证，所有写请求检查来源和页面 CSRF 令牌。训练提交只接受空对象，运行身份由服务端按用户、工作区、目标和输入指纹确定；训练 API 只返回用户运行、现场、任务与目标展示名、主指标、时间、取消请求及执行摘要，不返回持久意图、缓存路径、ClearML 任务身份或后端异常。启用训练时，应用 lifespan 先执行输入兼容处理，再启动一个 `training_coordinator` 线程接续持久意图，不要求浏览器读取或重发请求；每轮结束后等待 5 秒，关闭等待在途协调结束。仅标注模式不创建该线程。上传文件在独立运行目录暂存，页面计数和按钮由服务返回的数据库派生结果驱动；
- `xxtrain.workspace_data` 以工作区的 `images/` 原图和 `annotations.db` 为权威输入，并由组合根显式注入业务任务定义。图片接纳保存 SHA-256、尺寸及无损 64 位感知哈希；任务步骤选择原图或轴对齐矩形输入适配器，适配器从原图坐标中的权威标注派生 frame 映射，只有编辑入口需要图片时才生成可丢弃裁剪。摘要、正样本数和编辑指纹从同一映射及任务输入闭包派生；训练指纹再加入任务标识、有序输出标签、训练类型、负样本输出处理和显式转换键。对象级同步按持久 frame 映射还原几何与直接 parent UUID，通过 CVAT 原生 ID 保留未修改对象，按任务依赖清除受影响对象，并在同一事务内提交标注和映射。Point 旧指纹算法只供启动兼容处理计算迁移前身份；
- `xxtrain.business_tasks` 定义任务与步骤规则、输入适配器选择、标注策略、父来源、额外标签依赖、训练设置、有序输出标签、数据集转换键、样本编码回调、主要指标与交付内容；配置加载受信任的 Python 定义入口，Point 定义保留五种框标签及检测、分类和分割规则；
- `xxtrain.platform.training_contracts` 定义训练运行关联事实；`xxtrain.platform.training_store` 在调用者指定的平台元数据路径保存独立 SQLite 账本，按用户限制读取，并以可空的 `desired_action` 保存 `execute` 或 `cancel` 意图，但不存执行状态、计数或标注。每个新运行不可变地保存配置选择的任务入口；旧账本迁移为冻结的 Point 选择器。兼容输入别名以完整用户、工作区、目标和指纹键关联既有运行，不改写其指纹、缓存路径或任务 ID；`xxtrain.platform.runtime` 以编辑指纹保存可丢弃的 CVAT Job 引用，并只接受带精确清单及完整训练输入的新发布，无清单的旧 Point 发布也不由该通用入口接受；`xxtrain.platform.target_cache` 调用步骤定义的编码回调，从权威 frame 投影生成检测、分类、分割或嵌套矩形数据集，并写入精确目标与训练指纹清单后原子发布；
- `xxtrain.platform.training_service` 在标注写互斥内计算当前输入指纹，通过规范身份与兼容别名返回该用户和目标已有的运行，或在资格与缓存准备完成后创建带执行意图的服务端运行身份；提交和取消先持久保存意图，再由同一协调入口根据原 ClearML 任务事实创建、绑定、核验启动配置、入队或停止。只有命令和生命周期协调可以执行这些写操作；列表、详情和工作区投影只观察本地与远端事实，不修复关联或触发 ClearML 写入。启动兼容处理从一个权威标注快照验证冻结的 Point 转换语义与旧指纹；输入身份证明不依赖历史发布是否存在或完整，因此精确匹配始终添加别名，而不重建缓存或提交任务。工作区编辑保护检查全部用户的运行；活跃或未知执行锁定目标及其输入祖先，并对多个运行取并集，任何活跃运行仍禁止上传。未尝试的执行意图显示为待处理并锁定编辑，未尝试的取消意图可由本地事实确认未执行，旧 NULL 意图只供观察。终态执行与部署产物资格分别派生，远端缺失、歧义、启动配置不完整或执行查询失败继续锁定编辑；完成但缺少部署产物的运行解除编辑限制但不能下载。提供训练配置时，平台组合根在独立元数据目录构造运行账本和使用有限请求边界的 ClearML 适配器，在协调器和请求启动前执行一次兼容处理，并把实时编辑保护及运行引用缓存的重建保护绑定到 `AnnotationService`；省略训练配置时保留仅标注启动方式；
- `xxtrain.integrations.clearml` 通过可选 ClearML SDK 创建、核验、入队、观察和取消训练任务，并提供普通 Agent 执行入口；任务使用运行 UUID 作为初始名称恢复关联，创建部分成功时只补齐同一未执行任务的无冲突启动事实。Agent 从任务 `Args/*` 参数取得运行保存的任务入口和输入身份，通过兼容加载器把冻结的旧 Point 选择器映射到默认定义，其余入口直接加载配置工厂；定义选择训练设置、主要指标和交付内容，worker 不直接引用 Point 业务模块；
- `xxtrain.integrations.cvat.edit_codec` 按 `EditJob.frames` 的精确顺序转换 rectangle、分类 tag、polyline 和策略声明的图片级负样本 Tag，并核对每个 CVAT 标签的原生类型。需要对象身份的标注只在初始化时用临时 UUID 令牌关联 CVAT 原生 ID，常规回收只返回原生 ID；
- `xxtrain.integrations.cvat.CvatClient` 通过受限同源 HTTP 请求按显式 `AnnotationPolicy` 创建类型化任务，并只为 Task 响应明确证明无图片或未关联数据的新建 Task 顺序上传 `EditFrame`、轮询请求、核对实际 frame、单次初始化标注、解码原生对象映射并分配 Job；未知或非零图片数量在上传前被拒绝，只有完整对象映射建立后才返回可发布的 Job。浏览器会话与服务令牌隔离。平台依据运行目录中当前输入指纹对应的 Job 和 frame 来源映射回收当前步骤结果；同版本 CVAT UI 加载的返回插件负责保存、完成状态确认和返回平台，不写平台文件；
- `xxtrain.cli` 只把命令参数传给训练包 API，不重新实现数据或训练逻辑。

三个子系统页面完整描述各自契约；本文只维护它们之间的运行关系和所有权边界。

## Authoritative and generated data

Scenario 目录的 `src/` 是当前训练流程的权威输入。数据集目录、模型 YAML、临时裁剪和训练列表都可重新生成；具体完成信号和重建规则由 [训练工作流](subsystems/training-workflow.md) 所有。

预训练权重位于 checkout、安装包和 Scenario 之外的共享用户缓存。训练得到的 ONNX 和分类参考图片写入 Scenario 的 `weights/`；具体路径与失败行为由训练子系统所有。

## Extension points

常规任务通过组合 Source、Processor、Sink 和 `DatasetRecipe` 扩展；通用差异进入 Scenario 配置，特殊几何或业务转换进入代码。配置不承担任意程序逻辑，具体 Processor 和 Sink 也不是通用第三方插件 API。

平台业务任务通过 Python `TaskDefinition` 组合步骤规则、输入适配、标注策略、训练转换和交付内容。存储校验、frame 投影、同步与失效、输入身份、标注协调、CVAT 交换、训练缓存与保护、HTTP、页面和 worker 都解释同一定义；扩展边界与验证要求由[任务定义驱动决策](agent-notes/implemented/architecture/2026-09-19-task-definition-driven-platform.md)所有。Point 已通过该实现版本的[部署与人工验收](agent-notes/implemented/architecture/2026-09-19-task-definition-driven-platform.md#live-acceptance)。非 Point 合成任务已通过完整离线组件和安装 wheel 验证，但尚未通过真实 CVAT 页面和 ClearML Server、Agent、GPU 部署验收。

Ultralytics YOLO 是唯一训练后端。只有第二个真实后端形成共同边界后，才引入后端抽象。

## Future direction

现场管理、管理员数据治理和模型推理尚未实现。完整平台路线由 [内部自助训练平台 Agent Note](agent-notes/proposed/feature/2026-07-30-self-service-training-platform.md) 所有。
