# xxtrain 架构

> 本文只记录长期方向、核心边界和当前阶段的架构。具体接口、配置字段与算法参数随实现演进，在对应模块中定义。

## 1. 项目定位

xxtrain 面向垂类视觉任务，负责把原始标注转换为可训练数据集，并完成模型训练、验证和导出。长期目标是在不建设通用 AutoML 的前提下，为固定垂类任务提供轻量的标注、离线半监督训练和进度反馈闭环。项目按三个阶段演进：

1. **基础训练架构整理（当前实现完成）**：稳定标注转换、数据集构建与训练流程。
2. **离线半监督与轻量训练平台（方向设计完成）**：用少量目标种子标签、伪标签和固定任务模板减少人工工作量，并让客户独立提交单模型训练任务。
3. **在线 Teacher–Student 训练（远期规划）**：训练过程中利用未标注数据，由 Teacher 为 Student 提供监督。

第一阶段的数据、pipeline、training、打包和安装式 CLI 已经完成，迁移后的格式互转和分类无标签目录 review 也已落地。进入第二阶段实现前，先补齐目录源类别发现和源数据变化检测两项近端能力，再实现最小离线半监督核心，最后增加薄平台。

### 1.1 当前进度快照

| 能力域 | 状态 | 当前边界 |
| --- | --- | --- |
| 基础数据模型与格式 I/O | 完成 | LabelImg、LabelMe、YOLO、COCO 已接入统一 `Annotation`，支持各格式可表达范围内的读写与 round-trip |
| typed 数据转换管线 | 完成 | 标准 detect、segment、pose、obb、classify 与现有复合任务均通过 `Source -> Pipeline -> Sink` 执行 |
| 训练、导出与结果检查 | 完成 | Scenario 驱动的训练、独立 ONNX 导出、预测可视化、分类有标签及无标签目录 review 已可用 |
| 打包与安装式 CLI | 完成 | setuptools、`xxtrain train/export/review` 和用户级权重缓存已建立 |
| 按图片发现标注文件 | 完成 | 固定布局下按图片同名发现 YOLO、LabelImg、LabelMe 候选文件，并在任务边界校验 |
| 从目录标注汇总类别目录 | 待实现 | 普通 `DirectorySource` 尚不从 LabelImg/LabelMe 内容生成 `LabelCatalog`，标准任务仍读取 `src/labels.txt` |
| 源变化检测与强制重建 | 待设计 | 当前只以输出 `dataset.yaml` 存在作为转换完成信号 |
| 离线半监督训练 | 方向已确认 | 尚无种子训练、伪标签生成筛选、固定人工验证集和迭代停止闭环 |
| 轻量训练平台 | 方向已确认 | 尚无固定任务模板、标注步骤门禁、训练锁和客户进度界面 |
| 在线 Teacher–Student | 远期规划 | 等离线阶段产生稳定策略和评估证据后再设计 |

近端顺序为：

1. 定义并实现目录源类别发现，明确 LabelImg/LabelMe 跨文件去重、稳定排序、Pose schema 和冲突处理；
2. 设计源数据指纹、失效判断与显式强制重建；
3. 在上述输入与构建边界稳定后，实现最小离线半监督核心；
4. 用固定模板和单模型任务建立薄平台，以 Point 二级任务完成端到端验收；
5. 离线阶段验证有效后，再评估在线 Teacher–Student。

## 2. 第一阶段：基础训练架构整理

### 2.1 核心流程

```text
任务定义
  → 读取图像与外部标注
  → 转换为统一标注对象
  → 按任务组合处理管线
  → 生成训练集与模型配置
  → 训练 / 验证 / 导出
```

第一阶段不包含自动伪标注闭环，也不负责标注 GUI。人工标注继续由 labelimg、labelme、X-AnyLabeling 等外部工具完成。

### 2.2 核心组成

#### Scenario 组合根

`xxtrain.task.TaskType` 只描述 detect、segment、pose、classify、obb 五种基础模型类型。Python Scenario 是具体数据集转换与训练的组合根：

```text
Scenario file
├── DatasetRecipe: labels + Pipeline + Sink
├── model version/scale
├── split + reserve_no_label=False
└── YOLO.train() overrides
```

Scenario 文件必须导出 `SCENARIO: TrainingScenario`。其中 `DatasetRecipe` 明确数据集名、基础 `TaskType`、标签目录、Pipeline 和 Sink；`train_args` 只覆盖源码提供的标准训练参数，并原样传给 `YOLO.train()`。

所有相对 `Path` 值以 Scenario 文件所在目录解析，普通相对字符串不做路径转换。该目录同时定义固定布局：

```text
<scenario_dir>/
├── <scenario>.py
├── src/                    # 原始图像与标注
├── <recipe.name>/          # 可重新生成的数据集产物与模型 YAML
└── weights/                # ONNX 与分类参考图
```

`standard_recipe()` 由 `xxtrain.pipeline.recipes` 提供，负责 detect、segment、pose、obb、classify 五个标准 Recipe。point、knob、light 七个特殊 Recipe 直接由对应 `data/` Scenario 文件组合；具体 Processor 和 Sink 不是承诺给任意第三方插件的稳定 API。

配置不需要描述所有处理细节。通用差异进入配置；特殊几何处理或业务转换继续由代码实现，避免把配置发展成另一套编程语言。

#### 统一标注表示

外部标注格式先解析为内部统一的 `Annotation`。后续处理只依赖统一对象，不直接耦合 labelimg XML、labelme JSON 或具体训练格式。

统一表示负责表达标签、形状、坐标和实例关系。LabelImg、LabelMe、YOLO 和 COCO 在格式可表达且任务语义允许的范围内通过同一表示读写；无法无损表达的转换在边界明确失败。置信度、来源等字段在真正需要伪标签时再扩展，不在第一阶段预设完整 schema。

长期数据契约如下：

- Annotation、几何值和 pipeline 记录保持不可变，变换返回新对象；
- 标签目录保持显式声明或已定义发现规则产生的稳定顺序，数字形式的名称仍按字符串处理；
- 坐标、几何运算和虚拟裁剪范围使用浮点数，不在 data、Source、Processor 或编码阶段提前截断；
- 整数化只允许发生在 OpenCV 数组切片等明确栅格边界；
- 格式无法无损表达、几何不符合任务或标签不属于当前目录时尽早失败。

#### 可组合转换管线

标注转换采用 `Source -> Pipeline -> Sink`。`DirectorySource` 按稳定顺序发现样本；不可变的 `Sample`、`ImageRef` 和阶段专用 `*Input` / `*Output` 记录在 Processor 之间传递，不使用共享 payload 字典；Sink 是唯一的数据集写入边界。主要职责包括：

- 解析图像与标注；
- 过滤和变换标注；
- 匹配父子实例；
- 裁剪复合任务的局部区域；
- 生成训练标签并划分数据集。

对目录源，标注发现以图片为中心并遵循固定同名布局：`imgs/<name>.*` 分别映射到 `labels/<name>.txt`、`anns/<name>.xml` 和 `anns_seg/<name>.json`。缺失候选文件表示该格式没有标注；不会递归搜索任意目录，也不会为同一格式加载多个候选文件。Pose 可按已定义语义组合 LabelImg 框与 LabelMe 关键点；其他任务遇到多个非空标注格式时拒绝歧义输入。

“发现标注文件”不等于“发现类别目录”。普通目录源目前不扫描所有 XML/JSON 汇总 `LabelCatalog`；标准任务仍以 `src/labels.txt` 为类别名称与顺序的真值来源，并拒绝文件中未声明的标签。COCO 可从自身 category schema 提供目录，特殊 Recipe 则显式携带固定目录。

`ItemProcessor` 表示一对零或一，`ExpandProcessor` 表示一对多。每张图像和每个裁剪实例都使用独立值对象，裁剪内容延迟到 Sink 才物化，避免跨样本状态泄漏。Processor 保持单一职责；recipe 负责组合，不把任务分支堆入训练入口。

#### 数据集产物

转换结果是训练框架可直接消费的数据集，包括图像、标签、训练/验证划分和数据集描述文件。它们是可重新生成的构建产物，不是原始标注的替代品。

相同原始数据、任务定义和代码版本应产生一致结果。转换中遇到缺失输入、非法形状或尺寸不一致时应尽早失败，避免错误进入训练阶段。

#### 训练入口

训练层负责串联 Scenario 加载、数据转换、模型配置、预训练权重、训练和 ONNX 导出。当前以 Ultralytics YOLO 为唯一训练后端，保持薄封装。

安装后通过统一的 `xxtrain` 命令执行三个独立职责：

- `xxtrain train`：运行完整训练工作流；
- `xxtrain export`：从已有 checkpoint 独立导出 ONNX；
- `xxtrain review`：运行预测并保存可视化结果或分类错分报告。

预测结果检查不是 `model.val()` 指标评估。训练过程中的 Ultralytics 验证仍负责指标、曲线和样例；当前没有单独重新计算验证指标的入口。

只有出现第二个真实训练后端并确认公共边界后，才抽象统一后端接口。现阶段不为假设中的框架兼容性增加复杂度。

### 2.3 模块边界

当前代码的职责划分如下：

- `xxtrain.task`：只定义五种基础 `TaskType`；
- `xxtrain.data`：不可变标注、LabelImg/LabelMe/YOLO/COCO 格式 I/O、几何和数据集产物工具；
- `xxtrain.pipeline.discovery`：稳定顺序的样本发现；
- `xxtrain.pipeline.core` / `processors`：typed records、Pipeline 和具体转换步骤；
- `xxtrain.pipeline.recipes`：`DatasetRecipe` 和五条标准 Recipe；
- `xxtrain.pipeline.sinks` / `workflow`：数据集写入边界和 `convert_dataset()` 公共入口；
- `xxtrain.training.scenario`：`TrainingScenario`、Python Scenario 加载和相对 `Path` 解析；
- `xxtrain.training.model` / `workflow`：模型 YAML、预训练权重和固定训练工作流；
- `xxtrain.training.exporting` / `review`：ONNX 导出、分类参考图和预测结果检查；
- `xxtrain.cli`：安装后的 `train`、`export`、`review` 子命令入口。

项目通过 setuptools 按 `src` 布局安装，要求 Python 3.11 及以上。预训练权重位于 `platformdirs` 提供的 xxtrain 用户缓存目录；具体数据集的 Scenario 仍留在数据集目录，不作为包资源安装。

旧的 `annparser.py`、`annprocessor.py`、`annconverter.py` 已在 typed pipeline 承接全部既有任务后删除。

`xxtrain.pipeline` 包级受支持 API 为 `Context`、`ConversionConfig`、`ConversionReport`、`DatasetRecipe`、`ExpandProcessor`、`ImageRef`、`ItemProcessor`、`Pipeline`、`Sample`、`convert_dataset` 和 `standard_recipe`。`convert_dataset(recipe, root_path, *, split=10, reserve_no_label=False)` 直接接收 Recipe，不保留旧任务名注册表或旧转换签名。

`xxtrain.training` 包级受支持 API 为 `TrainingScenario`、`load_scenario`、`train`、`export` 和 `review`。模型模板处理、预训练下载、分类错分整理等细节保持内部实现。

### 2.4 第一阶段完成标准

- 标准任务和现有复合任务都通过统一管线构建数据集；
- 新增常规任务主要通过组合已有 Processor 完成；
- 转换结果可复现，关键解析与几何转换可独立验证；
- 训练入口只负责流程编排，不承载标注处理逻辑；
- 训练、验证和 ONNX 导出形成稳定可执行路径。

## 3. 第二阶段：离线半监督与轻量训练平台（方向设计完成）

### 3.1 固定模板与单模型任务

开发者预定义任务类型、标签 schema 和训练预设，客户只能选择，不能创建任务结构或直接修改底层超参数。每个训练任务只训练一个模型；需要多个模型时分别提交任务，不建设自动多模型 DAG。

每个模板只声明两类标注契约：

- `required_annotations`：训练当前模型不可替代的前置标签，必须完整并逐图人工确认；
- `target_annotations`：当前模型要学习的目标标签，只需达到人工种子门限，其余样本允许保持未标注并在训练时生成伪标签。

二级任务依赖合格的一级标签，不依赖一级模型或平台内的一级训练历史。前置标签可以由人工、外部数据集或已有模型取得，但模型候选必须人工审核后才能进入受管理数据集。

### 3.2 数据与过程边界

第二阶段明确区分三类状态：

1. **受管理数据集**：原始图片、人工标签、人工审核通过的伪标签和解释标签所需的任务 schema；
2. **标注过程工作区**：候选标签、未审核伪标签、置信度、模型来源和逐图审核进度，可持久化但不纳入数据集；
3. **训练运行产物**：冻结输入清单、临时裁剪、训练时伪标签、日志、指标和模型结果。

人工审核通过的伪标签等同于人工确认标签，不要求数据集永久保留其预测来源。未审核伪标签不得静默覆盖正式标签。

训练开始时锁定原始图片和正式标签；需要编辑必须先中断训练。平台不处理训练与标注并发修改，也不建设版本合并机制。

### 3.3 Point 代表场景

Point 提供 `point-detect`、`point-classify` 和 `point-segment` 三个独立模板。后二者各自只依赖完整、逐图审核的 Point 框，不互相依赖。

一个框只保存一次。`Point` 表示已确认一级框但尚未分类，`tl/tc/cl/cc` 是 Point 的二级分类：

- 检测任务把 `Point/tl/tc/cl/cc` 统一投影为 `Point`；
- 分类任务只把 `tl/tc/cl/cc` 作为人工或审核通过的目标标签，保留为 `Point` 的框是未标注样本；
- 分割任务使用所有已确认框的几何范围，二级线只需标注种子集。

一级标注完成的单位是整张原图，必须检查已有框、补充漏框并确认无框图片。Point 业务不允许一级框重叠，并保证每个合法一级框都存在一个真实二级目标；没有二级标签只表示尚未标注，确认没有二级目标则说明一级框错误。

原图及原图坐标标签是权威数据。裁剪只在预览或训练时临时生成，二级预测结果必须能映射回原图供人工审核。

### 3.4 半监督与验收边界

最小闭环为：

```text
人工种子标签
  → 快速试训
  → 未标注样本预测与伪标签筛选
  → 正式训练
  → 选择性人工审核
  → 审核通过的标签进入受管理数据集
  → 下一轮训练
```

固定验证集只使用人工确认标签；同一原图及其全部裁剪必须进入同一数据划分。客户不需要审核全部伪标签，选择策略可以考虑类别覆盖、不确定性、视觉多样性和随机抽检。

第一版产品以 Point 二级分类或分割任务端到端完成为代表性验收，但每次仍只训练一个模型。标注和训练模块可以拆分部署并复用外部标注工具；初期不建设通用标注器、复杂账号权限、计费系统或工作流编辑器。

每类种子硬门限已经确定为基本约束。相似度可以用于选择代表样本和调整额外种子量，但特征模型、聚类方式、门限公式、伪标签阈值和停止条件必须通过真实垂类数据实验确定，不在架构层硬编码。

## 4. 第三阶段：在线 Teacher–Student 训练（远期规划）

在线阶段不再先生成完整伪标签数据集。Teacher 在训练过程中为未标注样本生成监督信号，Student 同时学习人工标注与未标注数据，Teacher 再根据 Student 更新。

本阶段复用第一阶段的任务定义、样本处理、标注表示和评估能力，但训练循环将成为独立策略。Teacher 更新方式、增强一致性、损失设计和稳定性控制，需要通过实验决定，不在当前架构中固定。

## 5. 架构原则

- **先稳定基础路径**：数据转换和训练可靠后，再增加自训练机制。
- **统一内部表示**：格式差异停留在输入解析和数据集输出边界。
- **组合优于分支堆叠**：用小型 Processor 组成任务流程。
- **配置适度**：配置表达稳定参数，代码承载复杂行为。
- **权威数据最小化**：数据集只管理原图和人工确认标签，裁剪、候选标签和未审核伪标签留在可重建过程层。
- **单模型任务**：多级业务通过完整前置标注契约表达，不发展成自动模型 DAG。
- **训练期间不可编辑**：训练使用冻结输入，修改标签必须中断训练。
- **产物可追踪、可复现**：训练运行能关联到冻结输入、任务与代码版本。
- **按需求抽象**：存在真实复用点后再引入后端、存储或调度抽象。

具体配置 schema、伪标签策略、相似度算法、运行产物保存期限、原生库接入和 Teacher–Student 算法均属于后续实现决策，不由本文预先规定。
