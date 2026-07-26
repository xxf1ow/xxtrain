# xxtrain 架构

> 本文只记录长期方向、核心边界和当前阶段的架构。具体接口、配置字段与算法参数随实现演进，在对应模块中定义。

## 1. 项目定位

xxtrain 面向垂类视觉任务，负责把原始标注转换为可训练数据集，并完成模型训练、验证和导出。项目按三个阶段演进：

1. **基础训练架构整理（当前实现完成）**：稳定标注转换、数据集构建与训练流程。
2. **离线伪标签迭代训练（规划中）**：用已训练模型生成伪标签，筛选后重新训练。
3. **在线 Teacher–Student 训练（规划中）**：训练过程中利用未标注数据，由 Teacher 为 Student 提供监督。

第一阶段的数据、pipeline 和 training 源码边界已经完成；项目下一项迁移工作是打包和安装式 CLI。后续训练阶段建立在现有任务定义、数据处理和评估能力之上，不提前固定实现细节。

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

`standard_recipe()` 由 `xxtrain.pipeline.recipes` 提供，只负责 detect、segment、pose、classify 四个标准 Recipe。point、knob、light 七个特殊 Recipe 直接由对应 `data/` Scenario 文件组合；具体 Processor 和 Sink 不是承诺给任意第三方插件的稳定 API。

配置不需要描述所有处理细节。通用差异进入配置；特殊几何处理或业务转换继续由代码实现，避免把配置发展成另一套编程语言。

#### 统一标注表示

外部标注格式先解析为内部统一的 `Annotation`。后续处理只依赖统一对象，不直接耦合 labelimg XML、labelme JSON 或具体训练格式。

统一表示负责表达标签、形状、坐标和实例关系。置信度、来源等字段在真正需要伪标签时再扩展，不在第一阶段预设完整 schema。

#### 可组合转换管线

标注转换采用 `Source -> Pipeline -> Sink`。`DirectorySource` 按稳定顺序发现样本；不可变的 `Sample`、`ImageRef` 和阶段专用 `*Input` / `*Output` 记录在 Processor 之间传递，不使用共享 payload 字典；Sink 是唯一的数据集写入边界。主要职责包括：

- 解析图像与标注；
- 过滤和变换标注；
- 匹配父子实例；
- 裁剪复合任务的局部区域；
- 生成训练标签并划分数据集。

`ItemProcessor` 表示一对零或一，`ExpandProcessor` 表示一对多。每张图像和每个裁剪实例都使用独立值对象，裁剪内容延迟到 Sink 才物化，避免跨样本状态泄漏。Processor 保持单一职责；recipe 负责组合，不把任务分支堆入训练入口。

#### 数据集产物

转换结果是训练框架可直接消费的数据集，包括图像、标签、训练/验证划分和数据集描述文件。它们是可重新生成的构建产物，不是原始标注的替代品。

相同原始数据、任务定义和代码版本应产生一致结果。转换中遇到缺失输入、非法形状或尺寸不一致时应尽早失败，避免错误进入训练阶段。

#### 训练入口

训练层负责串联 Scenario 加载、数据转换、模型配置、预训练权重、训练和 ONNX 导出。当前以 Ultralytics YOLO 为唯一训练后端，保持薄封装。

三个源码入口职责独立：

- `src/train.py`：运行完整训练工作流；
- `src/export.py`：从已有 checkpoint 独立导出 ONNX；
- `src/review.py`：运行预测并保存可视化结果或分类错分报告。

预测结果检查不是 `model.val()` 指标评估。训练过程中的 Ultralytics 验证仍负责指标、曲线和样例；当前没有单独重新计算验证指标的入口。

只有出现第二个真实训练后端并确认公共边界后，才抽象统一后端接口。现阶段不为假设中的框架兼容性增加复杂度。

### 2.3 模块边界

当前代码的职责划分如下：

- `xxtrain.task`：只定义五种基础 `TaskType`；
- `xxtrain.data`：不可变标注、格式读取、几何、YOLO 编码和数据集产物工具；
- `xxtrain.pipeline.discovery`：稳定顺序的样本发现；
- `xxtrain.pipeline.core` / `processors`：typed records、Pipeline 和具体转换步骤；
- `xxtrain.pipeline.recipes`：`DatasetRecipe` 和四条标准 Recipe；
- `xxtrain.pipeline.sinks` / `workflow`：数据集写入边界和 `convert_dataset()` 公共入口；
- `xxtrain.training.scenario`：`TrainingScenario`、Python Scenario 加载和相对 `Path` 解析；
- `xxtrain.training.model` / `workflow`：模型 YAML、预训练权重和固定训练工作流；
- `xxtrain.training.exporting` / `review`：ONNX 导出、分类参考图和预测结果检查；
- `train.py`、`export.py`、`review.py`：当前源码检出中的薄命令入口；安装式 CLI 留到打包阶段。

旧的 `annparser.py`、`annprocessor.py`、`annconverter.py` 已在 typed pipeline 承接全部既有任务后删除。

`xxtrain.pipeline` 包级受支持 API 为 `Context`、`ConversionConfig`、`ConversionReport`、`DatasetRecipe`、`ExpandProcessor`、`ImageRef`、`ItemProcessor`、`Pipeline`、`Sample`、`convert_dataset` 和 `standard_recipe`。`convert_dataset(recipe, root_path, *, split=10, reserve_no_label=False)` 直接接收 Recipe，不保留旧任务名注册表或旧转换签名。

`xxtrain.training` 包级受支持 API 为 `TrainingScenario`、`load_scenario`、`train`、`export` 和 `review`。模型模板处理、预训练下载、分类错分整理等细节保持内部实现。

### 2.4 第一阶段完成标准

- 标准任务和现有复合任务都通过统一管线构建数据集；
- 新增常规任务主要通过组合已有 Processor 完成；
- 转换结果可复现，关键解析与几何转换可独立验证；
- 训练入口只负责流程编排，不承载标注处理逻辑；
- 训练、验证和 ONNX 导出形成稳定可执行路径。

## 3. 第二阶段：离线伪标签迭代训练（规划中）

离线阶段在训练轮次之间使用未标注数据：

```text
人工标注训练集 → 训练 → 未标注数据推理 → 筛选伪标签 → 合并训练集 → 下一轮训练
```

核心约束：

- 人工标注与伪标签可区分，伪标签不得静默覆盖人工标注；
- 每轮数据、模型和伪标签来源可追踪；
- 使用固定人工验证集监控质量，避免只依赖模型自身置信度；
- 伪标签筛选和停止条件由独立策略负责，不侵入基础转换管线。

置信度阈值、数据版本格式、回退方式和收敛规则留到本阶段实施时确定。

## 4. 第三阶段：在线 Teacher–Student 训练（规划中）

在线阶段不再先生成完整伪标签数据集。Teacher 在训练过程中为未标注样本生成监督信号，Student 同时学习人工标注与未标注数据，Teacher 再根据 Student 更新。

本阶段复用第一阶段的任务定义、样本处理、标注表示和评估能力，但训练循环将成为独立策略。Teacher 更新方式、增强一致性、损失设计和稳定性控制，需要通过实验决定，不在当前架构中固定。

## 5. 架构原则

- **先稳定基础路径**：数据转换和训练可靠后，再增加自训练机制。
- **统一内部表示**：格式差异停留在输入解析和数据集输出边界。
- **组合优于分支堆叠**：用小型 Processor 组成任务流程。
- **配置适度**：配置表达稳定参数，代码承载复杂行为。
- **产物可追踪、可复现**：数据集和模型能关联到输入、任务与代码版本。
- **按需求抽象**：存在真实复用点后再引入后端、存储或调度抽象。

具体配置 schema、伪标签策略、数据版本管理、原生库接入和 Teacher–Student 算法均属于后续实现决策，不由本文预先规定。
