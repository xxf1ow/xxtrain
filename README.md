# xxtrain

## What xxtrain does

xxtrain 是围绕 Ultralytics YOLO 的垂类视觉训练工具。它把 LabelImg、LabelMe、YOLO 或 COCO 标注转换为任务数据集，通过 Python Scenario 固定数据处理与训练参数，并提供训练、预测检查和 ONNX 导出入口。

## Current capabilities

- 支持 detect、segment、pose、classify 和 OBB 五种基础任务，以及由 Scenario 组合的 Point、knob、light 特殊任务；
- 使用格式无关的不可变标注值和 typed `Source -> Pipeline -> Sink` 转换流程；
- 生成训练/验证 split、Ultralytics 数据集描述和模型 YAML；
- 通过安装后的 `xxtrain train`、`xxtrain export` 和 `xxtrain review` 执行工作流；
- 对分类任务生成统一 `224×224` Letterbox 数据，并支持有标签错分检查和无标签预测归类。

当前仓库只包含本地训练核心，不包含用户平台、CVAT/ClearML 连接器或训练队列。完整边界见 [当前架构](docs/architecture.md)。

## Requirements

- uv；
- Python 3.12 或更高版本；
- 训练所需的本地 Python 与硬件环境；
- 一个导出 `SCENARIO: TrainingScenario` 的 Scenario 文件及其 `src/` 输入目录。

## Install

在仓库根目录同步包含开发依赖的环境并激活 `.venv`：

```powershell
uv sync --extra dev
.\.venv\Scripts\Activate.ps1
```

uv 会按照 `.python-version` 选择 Python，并以 editable 方式安装当前包。后续 `xxtrain`、`python` 和 `ruff` 命令都在已激活的环境中运行。

## Quick start

运行完整的转换、训练和 ONNX 导出流程：

```powershell
xxtrain train data/standard-detect/standard_detect.py
```

Scenario 的目录契约、独立导出和预测检查见 [训练工作流](docs/subsystems/training-workflow.md)。

## Documentation

- [架构](docs/architecture.md)：当前组成、运行流程和模块边界；
- [标注数据](docs/subsystems/annotation-data.md)：统一标注、几何、标签和格式边界；
- [数据集转换](docs/subsystems/dataset-pipeline.md)：发现、Processor、Sink 和转换报告；
- [训练工作流](docs/subsystems/training-workflow.md)：Scenario、训练、导出和预测检查；
- [开发](docs/development.md)：环境、日常工作流、格式化和 Scenario 管理；
- [测试](docs/testing.md)：测试层级、命令选择和文档检查。

## Project direction

项目计划在现有训练核心之外增加内部自助训练平台。该能力尚未实现；已批准方向、阶段边界和验收条件记录在 [内部自助训练平台 Agent Note](docs/agent-notes/proposed/feature/2026-07-30-self-service-training-platform.md)。
