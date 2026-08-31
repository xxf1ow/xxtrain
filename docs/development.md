# 开发指南

## Prerequisites

开发前需要安装 uv。根目录 `.python-version` 选择 Python 3.12，`pyproject.toml` 声明包支持 Python 3.12 及以上。源码采用 `src` 布局，构建和依赖元数据位于根目录 `pyproject.toml`；开发命令都从仓库根目录运行。

## Environment setup

创建或同步 `.venv`，以 editable 方式安装包和开发依赖，然后激活环境：

```powershell
uv sync --extra dev
.\.venv\Scripts\Activate.ps1
```

安装后使用 `xxtrain` 命令，不从 checkout 直接运行已删除的平铺脚本。本文及测试指南中的直接 `python`、`ruff` 和 `xxtrain` 命令都假定该环境已经激活。

## Repository layout

```text
src/xxtrain/       # 安装包源码
test/              # unittest 测试与受控 fixtures
data/              # 本地 Scenario 和数据，默认被 Git 忽略
docs/              # 项目权威文档与 Agent Notes
.superpowers/      # 本地规格、计划和工作记录，始终被 Git 忽略
```

仓库的 `src/` 是 Python 包目录；Scenario 的 `<scenario_dir>/src/` 是训练输入目录。两者名称相同但职责无关，开发和清理命令不得混淆它们。

## Daily workflow

修改前先确认当前工作树并保留无关用户改动。行为变更先运行 owning test，完成后按 [测试指南](testing.md) 的选择规则执行静态检查或完整测试；文档变更先更新事实所有者，再修复入口与引用。

本地提交只包含当前任务拥有的文件。提交或宣称可提交前检查完整差异并运行适用验证；任何 push、PR 创建、远程合并或远程 CI 触发都需要当前会话中的人工明确批准。

## Dependency changes

直接构建、运行和开发依赖在 `pyproject.toml` 中声明最低版本，完整解析结果由 `uv.lock` 固定。修改依赖声明后运行 `uv lock` 重新生成锁文件，并在提交前运行 `uv lock --check`；环境安装使用 `uv sync --locked --extra dev`，避免在验证时隐式改写锁文件。

## Scenario files

Scenario 是具体数据集转换与训练的组合根，目录契约见 [训练工作流](subsystems/training-workflow.md)。`data/` 默认被忽略；只强制添加经过审核的 Scenario 源文件，不添加原始数据、生成数据集、训练 run 或权重：

```powershell
git add -f data/<dataset>/<scenario>.py
```

生成目录可从同级 `src/` 重建。源数据变化后应删除对应生成目录再运行转换；当前工作流不会自动检测来源变化。

## Formatting

格式化受版本控制的包源码和测试：

```powershell
ruff format src test
```

检查或格式化被忽略的 Scenario 时显式关闭 respect-gitignore：

```powershell
ruff format --check --no-respect-gitignore data
```

## Documentation changes

当前架构、开发命令、测试规则、子系统契约和未来设计分别由 `docs/architecture.md`、本文、`docs/testing.md`、`docs/subsystems/` 和 `docs/agent-notes/` 所有。非平凡变更必须在同一逻辑变更中新增或更新 owning Agent Note；Superpowers 规格和计划只保存在 `.superpowers/`，不得进入 Git。

文档移动必须同时删除旧位置、创建新位置并修复所有仓库内入站引用。预算、相对链接和段落格式由 [测试指南](testing.md#documentation-checks) 中的项目文档校验命令检查。
