# 测试指南

## Scope

本页定义测试层级、命令选择、fixtures、人工审核快照、静态检查和文档检查。测试应观察实际包或 CLI 行为，不把模型报告、文件存在或实现细节当作更强行为的替代证据。

## Environment

本页中的直接 `python`、`ruff` 和 `xxtrain` 命令都要求先运行 `uv sync --extra dev` 并激活根目录 `.venv`。非交互式单次执行可以使用 `uv run --locked <command>`；已激活环境中的直接命令使用同一解释器和工具，是本文的标准写法。

## Test tiers

- **Owning test**：覆盖一个模块或局部契约，适用于普通行为修改和缺陷修复；
- **Consumer check**：公共或共享接口变更除 owning test 外，还要通过一个真实调用方；
- **Full suite**：适用于无法缩小的跨模块交互、CI 诊断或明确要求；
- **Static checks**：验证 Ruff、格式和 Python 编译；
- **Documentation checks**：验证 Agent Note 生命周期、字数预算、相对链接、标题片段和差异格式。

## Command selection

只运行能证明改动安全的最小证据集。局部 Python 行为运行对应 `test.test_*` 模块；包级公共接口同时运行使用它的测试；跨 data、pipeline 与 training 的工作流变化运行完整 suite。纯文档变化不要求无关 Python 单元测试，但必须运行两个文档校验器、适用的泄漏与重复扫描以及 `git diff --check`。

## Unit tests

运行一个 owning test 模块，例如：

```powershell
python -m unittest test.test_pipeline_core -v
```

运行完整 suite：

```powershell
python -m unittest discover -s test -t . -p 'test_*.py' -v
```

测试从 `test/` 作为包导入当前 `src/xxtrain` 安装，不使用兼容适配层代替真实公开入口。

## Static checks

```powershell
ruff check src test
ruff format --check src test
ruff check --no-respect-gitignore data
ruff format --check --no-respect-gitignore data
python -m compileall -q src test data
```

Windows 下若 .NET/MSBuild 构建进程同时收到 `Path` 和 `PATH`，应先规范化启动环境；该症状属于 Codex 进程环境冲突，不是仓库或编译器缺陷。

## Fixtures and reviewed snapshots

`test/fixtures/` 保存可重复的输入场景，不参与普通文档编辑。`test/expected/conversions/` 保存人工审核的数据转换语义快照；输出变化必须由 owning 测试重新生成或更新，并进行语义审核，不能只接受文本差异。

测试产生的临时目录和生成数据集不进入仓库。Scenario 的 `src/` 输入是权威数据，测试清理不得删除或修改它。

## Documentation checks

Agent Note 格式和生命周期：

```powershell
python "$env:USERPROFILE\.codex\skills\agent-notes\scripts\validate_agent_notes.py"
```

项目文档预算、段落、相对链接和源码引用：

```powershell
python "$env:USERPROFILE\.codex\skills\doc-standards\scripts\validate_project_docs.py"
```

完成文档编辑后还要按 `trim-cot-leakage` 的 recall batteries 检查不可解析的设计编号、评审过程、版本叙事和无主推迟，并运行：

```powershell
git diff --check
```

## CI relationship

仓库当前没有受版本控制的 CI 配置。上面的命令是本地权威验证入口；将来增加 CI 时，应调用这些相同命令或其明确子集，并在本页记录选择关系。
