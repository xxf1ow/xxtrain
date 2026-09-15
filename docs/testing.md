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

Point 检测入口的 HTTP 与离线页面检查使用 `platform` 可选依赖：

```powershell
uv run --locked --extra platform python -m unittest test.test_platform_http -v
uv run --locked --extra platform python -m unittest test.test_platform_browser -v
```

HTTP 测试通过 ASGI transport 调用真实 FastAPI 应用，其中一个成功路径组合真实 `AnnotationService`、`WorkspaceData`、`StateStore` 和 `CvatClient`，只在 CVAT 网络边界使用受控响应。离线页面测试通过包内 HTTP 资源检查布局，并在 Node.js 可用时执行页面返回同步和 CVAT 返回插件；缺少 Node.js 时这两项脚本检查明确跳过。真实 CVAT、浏览器布局、镜像注入和代理连通性属于部署验收，不由离线测试替代。

真实验收显式安装 `platform-test` 并提供全部三个环境变量；未配置时浏览器测试在启动浏览器或访问网络前跳过。只可使用独立生成的临时 fixture，并在验收后停用临时服务：

```powershell
uv run --locked --extra platform --extra platform-test python -m unittest test.test_platform_browser -v
```

修改平台 package-data 或入口后构建 wheel，并检查 wheel 包含三个页面资源、CVAT 返回插件及 `xxtrain-platform` console script。CVAT UI 基础镜像已在本机存在时，可以离线运行以下构建检查；该命令不得作为恢复或启动 CVAT 服务的替代授权：

```powershell
uv build --wheel --out-dir .superpowers/task-5-dist
docker build --file deploy/platform/cvat-ui.Dockerfile --tag xxtrain-cvat-ui:2.51.0 .
```

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
