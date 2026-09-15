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

开发 Point 检测标注入口时同时安装 `platform` 可选依赖：

```powershell
uv sync --extra dev --extra platform
```

## Repository layout

```text
src/xxtrain/       # 安装包源码
test/              # unittest 测试与受控 fixtures
deploy/platform/   # Point 入口的 CVAT UI 镜像、同源代理和示例配置
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

## Point detection portal

`deploy/platform/workspace.example.json` 展示完整配置字段；复制后只修改现场 ID、名称、所属 CVAT 用户 ID、三个本地路径和 CVAT 内部源地址。配置文件不得保存 CVAT 服务令牌。图片目录只读，标注目录和状态文件父目录必须可写。

在 PowerShell 中通过环境变量提供服务令牌，并把后端绑定到 loopback 或专用内网地址。进程固定使用一个 worker；示例端口 8000 是代理内网端口，不直接发布：

```powershell
$env:XXTRAIN_CVAT_SERVICE_TOKEN = '<service-token>'
xxtrain-platform --config '<workspace.json>' --host 127.0.0.1 --port 8000
```

正式 CVAT UI 镜像固定使用 2.51.0，并在构建时把导航隐藏样式和返回插件插入 `index.html`。基础 HTML 缺少唯一的 `head` 插入点时构建失败：

```powershell
docker build --file deploy/platform/cvat-ui.Dockerfile --tag xxtrain-cvat-ui:2.51.0 .
```

`deploy/platform/nginx.conf` 是唯一对浏览器开放的同源代理配置。部署编排需在同一私有网络提供 `xxtrain_platform:8000`、`cvat_server:8080` 和 `xxtrain_cvat_ui:8000`，把选定的外部端口映射到代理的 8080，而不发布后端或专用 CVAT UI。代理保留带端口的 Host，向 CVAT 响应添加隔离头，并为最长 120 秒的任务准备留出 130 秒读写超时。启动或恢复远程 CVAT 属于单独的受控验收步骤。

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
