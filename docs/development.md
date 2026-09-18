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

开发或部署 ClearML 训练适配时安装 `clearml` 可选依赖；服务地址和凭据使用 ClearML 支持的环境配置，不写入仓库配置：

```powershell
uv sync --extra dev --extra clearml
```

训练机的普通 ClearML Agent 使用预装且受控的 `/opt/xxtrain-agent` 环境。该环境同时安装当前构建 wheel 的 `clearml` extra 与 `clearml-agent==3.0.3`，并通过 `xxtrain-worker` 执行任务。`deploy/platform/training.example.json` 展示项目、队列、共享缓存根、平台元数据目录、Agent 运行目录和已安装 worker 路径；`deploy/platform/clearml-agent.example.conf` 展示普通 Agent 的已验证配置键。两个文件都不保存服务地址或凭据。

安装后从 `/opt/xxtrain-agent/bin/python` 导入 `xxtrain`、`clearml` 和训练依赖，并运行 `xxtrain-worker --help`。普通 Agent 默认另建任务虚拟环境；`python_binary` 只选择构建该环境的解释器。启动时把 `CLEARML_AGENT_SKIP_PIP_VENV_INSTALL` 设为已验证的预装解释器，ClearML Agent 便直接使用该环境，任务本身不从仓库、网络或 checkout 安装包。共享缓存根以只读方式提供给训练机，`run_root` 和 Agent 缓存目录保持可写且互相独立。服务地址及访问密钥通过 ClearML 官方环境变量或机器外部配置提供，不复制到示例文件。以下前台命令只监听一个共享队列；单个普通 daemon 同时执行一个任务，不使用 `--services-mode`、`--dynamic-gpus`、后台运行或开机自启：

```sh
CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=/opt/xxtrain-agent/bin/python \
  /opt/xxtrain-agent/bin/clearml-agent --config-file /etc/xxtrain/clearml-agent.conf daemon --foreground --queue training --gpus 0
```

部署前运行 `clearml-agent --version` 和 `clearml-agent --config-file /etc/xxtrain/clearml-agent.conf config` 检查实际安装版本与合并后的脱敏配置，并在清洁工作目录用上述解释器重复导入和 worker 帮助检查。ClearML Agent 3.0.3 的执行源码确认 `CLEARML_AGENT_SKIP_PIP_VENV_INSTALL` 可以指定直接执行任务的解释器；目标训练机仍需安装并现场核对 Agent 版本。启动 Agent 或设置服务、自启策略属于另行授权的部署操作。

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

## Point annotation portal

`deploy/platform/workspace.example.json` 展示完整配置字段；复制后修改现场 ID、名称、所属 CVAT 用户 ID、工作区与运行目录、CVAT 内部源地址。预先创建可写的 `workspace_dir/images`；服务在工作区根目录初始化 `annotations.db`。`runtime_dir` 独立于工作区，保存三个目标的可丢弃 Job 映射、共享裁剪和训练缓存，由服务按需创建。配置文件不得保存 CVAT 服务令牌。

页面按检测、分类、指针分割三行显示从 SQLite 事实派生的进度与操作资格。CVAT 返回页面使用当前标签页的 allowlisted 目标提示选择同步端点；运行目录中的 Job 和 frame 映射才是服务端来源权威。分类数量、指针点数、端点重合或越界时，页面说明业务规则及修正方法，保留返回状态并提供当前问题帧的修正链接；刷新可重试同步，不能通过请求体提交 Job、frame 或文件路径。

分类入口默认打开 Tag annotation；检测与指针分割使用 Standard。检测通过 CVAT 原生 Tag 工具选择 `无检测目标`，确认当前图片为有效负样本；删除该 Tag 撤销确认，不能同时保留检测框。该标签在新建检测 Task 时配置，旧 Task 需要完成回收后重建才能提供此入口。

重复真实部署验收时，必须从当前 checkout 运行 `python -m test.platform_fixture <authorized-parent> --owner-user-id <cvat-user-id> --receipt .superpowers/platform-acceptance/fixture.json` 新建独立工作区。fixture 创建 50 张有框原图，并为每个框写入明确的分类和指针标注；receipt 的 marker 必须匹配当前测试，并记录 `database_path`、每张图片的 `sample_id` 和三个步骤的初始标注 UUID。旧 receipt、旧数据库和旧运行目录不能复用于当前验收。原生身份操作使用正式 `CvatClient`、`AnnotationService` 和 repository，并在每次同步后从独立数据库回读断言；fixture、凭据、日志和截图只保存在忽略的 `.superpowers/` 或授权验收目录，不进入 Git。

在 PowerShell 中通过环境变量提供服务令牌，并把后端绑定到 loopback 或专用内网地址。进程固定使用一个 worker；示例端口 8000 是代理内网端口，不直接发布：

```powershell
$env:XXTRAIN_CVAT_SERVICE_TOKEN = '<service-token>'
xxtrain-platform --config '<workspace.json>' --host 127.0.0.1 --port 8000
```

启用训练 API 时另设 ClearML 官方环境变量 `CLEARML_API_ACCESS_KEY` 和 `CLEARML_API_SECRET_KEY`，并传入训练配置。工作区的 `runtime_dir/cache` 必须位于 `shared_root`，`metadata_dir` 必须位于可清理的工作区运行目录之外，`worker_script` 必须是可部署文件。`run_root` 是训练机写入独立运行产物的位置：

```powershell
$env:CLEARML_API_ACCESS_KEY = '<access-key>'
$env:CLEARML_API_SECRET_KEY = '<secret-key>'
xxtrain-platform --config '<workspace.json>' --training-config '<training.json>' --host 127.0.0.1 --port 8000
```

现场工作区位于 `/platform/`，用户训练历史位于 `/platform/training/`。训练页只显示当前 CVAT 会话用户的安全运行摘要；机器、队列和 ClearML 内部任务身份不在平台页面展示。

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
