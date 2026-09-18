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

训练 HTTP、严格运行配置和组合根使用 `platform` 与 `clearml` 可选依赖；该检查通过真实 ASGI 认证与 CSRF 路径覆盖安全投影、用户归属、取消、重试和下载，并验证仅标注启动保持可用：

```powershell
uv run --locked --extra platform --extra clearml python -m unittest test.test_platform_training_http test.test_platform_training_config test.test_platform_http -v
```

Point 三模型联合回归通过正式 HTTPX CVAT 适配器、平台服务、SQLite 数据层和训练缓存运行，覆盖原生身份、对象级失效、重启、回滚重试和缓存可读性：

```powershell
uv run --locked --extra platform python -m unittest test.test_platform_point_workflow -v
```

负样本 Tag 的往返、撤销、初始化核对、类型约束和冲突原子性运行 `python -m unittest test.test_platform_negative_tags -v`；分类默认布局及修正链接同时运行 `test.test_platform_downstream_service` 和 `test.test_platform_browser`。训练空标签与门槛由 `test.test_platform_data`、`test.test_platform_service` 覆盖，真实 CVAT 工具显示仍需部署验收。

HTTP 测试通过 ASGI transport 调用真实 FastAPI 应用，覆盖 multipart 上传的权限、临时文件清理、解析与错误脱敏、严格目标路由及安全修正响应，并组合真实数据与服务组件验证图片接纳和标注同步；CVAT 网络使用受控响应。服务测试覆盖 SQLite 派生计数、50 张有框原图门槛、裁剪图完成条件、稳定对象身份、下游 Job 失效、同步失败顺序、重启恢复、跨线程读取和三个缓存发布。离线页面测试检查包内 HTTP 资源，并在 Node.js 可用时执行页面脚本与 CVAT 返回插件，验证上传、三行计数、写操作互斥、逐行缓存按钮、每标签页返回目标、修正链接及同步失败后刷新重试；缺少 Node.js 时明确跳过。真实 CVAT 的 tag/polyline 编辑与返回修正、浏览器布局、镜像注入和代理连通性属于部署验收。

运行 Point 上传和检测缓存增量的完整离线验证：

```powershell
uv run --locked --extra platform python -m unittest test.test_platform_data test.test_platform_service test.test_platform_downstream_service test.test_platform_http test.test_platform_cvat_client test.test_platform_cvat_codec test.test_platform_browser test.test_task_transforms -v
```

真实验收显式安装 `platform-test` 并同时提供 `XXTRAIN_PLATFORM_URL`、`XXTRAIN_PLATFORM_TEST_USER` 和 `XXTRAIN_PLATFORM_TEST_PASSWORD`；未配置时浏览器测试在启动浏览器或访问网络前跳过。测试固定读取 `.superpowers/platform-acceptance/fixture.json`，只接受当前 marker、独立生成的临时根目录，以及根目录内的数据库、图片和基准路径。receipt 必须记录 `database_path`、50 张图片的 `sample_id` 和三个步骤的初始标注 UUID，测试通过正式 repository 回读保存结果。验收后停用本次启动的临时服务：

```powershell
uv run --locked --extra platform --extra platform-test python -m unittest test.test_platform_browser -v
```

同版本真实验收覆盖 tag 唯一选择、polyline 多指针、原生身份往返、裁剪来源和 frame 关联、分类数量及 line 点数修正、返回目标路由和两个下游缓存可读性。保存返回必须等待目标同步成功、返回参数清除且目标无错误后才能回读数据库；离线页面覆盖延迟与快速同步。改类后通过原生绘制补回该框被清除的指针；非法对象用 PATCH create 注入，回收检查当前 Job 的绑定。全量 PUT 预期按删除和新增处理，不要求坐标相同的对象保留身份。完整 fixture 的投影、同步和缓存消费者，以及真实服务到 HTTP、页面的业务原因均有离线验证。当前运行证据、范围和未覆盖项记录在 [Point 分类与指针分割 Agent Note](agent-notes/implemented/feature/2026-09-16-point-classification-segmentation.md#implementation)；离线 HTTPX fixture、旧 LabelMe 阶段的浏览器记录和 CVAT 源码检查不能替代同版本现场证据。

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
