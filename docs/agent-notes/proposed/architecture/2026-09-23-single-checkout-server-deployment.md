# Agent Note: 单一源码目录的服务端部署

Status: proposed

## Problem

标注平台的服务端由 xxtrain、CVAT、ClearML Server 和代理共同组成。多份来源不明的源码、安装脚本、容器和现场目录使操作者无法判断运行版本与数据位置，也无法从仓库中的命令重建服务。服务重启和版本切换必须保留权威数据，同时使部署入口、依赖、权限和验收步骤可查证。

## Proposal

服务端在目标 Linux 主机上只有一个 Git 源码 checkout。版本由这个 checkout 的提交身份确定，不设多版本发布目录、worktree 或临时源码副本。部署从已经推送的 `origin` 获取指定分支、标签或提交；运行版本必须显示完整提交哈希，选择最新版本时先 `git fetch origin`，再明确选择远端引用指向的提交，不以本地分支名或目录修改时间推断。功能分支在同一源码目录测试，升级或降级在停止服务、确认工作树干净后切换到已取得的提交并运行 `uv sync --locked --extra platform --extra clearml`；不得自动清理或覆盖 `.deployment/`。只有经过本地修改、本地验证、本地提交及人工审阅后，才可按本次对话的明确批准推送；远端只通过同步后的代码部署和测试，不直接编辑代码或在远端提交。

被 Git 忽略的 checkout 内 `.deployment/` 是现场配置、凭据、工作区、SQLite、可重建缓存和容器持久数据的唯一现场目录。编排文件、镜像定制和服务管理实现留在受版本控制的源码内；容器使用含义明确的服务名和绑定挂载目录名，所有持久挂载均位于 `.deployment/`，不以 Docker 命名卷暗藏现场数据。Docker 镜像、容器及系统服务本身是运行资源，不是第二套现场数据。`install`、`start`、`stop` 不删除工作区或容器持久数据。平台注册 systemd 单元是对目录边界的唯一窄例外：单元文件位于系统规定位置，指向当前唯一 checkout；操作服务单元不意味着将现场数据写到系统目录。目标主机已具备 uv 和 Docker；安装 uv 或 Docker 不属于本流程，也不要求部署文档执行 `source .bashrc`。

xxtrain 从源码 checkout 的 uv 环境直接运行，使用 systemd 托管单进程并仅监听宿主机 `127.0.0.1`；不为 xxtrain 再建服务镜像。CVAT（含定制 UI）、ClearML Server 和独立 nginx 代理运行在 Docker 中。仅 nginx 提供浏览器及 Agent 所需的对外入口，后端端口只向宿主机 loopback 开放；代理通过宿主网络连接平台的 loopback 监听以及后端 loopback 映射，不依赖容器访问另一个网络命名空间中的 `127.0.0.1`。平台与 CVAT 保持同源路径。CVAT 页面、API、静态资源和升级连接经过代理；代理调用 xxtrain 的会话检查接口，未登录的浏览器只能取得登录必需的资源，不能直接访问 CVAT 内容。xxtrain 复用 CVAT 身份，具体 Job 数据权限由 CVAT 原生权限控制。ClearML 的 Web、API 和文件服务入口独立说明，供服务端和后续多台 Agent 使用，不把 Agent 安装纳入服务端操作。

唯一服务端管理入口与 `xxtrain train` 并列，为 `uv run --locked --extra platform --extra clearml xxtrain serverctl install|start|stop|status|verify`。`install` 负责锁定依赖同步、拉取外部服务镜像、构建项目定制镜像、生成未存在的配置并检查挂载及系统服务前置条件，不启动服务；重复运行不得覆盖手工配置或已有持久数据。`start` 对同一服务集合幂等启动并设置开机自启；`stop` 幂等停止并取消开机自启。systemd 管理整个服务集合的开机状态，容器不得有绕过 `stop` 的独立开机自启策略。`status` 展示 checkout 提交、工作树状态、systemd、容器和健康状态；`verify` 从真实对外入口及后端验证可用性，非零退出指出失败位置。首次 `start` 在 `.deployment/administrator` 写入明文管理员密钥；文件已存在时保持原样，允许操作者在启动前直接修改，后续启动不得生成新密钥覆盖它。文档必须说明明文文件的权限与备份责任、每个命令的前置条件和成功信号，不能要求操作者猜哪个脚本负责安装或启动。

`xxtrain serverctl` 注册 `install`、`start`、`stop`、`status` 和 `verify` 五个动作。尚未实现的动作以非零状态和诊断退出，不会报告成功；真实 `start` 实现负责调用 `ensure_administrator`。该辅助函数只在 checkout 内 `.deployment/administrator` 不存在时用独占创建和 `0600` 权限写入随机密钥，已存在文件保持字节与权限不变。`site_root` 通过 Git 查找 checkout 顶层，并拒绝解析到 checkout 外的 `.deployment` 路径。

生命周期由单个 `xxtrain-server.service` 管理：启动前仅对 `xxtrain-server` Compose 项目执行 `up -d --no-build`，前台运行 checkout 的平台进程，停止后仅停止该项目的容器。容器没有独立重启策略。`install` 重复执行锁定依赖同步、镜像拉取和定制镜像构建，并仅创建缺失的 `.deployment/platform.env` 私有空文件；不生成需要操作者提供的 `workspace.json` 和 `training.json`，也不启动服务。Linux 上明确设置 `XXTRAIN_INSTALL_SYSTEMD=1` 才会通过特权命令注册单元并重载 systemd；`start` 缺少任一配置文件即报错，不生成管理员身份标识或 CVAT 令牌。`start` 启用并启动单元，`stop` 停止并取消启用状态；无论停止命令是否失败均尝试取消启用。

服务端操作步骤属于独立的零起点部署指南；[平台部署与数据集存储](../feature/2026-09-14-platform-dataset-storage.md)继续拥有业务数据组织、主机与 Agent 的数据关系及后续方向。本次只实现服务端部署文档与运行验证。ClearML Agent 安装和 GPU 训练验收属于另一份后续文档，不以单机 Agent 成功冒充服务端验收。

## Verification and recovery

编排由 `deploy/server/compose.yaml` 的 `xxtrain-server` 项目持有，镜像与上游提交列在 `deploy/server/versions.md`；`compose_files` 返回此唯一文件。CVAT 服务、八个 worker、Postgres、Redis、Kvrocks、OPA、ClickHouse、Vector 与定制 UI，以及 ClearML API、Web、文件、异步删除、MongoDB、Redis、Elasticsearch 共用编排项目。持久挂载位于 `.deployment/`，nginx 使用宿主网络访问仅绑定 loopback 的后端；对外监听 CVAT 同源入口 8080 和 ClearML API 8008、文件 8081、Web 8082。代理允许平台及精确登录端点绕过 CVAT 会话子请求；CVAT `/assets/` 和其余内容通过 `/platform/internal/cvat-auth` 检查，平台登录资源由 `/platform/` 提供。测试在 Docker Compose 可用时检查解析后的挂载来源，无 Docker 时检查原始清单的挂载和端口约束。实际镜像拉取和 nginx 语法须在安装 Docker 的 Linux 环境继续核验。

Task 1 的焦点回归由 `uv run --locked --extra platform --extra clearml python -m unittest test.test_cli test.test_serverctl -v` 覆盖 CLI 保持既有子命令、动作注册与退出码、Git 根发现、部署目录边界及密钥首次创建和保留。完整入口与配置测试仍需覆盖首次和重复 `install/start/stop`、启用与取消自启、无外部持久挂载、版本显示、脏工作树拒绝切换，以及代理未登录拒绝和登录后 CVAT 内置双用户数据隔离。实现代码在本地提交后同步到测试机；在 `/home/lxx/xxtest` 内按零起点指南真实安装、启动、复启、验证、停止及恢复，并记录镜像拉取、系统服务、网络和浏览器现场问题及解决步骤。现场部署或版本切换失败时保留 `.deployment/` 和可核对的运行提交，使用 Git 选择已取得的旧提交、`uv sync --locked` 和正常启动入口恢复；不得以额外源码副本或新发布目录回避失败。

## Alternatives considered

**每个版本独立发布目录。** 并存目录可以保留多套代码，但最新版本、依赖环境、现场配置和实际运行指向又需要额外的指针与清理规则；单一 checkout 由 Git 直接表达提交身份。

**把 xxtrain 打包为服务容器。** 可隔离 Python 环境，但会再次维护镜像构建、容器内代码与现场挂载的对应关系；单进程 Python 服务在已有 uv 环境和 systemd 下更轻量，依赖由锁文件确定。

**使用 shell 脚本作为安装与启停入口。** shell 适合小范围系统命令，但项目已有 Python 包入口，配置检查、状态收敛和可测试错误更适合放在 `xxtrain serverctl`，不另设需要猜测的脚本矩阵。

**让 CVAT 直接暴露。** 直接暴露绕过平台登录门槛；代理先确认 CVAT 会话有效，Job 可见性仍依赖 CVAT 自身权限。

## Acceptance criteria

- 测试机仅在 `/home/lxx/xxtest` 内放置源码和 `.deployment/` 现场数据；平台注册 systemd 单元是经批准的唯一目录外窄例外，无需修改其他目录或维护第二份源码。
- 从仓库零起点文档可明确选择远端提交、执行 `install/start/status/verify/stop`、复启、升级和降级；目录、脚本、已安装依赖及当前版本无需猜测。
- 重复安装和启停不覆盖手工密钥、配置或持久数据，启动启用自启，停止取消自启；实际服务健康且只有代理暴露需要的入口。
- 未登录不能浏览 CVAT 数据；两个已登录用户各自只能访问其 CVAT 权限允许的内容；ClearML Server 可由服务端访问并为以后多 Agent 提供入口。
- 实际部署问题与解决方法进入服务端指南，服务端验证结果与尚未执行的 Agent/GPU 验收明确区分。

## Risks

- 单一可变 checkout 在切换提交时需要短暂停机；未提交改动必须先在本地处理，远端脏工作树不能强行覆盖。滚回旧提交不自动回滚数据库格式；如版本含不兼容迁移，须先核对兼容路径和数据备份。
- `.deployment/` 与源码同处 checkout，整目录误删会同时删除权威数据；部署命令禁止执行删除 checkout 或清理 Git 忽略文件，备份和恢复责任须在指南中明确。
- 明文管理员密钥必须限制文件权限与读取人群；代理登录门槛不替代 CVAT 的任务权限或公网 TLS。目标是受控测试机和局域网部署，公网安全入口需单独设计。
