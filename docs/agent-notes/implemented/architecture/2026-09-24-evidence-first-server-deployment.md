# Agent Note: 实测优先的单一源码目录服务端部署

Status: implemented

## Problem

xxtrain、CVAT、ClearML Server 和入口代理共同构成标注平台服务端。来源不明的多份源码、现场目录、容器和脚本使运行版本、数据位置及启停入口无法判定。仅凭纸面编排实现自动化，会把未验证的镜像配置、身份初始化及启动时序当作既定事实；每次现场失败又迫使文档与代码来回修补，不能形成可靠的零起点部署流程。

## Decision

服务端在目标 Linux 主机的 `/home/lxx/xxtest/xxtrain` 使用唯一 Git checkout；Git 提交确定运行版本，不建发布版本目录、worktree 或源码临时副本。正式部署由维护者审阅并批准发布提交，目标机 `git fetch origin` 后选择远端引用的确定提交；不得凭本地分支或目录时间猜最新版本。升级时先停服并准备一致性数据备份，确认工作树干净，再切换目标提交并同步锁定依赖。不提供降级流程或旧提交与新数据库的兼容承诺。

唯一现场数据目录为 checkout 内被 Git 忽略的 `.deployment/`：配置、私有凭据、工作区、SQLite、运行缓存及容器持久绑定目录均放在其中；镜像与容器不是数据目录，不能用隐蔽的 Docker 命名卷代替现场绑定目录。测试机不直接修改或提交受版本控制的代码，也不使用另一份源码运行测试。除注册和管理指向该 checkout 的 systemd unit 外，测试机文件操作只在 `/home/lxx/xxtest` 内。源代码、镜像和其他项目不属于部署试验清理范围；清理前核对容器归属、挂载及目标真实路径，仅清除本项目容器与本项目 `.deployment/`，不得延伸到其他数据。正式升级保留 `.deployment/`，不得按试验清理程序操作。

宿主机从当前 checkout 的 uv 锁定环境运行单进程 xxtrain，由 systemd 托管且只监听 `127.0.0.1`。CVAT、ClearML Server 及独立 nginx 使用 Docker Compose；只有 nginx 在指定的受控内网地址监听对外入口，所有后端仅向宿主机 loopback 映射，不使用默认的全网卡端口发布。平台和 CVAT 同源：平台登录向 CVAT 创建浏览器会话并返回其 Cookie，代理向 xxtrain 核验该 Cookie 后原样转发给 CVAT；不另设平台身份、公共 CVAT 会话或代理注入 SSO 身份。具体 Job 可见性由 CVAT 内置权限决定，不建设按已领取 Job 动态授权。ClearML 的 Web、API 与文件端点独立提供给服务端及以后接入的 Agent，不经过平台登录门槛，只允许在受控网络开放，不能直接发布公网。部署服务端不安装 ClearML Agent，不以服务端验收替代 GPU/Agent 验收。[平台部署与数据集存储](../../proposed/feature/2026-09-14-platform-dataset-storage.md)继续负责业务数据组织、主机与训练机数据关系及后续 NFS 方向。

唯一管理入口与 `xxtrain train` 并列：`uv run --locked --extra platform --extra clearml xxtrain serverctl install|start|stop|status|verify`。操作者预先提供唯一 checkout 与目标提交、可用的 uv 和 Docker Compose、仓库及依赖和镜像源的网络访问、所需仓库认证、受控内网地址以及执行必要特权操作的 sudo 权限；部署命令不安装这些工具、不修复主机网络或 sudo 授权，也不代替操作者推送或拉取源码。服务间通信、端口和身份配置由本仓库负责。`install` 幂等同步锁定环境、拉取外部镜像、构建定制镜像、准备本地配置及离线凭据并注册 unit，但不启动服务；运行中才能签发的身份由 `start` 自动完成。不要求操作者交互式创建账号、密钥或配置。`serverctl` 由普通用户执行，通过 sudo 调用必要系统命令；需要密码时可在同一交互会话提前运行 `sudo -v`，不改写主机 sudo 授权。`start` 启用并重启 unit，等待真实入口可用；`stop` 停服并取消开机自启。systemd 是唯一开机自启控制者，Compose 容器不设置自动重启策略。systemd 启动与人工 `start` 使用同一套容器、数据库迁移、身份引导及平台前台路径；人工 `start` 额外等待对外入口核验完成，unit 的 `Type=simple` 本身不证明入口就绪。`status` 展示实际提交与服务状态；`verify` 从真实入口检查服务可用性。重复安装、启停和升级保留已有凭据及持久数据，部署命令不删除它们。

CVAT 管理员用户名固定为 `xxadmin`，该账号也可从平台登录；唯一可读密码来源是以 `0600` 权限明文保存的 `.deployment/.xxxxx`。`install` 与 `start` 共用缺失时随机创建、存在则保留的逻辑；每次 `start` 在 CVAT 数据库就绪后通过容器内管理入口创建或同步管理员密码，无须旧密码。启动前修改文件即可指定下次使用的密码；单独丢失文件时下次启动重新生成并同步，不等同于恢复丢失的数据库。CVAT 服务令牌与 ClearML 平台 API 密钥是独立的机器凭据，由部署流程无人值守地创建并私下持久保存；管理员密码不复用为机器凭据。ClearML Agent 的密钥签发和分发属于后续 Agent 部署设计。

## Runtime and verification

服务端操作者按[部署指南](../../../cookbook/server-deployment.md)从唯一 checkout 安装、启停、备份和升级。现场试验先验证镜像配置、数据库迁移、身份签发、代理 Cookie 及停止顺序，再把确认的行为收敛到同一控制器；本地 Git bundle 仅传送测试提交，不代表经审阅的 `origin` 发布。

生命周期控制器以 systemd `ExecStartPre` 执行本地 `prepare`，再由 `foreground` 启动同一 Compose 项目、完成身份引导并以锁定虚拟环境运行平台。引导在单调时钟期限内重试 CVAT `manage.py migrate --check`，通过容器管理命令从标准输入同步 `xxadmin` 密码，再用 CVAT 登录与 `/api/users/self` 验证并取得实际用户 ID；有效的既有服务令牌经验证后沿用，缺失或失效时才重新签发。ClearML 就绪要求服务密钥通过 `/auth.login` 认证。CVAT 服务令牌写入私有 `platform.env`，新建工作区以该 CVAT 用户 ID 为 owner，`workspace_id` 和 `display_name` 均使用 `xxtrain`；已有 `.deployment/workspace.json` 保持不变，仅校验 owner ID 与 `xxadmin` 的 CVAT ID 一致。平台进程的 ClearML SDK 配置与缓存、XDG 缓存以及代理旁路名单均限定在 `.deployment` 或本机服务地址。人工 `start` 启用并重启同一 unit，在 600 秒单调时钟期限内检查真实入口就绪；`stop` 停止后禁用该 unit；systemd 的停止后处理只停止 `xxtrain-server` Compose 项目，允许最长 300 秒供容器优雅停止。`status` 展示完整 Git 提交和各服务状态，`verify` 通过 loopback 登录 CVAT，显式把得到的会话 Cookie 转发到受控内网入口，检查平台、CVAT 与 ClearML。

离线测试覆盖配置保留、身份引导、权限边界、生命周期、CLI 和失败退出。实际测试主机经 bundle 导入精确提交后，重复 `install` 保持私有凭据和工作区、且不启动容器；`start` 在平台、CVAT 和 ClearML 入口通过核验后返回，管理员密码编辑后重启使旧密码失效、新密码可登录。两名 CVAT 普通用户各自任务返回 200，互访返回 403，管理员工作区返回 403；匿名 CVAT API 返回 401。24 个项目容器运行，代理四个端口仅绑定指定私有内网 IP，平台和后端仅绑定 loopback。`stop` 完成后 unit inactive/disabled 且项目容器全停；unit restart 复用同一引导路径。未执行整机重启、Agent/GPU 验收或正式 `origin` 发布提交的目标机验证。

## Alternatives considered

**先完成正式文档与自动化，再逐个现场修补。** 镜像、权限及启动依赖的关键细节尚需真实运行证据；先手工打通后再编码可以把已确认的操作固化为单一入口。

**每个版本独立目录或 worktree。** 并存源码让操作者重新猜测运行指向，并积累配置与现场副本；单一 checkout 已能用 Git 精确选择版本。

**把 xxtrain 也打包成容器。** 本服务可直接使用既有 uv 锁定环境和 systemd；额外镜像增加源代码、环境和持久挂载之间的对应关系，不符合当前轻量运行目标。

**用独立 shell 脚本矩阵管理生命周期。** Python CLI 与既有 `xxtrain train` 并列，更适合表达可测试的配置、身份、状态和就绪逻辑；不另建需要猜测的安装及启动脚本。

## Consequences

单一 checkout 和 `.deployment/` 明确运行版本与数据位置，代价是升级必须停机并先做一致性备份；Git 切换不恢复数据库，也没有降级兼容承诺。明文管理员密码必须保持 `0600` 并纳入备份。代理会话门槛不替代 CVAT 数据权限、公网 TLS 或 ClearML 的受控网络边界。

测试主机的 bundle 检查证明当时的镜像和服务端路径，不等于正式从 `origin` 获取经审阅提交后的验收。整机启动时的 unit 自启尚未通过重启主机验证；服务端验收不覆盖 ClearML Agent、GPU、真实训练提交或训练机共享存储。
