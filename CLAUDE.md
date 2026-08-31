# CLAUDE.md

本文件是 Claude Code 在本仓库中的精简指令入口。项目事实和操作命令由链接的权威文档所有，不在此重复维护。

## Repository constraints

- Superpowers 生成的规格、计划、问题记录和其他工作产物必须保存在被 Git 忽略的 `.superpowers/`，不得暂存或提交；
- 不为本仓库创建或使用 Git worktree，所有工作在当前 checkout 完成；
- 保留与当前任务无关的用户改动，编辑范围必须能追溯到已批准请求；
- 未经当前会话中的人工明确批准，不执行 push、PR 创建、远程合并或其他会触发远程状态与 CI 的操作；
- 未实现的平台能力只能记录在 proposed Agent Note；对应阶段设计批准前，不创建 SQLite schema、平台包、CVAT/ClearML 连接层、完整 UI、身份系统或其他平台抽象；
- 非平凡的行为、架构、流程、工具、测试策略或持久格式变更必须在同一逻辑变更中新增或更新 `docs/agent-notes/` 下的 owning Note。

## Project authorities

- [当前架构](docs/architecture.md)：已实现组成、运行流、模块边界和扩展点；
- [开发指南](docs/development.md)：环境、日常工作流、格式化、Scenario 和文档维护；
- [测试指南](docs/testing.md)：测试层级、命令选择、fixtures、静态检查和文档检查；
- `docs/subsystems/`：标注数据、数据集转换和训练工作流的稳定契约；
- `docs/agent-notes/`：提案、已实施决策、取舍、后果和验收依据。

修改当前事实时先更新其所有者，再修复入口文档和引用。无法从代码、测试或权威文档确认的事实必须停止并询问，不得自行推定。
