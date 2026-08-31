# Agent Note: uv Python 环境与依赖锁定

Status: implemented

## Problem

项目使用 `pyproject.toml`、`src` 布局、setuptools 构建后端和 `xxtrain` 命令入口定义 Python 包，但开发文档曾要求通过 pip 安装，仓库也没有声明 uv 使用的 Python 版本或提交依赖锁文件。不同开发环境可能选择不同的 Python 和依赖版本，文档中的环境入口与实际使用方式不一致。

## Decision

uv 是项目的 Python 环境和依赖管理入口。根目录 `.python-version` 选择 Python 3.12，`pyproject.toml` 声明包支持 Python 3.12 及以上，使开发解释器和依赖兼容范围一致。

`pyproject.toml` 中的构建依赖、运行依赖和 `dev` 可选依赖采用决策时开发环境中的版本作为 `>=` 下限。`uv.lock` 提交完整解析结果并记录精确版本；依赖声明表达允许范围，锁文件保证仓库环境可复现。

开发环境通过 `uv sync --extra dev` 创建和同步。README 与开发指南说明激活 uv 管理的 `.venv`，测试指南中的 `python`、`ruff` 和 `xxtrain` 命令都以该环境已经同步并激活为前提，不要求为每条命令重复添加 `uv run`。非交互式单次执行可以使用 `uv run --locked`，但它不是文档中的默认命令形式。

现有 `[build-system]`、`[project]`、`[project.scripts]` 和 `src` 包发现配置构成可安装 Python 包，不增加重复的 `[tool.uv] package = true`。

源码中的公开类型别名和 Processor 泛型使用 Python 3.12 的类型参数语法，使 Ruff 按项目最低 Python 版本检查时无需保留旧版 `TypeAlias` 和 `Generic` 声明。

## Dependency bounds

- 构建依赖：`setuptools>=83.0.0`；
- 运行依赖：`numpy>=2.5.1`、`opencv-python>=5.0.0.93`、`Pillow>=12.3.0`、`platformdirs>=4.11.0`、`ruamel.yaml>=0.19.1`、`tqdm>=4.70.0`、`ultralytics>=8.4.110`；
- 开发依赖：`PyYAML>=6.0.3`、`pycocotools>=2.0.11`、`ruff>=0.15.22`。

`numpy>=2.5.1` 要求 Python 3.12 及以上，因此项目不能同时保留 Python 3.11 兼容声明和全部当前环境版本下限。项目选择 Python 3.12 作为包的最低版本。

## Alternatives considered

**所有命令都通过 `uv run --locked` 执行。** 该形式能显式选择并检查项目环境，适合非交互式任务，但会让每条日常测试和格式化命令重复相同前缀。开发者已经激活同步后的 `.venv` 时，直接命令使用同一解释器和工具。

**继续使用 pip，只提交依赖下限。** 该方案会保留原有文档命令，但不能提供统一的解释器选择和完整依赖解析结果，也不能满足项目采用 uv 管理环境的要求。

**增加 `[tool.uv] package = true`。** uv 可以用该选项强制包模式，但现有构建系统已经表明项目需要构建并安装，额外声明不会增加包能力。

**保留 Python 3.11 支持并降低 NumPy 下限。** 该方案会扩大 Python 兼容范围，但不能以当前开发环境中的依赖版本作为全部直接依赖的下限。

## Consequences

开发者通过一个命令同步包、开发工具和精确依赖解析结果，激活环境后可以使用简洁的直接命令。依赖声明仍允许兼容更新，而锁文件使共享开发环境可复现。

锁文件必须在依赖声明变化时重新生成；只修改 `pyproject.toml` 而不更新 `uv.lock` 会被锁文件检查拒绝。直接命令依赖开发者先激活项目 `.venv`，未激活时可能调用系统解释器。项目要求 Python 3.12 及以上，不再声明对 Python 3.11 的兼容性。

## Verification

`uv lock --check` 验证锁文件与包元数据一致，`uv sync --locked --extra dev` 验证锁定环境可以安装。同步后的解释器从 `src/xxtrain` 导入包，安装后的 `xxtrain --help` 入口输出 train、export 和 review 子命令。完整 Python 测试、Ruff、编译、Agent Note 校验、项目文档校验和差异格式检查共同验证该决策。
