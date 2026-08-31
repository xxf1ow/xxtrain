# 数据集转换子系统

## Scope

`xxtrain.pipeline` 把来源样本转换为训练框架可消费的数据集。它拥有稳定发现顺序、阶段类型、Processor 基数、转换报告和唯一写入边界；标注值由 [`xxtrain.data`](annotation-data.md) 拥有，Scenario 与训练编排由 [`xxtrain.training`](training-workflow.md) 拥有。当前精确包级导出面由 [`src/xxtrain/pipeline/__init__.py`](../../src/xxtrain/pipeline/__init__.py) 定义。

## Data flow

```text
SampleSource
  → Sample / ImageRef
  → Pipeline[ItemProcessor | ExpandProcessor]
  → sink-specific output records
  → DatasetSink
  → ConversionReport
```

`Sample`、`ImageRef`、`Context`、配置、报告及各阶段输入输出都是 typed records，不使用共享 payload 字典。`DatasetRecipe` 组合任务名、基础 `TaskType`、可选标签目录、Source、Pipeline 和 Sink，并在构造时验证来源输出、相邻 Processor 与 Sink 输入的类型边界。

## Discovery contract

`DirectorySource` 从 `<scenario_root>/src/<group>/imgs/` 按组名和图片名稳定排序发现样本，不递归搜索任意目录。标注读取以图片为中心，固定候选为 `labels/<name>.txt`、`anns/<name>.xml` 和 `anns_seg/<name>.json`；缺失候选表示该格式没有标注，不会为同一格式加载多个文件。

Pose 可以组合 LabelImg 框和 LabelMe 关键点。其他任务遇到多个非空格式时拒绝歧义输入。`CocoSource` 按 COCO 图片记录读取，并校验显式目录与 schema 派生目录一致；Pose COCO 还要求单一 category 和统一关键点 schema。

## Processor semantics

`ItemProcessor` 把一个输入映射为零或一个输出，`ExpandProcessor` 把一个输入映射为多个有序输出。每个样本和裁剪实例使用独立不可变值，Processor 只负责解析、过滤、几何变换、父子匹配、裁剪描述或编码中的一个职责；Recipe 负责组合，训练入口不堆叠任务分支。

裁剪 Processor 只把 crop box 写入 `ImageRef`，不立即读取或修改图片。该延迟边界保留浮点坐标并避免中间阶段产生临时文件；Sink 在需要写出训练图片时才执行栅格切片。

## Sink and output contract

Sink 是数据集图片、标签、split 列表和 `dataset.yaml` 的唯一写入者。Whole-image 输出优先创建 symlink，失败时回退到 `shutil.copy2`；裁剪输出由 Sink 物化。分类 Sink 把整图或裁剪统一处理为居中的 `224×224` Letterbox 图片，插值使用 OpenCV linear，padding 值为 114。

`convert_dataset(recipe, root_path, *, split=10, reserve_no_label=False)` 解析标签目录，运行 Pipeline，逐项写入 Sink，验证最终目标类别覆盖，完成 Sink 并返回 `ConversionReport`。报告统计来源标签、过滤标签、输出标签、缺失标注和产物；目标类别经过完整处理链后没有输出样本时转换失败。

## Supported package API

调用者通过 `xxtrain.pipeline` 包级导出使用稳定的组合值与 `convert_dataset`、`standard_recipe`。具体 Source、阶段记录、Processor、Sink 和辅助函数仍是实现细节；仓库内受控 Scenario 可以从定义模块导入它们，但这不构成通用第三方插件承诺。

`standard_recipe()` 只提供 detect、segment、pose、OBB 和 classify 五条基础管线。Point、knob 和 light 的特殊流程由各自受版本控制的 Scenario 组合，不存在任务名注册表或通用自定义任务发现机制。

## Failures and limitations

缺失输入目录、分类目录与标签不一致、任务不支持的几何、相邻阶段类型不兼容、多个歧义标注格式、Recipe 与来源目录不一致或最终类别无样本都会尽早失败。生成目录是可丢弃产物；转换不得修改 Scenario 的 `src/` 权威输入。
