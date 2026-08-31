# 训练工作流子系统

## Scope

`xxtrain.training` 负责加载 Python Scenario、准备数据集与模型配置、调用 Ultralytics YOLO、导出 ONNX 并检查预测结果。它不拥有标注解析和数据集写入细节；这些职责分别属于 [`xxtrain.data`](annotation-data.md) 和 [`xxtrain.pipeline`](dataset-pipeline.md)。当前精确包级导出面由 [`src/xxtrain/training/__init__.py`](../../src/xxtrain/training/__init__.py) 定义。

## Scenario contract

一个 Scenario 文件必须导出 `SCENARIO: TrainingScenario`。该不可变值包含一个 `DatasetRecipe`、模型版本与规模、split、`reserve_no_label` 以及传给 `YOLO.train()` 的覆盖参数；规模只能是 `n`、`s`、`m`、`l` 或 `x`。

加载器把 `train_args` 中的相对 `Path` 递归解析到 Scenario 文件目录，普通字符串保持不变。文件不存在、模块无法加载、缺少 `SCENARIO` 或导出类型错误时，加载在开始训练前失败。

## Directory layout

```text
<scenario_dir>/
├── <scenario>.py
├── src/
│   ├── <group>/
│   │   ├── imgs/
│   │   ├── labels/
│   │   ├── anns/
│   │   └── anns_seg/
│   └── labels.txt
├── <recipe.name>/
└── weights/
```

仓库的 `src/` 包目录与 Scenario 的 `src/` 输入目录没有关系。标准 Recipe 使用 `src/labels.txt`；特殊 Recipe 可以在 Scenario 中携带固定目录。`<recipe.name>/` 保存可重建数据集与模型 YAML，`weights/` 保存导出的 ONNX 和分类参考图。

## Conversion and rebuild semantics

训练仅在 `<recipe.name>/dataset.yaml` 不存在时调用 `convert_dataset()`。该文件是转换完成信号，但当前流程不比较源文件时间或内容；源数据变化后，调用者必须删除生成目录并重新转换。分类输出缺少 `dataset.yaml` 时同样视为不完整。

`split=N` 在 `N > 0` 时把每个来源组中索引能被 N 整除的图片放入验证集，其余放入训练集；`N <= 0` 时所有图片同时进入两个 split。`reserve_no_label` 默认是 `False`，只有零标注图片确实应作为训练样本时才启用。

## Training

训练生成任务对应的模型 YAML，从 `platformdirs.user_cache_path('xxtrain') / 'weights'` 准备预训练权重，然后调用 Ultralytics。分类默认使用 `epochs=72`、`batch=64`、`imgsz=224`、`scale=0.0`；其他任务默认使用 `epochs=80`、`batch=32`、`imgsz=640`，Scenario 覆盖在默认值之后合并。

训练完成后，工作流把 Scenario 文件复制到 Ultralytics run 目录以保留运行配置；存在 best checkpoint 时从它导出，否则从当前模型导出并报告缺失。Ultralytics YOLO 是唯一后端，不为假设中的第二框架维护抽象。

## Export and review

`xxtrain export` 从已有 checkpoint 独立导出带时间戳的 ONNX。分类模型同时从生成数据集的每个训练类别复制一张参考图片；类别目录没有可用图片时导出失败。

`xxtrain review` 是预测结果检查，不调用 `model.val()`。Detect、segment、pose 和 OBB 把预测可视化交给 Ultralytics 保存；分类默认从图片父目录推断真实类别并整理错分，`--unlabeled` 则把无标签图片按预测类别分目录保存。无标签模式只支持分类模型。

## Supported package API

调用者通过 `xxtrain.training` 包级导出加载 Scenario，并执行 train、export 或 review。模型模板生成、预训练权重准备、分类参考图和错分整理保持内部实现；CLI 是这些包 API 的薄编排层。

## Failures and limitations

当前工作流不会自动识别源数据变化，不提供独立的指标重算入口，也不管理训练队列、远程状态或历史输入快照。训练、导出和 review 直接使用本地 Scenario 与 checkpoint；自助平台提案中的 ClearML 和不可变快照尚未实现。
