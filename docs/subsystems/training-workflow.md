# 训练工作流子系统

## Scope

`xxtrain.training` 负责加载 Python Scenario、准备数据集与模型配置、调用 Ultralytics YOLO、导出 ONNX 并检查预测结果。它不拥有标注解析和数据集写入细节；这些职责分别属于 [`xxtrain.data`](annotation-data.md) 和 [`xxtrain.pipeline`](dataset-pipeline.md)。当前精确包级导出面由 [`src/xxtrain/training/__init__.py`](../../src/xxtrain/training/__init__.py) 定义。

## Scenario contract

一个 Scenario 文件必须导出 `SCENARIO: TrainingScenario`。该不可变值包含一个 `DatasetRecipe`、模型版本与规模、split、`reserve_no_label` 以及传给 `YOLO.train()` 的覆盖参数；模型默认值与规模集合由训练设置共用，规模只能是 `n`、`s`、`m`、`l` 或 `x`。

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

`train_prepared()` 直接消费已发布的数据集目录。它在独立 `run_dir` 中以配置模型名生成 YAML，并生成 split 列表、标签副本和一次性图片副本；模型 basename 使 Ultralytics 选择 `model_scale` 对应的结构，其他副本使图片修复、cache、run、checkpoint 和 ONNX 写入都不触及发布树。每次运行仍只从共用默认权重缓存初始化，不使用历史训练 checkpoint。训练回调以从 1 开始的已完成 epoch 上报进度；best checkpoint 存在时，导出和验证指标均来自该 checkpoint。这些副本增加每次运行的传输与磁盘占用，但只属于当次 run，不是持久从机数据缓存。

图片列表保留缓存入口路径：图片可以是软链接，配套 `.txt` 标签仍从该入口旁读取，不能先解析图片链接再推导标签路径。`on_validation` 只在训练中的实际验证完成后返回该轮 epoch 和新指标；跳过验证的轮次不触发，最终 best checkpoint 的指标由返回值提供。ClearML worker 在每 5 轮的验证完成事件上更新主要指标，训练完成后以实际交付模型的验证结果覆盖；验证频率改为每 10 轮时，中间轮次保留上次已上报结果，不复制旧指标为新观测，也不额外执行验证。

## Export and review

`xxtrain export` 从已有 checkpoint 独立导出带时间戳的 ONNX。分类模型同时从生成数据集的每个训练类别复制一张参考图片；类别目录没有可用图片时导出失败。

`build_delivery()` 将单模型交付复制为 `model.onnx`。分类交付生成 ZIP，其中只有 `model.onnx`、按模型输出索引排序的 `labels.txt` 和 `references/<output_index>_<label>.<ext>`；索引目录名还原为业务标签，任何输出类别缺少参考图都使交付失败。

`xxtrain review` 是预测结果检查，不调用 `model.val()`。Detect、segment、pose 和 OBB 把预测可视化交给 Ultralytics 保存；分类默认从图片父目录推断真实类别并整理错分，`--unlabeled` 则把无标签图片按预测类别分目录保存。无标签模式只支持分类模型。

## Supported package API

调用者通过 `xxtrain.training` 包级导出加载 Scenario，并执行 train、export 或 review。模型模板生成、预训练权重准备、分类参考图和错分整理保持内部实现；CLI 是这些包 API 的薄编排层。

## Failures and limitations

当前工作流不会自动识别源数据变化，也不管理训练队列、远程状态或历史输入快照。Scenario 入口仍直接使用本地数据与 checkpoint；平台适配已离线实现 ClearML 提交、远程状态和任务产物传输，真实 Agent/GPU 执行仍待验收。
