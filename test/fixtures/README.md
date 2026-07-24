# Baseline test datasets

这些目录是从 `data/` 中筛选出的最小测试夹具。每组通常只保留两个样本，既能覆盖转换流程，也能形成训练集和验证集。所有 LabelMe JSON 标注统一存放在 `anns_seg/`。

| 目录 | 覆盖任务 | 内容范围 |
| --- | --- | --- |
| `standard-detect/` | 标准目标检测 | 图片、LabelImg XML 和类别表 |
| `standard-segment/` | 标准实例分割 | 图片、LabelMe JSON 和类别表 |
| `standard-pose/` | 标准姿态估计 | 图片、检测框 XML、关键点 JSON 和类别表 |
| `standard-classify/` | 标准图像分类 | 按类别分目录的图片和类别表 |
| `point/` | 点位检测、分类、分割 | 同一批图片及其 XML、JSON 标注 |
| `knob/` | 旋钮检测、分割 | 同一批图片及其 XML、JSON 标注 |
| `light/` | 指示灯两阶段检测 | 图片和包含灯组、灯位标注的 XML |

`scale` 暂不纳入测试基线。夹具不包含原始数据集中的标注工具、批处理文件、模型产物和未配对样本。

## 测试层次

- `fixtures/`：不可变的微型输入数据集。
- `expected/conversions/`：11 组经人工审阅的语义快照。
- `support/output_manifest.py`：统一规范化路径、文本、YAML 和图片尺寸。
- `test_*.py`：完整覆盖转换、解析、特殊变换、失败场景和可复现性。

测试直接导入 `xxtrain` 新包，不保留新旧模块兼容层。

完整测试命令：`.venv\Scripts\python.exe -m unittest discover -s test -t . -p 'test_*.py' -v`

显式更新快照：`.venv\Scripts\python.exe -m test.test_conversion_baseline --update-snapshots`

快照差异必须经过人工审阅；`scale` 不在本基线范围内。
