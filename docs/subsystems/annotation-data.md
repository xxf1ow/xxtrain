# 标注数据子系统

## Scope

`xxtrain.data` 提供格式无关的标注值、几何操作、稳定标签目录以及 LabelImg、LabelMe、YOLO、COCO 的读写边界。转换管线只依赖这些统一值，不直接携带外部文件格式差异。当前精确导出面由 [`src/xxtrain/data/__init__.py`](../../src/xxtrain/data/__init__.py) 定义。

## Values and identity

`Annotation` 是不可变值，具体形状包括 `Bbox`、`Polygon`、`Polyline`、`Points`、`Circle` 和 `Pose`。变换创建新对象；标注身份使用 UUID，`group` 独立于 `label`，可用于表达跨对象关系而不改变类别语义。

形状点集规范化为不可变 tuple。`Pose` 包含一个框和至少一个标签唯一的 `Keypoint`；关键点可见性只能是 0、1 或 2。非法标签、身份、分组和几何在值对象构造时失败。

## Coordinates and raster boundaries

标注坐标、平移量、几何结果和虚拟裁剪范围使用有限 `float`。`ImageInfo.width` 与 `height` 也统一为正有限 `float`，使裁剪后的分数范围不会在内存模型中丢失。

解析、匹配、Processor 和编码阶段不得提前截断或四舍五入。整数化只允许出现在 OpenCV 数组切片等明确的栅格边界；越过该边界后，坐标语义必须仍能由原始浮点范围解释。

## Label catalogs

`LabelCatalog` 保存非空、无重复的字符串 tuple，声明顺序就是类别索引顺序。数字形式的名称仍按字符串处理，不进行数值排序或隐式类型转换。

普通目录标注内容不拥有类别目录。标准 Recipe 从 Scenario 输入根的 `src/labels.txt` 读取目标名称和顺序；COCO 来源可从 category schema 提供目录；特殊 Recipe 可以显式携带固定目录。来源中出现非目标标签时，转换负责过滤和报告，不能隐式扩展目标目录。

## Format boundaries

LabelImg、LabelMe、YOLO 和 COCO 只在格式自身可表达且任务语义允许的范围内读写统一值。LabelImg 提供框，LabelMe 提供多种形状，YOLO 编解码任务训练行，COCO 同时携带图片、category schema 和标注集合。

Pose 可以把同图的 LabelImg 框与 LabelMe 关键点组合为统一 `Pose`。其他任务不会把多个非空格式的同图标注自动合并；该歧义由转换边界拒绝。

## Failures and limitations

非有限坐标、退化框、点数不足、重复关键点标签、图像范围外几何、未知标签或目标格式无法无损表达的语义都会在最接近所有权的边界失败。读取器不得静默丢弃调用者需要的形状语义，写入器也不得把不支持的对象伪装成有效输出。

统一值当前不预设模型置信度、候选来源或平台审核状态；这些字段只有在自助平台真正实现相应数据契约时才能加入。
