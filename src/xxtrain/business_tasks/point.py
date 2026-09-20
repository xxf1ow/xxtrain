from xxtrain.platform.cache_builders import encode_classification
from xxtrain.task import TaskType
from xxtrain.training.settings import TrainingSettings, standard_train_args
from xxtrain.workspace_data.inputs import AxisAlignedRectangleInputs, OriginalImageInputs

from .definition import AnnotationPolicy, DeliveryDefinition, StepDefinition, TargetTrainingDefinition, TaskDefinition
from .point_conversion import encode_point_detection, encode_point_segment

POINT_BOX_LABELS = ('Point', 'tl', 'tc', 'cl', 'cc')
MODEL_TARGETS = (('detect', True), ('classify', True), ('segment', True))
POINT_CLASSIFY_OVERRIDES = {'fliplr': 0.0, 'flipud': 0.0, 'degrees': 0.0, 'auto_augment': None}


def point_task_definition() -> TaskDefinition:
    """Return the Point annotation steps and their parent and dependency rules."""
    return TaskDefinition(
        (
            StepDefinition(
                key='detect',
                kinds=frozenset({'rectangle', 'negative'}),
                labels=frozenset(POINT_BOX_LABELS),
                parent_steps=frozenset(),
                depends_on=frozenset(),
                display_name='检测',
                sample_unit='张图片',
                annotation=AnnotationPolicy('rectangle', 'STANDARD', negative_label='无检测目标'),
                minimum_samples=50,
                input_adapter=OriginalImageInputs(),
                training=TargetTrainingDefinition(
                    settings=TrainingSettings(TaskType.DETECT, train_args=standard_train_args(TaskType.DETECT)),
                    metric_key='metrics/mAP50-95(B)',
                    metric_name='检测效果：mAP50-95',
                    delivery=DeliveryDefinition(labels=False, reference_images=False),
                    labels=('Point',),
                    encode_sample=encode_point_detection,
                ),
            ),
            StepDefinition(
                key='classify',
                kinds=frozenset({'classification'}),
                labels=frozenset({'tl', 'tc', 'cl', 'cc'}),
                parent_steps=frozenset({'detect'}),
                depends_on=frozenset(),
                display_name='分类',
                sample_unit='张裁剪图',
                annotation=AnnotationPolicy('tag', 'TAGS', maximum_annotations=1),
                input_adapter=AxisAlignedRectangleInputs(),
                training=TargetTrainingDefinition(
                    settings=TrainingSettings(
                        TaskType.CLASSIFY, train_args=standard_train_args(TaskType.CLASSIFY) | POINT_CLASSIFY_OVERRIDES
                    ),
                    metric_key='metrics/accuracy_top1',
                    metric_name='分类准确率：Top-1',
                    delivery=DeliveryDefinition(labels=True, reference_images=True),
                    labels=('tl', 'tc', 'cl', 'cc'),
                    encode_sample=encode_classification,
                ),
            ),
            StepDefinition(
                key='segment',
                kinds=frozenset({'polyline'}),
                labels=frozenset({'1'}),
                parent_steps=frozenset({'detect'}),
                depends_on=frozenset(),
                display_name='指针分割',
                sample_unit='张裁剪图',
                annotation=AnnotationPolicy('polyline', 'STANDARD', point_count=2),
                input_adapter=AxisAlignedRectangleInputs(),
                training=TargetTrainingDefinition(
                    settings=TrainingSettings(TaskType.SEGMENT, train_args=standard_train_args(TaskType.SEGMENT)),
                    metric_key='metrics/mAP50-95(M)',
                    metric_name='指针分割效果：Mask mAP50-95',
                    delivery=DeliveryDefinition(labels=False, reference_images=False),
                    labels=('Point',),
                    encode_sample=encode_point_segment,
                ),
            ),
        ),
        key='point',
        display_name='Point',
    )
