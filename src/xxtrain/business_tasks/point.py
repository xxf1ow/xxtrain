from xxtrain.data import LabelCatalog
from xxtrain.pipeline import DatasetRecipe, Pipeline
from xxtrain.pipeline.annotation_io import ReadAnnotations
from xxtrain.pipeline.processors import EncodeDetection, FilterLabels, ReadImageInfo, RelabelAnnotations
from xxtrain.pipeline.sinks import YoloDatasetSink
from xxtrain.task import TaskType
from xxtrain.training.settings import TrainingSettings, standard_train_args

from .definition import DeliveryDefinition, StepDefinition, TargetTrainingDefinition, TaskDefinition

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
                training=TargetTrainingDefinition(
                    settings=TrainingSettings(TaskType.DETECT, train_args=standard_train_args(TaskType.DETECT)),
                    metric_key='metrics/mAP50-95(B)',
                    metric_name='检测效果：mAP50-95',
                    delivery=DeliveryDefinition(labels=False, reference_images=False),
                ),
            ),
            StepDefinition(
                key='classify',
                kinds=frozenset({'classification'}),
                labels=frozenset({'tl', 'tc', 'cl', 'cc'}),
                parent_steps=frozenset({'detect'}),
                depends_on=frozenset({'detect'}),
                training=TargetTrainingDefinition(
                    settings=TrainingSettings(
                        TaskType.CLASSIFY, train_args=standard_train_args(TaskType.CLASSIFY) | POINT_CLASSIFY_OVERRIDES
                    ),
                    metric_key='metrics/accuracy_top1',
                    metric_name='分类准确率：Top-1',
                    delivery=DeliveryDefinition(labels=True, reference_images=True),
                ),
            ),
            StepDefinition(
                key='segment',
                kinds=frozenset({'polyline'}),
                labels=frozenset({'1'}),
                parent_steps=frozenset({'detect'}),
                depends_on=frozenset({'classify'}),
                training=TargetTrainingDefinition(
                    settings=TrainingSettings(TaskType.SEGMENT, train_args=standard_train_args(TaskType.SEGMENT)),
                    metric_key='metrics/mAP50-95(M)',
                    metric_name='指针分割效果：Mask mAP50-95',
                    delivery=DeliveryDefinition(labels=False, reference_images=False),
                ),
            ),
        )
    )


def point_detection_recipe() -> DatasetRecipe:
    """Return the Point detector projection used for disposable detection caches."""
    return DatasetRecipe(
        name='detect',
        task_type=TaskType.DETECT,
        labels=LabelCatalog(('Point',)),
        pipeline=Pipeline(
            (
                ReadImageInfo(),
                ReadAnnotations(),
                FilterLabels(POINT_BOX_LABELS),
                RelabelAnnotations('Point'),
                EncodeDetection(),
            )
        ),
        sink=YoloDatasetSink(),
    )
