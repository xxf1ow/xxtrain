from xxtrain.data import LabelCatalog
from xxtrain.pipeline import DatasetRecipe, Pipeline
from xxtrain.pipeline.annotation_io import ReadAnnotations
from xxtrain.pipeline.processors import EncodeDetection, FilterLabels, ReadImageInfo, RelabelAnnotations
from xxtrain.pipeline.sinks import YoloDatasetSink
from xxtrain.task import TaskType

from .definition import StepDefinition, TaskDefinition

POINT_BOX_LABELS = ('Point', 'tl', 'tc', 'cl', 'cc')
MODEL_TARGETS = (('detect', True), ('classify', True), ('segment', True))


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
            ),
            StepDefinition(
                key='classify',
                kinds=frozenset({'classification'}),
                labels=frozenset({'tl', 'tc', 'cl', 'cc'}),
                parent_steps=frozenset({'detect'}),
                depends_on=frozenset({'detect'}),
            ),
            StepDefinition(
                key='segment',
                kinds=frozenset({'polyline'}),
                labels=frozenset({'1'}),
                parent_steps=frozenset({'detect'}),
                depends_on=frozenset({'classify'}),
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
