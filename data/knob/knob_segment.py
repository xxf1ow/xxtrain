from xxtrain.data import LabelCatalog
from xxtrain.pipeline import DatasetRecipe, Pipeline
from xxtrain.pipeline.processors import (
    CropMatches,
    EncodeKnobSegment,
    FilterMatchingAnnotations,
    MatchAnnotations,
    PrepareMatchChildren,
    ReadImageInfo,
    ReadMatchingAnnotations,
)
from xxtrain.pipeline.sinks import YoloDatasetSink
from xxtrain.task import TaskType
from xxtrain.training import TrainingScenario

SCENARIO = TrainingScenario(
    dataset=DatasetRecipe(
        name='knob-segment',
        task_type=TaskType.SEGMENT,
        labels=LabelCatalog(('switch',)),
        pipeline=Pipeline(
            (
                ReadImageInfo(),
                ReadMatchingAnnotations(),
                FilterMatchingAnnotations(parent_labels=('switch',), child_labels=('switch',), strict=True),
                PrepareMatchChildren(TaskType.SEGMENT),
                MatchAnnotations(wide=0.15, strict=False),
                CropMatches(),
                EncodeKnobSegment(),
            )
        ),
        sink=YoloDatasetSink(),
    )
)
