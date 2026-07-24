from xxtrain.data import LabelCatalog
from xxtrain.pipeline import DatasetRecipe, Pipeline
from xxtrain.pipeline.processors import (
    CropMatches,
    EncodePointSegment,
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
        name='point-segment',
        task_type=TaskType.SEGMENT,
        labels=LabelCatalog(('Point',)),
        pipeline=Pipeline(
            (
                ReadImageInfo(),
                ReadMatchingAnnotations(),
                FilterMatchingAnnotations(
                    parent_labels=('tl', 'tc', 'cl', 'cc'),
                    child_labels=('1',),
                    strict=True,
                ),
                PrepareMatchChildren(TaskType.SEGMENT),
                MatchAnnotations(wide=0, strict=False),
                CropMatches(),
                EncodePointSegment(),
            )
        ),
        sink=YoloDatasetSink(),
    )
)
