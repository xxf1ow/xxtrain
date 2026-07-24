from xxtrain.data import LabelCatalog
from xxtrain.pipeline import DatasetRecipe, Pipeline
from xxtrain.pipeline.processors import (
    EncodeDetection,
    FilterLabels,
    ReadImageInfo,
    ReadLabelImg,
    RelabelAnnotations,
)
from xxtrain.pipeline.sinks import YoloDatasetSink
from xxtrain.task import TaskType
from xxtrain.training import TrainingScenario

SCENARIO = TrainingScenario(
    dataset=DatasetRecipe(
        name='point-detect',
        task_type=TaskType.DETECT,
        labels=LabelCatalog(('Point',)),
        pipeline=Pipeline(
            (
                ReadImageInfo(),
                ReadLabelImg(),
                FilterLabels(('tl', 'tc', 'cl', 'cc')),
                RelabelAnnotations('Point'),
                EncodeDetection(),
            )
        ),
        sink=YoloDatasetSink(),
    )
)
