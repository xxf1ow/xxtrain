from xxtrain.data import LabelCatalog
from xxtrain.pipeline import DatasetRecipe, Pipeline
from xxtrain.pipeline.processors import (
    CropDetectionBoxes,
    FilterLabels,
    ReadImageInfo,
    ReadLabelImg,
)
from xxtrain.pipeline.sinks import ClassificationDatasetSink
from xxtrain.task import TaskType
from xxtrain.training import TrainingScenario

SCENARIO = TrainingScenario(
    dataset=DatasetRecipe(
        name='point-classify',
        task_type=TaskType.CLASSIFY,
        labels=LabelCatalog(('tl', 'tc', 'cl', 'cc')),
        pipeline=Pipeline(
            (
                ReadImageInfo(),
                ReadLabelImg(),
                FilterLabels(('tl', 'tc', 'cl', 'cc')),
                CropDetectionBoxes(),
            )
        ),
        sink=ClassificationDatasetSink(indexed_class_directories=True),
    )
)
