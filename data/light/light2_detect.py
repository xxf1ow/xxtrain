from xxtrain.data import LabelCatalog
from xxtrain.pipeline import DatasetRecipe, Pipeline
from xxtrain.pipeline.processors import (
    CropMatches,
    EncodeCropDetection,
    MatchAnnotations,
    PartitionAnnotations,
    ReadImageInfo,
    ReadLabelImg,
    RelabelCropAnnotations,
)
from xxtrain.pipeline.sinks import YoloDatasetSink
from xxtrain.task import TaskType
from xxtrain.training import TrainingScenario

SCENARIO = TrainingScenario(
    dataset=DatasetRecipe(
        name='light2-detect',
        task_type=TaskType.DETECT,
        labels=LabelCatalog(('0',)),
        pipeline=Pipeline(
            (
                ReadImageInfo(),
                ReadLabelImg(),
                PartitionAnnotations(parent_labels=('1008',), child_labels=('0', '1', '2')),
                MatchAnnotations(wide=0.1, strict=False),
                CropMatches(),
                RelabelCropAnnotations('0'),
                EncodeCropDetection(),
            )
        ),
        sink=YoloDatasetSink(),
    )
)
