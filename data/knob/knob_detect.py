from xxtrain.data import LabelCatalog
from xxtrain.pipeline import DatasetRecipe, Pipeline
from xxtrain.pipeline.processors import EncodeDetection, FilterLabels, ReadImageInfo, ReadLabelImg
from xxtrain.pipeline.sinks import YoloDatasetSink
from xxtrain.task import TaskType
from xxtrain.training import TrainingScenario

SCENARIO = TrainingScenario(
    dataset=DatasetRecipe(
        name='knob-detect',
        task_type=TaskType.DETECT,
        labels=LabelCatalog(('switch',)),
        pipeline=Pipeline(
            (
                ReadImageInfo(),
                ReadLabelImg(),
                FilterLabels(('switch',)),
                EncodeDetection(),
            )
        ),
        sink=YoloDatasetSink(),
    )
)
