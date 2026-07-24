from xxtrain.data import LabelCatalog
from xxtrain.pipeline import DatasetRecipe, Pipeline
from xxtrain.pipeline.processors import EncodeDetection, FilterLabels, ReadImageInfo, ReadLabelImg
from xxtrain.pipeline.sinks import YoloDatasetSink
from xxtrain.task import TaskType
from xxtrain.training import TrainingScenario

SCENARIO = TrainingScenario(
    dataset=DatasetRecipe(
        name='light1-detect',
        task_type=TaskType.DETECT,
        labels=LabelCatalog(('1008',)),
        pipeline=Pipeline((ReadImageInfo(), ReadLabelImg(), FilterLabels(('1008',), strict=False), EncodeDetection())),
        sink=YoloDatasetSink(),
    )
)
