from xxtrain.data import LabelCatalog
from xxtrain.pipeline import DatasetRecipe, Pipeline
from xxtrain.pipeline.annotation_io import ReadAnnotations
from xxtrain.pipeline.processors import EncodeDetection, FilterLabels, ReadImageInfo, RelabelAnnotations
from xxtrain.pipeline.sinks import YoloDatasetSink
from xxtrain.task import TaskType

POINT_BOX_LABELS = ('Point', 'tl', 'tc', 'cl', 'cc')
MODEL_TARGETS = (('detect', True), ('classify', False), ('segment', False))


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
