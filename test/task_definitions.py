from dataclasses import replace

from xxtrain.business_tasks.definition import (
    AnnotationPolicy,
    DeliveryDefinition,
    StepDefinition,
    TargetTrainingDefinition,
    TaskDefinition,
)
from xxtrain.platform.cache_builders import encode_classification, encode_rectangles
from xxtrain.task import TaskType
from xxtrain.training.settings import TrainingSettings, standard_train_args
from xxtrain.workspace_data.inputs import AxisAlignedRectangleInputs, OriginalImageInputs


def synthetic_task_definition() -> TaskDefinition:
    return TaskDefinition(
        (
            StepDefinition(
                'regions',
                frozenset({'rectangle'}),
                frozenset({'part'}),
                frozenset(),
                frozenset(),
                display_name='Regions',
                sample_unit='images',
                annotation=AnnotationPolicy('rectangle', 'STANDARD'),
                input_adapter=OriginalImageInputs(),
            ),
            StepDefinition(
                'kind',
                frozenset({'classification'}),
                frozenset({'a', 'b'}),
                frozenset({'regions'}),
                frozenset(),
                display_name='Kind',
                sample_unit='crops',
                annotation=AnnotationPolicy('tag', 'TAGS', maximum_annotations=1),
                input_adapter=AxisAlignedRectangleInputs(),
            ),
            StepDefinition(
                'needles',
                frozenset({'polyline'}),
                frozenset({'line'}),
                frozenset({'regions'}),
                frozenset(),
                display_name='Needles',
                sample_unit='crops',
                annotation=AnnotationPolicy('polyline', 'STANDARD', point_count=2),
                input_adapter=AxisAlignedRectangleInputs(),
            ),
            StepDefinition(
                'subregions',
                frozenset({'rectangle'}),
                frozenset({'part'}),
                frozenset({'regions'}),
                frozenset(),
                display_name='Subregions',
                sample_unit='crops',
                annotation=AnnotationPolicy('rectangle', 'STANDARD'),
                input_adapter=AxisAlignedRectangleInputs(),
            ),
            StepDefinition(
                'details',
                frozenset({'classification'}),
                frozenset({'x', 'y'}),
                frozenset({'subregions'}),
                frozenset(),
                display_name='Details',
                sample_unit='crops',
                annotation=AnnotationPolicy('tag', 'TAGS', maximum_annotations=1),
                input_adapter=AxisAlignedRectangleInputs(),
            ),
        ),
        key='synthetic',
        display_name='Synthetic task',
    )


def not_a_definition() -> object:
    return object()


def synthetic_training_task_definition() -> TaskDefinition:
    task = synthetic_task_definition()
    training = TargetTrainingDefinition(
        settings=TrainingSettings(TaskType.CLASSIFY),
        metric_key='synthetic/kind',
        metric_name='Synthetic kind quality',
        delivery=DeliveryDefinition(labels=True, reference_images=True),
        labels=('a', 'b'),
        conversion_key='synthetic-kind-v1',
        encode_sample=encode_classification,
    )
    return replace(
        task, steps=tuple(replace(step, training=training) if step.key == 'kind' else step for step in task.steps)
    )


def workflow_task_definition() -> TaskDefinition:
    task = synthetic_task_definition()
    training = {
        'regions': TargetTrainingDefinition(
            settings=TrainingSettings(TaskType.DETECT, train_args=standard_train_args(TaskType.DETECT)),
            metric_key='synthetic/regions',
            metric_name='Synthetic region quality',
            delivery=DeliveryDefinition(labels=False, reference_images=False),
            labels=('part',),
            conversion_key='synthetic-regions-v1',
            encode_sample=encode_rectangles,
        ),
        'kind': TargetTrainingDefinition(
            settings=TrainingSettings(TaskType.CLASSIFY, train_args=standard_train_args(TaskType.CLASSIFY)),
            metric_key='synthetic/kind',
            metric_name='Synthetic kind quality',
            delivery=DeliveryDefinition(labels=True, reference_images=True),
            labels=('a', 'b'),
            conversion_key='synthetic-kind-v1',
            encode_sample=encode_classification,
        ),
        'subregions': TargetTrainingDefinition(
            settings=TrainingSettings(TaskType.DETECT, train_args=standard_train_args(TaskType.DETECT)),
            metric_key='synthetic/subregions',
            metric_name='Synthetic subregion quality',
            delivery=DeliveryDefinition(labels=False, reference_images=False),
            labels=('part',),
            conversion_key='synthetic-subregions-v1',
            encode_sample=encode_rectangles,
        ),
        'details': TargetTrainingDefinition(
            settings=TrainingSettings(TaskType.CLASSIFY, train_args=standard_train_args(TaskType.CLASSIFY)),
            metric_key='synthetic/details',
            metric_name='Synthetic detail quality',
            delivery=DeliveryDefinition(labels=True, reference_images=False),
            labels=('x', 'y'),
            conversion_key='synthetic-details-v1',
            encode_sample=encode_classification,
        ),
    }
    return replace(task, steps=tuple(replace(step, training=training.get(step.key)) for step in task.steps))
