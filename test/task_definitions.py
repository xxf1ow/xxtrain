from xxtrain.business_tasks.definition import AnnotationPolicy, StepDefinition, TaskDefinition


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
                annotation=AnnotationPolicy('rectangle', 'STANDARD'),
            ),
            StepDefinition(
                'kind',
                frozenset({'classification'}),
                frozenset({'a', 'b'}),
                frozenset({'regions'}),
                frozenset(),
                display_name='Kind',
                annotation=AnnotationPolicy('tag', 'TAGS', maximum_annotations=1),
            ),
            StepDefinition(
                'needles',
                frozenset({'polyline'}),
                frozenset({'line'}),
                frozenset({'regions'}),
                frozenset(),
                display_name='Needles',
                annotation=AnnotationPolicy('polyline', 'STANDARD', point_count=2),
            ),
            StepDefinition(
                'subregions',
                frozenset({'rectangle'}),
                frozenset({'part'}),
                frozenset({'regions'}),
                frozenset(),
                display_name='Subregions',
                annotation=AnnotationPolicy('rectangle', 'STANDARD'),
            ),
            StepDefinition(
                'details',
                frozenset({'classification'}),
                frozenset({'x', 'y'}),
                frozenset({'subregions'}),
                frozenset(),
                display_name='Details',
                annotation=AnnotationPolicy('tag', 'TAGS', maximum_annotations=1),
            ),
        ),
        key='synthetic',
        display_name='Synthetic task',
    )


def not_a_definition() -> object:
    return object()
