from importlib import import_module

from .definition import TaskDefinition

DEFAULT_TASK_ENTRY = 'xxtrain.business_tasks.point:point_task_definition'


def load_task_definition(entry: str) -> TaskDefinition:
    """Load a deployment-configured zero-argument task-definition factory."""
    if not isinstance(entry, str) or entry.count(':') != 1:
        raise ValueError('Task entry must use module:factory syntax')
    module_name, factory_name = entry.split(':')
    try:
        factory = getattr(import_module(module_name), factory_name)
        result = factory()
    except (ImportError, AttributeError, TypeError) as error:
        raise ValueError(f'Cannot load task definition: {entry!r}') from error
    if not isinstance(result, TaskDefinition):
        raise ValueError(f'Task factory {entry!r} did not return TaskDefinition')
    return result
