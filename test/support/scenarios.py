from pathlib import Path

from xxtrain.pipeline import DatasetRecipe, standard_recipe
from xxtrain.pipeline.recipes import Recipe, build_recipe
from xxtrain.task import TaskType

_STANDARD_TASK_TYPES = {
    'detect': TaskType.DETECT,
    'segment': TaskType.SEGMENT,
    'pose': TaskType.POSE,
    'classify': TaskType.CLASSIFY,
}


def recipe_for_case(task_name: str, root_path: str | Path) -> DatasetRecipe | Recipe:
    task_type = _STANDARD_TASK_TYPES.get(task_name)
    if task_type is not None:
        return standard_recipe(task_type)
    recipe, _ = build_recipe(task_name, root_path)
    return recipe
