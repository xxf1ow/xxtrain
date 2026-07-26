from xxtrain.pipeline import standard_recipe
from xxtrain.task import TaskType
from xxtrain.training import TrainingScenario

SCENARIO = TrainingScenario(dataset=standard_recipe(TaskType.POSE))
