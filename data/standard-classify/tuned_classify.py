from xxtrain.pipeline import standard_recipe
from xxtrain.task import TaskType
from xxtrain.training import TrainingScenario

SCENARIO = TrainingScenario(
    dataset=standard_recipe(TaskType.CLASSIFY),
    train_args={
        'batch': 16,
        'optimizer': 'AdamW',
        'lr0': 0.0005,
        'lrf': 0.05,
        'weight_decay': 0.001,
        'warmup_epochs': 3.0,
        'cos_lr': True,
        'dropout': 0.15,
        'fliplr': 0.0,
        'flipud': 0.0,
        'auto_augment': None,
        'erasing': 0.0,
    },
)
