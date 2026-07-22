from enum import Enum


class TaskType(Enum):
    CLASSIFY = 'classify'
    DETECT = 'detect'
    OBB = 'obb'
    POSE = 'pose'
    SEGMENT = 'segment'
