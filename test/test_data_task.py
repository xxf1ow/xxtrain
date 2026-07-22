import unittest

import xxtrain
from xxtrain.task import TaskType


class TaskTypeTest(unittest.TestCase):
    def test_defines_only_basic_task_values(self) -> None:
        self.assertEqual(
            ['classify', 'detect', 'obb', 'pose', 'segment'],
            [task.value for task in TaskType],
        )

    def test_package_root_does_not_reexport_task_type(self) -> None:
        self.assertFalse(hasattr(xxtrain, 'TaskType'))


if __name__ == '__main__':
    unittest.main()
