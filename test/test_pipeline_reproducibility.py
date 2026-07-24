import shutil
import tempfile
import unittest
from pathlib import Path

from test.support.output_manifest import collect_output_manifest
from test.support.scenarios import recipe_for_case
from test.test_conversion_baseline import BASELINE_CASES, FIXTURES_PATH
from test.test_reproducibility import snapshot_tree
from xxtrain.pipeline.workflow import convert_dataset


class NewPipelineReproducibilityTest(unittest.TestCase):
    def test_new_pipeline_is_reproducible_without_mutating_inputs(self) -> None:
        for task_name, fixture_name in BASELINE_CASES:
            with self.subTest(task_name=task_name):
                with tempfile.TemporaryDirectory(prefix='xxtrain-new-reproducibility-') as temp_dir:
                    manifests = []
                    for copy_name in ('first', 'second'):
                        root_path = Path(temp_dir) / copy_name
                        shutil.copytree(FIXTURES_PATH / fixture_name, root_path)
                        source_snapshot = snapshot_tree(root_path / 'src')
                        convert_dataset(
                            recipe_for_case(task_name, root_path),
                            root_path,
                            split=10,
                            reserve_no_label=False,
                        )
                        self.assertEqual(source_snapshot, snapshot_tree(root_path / 'src'))
                        manifests.append(collect_output_manifest(root_path, task_name))
                    self.assertEqual(manifests[0], manifests[1])


if __name__ == '__main__':
    unittest.main()
