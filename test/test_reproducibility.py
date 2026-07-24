import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from test.support.output_manifest import collect_output_manifest
from test.support.scenarios import recipe_for_case
from test.test_conversion_baseline import BASELINE_CASES, FIXTURES_PATH
from xxtrain.pipeline import convert_dataset


def snapshot_tree(root_path: Path) -> dict[str, str]:
    snapshot = {}
    paths = sorted(root_path.rglob('*'), key=lambda path: path.relative_to(root_path).as_posix())
    for path in paths:
        relative_path = path.relative_to(root_path).as_posix()
        if path.is_dir():
            snapshot[relative_path] = 'directory'
        elif path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            snapshot[relative_path] = f'file:{digest}'
    return snapshot


class ReproducibilityTest(unittest.TestCase):
    def test_tree_snapshot_detects_empty_directory_changes(self) -> None:
        with tempfile.TemporaryDirectory(prefix='xxtrain-tree-snapshot-') as temp_dir:
            root_path = Path(temp_dir)
            before = snapshot_tree(root_path)
            (root_path / 'empty').mkdir()

            self.assertNotEqual(before, snapshot_tree(root_path))

    def test_conversions_are_reproducible_without_mutating_inputs(self) -> None:
        for task_type, fixture_name in BASELINE_CASES:
            with self.subTest(task_type=task_type):
                with tempfile.TemporaryDirectory(prefix='xxtrain-reproducibility-') as temp_dir:
                    manifests = []
                    for copy_name in ('first', 'second'):
                        root_path = Path(temp_dir) / copy_name
                        shutil.copytree(FIXTURES_PATH / fixture_name, root_path)
                        source_snapshot = snapshot_tree(root_path / 'src')

                        convert_dataset(
                            recipe_for_case(task_type, root_path),
                            root_path,
                            split=10,
                            reserve_no_label=False,
                        )

                        self.assertEqual(
                            source_snapshot,
                            snapshot_tree(root_path / 'src'),
                            f'{task_type} mutated source tree during {copy_name} conversion',
                        )
                        manifests.append(collect_output_manifest(root_path, task_type))

                    self.assertEqual(manifests[0], manifests[1])


if __name__ == '__main__':
    unittest.main()
