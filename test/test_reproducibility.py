import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from test.support.current_api import convert_dataset
from test.support.output_manifest import collect_output_manifest
from test.test_conversion_baseline import BASELINE_CASES, FIXTURES_PATH


def hash_tree(root_path: Path) -> dict[str, str]:
    files = sorted(
        ((path.relative_to(root_path).as_posix(), path) for path in root_path.rglob('*') if path.is_file()),
        key=lambda item: item[0],
    )
    return {relative_path: hashlib.sha256(path.read_bytes()).hexdigest() for relative_path, path in files}


class ReproducibilityTest(unittest.TestCase):
    def test_conversions_are_reproducible_without_mutating_inputs(self) -> None:
        for task_type, fixture_name in BASELINE_CASES:
            with self.subTest(task_type=task_type):
                with tempfile.TemporaryDirectory(prefix='xxtrain-reproducibility-') as temp_dir:
                    manifests = []
                    for copy_name in ('first', 'second'):
                        root_path = Path(temp_dir) / copy_name
                        shutil.copytree(FIXTURES_PATH / fixture_name, root_path)
                        source_hashes = hash_tree(root_path / 'src')

                        convert_dataset(task_type, str(root_path), split=10, reserve_no_label=False)

                        self.assertEqual(
                            source_hashes,
                            hash_tree(root_path / 'src'),
                            f'{task_type} mutated source files during {copy_name} conversion',
                        )
                        manifests.append(collect_output_manifest(root_path, task_type))

                    self.assertEqual(manifests[0], manifests[1])


if __name__ == '__main__':
    unittest.main()
