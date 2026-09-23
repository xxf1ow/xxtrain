import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from xxtrain.serverctl import ensure_administrator, main, site_root


class ServerctlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_site_root_returns_git_top_level(self) -> None:
        nested = self.root / 'nested'
        nested.mkdir()

        self.assertEqual(self.root.resolve(), site_root(nested))

    def test_site_root_rejects_deployment_symlink_outside_checkout(self) -> None:
        deployment = self.root / '.deployment'
        outside = self.root.parent / f'{self.root.name}-outside'
        outside.mkdir()
        try:
            deployment.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, r'\.deployment.*outside'):
                site_root(self.root)
        finally:
            deployment.unlink(missing_ok=True)
            outside.rmdir()

    def test_existing_administrator_secret_is_never_replaced(self) -> None:
        path = self.root / '.deployment' / 'administrator'
        path.parent.mkdir()
        path.write_text('operator-chosen-secret\n', encoding='utf-8')
        original_mode = os.stat(path).st_mode & 0o777

        self.assertEqual(path, ensure_administrator(self.root))
        self.assertEqual('operator-chosen-secret\n', path.read_text(encoding='utf-8'))
        self.assertEqual(original_mode, os.stat(path).st_mode & 0o777)

    def test_missing_administrator_secret_is_created_private_and_stable(self) -> None:
        path = ensure_administrator(self.root)
        first_content = path.read_bytes()
        first_mode = os.stat(path).st_mode & 0o777

        if os.name != 'nt':
            self.assertEqual(0o600, first_mode)
        self.assertTrue(first_content)
        self.assertEqual(path, ensure_administrator(self.root))
        self.assertEqual(first_content, path.read_bytes())
        self.assertEqual(first_mode, os.stat(path).st_mode & 0o777)

    def test_unimplemented_actions_fail_without_preparing_start(self) -> None:
        for action in ('install', 'start', 'stop', 'status', 'verify'):
            with self.subTest(action=action), redirect_stderr(StringIO()) as errors:
                self.assertNotEqual(0, main(action, root=self.root))
                self.assertIn(action, errors.getvalue())
        self.assertFalse((self.root / '.deployment').exists())


if __name__ == '__main__':
    unittest.main()
