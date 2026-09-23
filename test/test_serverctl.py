import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import yaml

from xxtrain.serverctl import compose_files, ensure_administrator, main, site_root


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

    def test_secret_file_failure_does_not_delete_replacement_file(self) -> None:
        path = self.root / '.deployment' / 'administrator'

        def replace_then_fail(descriptor: int, *_args: str, **_kwargs: str) -> None:
            os.close(descriptor)
            path.unlink()
            path.write_text('operator-replacement-secret\n', encoding='utf-8')
            raise OSError('secret file open failed')

        with patch('xxtrain.serverctl.os.fdopen', side_effect=replace_then_fail):
            with self.assertRaisesRegex(OSError, 'secret file open failed'):
                ensure_administrator(self.root)

        self.assertEqual('operator-replacement-secret\n', path.read_text(encoding='utf-8'))

    def test_unimplemented_actions_fail_without_preparing_start(self) -> None:
        for action in ('install', 'start', 'stop', 'status', 'verify'):
            with self.subTest(action=action), redirect_stderr(StringIO()) as errors:
                self.assertNotEqual(0, main(action, root=self.root))
                self.assertIn(action, errors.getvalue())
        self.assertFalse((self.root / '.deployment').exists())

    def test_compose_topology_keeps_all_persistent_data_under_site_root(self) -> None:
        checkout = Path(__file__).resolve().parents[1]
        files = compose_files(checkout)
        self.assertEqual((checkout / 'deploy/server/compose.yaml',), files)
        manifest = yaml.safe_load(files[0].read_text(encoding='utf-8'))
        self.assertEqual('xxtrain-server', manifest['name'])
        services = manifest['services']
        self.assertEqual('host', services['nginx']['network_mode'])
        for worker in ('utils', 'import', 'export', 'annotation', 'webhooks', 'quality_reports', 'chunks', 'consensus'):
            self.assertIn(f'cvat_worker_{worker}', services)
        expected = {
            'cvat_db': '/var/lib/postgresql/data',
            'cvat_redis_inmem': '/data',
            'cvat_redis_ondisk': '/var/lib/kvrocks',
            'cvat_server': '/home/django/data',
            'cvat_clickhouse': '/var/lib/clickhouse',
            'clearml_mongo': '/data/db',
            'clearml_redis': '/data',
            'clearml_elasticsearch': '/usr/share/elasticsearch/data',
            'clearml_fileserver': '/mnt/fileserver',
        }
        for service, target in expected.items():
            with self.subTest(service=service):
                mounts = services[service]['volumes']
                self.assertTrue(
                    any(
                        (
                            v.get('target') == target
                            if isinstance(v, dict)
                            else v.replace('${XXTRAIN_SITE_ROOT:?}', 'SITE_ROOT').split(':')[1] == target
                        )
                        for v in mounts
                    )
                )
        for name, service in services.items():
            self.assertNotIn('restart', service, name)
            for mount in service.get('volumes', []):
                if isinstance(mount, str):
                    parts = mount.replace('${XXTRAIN_SITE_ROOT:?}', 'SITE_ROOT').split(':')
                    source, mode = (
                        parts[0].replace('SITE_ROOT', '${XXTRAIN_SITE_ROOT:?}'),
                        (parts[2] if len(parts) > 2 else ''),
                    )
                    self.assertTrue(source.startswith('${XXTRAIN_SITE_ROOT:?}/') or mode == 'ro', (name, mount))
                elif mount.get('type') != 'tmpfs':
                    self.assertTrue(
                        mount['source'].startswith('${XXTRAIN_SITE_ROOT:?}/') or mount.get('read_only'), (name, mount)
                    )
            for port in service.get('ports', []):
                self.assertTrue(str(port).startswith('127.0.0.1:'), (name, port))
        self.assertNotIn('volumes', manifest)


if __name__ == '__main__':
    unittest.main()
