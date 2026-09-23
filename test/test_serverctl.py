import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import yaml

from xxtrain.serverctl import compose_files, ensure_administrator, install, main, site_root, start, stop


class ServerctlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_install_is_repeatable_and_does_not_start_services(self) -> None:
        calls = []

        def run(args, **kwargs):
            calls.append((args, kwargs))

        with patch('xxtrain.serverctl.site_root', return_value=self.root):
            install(self.root, run)
            env_file = self.root / '.deployment/platform.env'
            env_file.write_text('OPERATOR_SECRET=kept\n', encoding='utf-8')
            install(self.root, run)
        self.assertEqual('OPERATOR_SECRET=kept\n', env_file.read_text(encoding='utf-8'))
        expected = [
            ['uv', 'sync', '--locked', '--extra', 'platform', '--extra', 'clearml'],
            ['docker', 'compose', '-p', 'xxtrain-server', '-f', str(self.root / 'deploy/server/compose.yaml'), 'pull'],
            ['docker', 'compose', '-p', 'xxtrain-server', '-f', str(self.root / 'deploy/server/compose.yaml'), 'build'],
        ]
        self.assertEqual(expected * 2, [args for args, _ in calls])
        self.assertTrue(all(kwargs['check'] and kwargs['cwd'] == self.root for _, kwargs in calls))
        self.assertTrue(
            all(kwargs['env']['XXTRAIN_SITE_ROOT'] == str(self.root / '.deployment') for _, kwargs in calls)
        )
        self.assertFalse((self.root / '.deployment/administrator').exists())
        self.assertFalse((self.root / '.deployment/workspace.json').exists())

    def test_start_requires_operator_configs_before_secret_or_service_calls(self) -> None:
        calls = []
        with patch('xxtrain.serverctl.site_root', return_value=self.root):
            with self.assertRaisesRegex(ValueError, 'workspace.json'):
                start(self.root, lambda args, **kwargs: calls.append(args))
            (self.root / '.deployment/workspace.json').parent.mkdir()
            (self.root / '.deployment/workspace.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'training.json'):
                start(self.root, lambda args, **kwargs: calls.append(args))
        self.assertFalse(calls)
        self.assertFalse((self.root / '.deployment/administrator').exists())

    def test_start_and_stop_repeat_without_replacing_secret(self) -> None:
        deployment = self.root / '.deployment'
        deployment.mkdir()
        for name in ('workspace.json', 'training.json', 'platform.env'):
            (deployment / name).write_text('operator-owned', encoding='utf-8')
        secret = deployment / 'administrator'
        secret.write_text('operator-secret\n', encoding='utf-8')
        calls = []
        with patch('xxtrain.serverctl.site_root', return_value=self.root):
            for _ in range(2):
                start(self.root, lambda args, **kwargs: calls.append(args))
            for _ in range(2):
                stop(self.root, lambda args, **kwargs: calls.append(args))
        self.assertEqual('operator-secret\n', secret.read_text(encoding='utf-8'))
        self.assertEqual(
            [
                ['sudo', 'systemctl', 'enable', 'xxtrain-server.service'],
                ['sudo', 'systemctl', 'start', 'xxtrain-server.service'],
            ]
            * 2
            + [
                ['sudo', 'systemctl', 'stop', 'xxtrain-server.service'],
                ['sudo', 'systemctl', 'disable', 'xxtrain-server.service'],
            ]
            * 2,
            calls,
        )

    def test_linux_install_registers_checkout_unit_without_hidden_opt_in(self) -> None:
        unit_path = self.root / 'deploy/server/xxtrain-server.service'
        unit_path.parent.mkdir(parents=True)
        unit_path.write_text('WorkingDirectory={{ROOT}}\n', encoding='utf-8')
        calls = []
        with (
            patch('xxtrain.serverctl.site_root', return_value=self.root),
            patch.dict(os.environ, {'XXTRAIN_INSTALL_SYSTEMD': '0'}),
            patch('xxtrain.serverctl.sys.platform', 'linux'),
        ):
            install(self.root, lambda args, **kwargs: calls.append((args, kwargs)))
        self.assertEqual(
            ['sudo', 'install', '-m', '0644', '/dev/stdin', '/etc/systemd/system/xxtrain-server.service'], calls[-2][0]
        )
        self.assertEqual(f'WorkingDirectory={self.root}\n', calls[-2][1]['input'])
        self.assertEqual(['sudo', 'systemctl', 'daemon-reload'], calls[-1][0])

    def test_install_rejects_existing_permissive_env_without_changing_contents(self) -> None:
        deployment = self.root / '.deployment'
        deployment.mkdir()
        env_file = deployment / 'platform.env'
        env_file.write_text('API_SECRET=keep\n', encoding='utf-8')
        os.chmod(env_file, 0o644)
        calls = []
        with (
            patch('xxtrain.serverctl.site_root', return_value=self.root),
            patch('xxtrain.serverctl.sys.platform', 'linux'),
        ):
            with self.assertRaisesRegex(ValueError, r'platform.env.*0600'):
                install(self.root, lambda args, **kwargs: calls.append(args))
        self.assertEqual('API_SECRET=keep\n', env_file.read_text(encoding='utf-8'))
        self.assertEqual([], calls)

    def test_start_rejects_env_symlink_escaping_checkout(self) -> None:
        deployment = self.root / '.deployment'
        deployment.mkdir()
        for name in ('workspace.json', 'training.json'):
            (deployment / name).write_text('{}', encoding='utf-8')
        outside = self.root.parent / f'{self.root.name}-outside-env'
        outside.write_text('API_SECRET=outside\n', encoding='utf-8')
        (deployment / 'platform.env').symlink_to(outside)
        calls = []
        try:
            with self.assertRaisesRegex(ValueError, r'platform.env.*outside'):
                start(self.root, lambda args, **kwargs: calls.append(args))
        finally:
            outside.unlink()
        self.assertEqual([], calls)
        self.assertFalse((deployment / 'administrator').exists())

    def test_install_rejects_env_symlink_escaping_checkout(self) -> None:
        deployment = self.root / '.deployment'
        deployment.mkdir()
        outside = self.root.parent / f'{self.root.name}-outside-env'
        outside.write_text('API_SECRET=outside\n', encoding='utf-8')
        (deployment / 'platform.env').symlink_to(outside)
        calls = []
        try:
            with self.assertRaisesRegex(ValueError, r'platform.env.*outside'):
                install(self.root, lambda args, **kwargs: calls.append(args))
        finally:
            outside.unlink()
        self.assertEqual([], calls)

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
        for action in ('status', 'verify'):
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

    def test_resolved_compose_bind_sources_stay_under_site_root(self) -> None:
        if not shutil.which('docker'):
            self.skipTest('Docker Compose unavailable; manifest parse check covers volume and port policy')
        checkout = Path(__file__).resolve().parents[1]
        site = (self.root / '.deployment').resolve()
        env = dict(os.environ, XXTRAIN_SITE_ROOT=str(site))
        output = subprocess.run(
            ['docker', 'compose', '-f', str(compose_files(checkout)[0]), 'config', '--format', 'json'],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        services = json.loads(output.stdout)['services']
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
                mounts = [v for v in services[service]['volumes'] if v['target'] == target]
                self.assertEqual(1, len(mounts))
                self.assertEqual('bind', mounts[0]['type'])
                self.assertTrue(Path(mounts[0]['source']).is_relative_to(site), mounts[0])
        for name, service in services.items():
            for mount in service.get('volumes', []):
                if mount['type'] == 'bind' and not mount.get('read_only'):
                    self.assertTrue(Path(mount['source']).is_relative_to(site), (name, mount))
                self.assertNotEqual('volume', mount['type'], (name, mount))
            for port in service.get('ports', []):
                self.assertEqual('127.0.0.1', port['host_ip'], (name, port))

    def test_cvat_assets_require_session_but_platform_login_does_not(self) -> None:
        config = (Path(__file__).resolve().parents[1] / 'deploy/server/nginx.conf').read_text(encoding='utf-8')

        def location(path: str) -> str:
            match = re.search(r'location ' + re.escape(path) + r' \{([^{}]*)\}', config)
            self.assertIsNotNone(match, path)
            return match.group(1)

        self.assertIn('auth_request /_cvat_session;', location('/assets/'))
        self.assertNotIn('auth_request', location('/platform/'))
        self.assertNotIn('auth_request', location('= /auth/login'))


if __name__ == '__main__':
    unittest.main()
