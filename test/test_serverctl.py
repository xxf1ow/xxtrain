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

from xxtrain import serverctl

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / 'deploy' / 'server'


class ServerInstallTests(unittest.TestCase):
    def setUp(self):
        self.controller = serverctl
        self.operator = patch('xxtrain.serverctl.os.geteuid', return_value=1000, create=True)
        self.operator.start()
        self.addCleanup(self.operator.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        subprocess.run(['git', 'init', '--quiet', str(self.root)], check=True, capture_output=True)
        (self.root / 'deploy/server').mkdir(parents=True)
        for name in ('nginx.conf', 'xxtrain-server.service'):
            source = DEPLOY / name
            if source.exists():
                shutil.copyfile(source, self.root / 'deploy/server' / name)
        self.calls = []
        self.real_run = subprocess.run

    def run_command(self, args, **kwargs):
        if args[0] == 'git':
            return self.real_run(args, **kwargs)
        self.calls.append((args, kwargs))
        output = ''
        if args[:3] == ['ip', '-j', '-4']:
            output = json.dumps([{'addr_info': [{'local': '192.168.0.109', 'scope': 'global'}]}])
        return subprocess.CompletedProcess(args, 0, output, '')

    def install(self, address='192.168.0.109'):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('XXTRAIN_SERVER_LISTEN_IP', None)
            if address is not None:
                os.environ['XXTRAIN_SERVER_LISTEN_IP'] = address
            with patch('xxtrain.serverctl.subprocess.run', side_effect=self.run_command):
                return self.controller.main('install', root=self.root)

    def test_checkout_resolution_and_password_preservation(self):
        self.assertEqual(self.controller.site_root(self.root / 'deploy'), self.root)
        password = self.controller.ensure_administrator(self.root)
        original = password.read_bytes()
        self.assertTrue(original.strip())
        self.controller.ensure_administrator(self.root)
        self.assertEqual(password.read_bytes(), original)
        password.write_bytes(b'operator-edited-password\n')
        self.controller.ensure_administrator(self.root)
        self.assertEqual(password.read_bytes(), b'operator-edited-password\n')
        if os.name == 'posix':
            self.assertEqual(password.stat().st_mode & 0o777, 0o600)

    def test_escaping_deployment_is_rejected(self):
        outside = self.root / 'outside'
        outside.mkdir()
        link = self.root / '.deployment'
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest('directory symlinks require OS permission')
        with self.assertRaisesRegex(ValueError, 'deployment'):
            self.controller.ensure_administrator(self.root)
        self.assertFalse((outside / '.xxxxx').exists())

    def test_install_prepares_without_starting_and_preserves_configuration(self):
        self.assertEqual(self.install(), 0)
        site = self.root / '.deployment'
        names = ('.xxxxx', 'platform.env', 'clearml/config/secure.conf', 'training.json', 'listen-ip')
        saved = {name: (site / name).read_bytes() for name in names}
        secure = json.loads(saved['clearml/config/secure.conf'])['credentials']['user']
        self.assertEqual(secure['role'], 'user')
        self.assertIn(f'CLEARML_API_ACCESS_KEY={secure["user_key"]}', saved['platform.env'].decode())
        self.assertIn(f'CLEARML_API_SECRET_KEY={secure["user_secret"]}', saved['platform.env'].decode())
        self.assertNotEqual(secure['user_key'], secure['user_secret'])
        self.assertNotEqual(secure['user_secret'], saved['.xxxxx'].decode().strip())
        training = json.loads(saved['training.json'])
        self.assertEqual(training['shared_root'], 'training-shared')
        self.assertEqual(training['metadata_dir'], 'training-metadata')
        self.assertEqual(training['run_root'], 'training-runs')
        self.assertFalse((site / 'workspace.json').exists())
        self.assertIn('listen 192.168.0.109:8080;', (site / 'nginx.conf').read_text())
        self.assertNotIn('{{LAN_ADDRESS}}', (site / 'nginx.conf').read_text())
        (site / 'workspace.json').write_bytes(b'operator-owned workspace')
        self.assertEqual(self.install(None), 0)
        self.assertEqual({name: (site / name).read_bytes() for name in names}, saved)
        self.assertEqual((site / 'workspace.json').read_bytes(), b'operator-owned workspace')
        commands = [args for args, _ in self.calls]
        self.assertEqual(commands.count(['uv', 'sync', '--locked', '--extra', 'platform', '--extra', 'clearml']), 2)
        self.assertEqual(sum(args[-2:] == ['pull', '--ignore-buildable'] for args in commands), 2)
        self.assertEqual(sum(args[-2:] == ['build', 'cvat_ui'] for args in commands), 2)
        registrations = [(args, kwargs) for args, kwargs in self.calls if args[:2] == ['sudo', 'install']]
        self.assertEqual(len(registrations), 2)
        for _, kwargs in registrations:
            self.assertIn(str(self.root), kwargs['input'])
        self.assertFalse(any(action in args for args in commands for action in ('start', 'enable', 'up', 'restart')))
        for args, kwargs in self.calls:
            if args[0] == 'uv':
                self.assertEqual(kwargs['env']['UV_CACHE_DIR'], str(site / 'uv-cache'))
                self.assertEqual(kwargs['env']['UV_PYTHON_INSTALL_DIR'], str(site / 'uv-python'))
            self.assertNotIn(secure['user_secret'], ' '.join(args))

    def test_invalid_or_unassigned_listen_ip_is_rejected(self):
        for address in (None, '0.0.0.0', '127.0.0.1', '8.8.8.8', '192.168.0.110', 'not-an-ip'):
            with self.subTest(address=address), redirect_stderr(StringIO()):
                self.assertNotEqual(self.install(address), 0)
        self.assertFalse(any(args[0] == 'uv' for args, _ in self.calls))

    def test_incomplete_existing_credentials_are_not_replaced(self):
        site = self.root / '.deployment'
        site.mkdir()
        env_file = site / 'platform.env'
        env_file.write_bytes(b'CLEARML_API_ACCESS_KEY=existing\n')
        if os.name == 'posix':
            env_file.chmod(0o600)
        with redirect_stderr(StringIO()):
            self.assertNotEqual(self.install(), 0)
        self.assertEqual(env_file.read_bytes(), b'CLEARML_API_ACCESS_KEY=existing\n')
        self.assertFalse(any(args[0] == 'uv' for args, _ in self.calls))

    def test_changed_listen_ip_does_not_reconfigure_site(self):
        self.assertEqual(self.install(), 0)
        original = (self.root / '.deployment/nginx.conf').read_bytes()
        self.calls.clear()
        with redirect_stderr(StringIO()):
            self.assertNotEqual(self.install('192.168.0.110'), 0)
        self.assertEqual((self.root / '.deployment/nginx.conf').read_bytes(), original)
        self.assertEqual(self.calls, [])

    def test_conflicting_clearml_credentials_are_preserved_and_rejected(self):
        self.assertEqual(self.install(), 0)
        path = self.root / '.deployment/clearml/config/secure.conf'
        original = json.loads(path.read_text())
        original['credentials']['user']['user_secret'] = 'different-private-value'
        path.write_text(json.dumps(original))
        saved = path.read_bytes()
        self.calls.clear()
        with redirect_stderr(StringIO()) as output:
            self.assertNotEqual(self.install(None), 0)
        self.assertEqual(path.read_bytes(), saved)
        self.assertNotIn('different-private-value', output.getvalue())
        self.assertFalse(any(args[0] == 'uv' for args, _ in self.calls))

    def test_escaping_bind_directory_is_rejected_before_privileged_commands(self):
        site = self.root / '.deployment'
        site.mkdir()
        outside = self.root / 'outside'
        outside.mkdir()
        try:
            (site / 'cvat').symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest('directory symlinks require OS permission')
        with redirect_stderr(StringIO()):
            self.assertNotEqual(self.install(), 0)
        self.assertEqual(self.calls, [])
        self.assertEqual(list(outside.iterdir()), [])

    def test_dependency_failure_does_not_register_or_activate_unit(self):
        def fail_uv(args, **kwargs):
            if args[0] == 'uv':
                raise subprocess.CalledProcessError(1, args, output='private dependency output')
            return self.run_command(args, **kwargs)

        with patch.dict(os.environ, XXTRAIN_SERVER_LISTEN_IP='192.168.0.109'):
            with patch('xxtrain.serverctl.subprocess.run', side_effect=fail_uv), redirect_stderr(StringIO()) as output:
                self.assertNotEqual(self.controller.main('install', root=self.root), 0)
        self.assertNotIn('private dependency output', output.getvalue())
        self.assertFalse(any(args[:2] == ['sudo', 'install'] for args, _ in self.calls))

    def test_training_storage_cannot_escape_deployment(self):
        self.assertEqual(self.install(), 0)
        path = self.root / '.deployment/training.json'
        configuration = json.loads(path.read_text())
        configuration['metadata_dir'] = '../outside'
        path.write_text(json.dumps(configuration))
        with redirect_stderr(StringIO()):
            self.assertNotEqual(self.install(None), 0)

    def test_unavailable_actions_fail_explicitly(self):
        for action in ('start', 'stop', 'status', 'verify'):
            with self.subTest(action=action), redirect_stderr(StringIO()) as output:
                self.assertNotEqual(self.controller.main(action, root=self.root), 0)
                self.assertIn('unavailable', output.getvalue())

    def test_install_rejects_root_before_writing_configuration(self):
        with patch('xxtrain.serverctl.os.geteuid', return_value=0), redirect_stderr(StringIO()):
            self.assertNotEqual(self.install(), 0)
        self.assertFalse((self.root / '.deployment').exists())
        self.assertEqual(self.calls, [])


class ServerTopologyTests(unittest.TestCase):
    def test_compose_bind_and_port_isolation(self):
        site = (ROOT / '.superpowers' / 'server-deployment' / 'test-site').resolve()
        if shutil.which('docker'):
            result = subprocess.run(
                ['docker', 'compose', '-f', str(DEPLOY / 'compose.yaml'), 'config', '--format', 'json'],
                env={**os.environ, 'XXTRAIN_SITE_ROOT': str(site)},
                capture_output=True,
                text=True,
                check=True,
            )
            config = json.loads(result.stdout)
        else:
            config = yaml.safe_load((DEPLOY / 'compose.yaml').read_text())
        self.assertEqual(config['name'], 'xxtrain-server')
        services = config['services']
        self.assertEqual(len(services), 24)
        for name, service in services.items():
            self.assertNotIn(service.get('restart'), ('always', 'unless-stopped'), name)
            for port in service.get('ports', []):
                host = port['host_ip'] if isinstance(port, dict) else port.split(':')[0]
                self.assertEqual(host, '127.0.0.1', (name, port))
            for mount in service.get('volumes', []):
                if isinstance(mount, dict):
                    if mount['type'] == 'tmpfs':
                        continue
                    self.assertEqual(mount['type'], 'bind')
                    source = mount['source']
                    target = mount['target']
                    readonly = mount.get('read_only', False)
                else:
                    source, target = mount.split(':/', 1)
                    target = '/' + target.split(':')[0]
                    readonly = mount.endswith(':ro')
                if source.startswith('${XXTRAIN_SITE_ROOT:?}/'):
                    source = str(site / source.removeprefix('${XXTRAIN_SITE_ROOT:?}/'))
                elif source.startswith('./'):
                    source = str((DEPLOY / source).resolve())
                self.assertTrue(Path(source).is_relative_to(site) or Path(source).is_relative_to(DEPLOY), (name, mount))
                if Path(source).is_relative_to(DEPLOY):
                    self.assertTrue(readonly, mount)
                if name == 'nginx' and target == '/etc/nginx/conf.d/default.conf':
                    self.assertEqual(Path(source), site / 'nginx.conf')
        self.assertEqual(
            Path(services['cvat_ui']['build']['context']).resolve()
            if shutil.which('docker')
            else (DEPLOY / services['cvat_ui']['build']['context']).resolve(),
            ROOT,
        )
        self.assertEqual(services['cvat_server']['image'], 'cvat/server:v2.51.0')
        self.assertEqual(services['clearml_apiserver']['image'], 'clearml/server:2.4.0')
        self.assertEqual(services['nginx']['image'], 'nginx:1.27-alpine')
        for name, published in [('cvat_server', '18080'), ('clearml_apiserver', '18083')]:
            self.assertIn(published, str(services[name]['ports']))
        for name, target in [
            ('cvat_redis_ondisk', '/var/lib/kvrocks'),
            ('cvat_clickhouse', '/var/log/clickhouse-server'),
        ]:
            mounts = services[name]['volumes']
            targets = [
                mount['target'] if isinstance(mount, dict) else '/' + mount.split(':/', 1)[1].split(':')[0]
                for mount in mounts
            ]
            self.assertIn(target, targets)

    def test_nginx_session_gate_and_listeners(self):
        template = (DEPLOY / 'nginx.conf').read_text()
        self.assertNotIn('192.168.0.109', template)
        self.assertEqual(len(re.findall(r'listen \{\{LAN_ADDRESS\}\}:\d+;', template)), 4)
        session = re.search(r'location = /_cvat_session\s*\{([^}]+)\}', template).group(1)
        self.assertIn('internal;', session)
        self.assertIn('proxy_pass http://127.0.0.1:18001/platform/api/session;', session)
        self.assertIn('proxy_set_header Cookie $http_cookie;', session)
        for path in ('/assets/', '/api/', '/static/', '/'):
            body = re.search(r'location ' + re.escape(path) + r'\s*\{([^}]+)\}', template).group(1)
            self.assertIn('auth_request /_cvat_session;', body, path)
        platform = re.search(r'location /platform/\s*\{([^}]+)\}', template).group(1)
        self.assertNotIn('auth_request', platform)

    def test_websocket_proxy_uses_http_1_1(self):
        template = (DEPLOY / 'nginx.conf').read_text()
        for path in ('/api/', '/'):
            body = re.search(r'location ' + re.escape(path) + r'\s*\{([^}]+)\}', template).group(1)
            self.assertIn('proxy_set_header Upgrade $http_upgrade;', body)
            self.assertIn('proxy_http_version 1.1;', body)


if __name__ == '__main__':
    unittest.main()
