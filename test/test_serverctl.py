import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import yaml

from xxtrain.platform.training_config import load_training_config
from xxtrain.serverctl import compose_files, ensure_administrator, install, main, site_root, start, status, stop, verify


class BootstrapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        self.deployment = self.root / '.deployment'
        self.deployment.mkdir()
        (self.deployment / '.xxxxx').write_text('private-password\n', encoding='utf-8')
        (self.deployment / 'platform.env').write_text(
            'CLEARML_API_ACCESS_KEY=access\nCLEARML_API_SECRET_KEY=secret\n', encoding='utf-8'
        )
        if os.name != 'nt':
            os.chmod(self.deployment / '.xxxxx', 0o600)
            os.chmod(self.deployment / 'platform.env', 0o600)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_bootstrap_waits_for_migrations_and_uses_actual_cvat_identity(self) -> None:
        from xxtrain.serverctl import bootstrap

        commands = []
        clock = [0.0]

        def run(args, **kwargs):
            commands.append((args, kwargs))
            if len(commands) == 1:
                raise subprocess.CalledProcessError(1, args)
            return subprocess.CompletedProcess(args, 0)

        class Response:
            status = 200

            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def read(self):
                return json.dumps(self.payload).encode()

        requests = []

        def request(req, timeout=5):
            requests.append(req)
            if req.full_url.endswith('/api/auth/login'):
                return Response({'key': 'issued-key'})
            if req.full_url.endswith('/api/users/self'):
                return Response({'id': 89})
            return Response({'meta': {'result_code': 200}, 'data': {'token': 'jwt'}})

        def sleep(seconds):
            clock[0] += seconds

        with patch('xxtrain.serverctl.site_root', return_value=self.root):
            bootstrap(self.root, run=run, request=request, monotonic=lambda: clock[0], sleep=sleep, timeout=8)
        self.assertGreaterEqual(len(commands), 3)
        self.assertTrue(all('private-password' not in ' '.join(args) for args, _ in commands))
        self.assertEqual('private-password\n', commands[-1][1]['input'])
        self.assertIn('xxadmin', ' '.join(commands[-1][0]))
        self.assertEqual(
            89, json.loads((self.deployment / 'workspace.json').read_text(encoding='utf-8'))['owner_user_id']
        )
        self.assertIn('XXTRAIN_CVAT_SERVICE_TOKEN=issued-key\n', (self.deployment / 'platform.env').read_text())
        self.assertEqual('Token issued-key', requests[-2].get_header('Authorization'))

    def test_bootstrap_timeout_does_not_print_password(self) -> None:
        from xxtrain.serverctl import bootstrap

        clock = [0.0]

        def sleep(seconds):
            clock[0] += seconds

        with patch('xxtrain.serverctl.site_root', return_value=self.root):
            with self.assertRaisesRegex(ValueError, 'CVAT') as error:
                bootstrap(
                    self.root,
                    run=lambda args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, args)),
                    monotonic=lambda: clock[0],
                    sleep=sleep,
                    timeout=2,
                )
        self.assertNotIn('private-password', str(error.exception))
        self.assertFalse((self.deployment / 'workspace.json').exists())

    def test_valid_stored_token_and_existing_workspace_survive_bootstrap(self) -> None:
        from xxtrain.serverctl import bootstrap

        env_file = self.deployment / 'platform.env'
        original_env = env_file.read_bytes() + b'XXTRAIN_CVAT_SERVICE_TOKEN=stable\n'
        env_file.write_bytes(original_env)
        workspace = self.deployment / 'workspace.json'
        workspace.write_bytes(b'{ "preserved": true }\n')

        class Response:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def read(self):
                return json.dumps(self.payload).encode()

        def request(req, timeout=5):
            if req.full_url.endswith('/api/users/self'):
                self.assertEqual('Token stable', req.get_header('Authorization'))
                return Response({'id': 17})
            self.assertTrue(req.full_url.endswith('/auth.login'))
            self.assertTrue(req.get_header('Authorization').startswith('Basic '))
            return Response({'meta': {'result_code': 200}, 'data': {'token': 'jwt'}})

        with patch('xxtrain.serverctl.site_root', return_value=self.root):
            bootstrap(self.root, run=lambda args, **kwargs: subprocess.CompletedProcess(args, 0), request=request)
        self.assertEqual(original_env, env_file.read_bytes())
        self.assertEqual(b'{ "preserved": true }\n', workspace.read_bytes())

    def test_clearml_anonymous_response_cannot_authenticate_and_token_is_not_saved(self) -> None:
        from xxtrain.serverctl import bootstrap

        class Response:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def read(self):
                return json.dumps(self.payload).encode()

        def request(req, timeout=5):
            if req.full_url.endswith('/api/auth/login'):
                return Response({'key': 'unsaved'})
            if req.full_url.endswith('/api/users/self'):
                return Response({'id': 7})
            return Response({'meta': {'result_code': 200}, 'data': {}})

        original = (self.deployment / 'platform.env').read_bytes()
        with patch('xxtrain.serverctl.site_root', return_value=self.root):
            with self.assertRaisesRegex(ValueError, 'ClearML'):
                bootstrap(self.root, run=lambda args, **kwargs: subprocess.CompletedProcess(args, 0), request=request)
        self.assertEqual(original, (self.deployment / 'platform.env').read_bytes())
        self.assertFalse((self.deployment / 'workspace.json').exists())

    def test_clearml_api_unavailable_is_retried_within_deadline(self) -> None:
        from xxtrain.serverctl import bootstrap

        clock = [0.0]
        attempts = [0]
        (self.deployment / 'platform.env').write_bytes(
            (self.deployment / 'platform.env').read_bytes() + b'XXTRAIN_CVAT_SERVICE_TOKEN=stable\n'
        )

        class Response:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def read(self):
                return json.dumps(self.payload).encode()

        def request(req, timeout=5):
            if req.full_url.endswith('/api/users/self'):
                return Response({'id': 7})
            attempts[0] += 1
            if attempts[0] == 1:
                raise OSError('not ready')
            return Response({'meta': {'result_code': 200}, 'data': {'token': 'jwt'}})

        def sleep(seconds):
            clock[0] += seconds

        with patch('xxtrain.serverctl.site_root', return_value=self.root):
            bootstrap(
                self.root,
                run=lambda args, **kwargs: subprocess.CompletedProcess(args, 0),
                request=request,
                monotonic=lambda: clock[0],
                sleep=sleep,
                timeout=3,
            )
        self.assertEqual(2, attempts[0])

    def test_clearml_rejected_key_secret_fails_without_retry(self) -> None:
        from urllib.error import HTTPError

        from xxtrain.serverctl import bootstrap

        (self.deployment / 'platform.env').write_bytes(
            (self.deployment / 'platform.env').read_bytes() + b'XXTRAIN_CVAT_SERVICE_TOKEN=stable\n'
        )

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def read(self):
                return b'{"id": 7}'

        attempts = []

        def request(req, timeout=5):
            if req.full_url.endswith('/api/users/self'):
                return Response()
            attempts.append(req.get_header('Authorization'))
            raise HTTPError(req.full_url, 401, 'wrong credentials', {}, None)

        with patch('xxtrain.serverctl.site_root', return_value=self.root):
            with self.assertRaisesRegex(ValueError, 'ClearML') as failure:
                bootstrap(self.root, run=lambda args, **kwargs: subprocess.CompletedProcess(args, 0), request=request)
        self.assertEqual(1, len(attempts))
        self.assertTrue(attempts[0].startswith('Basic '))
        self.assertNotIn('secret', str(failure.exception))

    def test_rejected_stored_token_is_replaced_only_after_login_and_authentication(self) -> None:
        from urllib.error import HTTPError

        from xxtrain.serverctl import bootstrap

        env_file = self.deployment / 'platform.env'
        env_file.write_bytes(env_file.read_bytes() + b'XXTRAIN_CVAT_SERVICE_TOKEN=expired\n')

        class Response:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def read(self):
                return json.dumps(self.payload).encode()

        def request(req, timeout=5):
            if req.full_url.endswith('/api/users/self'):
                if req.get_header('Authorization') == 'Token expired':
                    raise HTTPError(req.full_url, 401, 'invalid', {}, None)
                self.assertEqual('Token fresh', req.get_header('Authorization'))
                return Response({'id': 15})
            if req.full_url.endswith('/api/auth/login'):
                return Response({'key': 'fresh'})
            return Response({'meta': {'result_code': 200}, 'data': {'token': 'jwt'}})

        with patch('xxtrain.serverctl.site_root', return_value=self.root):
            bootstrap(self.root, run=lambda args, **kwargs: subprocess.CompletedProcess(args, 0), request=request)
        self.assertEqual(1, env_file.read_text(encoding='utf-8').count('XXTRAIN_CVAT_SERVICE_TOKEN='))
        self.assertIn('XXTRAIN_CVAT_SERVICE_TOKEN=fresh\n', env_file.read_text(encoding='utf-8'))


class ServerctlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_install_prepares_local_configuration_once_without_starting_services(self) -> None:
        calls = []

        def run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:4] == ['sudo', 'mkdir', '-p', '--']:
                Path(args[-1]).mkdir(parents=True, exist_ok=True)

        (self.root / '.deployment').mkdir()
        env_file = self.root / '.deployment/platform.env'
        env_file.write_text('OPERATOR_SECRET=kept\nXXTRAIN_CVAT_SERVICE_TOKEN=preserve-me\n', encoding='utf-8')
        if os.name != 'nt':
            os.chmod(env_file, 0o600)
        with patch('xxtrain.serverctl.site_root', return_value=self.root):
            install(self.root, run)
            original = {
                path.relative_to(self.root / '.deployment').as_posix(): path.read_bytes()
                for path in (self.root / '.deployment').rglob('*')
                if path.is_file()
            }
            install(self.root, run)
        self.assertEqual(
            original,
            {
                path.relative_to(self.root / '.deployment').as_posix(): path.read_bytes()
                for path in (self.root / '.deployment').rglob('*')
                if path.is_file()
            },
        )
        expected = [
            ['uv', 'sync', '--locked', '--extra', 'platform', '--extra', 'clearml'],
            [
                'docker',
                'compose',
                '-p',
                'xxtrain-server',
                '-f',
                str(self.root / 'deploy/server/compose.yaml'),
                'pull',
                '--ignore-buildable',
            ],
            ['docker', 'compose', '-p', 'xxtrain-server', '-f', str(self.root / 'deploy/server/compose.yaml'), 'build'],
        ]
        self.assertEqual(expected * 2, [args for args, _ in calls if args[0] != 'sudo'])
        self.assertTrue(all(kwargs['check'] and kwargs['cwd'] == self.root for _, kwargs in calls))
        self.assertTrue(
            all(kwargs['env']['XXTRAIN_SITE_ROOT'] == str(self.root / '.deployment') for _, kwargs in calls)
        )
        self.assertTrue((self.root / '.deployment/.xxxxx').is_file())
        self.assertFalse((self.root / '.deployment/workspace.json').exists())
        training_path = self.root / '.deployment/training.json'
        self.assertEqual(6, len(json.loads(training_path.read_text(encoding='utf-8'))))
        training = load_training_config(training_path)
        self.assertTrue(training.shared_root.is_relative_to((self.root / '.deployment').resolve()))
        self.assertTrue(training.metadata_dir.is_relative_to((self.root / '.deployment').resolve()))
        self.assertTrue(training.run_root.is_relative_to((self.root / '.deployment').resolve()))
        self.assertEqual((self.root / 'src/xxtrain/integrations/clearml/worker.py').resolve(), training.worker_script)
        if os.name != 'nt':
            self.assertEqual(0o600, os.stat(self.root / '.deployment/.xxxxx').st_mode & 0o777)
        env = dict(line.split('=', 1) for line in env_file.read_text(encoding='utf-8').splitlines())
        self.assertEqual(
            {
                'CLEARML_API_HOST': 'http://127.0.0.1:18083',
                'CLEARML_WEB_HOST': 'http://127.0.0.1:18084',
                'CLEARML_FILES_HOST': 'http://127.0.0.1:18082',
            },
            {name: env[name] for name in ('CLEARML_API_HOST', 'CLEARML_WEB_HOST', 'CLEARML_FILES_HOST')},
        )
        self.assertTrue(env['CLEARML_API_ACCESS_KEY'])
        self.assertTrue(env['CLEARML_API_SECRET_KEY'])
        self.assertTrue(env['CLEARML_API_ACCESS_KEY'] != env['CLEARML_API_SECRET_KEY'])
        self.assertEqual('preserve-me', env['XXTRAIN_CVAT_SERVICE_TOKEN'])
        self.assertFalse(any('up' in args or ('systemctl' in args and 'start' in args) for args, _ in calls))
        secure = (self.root / '.deployment/clearml/config/secure.conf').read_text(encoding='utf-8')
        self.assertIn('      user {', secure)
        self.assertTrue(env['CLEARML_API_ACCESS_KEY'] in secure)
        self.assertTrue(env['CLEARML_API_SECRET_KEY'] in secure)
        if os.name != 'nt':
            self.assertEqual(0o600, os.stat(self.root / '.deployment/clearml/config/secure.conf').st_mode & 0o777)

    def test_lifecycle_actions_require_noninteractive_sudo_before_site_changes(self) -> None:
        for action in (install, start, stop):
            with self.subTest(action=action.__name__):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    calls = []

                    def run(args, **kwargs):
                        calls.append(args)
                        result = subprocess.CompletedProcess(args, 1 if args == ['sudo', '-n', '-v'] else 0)
                        if result.returncode and kwargs.get('check'):
                            raise subprocess.CalledProcessError(result.returncode, args)
                        return result

                    with (
                        patch('xxtrain.serverctl.site_root', return_value=root),
                        patch('xxtrain.serverctl.sys.platform', 'linux'),
                        patch('xxtrain.serverctl._check_private_env'),
                        self.assertRaises(subprocess.CalledProcessError),
                    ):
                        action(root, run)

                    self.assertEqual([['sudo', '-n', '-v']], calls)
                    self.assertFalse((root / '.deployment').exists())

    def test_cli_lifecycle_entry_runs_real_install_start_and_stop_controllers(self) -> None:
        from xxtrain import serverctl
        from xxtrain.cli import main as cli_main
        from xxtrain.serverctl import install as controller_install
        from xxtrain.serverctl import start as controller_start
        from xxtrain.serverctl import stop as controller_stop

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls = []

            def run(args, **kwargs):
                calls.append(args)
                if args[:4] == ['sudo', 'mkdir', '-p', '--']:
                    Path(args[-1]).mkdir(parents=True, exist_ok=True)
                if args[:3] == ['systemctl', 'is-active', '--quiet']:
                    return subprocess.CompletedProcess(args, 3)
                return subprocess.CompletedProcess(args, 0)

            with (
                patch('xxtrain.serverctl.site_root', return_value=root),
                patch('xxtrain.serverctl.sys.platform', 'win32'),
                patch.object(serverctl, 'install', side_effect=lambda path: controller_install(path, run)),
                patch.object(serverctl, 'start', side_effect=lambda path: controller_start(path, run)),
                patch.object(serverctl, 'stop', side_effect=lambda path: controller_stop(path, run)),
            ):
                for action in ('install', 'start', 'stop'):
                    cli_main(['serverctl', action])

            deployment = root / '.deployment'
            self.assertTrue((deployment / '.xxxxx').is_file())
            self.assertTrue((deployment / 'platform.env').is_file())
            self.assertTrue((deployment / 'training.json').is_file())
            self.assertFalse((deployment / 'workspace.json').exists())
            self.assertFalse(any('up' in args for args in calls))
            self.assertIn(['sudo', 'systemctl', 'enable', 'xxtrain-server.service'], calls)
            self.assertIn(['sudo', 'systemctl', 'start', 'xxtrain-server.service'], calls)
            self.assertIn(['sudo', 'systemctl', 'stop', 'xxtrain-server.service'], calls)
            self.assertIn(['sudo', 'systemctl', 'disable', 'xxtrain-server.service'], calls)

    def test_install_rejects_partial_clearml_pair_without_changing_environment(self) -> None:
        deployment = self.root / '.deployment'
        deployment.mkdir()
        env_file = deployment / 'platform.env'
        original = b'CLEARML_API_ACCESS_KEY=existing-key\nKEEP=value\n'
        env_file.write_bytes(original)
        calls = []

        with (
            patch('xxtrain.serverctl.site_root', return_value=self.root),
            patch('xxtrain.serverctl.sys.platform', 'win32'),
        ):
            with self.assertRaisesRegex(ValueError, 'CLEARML_API_SECRET_KEY'):
                install(self.root, lambda args, **kwargs: calls.append(args))

        self.assertEqual(original, env_file.read_bytes())
        self.assertEqual([], calls)

    def test_install_rejects_empty_clearml_pair_without_changing_environment(self) -> None:
        deployment = self.root / '.deployment'
        deployment.mkdir()
        env_file = deployment / 'platform.env'
        original = b'CLEARML_API_ACCESS_KEY=\nCLEARML_API_SECRET_KEY=\n'
        env_file.write_bytes(original)
        if os.name != 'nt':
            os.chmod(env_file, 0o600)
        calls = []

        # Exercise pair validation with empty parsed values; the env parser rejects blank assignments earlier.
        for values in (
            {'CLEARML_API_ACCESS_KEY': '', 'CLEARML_API_SECRET_KEY': ''},
            {'CLEARML_API_ACCESS_KEY': '', 'CLEARML_API_SECRET_KEY': 'existing-secret'},
            {'CLEARML_API_ACCESS_KEY': 'existing-key', 'CLEARML_API_SECRET_KEY': ''},
        ):
            with self.subTest(values=tuple(values.values())):
                env_file.write_bytes(original)
                calls.clear()
                with (
                    patch('xxtrain.serverctl.site_root', return_value=self.root),
                    patch(
                        'xxtrain.serverctl._platform_environment',
                        return_value=(original.decode().splitlines(True), values),
                    ),
                ):
                    with self.assertRaisesRegex(ValueError, 'ClearML keys must not be empty'):
                        install(self.root, lambda args, **kwargs: calls.append(args))

                self.assertEqual(original, env_file.read_bytes())
                self.assertEqual([], calls)

    def test_start_regenerates_clearml_derivative_from_authoritative_pair(self) -> None:
        deployment = self.root / '.deployment'
        deployment.mkdir()
        (deployment / 'workspace.json').write_text('{}', encoding='utf-8')
        env_file = deployment / 'platform.env'
        env_file.write_text('CLEARML_API_ACCESS_KEY=first-key\nCLEARML_API_SECRET_KEY=first-secret\n', encoding='utf-8')
        if os.name != 'nt':
            os.chmod(env_file, 0o600)
        calls = []

        expected_pair = ('first-key', 'first-secret')

        def run(args, **kwargs):
            calls.append(args)
            if args[-2:] == ['systemctl', 'start']:
                configured = (deployment / 'clearml/config/secure.conf').read_text(encoding='utf-8')
                self.assertTrue(expected_pair[0] in configured and expected_pair[1] in configured)

        with patch('xxtrain.serverctl.site_root', return_value=self.root):
            start(self.root, run)
            first_secure = (deployment / 'clearml/config/secure.conf').read_bytes()
            env_file.write_text(
                'CLEARML_API_ACCESS_KEY=changed-key\nCLEARML_API_SECRET_KEY=changed-secret\n', encoding='utf-8'
            )
            expected_pair = ('changed-key', 'changed-secret')
            start(self.root, run)

        regenerated = (deployment / 'clearml/config/secure.conf').read_text(encoding='utf-8')
        self.assertTrue(first_secure != regenerated.encode())
        self.assertTrue('changed-key' in regenerated)
        self.assertTrue('changed-secret' in regenerated)
        self.assertEqual(
            [
                ['systemctl', 'is-active', '--quiet', 'xxtrain-server.service'],
                ['sudo', 'systemctl', 'enable', 'xxtrain-server.service'],
                ['sudo', 'systemctl', 'start', 'xxtrain-server.service'],
            ]
            * 2,
            calls,
        )

    def test_install_prepares_only_pinned_non_root_bind_directories_idempotently(self) -> None:
        deployment = self.root / '.deployment'
        existing = deployment / 'cvat/keys/secret_key.py'
        existing.parent.mkdir(parents=True)
        existing.write_text('preserve-me', encoding='utf-8')
        calls = []

        def run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:4] == ['sudo', 'mkdir', '-p', '--']:
                Path(args[-1]).mkdir(parents=True, exist_ok=True)

        with (
            patch('xxtrain.serverctl.site_root', return_value=self.root),
            patch('xxtrain.serverctl.sys.platform', 'win32'),
        ):
            install(self.root, run)
            install(self.root, run)

        managed = {
            deployment / 'cvat/data': ('1000:1000',),
            deployment / 'cvat/keys': ('1000:1000',),
            deployment / 'cvat/logs': ('1000:1000',),
            deployment / 'cvat/kvrocks/data': ('999:999',),
            deployment / 'clearml/elasticsearch': ('1000:0',),
            deployment / 'clearml/elasticsearch-logs': ('1000:0',),
        }
        permission_calls = [args for args, _ in calls if args[:2] in (['sudo', 'chown'], ['sudo', 'chmod'])]
        for path, (owner,) in managed.items():
            self.assertTrue(path.is_dir())
            self.assertEqual(2, permission_calls.count(['sudo', 'chown', f'{owner}', str(path)]))
            self.assertEqual(2, permission_calls.count(['sudo', 'chmod', '0770', str(path)]))
        self.assertEqual('preserve-me', existing.read_text(encoding='utf-8'))

    def test_install_rejects_bind_directory_symlink_escape_before_privileged_calls(self) -> None:
        deployment = self.root / '.deployment'
        deployment.mkdir()
        outside = self.root.parent / f'{self.root.name}-outside-data'
        outside.mkdir()
        (deployment / 'cvat').mkdir()
        (deployment / 'cvat/keys').symlink_to(outside, target_is_directory=True)
        calls = []
        try:
            with (
                patch('xxtrain.serverctl.site_root', return_value=self.root),
                patch('xxtrain.serverctl.sys.platform', 'win32'),
            ):
                with self.assertRaisesRegex(ValueError, r'\.deployment/cvat/keys.*outside'):
                    install(self.root, lambda args, **kwargs: calls.append(args))
        finally:
            (deployment / 'cvat/keys').unlink()
            outside.rmdir()
        self.assertFalse(any(args[0] == 'sudo' for args in calls))

    def test_start_prepares_local_configuration_before_unit_bootstrap_creates_workspace(self) -> None:
        calls = []
        with patch('xxtrain.serverctl.site_root', return_value=self.root):
            start(self.root, lambda args, **kwargs: calls.append(args))
        self.assertEqual(
            [
                ['systemctl', 'is-active', '--quiet', 'xxtrain-server.service'],
                ['sudo', 'systemctl', 'enable', 'xxtrain-server.service'],
                ['sudo', 'systemctl', 'start', 'xxtrain-server.service'],
            ],
            calls,
        )
        self.assertTrue((self.root / '.deployment/training.json').is_file())
        self.assertTrue((self.root / '.deployment/.xxxxx').is_file())
        self.assertFalse((self.root / '.deployment/workspace.json').exists())

    def test_start_and_stop_repeat_without_replacing_secret(self) -> None:
        deployment = self.root / '.deployment'
        deployment.mkdir()
        (deployment / 'workspace.json').write_text('{}', encoding='utf-8')
        (deployment / 'platform.env').write_text(
            'CLEARML_API_ACCESS_KEY=access\nCLEARML_API_SECRET_KEY=secret\n', encoding='utf-8'
        )
        secret = deployment / '.xxxxx'
        secret.write_text('operator-secret\n', encoding='utf-8')
        if os.name != 'nt':
            os.chmod(deployment / 'platform.env', 0o600)
            os.chmod(secret, 0o600)
        calls = []
        with patch('xxtrain.serverctl.site_root', return_value=self.root):
            for _ in range(2):
                start(self.root, lambda args, **kwargs: calls.append(args))
            for _ in range(2):
                stop(self.root, lambda args, **kwargs: calls.append(args))
        self.assertEqual('operator-secret\n', secret.read_text(encoding='utf-8'))
        self.assertEqual(
            [
                ['systemctl', 'is-active', '--quiet', 'xxtrain-server.service'],
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
            patch('xxtrain.serverctl._check_private_env'),
        ):
            install(self.root, lambda args, **kwargs: calls.append((args, kwargs)))
        self.assertEqual(
            ['sudo', 'install', '-m', '0644', '/dev/stdin', '/etc/systemd/system/xxtrain-server.service'], calls[-2][0]
        )
        self.assertEqual(f'WorkingDirectory={self.root}\n', calls[-2][1]['input'])
        self.assertEqual(['sudo', 'systemctl', 'daemon-reload'], calls[-1][0])

    def test_installed_unit_orders_private_prepare_compose_bootstrap_and_foreground(self) -> None:
        actual = Path(__file__).resolve().parents[1] / 'deploy/server/xxtrain-server.service'
        destination = self.root / 'deploy/server/xxtrain-server.service'
        destination.parent.mkdir(parents=True)
        destination.write_bytes(actual.read_bytes())
        calls = []
        with (
            patch('xxtrain.serverctl.site_root', return_value=self.root),
            patch('xxtrain.serverctl.sys.platform', 'linux'),
            patch('xxtrain.serverctl._check_private_env'),
            patch('xxtrain.serverctl.getpass.getuser', return_value='site-operator'),
        ):
            install(self.root, lambda args, **kwargs: calls.append((args, kwargs)))
        unit = calls[-2][1]['input']
        lines = unit.splitlines()
        self.assertIn('User=site-operator', lines)
        self.assertLess(
            next(i for i, line in enumerate(lines) if 'serverctl prepare' in line),
            next(i for i, line in enumerate(lines) if 'up -d --no-build' in line),
        )
        self.assertLess(
            next(i for i, line in enumerate(lines) if 'up -d --no-build' in line),
            next(i for i, line in enumerate(lines) if 'serverctl bootstrap --timeout 120' in line),
        )
        self.assertLess(
            next(i for i, line in enumerate(lines) if 'serverctl bootstrap' in line),
            next(i for i, line in enumerate(lines) if 'serverctl foreground' in line),
        )
        self.assertNotIn('EnvironmentFile=', unit)
        self.assertIn('ExecStopPost=/usr/bin/docker compose', unit)

    def test_active_start_repeats_password_synchronization(self) -> None:
        calls = []
        secret = self.root / '.deployment/.xxxxx'

        def run(args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0)

        with (
            patch('xxtrain.serverctl.site_root', return_value=self.root),
            patch('xxtrain.serverctl.bootstrap') as synchronize,
        ):
            start(self.root, run)
            secret.write_text('edited\n', encoding='utf-8')
            start(self.root, run)
        self.assertEqual(2, synchronize.call_count)
        self.assertEqual('edited\n', secret.read_text(encoding='utf-8'))

    def test_cli_start_propagates_bootstrap_failure_and_stops_active_unit(self) -> None:
        from xxtrain.cli import main as cli_main
        from xxtrain.serverctl import start as controller_start

        calls = []

        def run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0)

        with (
            patch('xxtrain.serverctl.site_root', return_value=self.root),
            patch('xxtrain.serverctl._prepare_local_configuration'),
            patch('xxtrain.serverctl.start', side_effect=lambda root: controller_start(root, run)),
            patch('xxtrain.serverctl.bootstrap', side_effect=ValueError('CVAT not ready')),
            self.assertRaises(SystemExit) as raised,
        ):
            cli_main(['serverctl', 'start'])

        self.assertEqual(1, raised.exception.code)
        self.assertIn(['sudo', 'systemctl', 'stop', 'xxtrain-server.service'], calls)

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
        self.assertEqual([['sudo', '-n', '-v']], calls)

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
        self.assertFalse((deployment / '.xxxxx').exists())
        self.assertFalse((deployment / 'clearml').exists())

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
        self.assertEqual(['platform.env'], [path.name for path in deployment.iterdir()])

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
            with self.assertRaisesRegex(ValueError, r'\.deployment.*symlink'):
                site_root(self.root)
        finally:
            deployment.unlink(missing_ok=True)
            outside.rmdir()

    def test_site_root_rejects_deployment_symlink_to_another_checkout_directory(self) -> None:
        deployment = self.root / '.deployment'
        inside = self.root / 'other-data'
        inside.mkdir()
        deployment.symlink_to(inside, target_is_directory=True)

        with self.assertRaisesRegex(ValueError, r'\.deployment.*symlink'):
            site_root(self.root)

    def test_existing_administrator_secret_is_never_replaced(self) -> None:
        path = self.root / '.deployment' / '.xxxxx'
        path.parent.mkdir()
        path.write_text('operator-chosen-secret\n', encoding='utf-8')
        if os.name != 'nt':
            os.chmod(path, 0o600)
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
        path = self.root / '.deployment' / '.xxxxx'

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
        with redirect_stderr(StringIO()) as errors:
            self.assertNotEqual(0, main('unknown', root=self.root))
            self.assertIn('unknown', errors.getvalue())
        self.assertFalse((self.root / '.deployment').exists())

    def test_status_reports_git_systemd_and_each_container_state_without_secrets(self) -> None:
        sha = 'a' * 40

        def run(args, **kwargs):
            if args[:2] == ['git', 'rev-parse']:
                return subprocess.CompletedProcess(args, 0, sha + '\n', '')
            if args[:2] == ['git', 'status']:
                return subprocess.CompletedProcess(args, 0, ' M src/xxtrain/serverctl.py\n', '')
            if args[:2] == ['systemctl', 'is-enabled']:
                return subprocess.CompletedProcess(args, 0, 'enabled\n', '')
            if args[:2] == ['systemctl', 'is-active']:
                return subprocess.CompletedProcess(args, 3, 'inactive\n', '')
            if args[-2:] == ['config', '--services']:
                return subprocess.CompletedProcess(args, 0, 'nginx\ncvat_server\n', '')
            if args[-4:] == ['ps', '--all', '--format', 'json']:
                return subprocess.CompletedProcess(
                    args,
                    0,
                    json.dumps(
                        [
                            {'Service': 'nginx', 'State': 'running', 'Health': 'healthy'},
                            {'Service': 'cvat_server', 'State': 'exited', 'Health': ''},
                        ]
                    ),
                    '',
                )
            raise AssertionError(args)

        output = StringIO()
        with patch('xxtrain.serverctl.site_root', return_value=self.root), redirect_stdout(output):
            status(self.root, run)
        result = output.getvalue()
        for value in (
            sha,
            'dirty',
            'enabled',
            'inactive',
            'nginx: running (healthy)',
            'cvat_server: exited (health unavailable)',
        ):
            self.assertIn(value, result)
        self.assertNotIn('OPERATOR_SECRET', result)

    def test_status_reports_missing_compose_service_as_failure(self) -> None:
        def run(args, **kwargs):
            if args[:2] == ['git', 'rev-parse']:
                value = 'a' * 40
            elif args[:2] == ['git', 'status']:
                value = ''
            elif args[-2:] == ['config', '--services']:
                value = 'nginx\ncvat_server\n'
            elif args[-4:] == ['ps', '--all', '--format', 'json']:
                value = json.dumps([{'Service': 'nginx', 'State': 'running', 'Health': 'healthy'}])
            else:
                value = 'active' if 'is-active' in args else 'enabled'
            return subprocess.CompletedProcess(args, 0, value, '')

        output = StringIO()
        with patch('xxtrain.serverctl.site_root', return_value=self.root), redirect_stdout(output):
            self.assertFalse(status(self.root, run))
        self.assertIn('cvat_server: missing', output.getvalue())

    def test_verify_reports_failed_health_and_returns_failure(self) -> None:
        import urllib.error

        def request(url, timeout):
            if url.endswith('/api/server/health/'):
                raise urllib.error.URLError('backend unavailable')
            return type(
                'Reply', (), {'status': 200, '__enter__': lambda self: self, '__exit__': lambda self, *args: None}
            )()

        output = StringIO()
        with patch('xxtrain.serverctl.site_root', return_value=self.root), redirect_stdout(output):
            self.assertFalse(verify(self.root, request))
        self.assertIn('FAIL', output.getvalue())
        self.assertNotIn('backend unavailable', output.getvalue())

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
            'cvat_redis_ondisk': '/var/lib/kvrocks/data',
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
        self.assertEqual(
            '${XXTRAIN_SITE_ROOT:?}/cvat/kvrocks/data:/var/lib/kvrocks/data',
            services['cvat_redis_ondisk']['volumes'][0],
        )
        self.assertEqual(
            [
                'kvrocks',
                '-c',
                '/var/lib/kvrocks/kvrocks.conf',
                '--dir',
                '/var/lib/kvrocks/data',
                '--pidfile',
                '/var/run/kvrocks/kvrocks.pid',
                '--bind',
                '0.0.0.0',
            ],
            services['cvat_redis_ondisk']['entrypoint'],
        )
        self.assertIn('apiserver', services['clearml_apiserver']['networks']['default']['aliases'])
        self.assertIn('fileserver', services['clearml_fileserver']['networks']['default']['aliases'])
        self.assertEqual('', services['cvat_server']['environment']['SMOKESCREEN_OPTS'])
        self.assertNotIn('ENV_SMOKESCREEN_OPTS', services['cvat_server']['environment'])
        for worker in ('utils', 'import', 'export', 'annotation', 'webhooks', 'quality_reports', 'chunks', 'consensus'):
            with self.subTest(worker=worker):
                self.assertEqual('', services[f'cvat_worker_{worker}']['environment']['SMOKESCREEN_OPTS'])
        self.assertIn('--set=services.cvat.url=http://cvat-server:8080', services['cvat_opa']['command'])
        self.assertIn('redis', services['clearml_redis']['networks']['default']['aliases'])
        self.assertEqual(
            {'condition': 'service_healthy'}, services['clearml_fileserver']['depends_on']['clearml_apiserver']
        )
        self.assertEqual(
            [
                'CMD',
                'python3',
                '-c',
                "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8008/debug.ping', timeout=2)",
            ],
            services['clearml_apiserver']['healthcheck']['test'],
        )
        self.assertNotIn('clearml_fileserver', services['clearml_apiserver']['depends_on'])
        self.assertEqual(
            {'condition': 'service_healthy'}, services['clearml_webserver']['depends_on']['clearml_apiserver']
        )
        self.assertEqual(
            {'condition': 'service_started'}, services['clearml_webserver']['depends_on']['clearml_fileserver']
        )
        self.assertEqual(
            '${XXTRAIN_SITE_ROOT:?}/clearml/elasticsearch-logs:/usr/share/elasticsearch/logs',
            services['clearml_elasticsearch']['volumes'][1],
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
            'cvat_redis_ondisk': '/var/lib/kvrocks/data',
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

    def test_all_cvat_paths_require_session_and_platform_login_remains_public(self) -> None:
        config = (Path(__file__).resolve().parents[1] / 'deploy/server/nginx.conf').read_text(encoding='utf-8')

        def location(path: str) -> str:
            match = re.search(r'location ' + re.escape(path) + r' \{([^{}]*)\}', config)
            self.assertIsNotNone(match, path)
            return match.group(1)

        self.assertIn('auth_request /_cvat_session;', location('/assets/'))
        for path in ('/api/', '/static/', '/'):
            self.assertIn('auth_request /_cvat_session;', location(path))
        self.assertNotIn('auth_request', location('/platform/'))
        self.assertNotIn('location = /auth/login {', config)
        self.assertNotIn('location = /api/auth/login {', config)
        self.assertIn('error_page 401 =302 /platform/;', location('/'))


if __name__ == '__main__':
    unittest.main()
