"""Prepare one checkout's server configuration and dependencies without activating services."""

import getpass
import ipaddress
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

_CLEARML_KEYS = ('CLEARML_API_ACCESS_KEY', 'CLEARML_API_SECRET_KEY')
_CLEARML_ENDPOINTS = {
    'CLEARML_API_HOST': 'http://127.0.0.1:18083',
    'CLEARML_WEB_HOST': 'http://127.0.0.1:18084',
    'CLEARML_FILES_HOST': 'http://127.0.0.1:18082',
}
_BIND_DIRECTORIES = (
    'cvat/postgres',
    'cvat/redis',
    'cvat/kvrocks',
    'cvat/clickhouse',
    'cvat/clickhouse-logs',
    'cvat/data',
    'cvat/keys',
    'cvat/logs',
    'clearml/mongo',
    'clearml/mongo-config',
    'clearml/redis',
    'clearml/elasticsearch',
    'clearml/elasticsearch-logs',
    'clearml/logs',
    'clearml/files',
    'clearml/config',
)


def site_root(root: Path | None = None) -> Path:
    """Resolve the containing Git checkout; reject a symlinked or escaping deployment directory."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'], cwd=root, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError('serverctl must run within a Git checkout') from exc
    checkout = Path(result.stdout.strip()).resolve()
    _deployment_path(checkout, '')
    return checkout


def _deployment_path(root: Path, name: str) -> Path:
    site = root / '.deployment'
    path = site / name
    if site.resolve() != site or path.resolve() != path or not path.resolve().is_relative_to(site):
        raise ValueError(f'.deployment/{name} must stay within .deployment without symlinks')
    return path


def _create_private(path: Path, content: str, *, require_private: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if require_private:
            _check_private(path)
        return
    with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
        stream.write(content)


def _check_private(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f'{path.name} must be a regular private file')
    if os.name == 'posix' and path.stat().st_mode & 0o777 != 0o600:
        raise ValueError(f'{path.name} must have mode 0600')


def ensure_administrator(root: Path) -> Path:
    """Exclusively create a missing 0600 administrator password; preserve existing bytes."""
    path = _deployment_path(site_root(root), '.xxxxx')
    _create_private(path, secrets.token_urlsafe(32) + '\n')
    if not path.read_text(encoding='utf-8').strip():
        raise ValueError('.deployment/.xxxxx must contain a nonempty password')
    return path


def _environment(root: Path) -> dict[str, str]:
    site = root / '.deployment'
    return dict(
        os.environ,
        XXTRAIN_SITE_ROOT=str(site),
        UV_CACHE_DIR=str(site / 'uv-cache'),
        UV_PYTHON_INSTALL_DIR=str(site / 'uv-python'),
    )


def _run(root: Path, args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=root, env=_environment(root), check=True, **kwargs)


def _compose(root: Path) -> list[str]:
    return ['docker', 'compose', '-p', 'xxtrain-server', '-f', str(root / 'deploy/server/compose.yaml')]


def _listen_address(root: Path) -> str:
    path = _deployment_path(root, 'listen-ip')
    saved = path.read_text(encoding='utf-8').strip() if path.exists() else None
    supplied = os.environ.get('XXTRAIN_SERVER_LISTEN_IP')
    if saved and supplied and supplied != saved:
        raise ValueError('XXTRAIN_SERVER_LISTEN_IP differs from .deployment/listen-ip')
    value = saved or supplied
    if not value:
        raise ValueError('first install requires XXTRAIN_SERVER_LISTEN_IP with a local private-LAN IPv4 address')
    try:
        address = ipaddress.IPv4Address(value)
    except ipaddress.AddressValueError as exc:
        raise ValueError('XXTRAIN_SERVER_LISTEN_IP must be a private-LAN IPv4 address') from exc
    networks = ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16')
    if not any(address in ipaddress.IPv4Network(network) for network in networks):
        raise ValueError('XXTRAIN_SERVER_LISTEN_IP must be a private-LAN IPv4 address')
    interfaces = json.loads(_run(root, ['ip', '-j', '-4', 'address', 'show'], capture_output=True, text=True).stdout)
    if not any(
        entry.get('local') == str(address) and entry.get('scope') == 'global'
        for interface in interfaces
        for entry in interface.get('addr_info', [])
    ):
        raise ValueError('XXTRAIN_SERVER_LISTEN_IP is not assigned to a local private-LAN interface')
    _create_private(path, str(address) + '\n', require_private=False)
    return str(address)


def _platform_environment(path: Path) -> dict[str, str]:
    _check_private(path)
    values = {}
    owned = {*_CLEARML_KEYS, *_CLEARML_ENDPOINTS, 'XXTRAIN_CVAT_SERVICE_TOKEN'}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'):
            continue
        name, separator, value = line.partition('=')
        if name not in owned:
            if any(key in line for key in owned):
                raise ValueError('platform.env has an invalid service credential assignment')
            continue
        if not separator or not value or name in values or value != value.strip() or any(c in value for c in '\'"\\'):
            raise ValueError('platform.env has an invalid or duplicate service credential assignment')
        values[name] = value
    if any(not values.get(key) for key in _CLEARML_KEYS):
        raise ValueError('platform.env requires both ClearML service credentials')
    if any(values.get(key) != value for key, value in _CLEARML_ENDPOINTS.items()):
        raise ValueError('platform.env requires the local ClearML backend endpoints')
    return values


def _prepare_configuration(root: Path) -> None:
    env_file = _deployment_path(root, 'platform.env')
    secure_file = _deployment_path(root, 'clearml/config/secure.conf')
    training_file = _deployment_path(root, 'training.json')
    if not env_file.exists():
        if secure_file.exists():
            raise ValueError('platform.env is missing for existing ClearML credentials')
        values = dict(zip(_CLEARML_KEYS, (secrets.token_urlsafe(32), secrets.token_urlsafe(48)), strict=True))
        values.update(_CLEARML_ENDPOINTS)
        _create_private(env_file, ''.join(f'{key}={value}\n' for key, value in values.items()))
    values = _platform_environment(env_file)
    expected = {
        'credentials': {
            'user': {
                'role': 'user',
                'user_key': values[_CLEARML_KEYS[0]],
                'user_secret': values[_CLEARML_KEYS[1]],
                'display_name': 'xxtrain service',
            }
        }
    }
    _create_private(secure_file, json.dumps(expected) + '\n')
    if json.loads(secure_file.read_text(encoding='utf-8')) != expected:
        raise ValueError('ClearML secure.conf does not match platform.env credentials')
    _create_private(
        training_file,
        json.dumps(
            {
                'project': 'xxtrain',
                'queue': 'training',
                'shared_root': 'training-shared',
                'metadata_dir': 'training-metadata',
                'run_root': 'training-runs',
                'worker_script': str(root / 'src/xxtrain/integrations/clearml/worker.py'),
            },
            indent=2,
        )
        + '\n',
        require_private=False,
    )
    from xxtrain.platform.training_config import load_training_config

    training = load_training_config(training_file)
    for path in (training.shared_root, training.metadata_dir, training.run_root):
        if not path.is_relative_to(root / '.deployment'):
            raise ValueError('training storage paths must remain within .deployment')


def prepare(root: Path) -> None:
    """Create local credentials and configuration before Compose services start."""
    root = site_root(root)
    _prepare_configuration(root)
    ensure_administrator(root)


def _save_service_token(path: Path, token: str) -> None:
    existing = path.read_text(encoding='utf-8').splitlines()
    if f'XXTRAIN_CVAT_SERVICE_TOKEN={token}' in existing:
        return
    lines = [line for line in existing if not line.startswith('XXTRAIN_CVAT_SERVICE_TOKEN=')]
    lines.append(f'XXTRAIN_CVAT_SERVICE_TOKEN={token}')
    temporary = path.with_name(f'.{path.name}.{secrets.token_hex(8)}')
    _create_private(temporary, '\n'.join(lines) + '\n')
    temporary.replace(path)


def _management(root: Path, password: str) -> int:
    expression = (
        'import sys; from django.contrib.auth import get_user_model; '
        "u, _ = get_user_model().objects.get_or_create(username='xxadmin'); "
        'u.is_staff = True; u.is_superuser = True; u.set_password(sys.stdin.read().strip()); '
        'u.save(); print(u.pk)'
    )
    result = _run(
        root,
        [*_compose(root), 'exec', '-T', 'cvat_server', 'python', 'manage.py', 'shell', '-c', expression],
        input=password,
        capture_output=True,
        text=True,
    )
    try:
        return int(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        raise ValueError('CVAT management command returned no user ID') from exc


def bootstrap(root: Path, *, timeout: float = 600) -> int:
    """Wait for authenticated services, synchronize xxadmin, and persist their identities."""
    import httpx

    root = site_root(root)
    deadline = time.monotonic() + timeout
    password_path = ensure_administrator(root)
    password = password_path.read_text(encoding='utf-8').strip()
    migration = [*_compose(root), 'exec', '-T', 'cvat_server', 'python', 'manage.py', 'migrate', '--check']
    while True:
        try:
            _run(root, migration, capture_output=True, text=True)
            break
        except (OSError, subprocess.CalledProcessError):
            if time.monotonic() >= deadline:
                raise TimeoutError('CVAT migrations did not become ready') from None
            time.sleep(min(5, max(0, deadline - time.monotonic())))

    user_id = _management(root, password)
    values = _platform_environment(_deployment_path(root, 'platform.env'))
    cvat_root = 'http://127.0.0.1:18080'
    clearml_root = values['CLEARML_API_HOST']
    while True:
        try:
            with httpx.Client(timeout=10) as client:
                token = values.get('XXTRAIN_CVAT_SERVICE_TOKEN')
                identity = None
                if token:
                    identity = client.get(f'{cvat_root}/api/users/self', headers={'Authorization': f'Token {token}'})
                    if identity.status_code in (401, 403):
                        token = None
                if token is None:
                    login = client.post(
                        f'{cvat_root}/api/auth/login', json={'username': 'xxadmin', 'password': password}
                    )
                    login.raise_for_status()
                    token = login.json()['key']
                    identity = client.get(f'{cvat_root}/api/users/self', headers={'Authorization': f'Token {token}'})
                identity.raise_for_status()
                authenticated_user_id = int(identity.json()['id'])
                if authenticated_user_id != user_id:
                    raise ValueError('CVAT service token did not authenticate as xxadmin')
                user_id = authenticated_user_id
                clearml = client.get(
                    f'{clearml_root}/auth.login', auth=(values[_CLEARML_KEYS[0]], values[_CLEARML_KEYS[1]])
                )
                clearml.raise_for_status()
                clearml_response = clearml.json()
                if clearml_response.get('meta', {}).get('result_code') != 200 or not clearml_response.get(
                    'data', {}
                ).get('token'):
                    raise ValueError('ClearML service credentials were rejected')
            break
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            if time.monotonic() >= deadline:
                raise TimeoutError('authenticated CVAT and ClearML identities did not become ready') from None
            time.sleep(min(5, max(0, deadline - time.monotonic())))

    _save_service_token(_deployment_path(root, 'platform.env'), token)
    workspace = _deployment_path(root, 'workspace.json')
    _create_private(
        workspace,
        json.dumps(
            {
                'workspace_id': 'xxtrain',
                'display_name': 'xxtrain',
                'owner_user_id': user_id,
                'workspace_dir': 'workspace',
                'runtime_dir': 'training-shared/runtime',
                'cvat_internal_url': 'http://127.0.0.1:18080',
                'task_entry': 'xxtrain.business_tasks.point:point_task_definition',
            },
            indent=2,
        )
        + '\n',
        require_private=False,
    )
    if int(json.loads(workspace.read_text(encoding='utf-8'))['owner_user_id']) != user_id:
        raise ValueError('workspace.json owner_user_id does not match xxadmin')
    return user_id


def foreground(root: Path) -> None:
    """Start the project services, bootstrap identities, and replace this process with xxtrain-platform."""
    root = site_root(root)
    _run(root, [*_compose(root), 'up', '-d'])
    bootstrap(root)
    values = _platform_environment(_deployment_path(root, 'platform.env'))
    environment = _environment(root)
    environment.update(values)
    site = _deployment_path(root, '')
    environment.update(
        {
            'CLEARML_CACHE_DIR': str(site / 'clearml/sdk-cache'),
            'CLEARML_CONFIG_FILE': str(site / 'clearml/client.conf'),
            'XDG_CACHE_HOME': str(site / 'cache'),
        }
    )
    bypass = {'localhost', '127.0.0.1', '::1', _deployment_path(root, 'listen-ip').read_text(encoding='utf-8').strip()}
    for key in ('NO_PROXY', 'no_proxy'):
        bypass.update(item for item in environment.get(key, '').split(',') if item)
        environment[key] = ','.join(sorted(bypass))
    os.execve(
        sys.executable,
        [
            sys.executable,
            '-m',
            'xxtrain.platform',
            '--config',
            str(_deployment_path(root, 'workspace.json')),
            '--training-config',
            str(_deployment_path(root, 'training.json')),
            '--host',
            '127.0.0.1',
            '--port',
            '18001',
        ],
        environment,
    )


def start(root: Path) -> None:
    root = site_root(root)
    if hasattr(os, 'geteuid') and os.geteuid() == 0:
        raise ValueError('run serverctl as the normal checkout user; privileged commands use sudo')
    _run(root, ['sudo', 'systemctl', 'enable', 'xxtrain-server.service'])
    _run(root, ['sudo', 'systemctl', 'restart', 'xxtrain-server.service'])


def stop(root: Path) -> None:
    root = site_root(root)
    try:
        _run(root, ['sudo', 'systemctl', 'stop', 'xxtrain-server.service'])
    finally:
        _run(root, ['sudo', 'systemctl', 'disable', 'xxtrain-server.service'])


def status(root: Path) -> int:
    root = site_root(root)
    revision = _run(root, ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    unit = _run(
        root,
        ['sudo', 'systemctl', 'show', '--property=ActiveState', '--value', 'xxtrain-server.service'],
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = _run(root, [*_compose(root), 'ps', '--format', 'json'], capture_output=True, text=True)
    try:
        payload = result.stdout.strip()
        services = (
            json.loads(payload)
            if payload.startswith('[')
            else [json.loads(line) for line in payload.splitlines() if line]
        )
    except json.JSONDecodeError as exc:
        raise ValueError('Compose returned invalid service status') from exc
    print(f'Git: {revision}')
    print(f'systemd: {unit}')
    for service in services:
        name = service.get('Service', 'unknown')
        state = service.get('State', 'unknown')
        health = service.get('Health', 'no healthcheck')
        print(f'{name}: {state} ({health})')
    return 0


def verify_endpoints(root: Path) -> None:
    import httpx

    root = site_root(root)
    address = _deployment_path(root, 'listen-ip').read_text(encoding='utf-8').strip()
    values = _platform_environment(_deployment_path(root, 'platform.env'))
    password_path = _deployment_path(root, '.xxxxx')
    _check_private(password_path)
    password = password_path.read_text(encoding='utf-8').strip()
    if not password:
        raise ValueError('.deployment/.xxxxx must contain a nonempty password')
    base = f'http://{address}'
    try:
        with httpx.Client(timeout=10) as client:
            if client.get(f'{base}:8080/platform/').status_code != 200:
                raise ValueError('platform endpoint is unavailable')
            login = client.post(
                'http://127.0.0.1:18080/api/auth/login', json={'username': 'xxadmin', 'password': password}
            )
            login.raise_for_status()
            session_cookie = '; '.join(f'{name}={value}' for name, value in login.cookies.items())
            identity = client.get(f'{base}:8080/api/users/self', headers={'Cookie': session_cookie})
            identity.raise_for_status()
            clearml_api = f'{base}:8008'
            for url in (f'{base}:8082/', f'{base}:8081/', f'{clearml_api}/debug.ping'):
                response = client.get(url)
                response.raise_for_status()
            clearml = client.get(f'{clearml_api}/auth.login', auth=(values[_CLEARML_KEYS[0]], values[_CLEARML_KEYS[1]]))
            clearml.raise_for_status()
            if clearml.json().get('meta', {}).get('result_code') != 200:
                raise ValueError('ClearML service credentials were rejected')
    except httpx.HTTPError:
        raise OSError('verification endpoint request failed') from None


def install(root: Path) -> None:
    """Prepare dependencies, private configuration and the unit without changing activation state."""
    root = site_root(root)
    if hasattr(os, 'geteuid') and os.geteuid() == 0:
        raise ValueError('run serverctl as the normal checkout user; privileged commands use sudo')
    for name in (
        *_BIND_DIRECTORIES,
        'uv-cache',
        'uv-python',
        '.xxxxx',
        'platform.env',
        'training.json',
        'clearml/config/secure.conf',
        'nginx.conf',
        'listen-ip',
    ):
        _deployment_path(root, name)
    address = _listen_address(root)
    _prepare_configuration(root)
    ensure_administrator(root)
    template = (root / 'deploy/server/nginx.conf').read_text(encoding='utf-8')
    _deployment_path(root, 'nginx.conf').write_text(template.replace('{{LAN_ADDRESS}}', address), encoding='utf-8')
    for name in _BIND_DIRECTORIES:
        _deployment_path(root, name).mkdir(parents=True, exist_ok=True)
    ownership = {
        'cvat/data': '1000:1000',
        'cvat/keys': '1000:1000',
        'cvat/logs': '1000:1000',
        'cvat/kvrocks': '999:999',
        'clearml/elasticsearch': '1000:0',
        'clearml/elasticsearch-logs': '1000:0',
    }
    for name, owner in ownership.items():
        path = str(_deployment_path(root, name))
        _run(root, ['sudo', 'chown', owner, path])
        _run(root, ['sudo', 'chmod', '0770', path])
    _run(root, ['uv', 'sync', '--locked', '--extra', 'platform', '--extra', 'clearml'])
    _run(root, [*_compose(root), 'pull', '--ignore-buildable'])
    _run(root, [*_compose(root), 'build', 'cvat_ui'])
    unit = (root / 'deploy/server/xxtrain-server.service').read_text(encoding='utf-8')
    unit = unit.replace('{{ROOT}}', str(root)).replace('{{USER}}', getpass.getuser())
    _run(
        root,
        ['sudo', 'install', '-m', '0644', '/dev/stdin', '/etc/systemd/system/xxtrain-server.service'],
        input=unit,
        text=True,
    )
    _run(root, ['sudo', 'systemctl', 'daemon-reload'])


def main(command: str, *, root: Path | None = None) -> int:
    """Run an available server action; report failures without exposing subprocess output or secrets."""
    try:
        checkout = site_root(root)
        if command == 'install':
            install(checkout)
        elif command == 'prepare':
            prepare(checkout)
        elif command == 'foreground':
            foreground(checkout)
        elif command == 'start':
            start(checkout)
        elif command == 'stop':
            stop(checkout)
        elif command == 'status':
            return status(checkout)
        elif command == 'verify':
            verify_endpoints(checkout)
        else:
            print(f'serverctl {command}: unavailable', file=sys.stderr)
            return 2
    except (OSError, ValueError, TimeoutError, subprocess.CalledProcessError) as exc:
        message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        print(f'serverctl {command}: {message}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1] if len(sys.argv) == 2 else ''))
