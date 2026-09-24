"""Prepare one checkout's server configuration and dependencies without activating services."""

import getpass
import ipaddress
import json
import os
import secrets
import subprocess
import sys
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
    if command != 'install':
        print(f'serverctl {command}: unavailable', file=sys.stderr)
        return 2
    try:
        install(site_root(root))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        print(f'serverctl {command}: {message}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1] if len(sys.argv) == 2 else ''))
