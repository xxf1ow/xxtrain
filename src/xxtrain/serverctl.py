import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from subprocess import CompletedProcess
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from xxtrain.platform.training_config import load_training_config

_CLEARML_KEYS = ('CLEARML_API_ACCESS_KEY', 'CLEARML_API_SECRET_KEY')
_CLEARML_ENDPOINTS = {
    'CLEARML_API_HOST': 'http://127.0.0.1:18083',
    'CLEARML_WEB_HOST': 'http://127.0.0.1:18084',
    'CLEARML_FILES_HOST': 'http://127.0.0.1:18082',
}
_OWNED_ENV_KEYS = set(_CLEARML_KEYS) | set(_CLEARML_ENDPOINTS)
_ENV_ASSIGNMENT = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$')


def site_root(root: Path | None = None) -> Path:
    """Return the Git checkout containing root and reject an escaping deployment directory."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'], cwd=root, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError('serverctl must run from within a Git checkout') from exc

    checkout = Path(result.stdout.strip()).resolve()
    deployment = checkout / '.deployment'
    if deployment.resolve() != deployment.absolute():
        raise ValueError('.deployment must not be a symlink')
    if not deployment.resolve().is_relative_to(checkout):
        raise ValueError(f'{deployment} resolves outside the Git checkout')
    return checkout


def compose_files(root: Path) -> tuple[Path, ...]:
    """Return the tracked Compose manifest for a Git checkout."""
    return (site_root(root) / 'deploy' / 'server' / 'compose.yaml',)


def ensure_administrator(root: Path) -> Path:
    """Create a private administrator secret only when the file is absent."""
    checkout = site_root(root)
    path = _check_deployment_file(checkout, '.xxxxx')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if sys.platform == 'linux' and path.stat().st_mode & 0o077:
            raise ValueError('.deployment/.xxxxx must have mode 0600; run chmod 600 .deployment/.xxxxx')
        return path

    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        stream.write(secrets.token_urlsafe(32))
        stream.write('\n')
    return path


def _run(args: list[str], root: Path, run: Callable[..., CompletedProcess], env: dict[str, str]) -> None:
    run(args, cwd=root, env=env, check=True)


def _environment(root: Path) -> dict[str, str]:
    return dict(os.environ, XXTRAIN_SITE_ROOT=str(root / '.deployment'))


def _compose(root: Path) -> list[str]:
    return ['docker', 'compose', '-p', 'xxtrain-server', '-f', str(compose_files(root)[0])]


def _check_deployment_file(root: Path, name: str) -> Path:
    path = root / '.deployment' / name
    if not path.resolve().is_relative_to(root):
        raise ValueError(f'.deployment/{name} resolves outside the Git checkout')
    return path


def _write_private_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(content)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _platform_environment(path: Path) -> tuple[list[str], dict[str, str]]:
    try:
        lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
    except UnicodeDecodeError as exc:
        raise ValueError('.deployment/platform.env must be UTF-8') from exc
    values: dict[str, str] = {}
    for line in lines:
        raw_line = line.rstrip('\r\n')
        assignment = _ENV_ASSIGNMENT.fullmatch(raw_line)
        if assignment is None:
            if any(re.match(rf'^\s*(?:export\s+)?{re.escape(name)}(?:\s|$|:|=)', raw_line) for name in _OWNED_ENV_KEYS):
                raise ValueError('.deployment/platform.env contains an invalid owned assignment')
            continue
        name, value = assignment.groups()
        if name in _OWNED_ENV_KEYS:
            if name in values:
                raise ValueError(f'.deployment/platform.env contains duplicate {name}')
            if not value or value != value.strip() or any(char in value for char in '\'"\\'):
                raise ValueError(f'.deployment/platform.env contains an invalid {name}')
            values[name] = value
    return lines, values


def _training_configuration(root: Path) -> bytes:
    payload = {
        'project': 'xxtrain',
        'queue': 'training',
        'shared_root': 'workspace',
        'metadata_dir': 'clearml/metadata',
        'worker_script': str(root / 'src/xxtrain/integrations/clearml/worker.py'),
        'run_root': 'clearml/runs',
    }
    content = (json.dumps(payload, indent=2) + '\n').encode('utf-8')
    path = root / '.deployment/training.json'
    descriptor, temporary_name = tempfile.mkstemp(prefix='.training-config-', dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(content)
        load_training_config(temporary)
    finally:
        temporary.unlink(missing_ok=True)
    return content


def _secure_configuration(access_key: str, secret_key: str) -> bytes:
    return (
        'secure {\n'
        '  credentials {\n'
        '    users {\n'
        '      user {\n'
        f'        username: {json.dumps(access_key)}\n'
        f'        password: {json.dumps(secret_key)}\n'
        '      }\n'
        '    }\n'
        '  }\n'
        '}\n'
    ).encode()


def _prepare_local_configuration(root: Path) -> None:
    """Create missing private configuration and derive ClearML credentials under .deployment."""
    checkout = site_root(root)
    deployment = checkout / '.deployment'
    paths = (
        _check_deployment_file(checkout, 'platform.env'),
        _check_deployment_file(checkout, '.xxxxx'),
        _check_deployment_file(checkout, 'training.json'),
        _check_deployment_file(checkout, 'clearml/config/secure.conf'),
    )
    deployment.mkdir(parents=True, exist_ok=True)
    env_file, _, training_file, secure_file = paths
    try:
        descriptor = os.open(env_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        _check_private_env(env_file)
    else:
        os.close(descriptor)

    ensure_administrator(checkout)
    if training_file.exists():
        load_training_config(training_file)
    else:
        _write_private_file(training_file, _training_configuration(checkout))

    lines, values = _platform_environment(env_file)
    access_key, secret_key = (values.get(name) for name in _CLEARML_KEYS)
    if access_key == '' or secret_key == '':
        raise ValueError('.deployment/platform.env ClearML keys must not be empty')
    if (access_key is None) != (secret_key is None):
        missing = _CLEARML_KEYS[0] if access_key is None else _CLEARML_KEYS[1]
        raise ValueError(f'.deployment/platform.env must contain both ClearML keys; missing {missing}')
    additions: dict[str, str] = {}
    if access_key is None:
        access_key = secrets.token_urlsafe(32)
        secret_key = secrets.token_urlsafe(48)
        additions.update(zip(_CLEARML_KEYS, (access_key, secret_key), strict=True))
    for name, endpoint in _CLEARML_ENDPOINTS.items():
        configured = values.get(name)
        if configured is None:
            additions[name] = endpoint
        elif configured != endpoint:
            raise ValueError(f'.deployment/platform.env {name} must equal {endpoint}')
    if additions:
        content = ''.join(lines)
        if content and not content.endswith(('\n', '\r')):
            content += '\n'
        content += ''.join(f'{name}={value}\n' for name, value in additions.items())
        _write_private_file(env_file, content.encode('utf-8'))
    if sys.platform == 'linux':
        _check_private_env(env_file)
    secure_content = _secure_configuration(access_key, secret_key)
    if not secure_file.exists() or secure_file.read_bytes() != secure_content:
        secure_file.parent.mkdir(parents=True, exist_ok=True)
        _write_private_file(secure_file, secure_content)


def _check_private_env(path: Path) -> None:
    if sys.platform == 'linux' and path.stat().st_mode & 0o077:
        raise ValueError('.deployment/platform.env must have mode 0600; run chmod 600 .deployment/platform.env')


def _prepare_writable_directories(root: Path, run: Callable[..., CompletedProcess]) -> None:
    deployment = root / '.deployment'
    directories = (
        (Path('cvat/data'), '1000:1000'),
        (Path('cvat/keys'), '1000:1000'),
        (Path('cvat/logs'), '1000:1000'),
        (Path('cvat/kvrocks/data'), '999:999'),
        (Path('clearml/elasticsearch'), '1000:0'),
        (Path('clearml/elasticsearch-logs'), '1000:0'),
    )
    resolved_deployment = deployment.resolve()
    if resolved_deployment != deployment.absolute():
        raise ValueError('.deployment must not be a symlink')
    paths = [(deployment / relative, owner) for relative, owner in directories]
    for path, _ in paths:
        if path.resolve() != path.absolute() or not path.resolve().is_relative_to(resolved_deployment):
            raise ValueError(f'{path.relative_to(root).as_posix()} resolves outside .deployment or through a symlink')
    env = _environment(root)
    for path, _ in paths:
        run(['sudo', 'mkdir', '-p', '--', str(path)], cwd=root, env=env, check=True)
    for path, owner in paths:
        if path.resolve() != path.absolute() or not path.resolve().is_relative_to(resolved_deployment):
            raise ValueError(f'{path.relative_to(root).as_posix()} resolves outside .deployment or through a symlink')
    for path, owner in paths:
        run(['sudo', 'chown', owner, str(path)], cwd=root, env=env, check=True)
        run(['sudo', 'chmod', '0770', str(path)], cwd=root, env=env, check=True)


def install(root: Path, run: Callable[..., CompletedProcess] = subprocess.run) -> None:
    """Prepare dependencies and this checkout's private data without starting services."""
    root = site_root(root)
    _prepare_local_configuration(root)
    env = _environment(root)
    _prepare_writable_directories(root, run)
    _run(['uv', 'sync', '--locked', '--extra', 'platform', '--extra', 'clearml'], root, run, env)
    _run([*_compose(root), 'pull', '--ignore-buildable'], root, run, env)
    _run([*_compose(root), 'build'], root, run, env)
    if sys.platform == 'linux':
        template = (root / 'deploy/server/xxtrain-server.service').read_text(encoding='utf-8')
        unit = template.replace('{{ROOT}}', str(root))
        run(
            ['sudo', 'install', '-m', '0644', '/dev/stdin', '/etc/systemd/system/xxtrain-server.service'],
            cwd=root,
            env=env,
            check=True,
            input=unit,
            text=True,
        )
        _run(['sudo', 'systemctl', 'daemon-reload'], root, run, env)


def start(root: Path, run: Callable[..., CompletedProcess] = subprocess.run) -> None:
    """Prepare local configuration, then enable and start the site."""
    root = site_root(root)
    workspace_file = _check_deployment_file(root, 'workspace.json')
    _prepare_local_configuration(root)
    if not workspace_file.is_file():
        raise ValueError('missing .deployment/workspace.json; create the workspace configuration before start')
    env = _environment(root)
    _run(['sudo', 'systemctl', 'enable', 'xxtrain-server.service'], root, run, env)
    _run(['sudo', 'systemctl', 'start', 'xxtrain-server.service'], root, run, env)


def stop(root: Path, run: Callable[..., CompletedProcess] = subprocess.run) -> None:
    """Stop this site's service and disable its boot startup without deleting data."""
    root = site_root(root)
    env = _environment(root)
    try:
        _run(['sudo', 'systemctl', 'stop', 'xxtrain-server.service'], root, run, env)
    finally:
        _run(['sudo', 'systemctl', 'disable', 'xxtrain-server.service'], root, run, env)


def status(root: Path, run: Callable[..., CompletedProcess] = subprocess.run) -> bool:
    """Print checkout, service and container states; return false if observations fail."""
    root = site_root(root)
    env = _environment(root)
    commands = (
        ('revision', ['git', 'rev-parse', 'HEAD']),
        ('worktree', ['git', 'status', '--porcelain', '--untracked-files=normal']),
        ('systemd enabled', ['systemctl', 'is-enabled', 'xxtrain-server.service']),
        ('systemd active', ['systemctl', 'is-active', 'xxtrain-server.service']),
        ('services', [*_compose(root), 'config', '--services']),
        ('containers', [*_compose(root), 'ps', '--all', '--format', 'json']),
    )
    success = True
    services: set[str] = set()
    for label, args in commands:
        try:
            result = run(args, cwd=root, env=env, check=False, capture_output=True, text=True)
            value = result.stdout.strip()
            if label == 'worktree':
                value = 'dirty' if value else 'clean'
            elif label == 'services':
                services = set(value.splitlines())
                value = f'{len(services)} configured'
            elif label == 'containers':
                entries = (
                    json.loads(value) if value.startswith('[') else [json.loads(line) for line in value.splitlines()]
                )
                present = {entry['Service']: entry for entry in entries}
                value = (
                    ', '.join(
                        f'{name}: {present[name]["State"]}'
                        + f' ({present[name].get("Health") or "health unavailable"})'
                        if name in present
                        else f'{name}: missing'
                        for name in sorted(services | present.keys())
                    )
                    or 'none'
                )
            elif label == 'revision' and (len(value) != 40 or any(char not in '0123456789abcdef' for char in value)):
                raise ValueError('invalid full Git SHA')
            if result.returncode and label not in ('systemd enabled', 'systemd active'):
                raise ValueError('command failed')
            print(f'{label}: {value}')
            if label == 'containers' and (
                not services
                or services != present.keys()
                or any(entry['State'] != 'running' or entry.get('Health') == 'unhealthy' for entry in entries)
            ):
                success = False
            if label.startswith('systemd') and result.returncode:
                success = False
        except (OSError, ValueError, KeyError) as exc:
            print(f'{label}: unavailable ({type(exc).__name__})')
            success = False
    return success


def verify(root: Path, request: Callable[..., object] = urlopen) -> bool:
    """Probe real local public and backend HTTP endpoints; return false on any failed check."""
    site_root(root)
    checks = (
        ('platform', 'http://127.0.0.1:8080/platform/', 200),
        ('CVAT unauthenticated', 'http://127.0.0.1:8080/api/users/self', 401),
        ('CVAT backend', 'http://127.0.0.1:18080/api/server/health/', 200),
        ('ClearML API', 'http://127.0.0.1:8008/', 200),
        ('ClearML files', 'http://127.0.0.1:8081/', 200),
        ('ClearML web', 'http://127.0.0.1:8082/', 200),
    )
    success = True
    for label, url, expected in checks:
        try:
            with request(url, timeout=5) as response:
                code = response.status
        except HTTPError as exc:
            code = exc.code
        except (OSError, URLError):
            code = None
        matched = code == expected
        observed = code if code is not None else 'unavailable'
        print(f'{label}: {"PASS" if matched else "FAIL"} (HTTP {observed}; expected {expected})')
        success &= matched
    return success


def main(command: str, *, root: Path | None = None) -> int:
    """Run the selected server lifecycle operation or diagnose unavailable actions."""
    if command in ('install', 'start', 'stop', 'status', 'verify'):
        try:
            operation = {'install': install, 'start': start, 'stop': stop, 'status': status, 'verify': verify}[command]
            result = operation(site_root(root))
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            print(f'serverctl {command}: {exc}', file=sys.stderr)
            return 1
        return 0 if result is not False else 1
    print(f'serverctl {command} is not implemented yet', file=sys.stderr)
    return 2
