import os
import secrets
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from subprocess import CompletedProcess


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
    if not deployment.resolve().is_relative_to(checkout):
        raise ValueError(f'{deployment} resolves outside the Git checkout')
    return checkout


def compose_files(root: Path) -> tuple[Path, ...]:
    """Return the tracked Compose manifest for a Git checkout."""
    return (site_root(root) / 'deploy' / 'server' / 'compose.yaml',)


def ensure_administrator(root: Path) -> Path:
    """Create a private administrator secret only when the file is absent."""
    checkout = site_root(root)
    path = checkout / '.deployment' / 'administrator'
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
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


def _check_private_env(path: Path) -> None:
    if sys.platform == 'linux' and path.stat().st_mode & 0o077:
        raise ValueError('.deployment/platform.env must have mode 0600; run chmod 600 .deployment/platform.env')


def install(root: Path, run: Callable[..., CompletedProcess] = subprocess.run) -> None:
    """Prepare dependencies and this checkout's private data without starting services."""
    root = site_root(root)
    deployment = root / '.deployment'
    deployment.mkdir(parents=True, exist_ok=True)
    env_file = _check_deployment_file(root, 'platform.env')
    try:
        descriptor = os.open(env_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        _check_private_env(env_file)
    else:
        os.close(descriptor)
    env = _environment(root)
    _run(['uv', 'sync', '--locked', '--extra', 'platform', '--extra', 'clearml'], root, run, env)
    _run([*_compose(root), 'pull'], root, run, env)
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
    """Enable and start the site only after operator configuration is present."""
    root = site_root(root)
    for name in ('workspace.json', 'training.json', 'platform.env'):
        if not _check_deployment_file(root, name).is_file():
            raise ValueError(f'missing .deployment/{name}; create the operator configuration before start')
    _check_private_env(root / '.deployment/platform.env')
    _check_deployment_file(root, 'administrator')
    ensure_administrator(root)
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


def main(command: str, *, root: Path | None = None) -> int:
    """Run the selected server lifecycle operation or diagnose unavailable actions."""
    if command in ('install', 'start', 'stop'):
        try:
            {'install': install, 'start': start, 'stop': stop}[command](site_root(root))
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            print(f'serverctl {command}: {exc}', file=sys.stderr)
            return 1
        return 0
    print(f'serverctl {command} is not implemented yet', file=sys.stderr)
    return 2
