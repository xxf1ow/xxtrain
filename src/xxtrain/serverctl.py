import os
import secrets
import subprocess
import sys
from pathlib import Path


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


def main(command: str, *, root: Path | None = None) -> int:
    """Report that a server lifecycle action awaits its implementation."""
    del root
    print(f'serverctl {command} is not implemented yet', file=sys.stderr)
    return 2
