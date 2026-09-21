import argparse
import ipaddress
import os
from collections.abc import Sequence
from pathlib import Path

import httpx
import uvicorn

from xxtrain.business_tasks.loader import load_task_definition
from xxtrain.integrations.clearml import ClearMLClient
from xxtrain.integrations.cvat import CvatClient
from xxtrain.platform.app import create_app
from xxtrain.platform.config import load_config
from xxtrain.platform.runtime import RuntimeCache
from xxtrain.platform.service import AnnotationService
from xxtrain.platform.training_config import load_training_config
from xxtrain.platform.training_service import TrainingService
from xxtrain.platform.training_store import TrainingRunStore
from xxtrain.workspace_data import WorkspaceData

_SERVICE_TOKEN_ENV = 'XXTRAIN_CVAT_SERVICE_TOKEN'
_CLEARML_KEY_ENVS = ('CLEARML_API_ACCESS_KEY', 'CLEARML_API_SECRET_KEY')


def _bind_host(value: str) -> str:
    if value == 'localhost':
        return value
    try:
        address = ipaddress.ip_address(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('host must be localhost or a private IP address') from error
    if address.is_unspecified or not (address.is_loopback or address.is_private):
        raise argparse.ArgumentTypeError('host must be localhost or a private IP address')
    return value


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('port must be an integer') from error
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError('port must be between 1 and 65535')
    return port


def main(argv: Sequence[str] | None = None) -> None:
    """Run the single-worker portal for one configured workspace.

    The CVAT service token comes only from ``XXTRAIN_CVAT_SERVICE_TOKEN``. The process creates one dedicated HTTPX
    client for the CVAT adapter and closes it when Uvicorn stops. Invalid arguments or a missing token terminate with
    argparse's nonzero usage error; configuration and startup failures remain visible to the operator.
    """
    parser = argparse.ArgumentParser(prog='xxtrain-platform', description='Run the configured annotation portal')
    parser.add_argument('--config', type=Path, required=True, help='workspace JSON configuration')
    parser.add_argument('--training-config', type=Path, help='training JSON configuration')
    parser.add_argument('--host', type=_bind_host, default='127.0.0.1', help='private or loopback bind address')
    parser.add_argument('--port', type=_port, default=8000, help='internal HTTP port')
    args = parser.parse_args(argv)

    token = os.environ.get(_SERVICE_TOKEN_ENV)
    if not token:
        parser.error(f'{_SERVICE_TOKEN_ENV} must be set')

    config = load_config(args.config)
    training_config = load_training_config(args.training_config) if args.training_config is not None else None
    if training_config is not None:
        missing_keys = [name for name in _CLEARML_KEY_ENVS if not os.environ.get(name)]
        if missing_keys:
            parser.error(f'{", ".join(missing_keys)} must be set')
        cache_root = (config.runtime_dir / 'cache').resolve()
        if not cache_root.is_relative_to(training_config.shared_root):
            parser.error('workspace runtime cache must be inside training shared_root')
        if training_config.metadata_dir.is_relative_to(config.runtime_dir.resolve()):
            parser.error('training metadata_dir must be outside the disposable workspace runtime')
        if not training_config.worker_script.is_file():
            parser.error('training worker_script must name a deployable file')
    task = load_task_definition(config.task_entry)
    data = WorkspaceData(config.workspace_dir, task)
    runtime = RuntimeCache(config.runtime_dir)
    with httpx.Client() as http:
        cvat = CvatClient(config.cvat_internal_url, token, http)
        service = AnnotationService(config, data, cvat, runtime)
        training_service = None
        if training_config is not None:
            store = TrainingRunStore(training_config.metadata_dir / 'training-runs.db')
            clearml = ClearMLClient(
                training_config.project,
                training_config.queue,
                training_config.worker_script,
                training_config.shared_root,
                run_root=training_config.run_root,
            )
            training_service = TrainingService(config, service, store, clearml, training_config.shared_root)
            service.require_editable = training_service.require_editable
            service.require_cache_rebuild = training_service.require_cache_rebuild
        app = create_app(config, service, cvat, task=task, training_service=training_service)
        uvicorn.run(app, host=args.host, port=args.port, workers=1, proxy_headers=True, forwarded_allow_ips='*')


if __name__ == '__main__':
    main()
