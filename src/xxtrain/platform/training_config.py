import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Operator-owned ClearML routing and platform storage paths."""

    project: str
    queue: str
    shared_root: Path
    metadata_dir: Path
    worker_script: Path
    run_root: Path


def load_training_config(path: Path) -> TrainingConfig:
    """Load the exact training JSON schema and resolve paths relative to it."""
    path = Path(path)
    with path.open(encoding='utf-8') as stream:
        payload = json.load(stream)
    expected = {'project', 'queue', 'shared_root', 'metadata_dir', 'worker_script', 'run_root'}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError('Training configuration fields do not match the required schema')
    if any(not isinstance(payload[field], str) or not payload[field] for field in expected):
        raise ValueError('Training configuration fields must be non-empty strings')

    def resolve(field: str) -> Path:
        return (path.parent / payload[field]).resolve()

    return TrainingConfig(
        project=payload['project'],
        queue=payload['queue'],
        shared_root=resolve('shared_root'),
        metadata_dir=resolve('metadata_dir'),
        worker_script=resolve('worker_script'),
        run_root=resolve('run_root'),
    )
