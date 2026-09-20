from dataclasses import dataclass
from pathlib import Path

EXECUTION_STATUSES = frozenset({'pending', 'queued', 'running', 'completed', 'failed', 'cancelled', 'unknown'})


@dataclass(frozen=True)
class TrainingRun:
    """Durable association between a user's prepared training input and its ClearML task, when bound."""

    id: str
    user_id: int
    workspace_id: str
    workspace_name: str
    target: str
    fingerprint: str
    cache_relative_path: str
    submitted_at: str
    create_attempted_at: str | None
    clearml_task_id: str | None
    desired_action: str | None = None


@dataclass(frozen=True)
class ExecutionView:
    """Read-only ClearML execution facts; ``unknown`` remains active until ClearML reports otherwise."""

    task_id: str
    status: str
    active: bool
    epoch: int | None
    total_epochs: int | None
    elapsed_seconds: float | None
    metric: float | None
    download_ready: bool
    detail: str | None


@dataclass(frozen=True)
class TrainingRunView:
    """One durable training association and its latest execution observation, if available."""

    run: TrainingRun
    execution: ExecutionView | None
    cancellation_requested: bool = False


@dataclass(frozen=True)
class DownloadFile:
    """One authorized deployment file prepared for download."""

    path: Path
    filename: str
    media_type: str
