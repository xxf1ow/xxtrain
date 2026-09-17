from dataclasses import dataclass


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
