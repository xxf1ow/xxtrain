import json
import re
from dataclasses import dataclass
from pathlib import Path

from xxtrain.business_tasks.loader import DEFAULT_TASK_ENTRY


@dataclass(frozen=True)
class WorkspaceConfig:
    """Configuration for one preselected workspace and its CVAT origin."""

    workspace_id: str
    display_name: str
    owner_user_id: int
    workspace_dir: Path
    runtime_dir: Path
    cvat_internal_url: str
    task_entry: str = DEFAULT_TASK_ENTRY


def load_config(path: Path) -> WorkspaceConfig:
    """Load one exact-schema JSON object, resolving filesystem paths relative to its file.

    Invalid JSON, fields, or field values raise ``ValueError``. Filesystem access errors remain visible to the
    composition root.
    """
    path = Path(path)
    with path.open(encoding='utf-8') as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError('Workspace configuration must be a JSON object')

    expected = {'workspace_id', 'display_name', 'owner_user_id', 'workspace_dir', 'runtime_dir', 'cvat_internal_url'}
    if not expected <= set(payload) or set(payload) - expected - {'task_entry'}:
        raise ValueError('Workspace configuration fields do not match the required schema')

    strings = ('workspace_id', 'display_name', 'workspace_dir', 'runtime_dir', 'cvat_internal_url')
    if any(not isinstance(payload[field], str) or not payload[field] for field in strings):
        raise ValueError('Workspace configuration string fields must be non-empty')
    task_entry = payload.get('task_entry', DEFAULT_TASK_ENTRY)
    entry_pattern = r'(?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*:[A-Za-z_][A-Za-z0-9_]*'
    if not isinstance(task_entry, str) or re.fullmatch(entry_pattern, task_entry) is None:
        raise ValueError('Workspace configuration task_entry must use module:factory syntax')
    owner_user_id = payload['owner_user_id']
    if isinstance(owner_user_id, bool) or not isinstance(owner_user_id, int) or owner_user_id <= 0:
        raise ValueError('Workspace owner_user_id must be a positive integer')

    def resolve(field: str) -> Path:
        return (path.parent / payload[field]).resolve()

    return WorkspaceConfig(
        workspace_id=payload['workspace_id'],
        display_name=payload['display_name'],
        owner_user_id=owner_user_id,
        workspace_dir=resolve('workspace_dir'),
        runtime_dir=resolve('runtime_dir'),
        cvat_internal_url=payload['cvat_internal_url'],
        task_entry=task_entry,
    )
