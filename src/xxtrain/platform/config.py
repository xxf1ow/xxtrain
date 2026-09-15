import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspaceConfig:
    """Configuration for one preselected workspace and its CVAT origin."""

    workspace_id: str
    display_name: str
    owner_user_id: int
    images_dir: Path
    annotations_dir: Path
    state_path: Path
    cvat_internal_url: str


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

    expected = {
        'workspace_id',
        'display_name',
        'owner_user_id',
        'images_dir',
        'annotations_dir',
        'state_path',
        'cvat_internal_url',
    }
    if set(payload) != expected:
        raise ValueError('Workspace configuration fields do not match the required schema')

    strings = ('workspace_id', 'display_name', 'images_dir', 'annotations_dir', 'state_path', 'cvat_internal_url')
    if any(not isinstance(payload[field], str) or not payload[field] for field in strings):
        raise ValueError('Workspace configuration string fields must be non-empty')
    owner_user_id = payload['owner_user_id']
    if isinstance(owner_user_id, bool) or not isinstance(owner_user_id, int) or owner_user_id <= 0:
        raise ValueError('Workspace owner_user_id must be a positive integer')

    def resolve(field: str) -> Path:
        return (path.parent / payload[field]).resolve()

    return WorkspaceConfig(
        workspace_id=payload['workspace_id'],
        display_name=payload['display_name'],
        owner_user_id=owner_user_id,
        images_dir=resolve('images_dir'),
        annotations_dir=resolve('annotations_dir'),
        state_path=resolve('state_path'),
        cvat_internal_url=payload['cvat_internal_url'],
    )
