import json
import os
import tempfile
from pathlib import Path


class StateStore:
    """Persist the workflow's private JSON object with atomic single-file replacement."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> dict:
        """Return the persisted object, or an empty object when no state file exists.

        Invalid JSON or a non-object root raises ``ValueError``; filesystem access errors remain visible.
        """
        if not self.path.is_file():
            return {}
        with self.path.open(encoding='utf-8') as stream:
            state = json.load(stream)
        if not isinstance(state, dict):
            raise ValueError('Workspace state must be a JSON object')
        return state

    def save(self, state: dict) -> None:
        """Atomically replace the state file after encoding the complete object.

        Encoding and filesystem failures remain visible to the caller. A failed replacement leaves the prior
        destination intact and removes its temporary file.
        """
        if not isinstance(state, dict):
            raise ValueError('Workspace state must be a JSON object')
        payload = json.dumps(state, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
        with tempfile.NamedTemporaryFile(dir=self.path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
