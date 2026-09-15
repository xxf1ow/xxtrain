from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

type PreparationStage = Literal['new', 'uploading', 'uploaded', 'initializing', 'initialized']
type PreparationCheckpoint = Callable[['PreparationState'], None]


@dataclass(frozen=True)
class PreparationState:
    """Workflow-owned checkpoint for retrying CVAT task preparation.

    ``request_id`` is present only while an uploaded data request is being polled. An ``uploading`` state without
    that ID and every ``initializing`` state require administrator reconciliation because repeating the preceding
    write could replace intent or duplicate data.
    """

    stage: PreparationStage = 'new'
    request_id: str | None = None

    def __post_init__(self) -> None:
        if self.stage not in {'new', 'uploading', 'uploaded', 'initializing', 'initialized'}:
            raise ValueError(f'Unknown CVAT preparation stage: {self.stage!r}')
        if self.request_id is not None and (self.stage != 'uploading' or not self.request_id):
            raise ValueError('A CVAT request ID is valid only for the uploading stage')
