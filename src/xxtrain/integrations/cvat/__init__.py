from typing import TYPE_CHECKING

from .preparation import PreparationCheckpoint, PreparationStage, PreparationState

if TYPE_CHECKING:
    from .client import CvatClient

__all__ = ['CvatClient', 'PreparationCheckpoint', 'PreparationStage', 'PreparationState']


def __getattr__(name: str) -> object:
    if name == 'CvatClient':
        from .client import CvatClient

        return CvatClient
    raise AttributeError(name)
