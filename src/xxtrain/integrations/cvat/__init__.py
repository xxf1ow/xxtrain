from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import CvatClient

__all__ = ['CvatClient']


def __getattr__(name: str) -> object:
    if name == 'CvatClient':
        from .client import CvatClient

        return CvatClient
    raise AttributeError(name)
