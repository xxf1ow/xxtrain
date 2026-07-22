from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LabelCatalog:
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError('Label catalog must not be empty')
        if any(not isinstance(name, str) or not name for name in self.names):
            raise ValueError('Label names must be non-empty strings')
        if len(self.names) != len(set(self.names)):
            raise ValueError('Label names must not contain duplicates')

    def index(self, name: str) -> int:
        return self.names.index(name)

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)
