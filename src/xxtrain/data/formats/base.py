from pathlib import Path
from typing import Protocol

from ..annotation import Annotation, ImageInfo


class AnnotationReader(Protocol):
    def __call__(self, path: str | Path, image_info: ImageInfo) -> list[Annotation]: ...
