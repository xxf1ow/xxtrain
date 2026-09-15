import json
import os
import tempfile
from pathlib import Path

from PIL import Image

from xxtrain.platform.contracts import FrameResult, ImageInput, JsonObject, PlatformAccessError
from xxtrain.workspace_data.labelme import _detection_boxes, _empty_document, merge_detection

_IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}


class WorkspaceData:
    """Access one flat, preconfigured image directory and its matching-stem LabelMe files."""

    def __init__(self, images_dir: Path, annotations_dir: Path) -> None:
        self._images_dir = Path(images_dir)
        self._annotations_dir = Path(annotations_dir)
        self._require_directory(self._images_dir, writable=False)
        self._require_directory(self._annotations_dir, writable=True)
        try:
            images = tuple(
                sorted(
                    (
                        path
                        for path in self._images_dir.iterdir()
                        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
                    ),
                    key=lambda path: path.name,
                )
            )
        except OSError as error:
            raise PlatformAccessError(f'Cannot read image directory: {self._images_dir}') from error

        stems: dict[str, Path] = {}
        for path in images:
            key = path.stem.casefold()
            if key in stems:
                raise ValueError(f'Duplicate image stem: {path.stem}')
            stems[key] = path
        self._images = images

    @staticmethod
    def _require_directory(path: Path, *, writable: bool) -> None:
        access = os.R_OK | os.X_OK | (os.W_OK if writable else 0)
        if not path.is_dir():
            raise PlatformAccessError(f'Directory does not exist: {path}')
        if not os.access(path, access):
            raise PlatformAccessError(f'Directory is not accessible: {path}')

    def images(self) -> tuple[ImageInput, ...]:
        """Return actual image dimensions and validated detection rectangles in filename order."""
        result = []
        for image_path in self._images:
            with Image.open(image_path) as image:
                width, height = image.size
            document = self._read_document(self._annotation_path(image_path.stem))
            result.append(
                ImageInput(
                    sample_id=image_path.stem,
                    image_path=image_path,
                    width=width,
                    height=height,
                    boxes=_detection_boxes(document),
                )
            )
        return tuple(result)

    def save_detection(self, results: tuple[FrameResult, ...]) -> None:
        """Replace rectangles after exact-set validation and encoding, using one atomic replacement per JSON.

        Earlier replacements remain if a later replacement fails; retrying overwrites the same paths.
        """
        expected = {path.stem for path in self._images}
        received = [result.sample_id for result in results]
        if len(received) != len(set(received)):
            raise ValueError('Detection results contain duplicate sample IDs')
        if set(received) != expected:
            raise ValueError('Detection results must cover every workspace image exactly once')

        by_sample = {result.sample_id: result for result in results}
        encoded = []
        for image_path in self._images:
            destination = self._annotation_path(image_path.stem)
            document = self._read_document(destination)
            if not destination.is_file():
                with Image.open(image_path) as image:
                    width, height = image.size
                document = _empty_document(
                    image_path=os.path.relpath(image_path, self._annotations_dir).replace('\\', '/'),
                    width=width,
                    height=height,
                )
            merged = merge_detection(document, by_sample[image_path.stem].boxes)
            payload = json.dumps(merged, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
            encoded.append((destination, payload))

        for destination, payload in encoded:
            self._replace(destination, payload)

    def _annotation_path(self, sample_id: str) -> Path:
        return self._annotations_dir / f'{sample_id}.json'

    @staticmethod
    def _read_document(path: Path) -> JsonObject:
        if not path.is_file():
            return {'shapes': []}
        with path.open(encoding='utf-8') as stream:
            document = json.load(stream)
        if not isinstance(document, dict):
            raise ValueError('LabelMe document must be an object')
        return document

    @staticmethod
    def _replace(destination: Path, payload: bytes) -> None:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
