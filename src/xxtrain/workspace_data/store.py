import json
import os
import tempfile
from hashlib import sha256
from pathlib import Path

from PIL import Image

from xxtrain.platform.contracts import (
    DetectionSummary,
    FrameResult,
    ImageInput,
    JsonObject,
    PlatformAccessError,
    UploadResult,
)
from xxtrain.workspace_data.dedup import SIMILARITY_DISTANCE, hamming_distance, image_sha256, perceptual_hash
from xxtrain.workspace_data.labelme import _detection_boxes, _empty_document, detection_complete, merge_detection

_IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}


class WorkspaceData:
    """Access workspace ``images`` and ``annotations`` directories from a root or explicit paths."""

    def __init__(self, workspace_dir: Path, annotations_dir: Path | None = None) -> None:
        if annotations_dir is None:
            self._images_dir = Path(workspace_dir) / 'images'
            self._annotations_dir = Path(workspace_dir) / 'annotations'
        else:
            self._images_dir = Path(workspace_dir)
            self._annotations_dir = Path(annotations_dir)
        self._require_directory(self._images_dir, writable=True)
        self._require_directory(self._annotations_dir, writable=True)

    @staticmethod
    def _require_directory(path: Path, *, writable: bool) -> None:
        access = os.R_OK | os.X_OK | (os.W_OK if writable else 0)
        if not path.is_dir():
            raise PlatformAccessError(f'Directory does not exist: {path}')
        if not os.access(path, access):
            raise PlatformAccessError(f'Directory is not accessible: {path}')

    def _image_paths(self) -> tuple[Path, ...]:
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
        return images

    def images(self) -> tuple[ImageInput, ...]:
        """Return actual image dimensions and validated detection rectangles in filename order."""
        result = []
        for image_path in self._image_paths():
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

    def admit(self, staged: tuple[Path, ...]) -> UploadResult:
        """Admit decodable staged images once, deleting every staged file before returning."""
        try:
            existing = tuple((image_sha256(path), perceptual_hash(path)) for path in self._image_paths())
            existing_shas = {digest for digest, _ in existing}
            existing_hashes = tuple(image_hash for _, image_hash in existing)
            candidates = []
            for path in staged:
                path = Path(path)
                if path.suffix.lower() not in _IMAGE_SUFFIXES:
                    continue
                try:
                    candidates.append((image_sha256(path), perceptual_hash(path), path))
                except ValueError:
                    continue

            accepted_count = 0
            exact_duplicate_count = 0
            similar_duplicate_count = 0
            accepted_hashes = []
            batch_shas = set()
            for digest, image_hash, path in sorted(candidates, key=lambda candidate: candidate[0]):
                if digest in existing_shas or digest in batch_shas:
                    exact_duplicate_count += 1
                    continue
                batch_shas.add(digest)
                if any(
                    hamming_distance(image_hash, compared_hash) <= SIMILARITY_DISTANCE
                    for compared_hash in (*existing_hashes, *accepted_hashes)
                ):
                    similar_duplicate_count += 1
                    continue
                os.replace(path, self._images_dir / f'{digest}{path.suffix.lower()}')
                existing_shas.add(digest)
                accepted_hashes.append(image_hash)
                accepted_count += 1

            return UploadResult(
                received_count=len(staged),
                accepted_count=accepted_count,
                exact_duplicate_count=exact_duplicate_count,
                similar_duplicate_count=similar_duplicate_count,
            )
        finally:
            for path in staged:
                Path(path).unlink(missing_ok=True)

    def detection_summary(self) -> DetectionSummary:
        """Derive completed and boxed detection-image counts from current LabelMe files."""
        image_count = 0
        annotated_image_count = 0
        boxed_image_count = 0
        for image_path in self._image_paths():
            image_count += 1
            document = self._read_document(self._annotation_path(image_path.stem))
            boxes = _detection_boxes(document)
            if detection_complete(document):
                annotated_image_count += 1
            if boxes:
                boxed_image_count += 1
        return DetectionSummary(image_count, annotated_image_count, boxed_image_count)

    def detection_fingerprint(self) -> str:
        """Hash current image bytes and canonical detection boxes or negative markers."""
        records = []
        for image_path in self._image_paths():
            document = self._read_document(self._annotation_path(image_path.stem))
            boxes = _detection_boxes(document)
            if boxes:
                detection = {
                    'boxes': sorted(
                        (
                            {
                                'label': box.geometry.label,
                                'points': [[box.geometry.x1, box.geometry.y1], [box.geometry.x2, box.geometry.y2]],
                                'extra': box.extra,
                            }
                            for box in boxes
                        ),
                        key=lambda box: json.dumps(box, ensure_ascii=False, sort_keys=True, separators=(',', ':')),
                    )
                }
            elif detection_complete(document):
                detection = {'negative': True}
            else:
                detection = None
            records.append({'sha256': image_sha256(image_path), 'detection': detection})
        payload = json.dumps(
            sorted(records, key=lambda record: record['sha256']),
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
            allow_nan=False,
        ).encode('utf-8')
        return sha256(payload).hexdigest()

    def save_detection(self, results: tuple[FrameResult, ...]) -> None:
        """Replace rectangles after exact-set validation and encoding, using one atomic replacement per JSON.

        Earlier replacements remain if a later replacement fails; retrying overwrites the same paths.
        """
        images = self._image_paths()
        expected = {path.stem for path in images}
        received = [result.sample_id for result in results]
        if len(received) != len(set(received)):
            raise ValueError('Detection results contain duplicate sample IDs')
        if set(received) != expected:
            raise ValueError('Detection results must cover every workspace image exactly once')

        by_sample = {result.sample_id: result for result in results}
        encoded = []
        for image_path in images:
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
