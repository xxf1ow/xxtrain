import json
import os
import tempfile
from pathlib import Path
from uuid import UUID

from xxtrain.platform.contracts import EditJob, FrameMapping, JobRef

type RuntimeJob = JobRef | EditJob


class RuntimeCache:
    """Persist disposable CVAT job references and locate generated training caches."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._jobs_path = self._root / 'jobs.json'

    def job_for(self, target: str, fingerprint: str) -> JobRef | None:
        """Return the cached job for one target and input fingerprint, if present."""
        jobs = self._load_jobs()
        job = jobs.get(target, {}).get(fingerprint)
        return job.ref if isinstance(job, EditJob) else job

    def edit_job_for(self, target: str, fingerprint: str) -> EditJob | None:
        """Return a persisted edit job, or miss when the entry has no frame-source mapping.

        Malformed persisted mappings raise ``ValueError`` instead of inferring their missing source identity.
        """
        job = self._load_jobs().get(target, {}).get(fingerprint)
        return job if isinstance(job, EditJob) else None

    def remember_job(self, target: str, fingerprint: str, ref: JobRef) -> None:
        """Atomically record the CVAT job associated with one target and fingerprint."""
        jobs = self._load_jobs()
        jobs.setdefault(target, {})[fingerprint] = ref
        self._replace(self._encode_jobs(jobs))

    def remember_edit_job(self, target: str, fingerprint: str, job: EditJob) -> None:
        """Atomically record a validated edit job and its ordered frame-to-original mappings."""
        _validate_edit_job(job)
        jobs = self._load_jobs()
        jobs.setdefault(target, {})[fingerprint] = job
        self._replace(self._encode_jobs(jobs))

    def forget_targets(self, targets: frozenset[str]) -> None:
        """Atomically remove every cached Job reference for the selected targets."""
        if not targets:
            return
        jobs = self._load_jobs()
        changed = False
        for target in targets:
            if target in jobs:
                del jobs[target]
                changed = True
        if changed:
            self._replace(self._encode_jobs(jobs))

    def has_detection_cache(self, fingerprint: str) -> bool:
        """Return whether a complete Point detection dataset exists for the fingerprint."""
        try:
            self.cache_path('detect', fingerprint)
        except ValueError:
            return False
        return True

    def has_target_cache(self, target: str, fingerprint: str) -> bool:
        """Return whether a downstream cache has its exact manifest and target directory.

        Missing, unreadable, and malformed disposable manifests are cache misses. Unsupported targets raise
        ``ValueError``.
        """
        if target not in {'classify', 'segment'}:
            raise ValueError(f'Unsupported target cache: {target!r}')
        try:
            self.cache_path(target, fingerprint)
        except ValueError:
            return False
        return True

    def cache_path(self, target: str, fingerprint: str) -> Path:
        """Return one complete published target cache or reject an invalid, incomplete, or escaping reference."""
        if target not in {'detect', 'classify', 'segment'}:
            raise ValueError(f'Unsupported target cache: {target!r}')
        if not _safe_component(fingerprint):
            raise ValueError('Training cache fingerprint must name one relative cache directory')
        cache_root = self._root / 'cache'
        publication = cache_root / fingerprint
        path = publication / target
        if not _is_within(path, cache_root):
            raise ValueError('Training cache is not complete')
        if target != 'detect' and not _has_target_manifest(publication, target, fingerprint):
            raise ValueError('Training cache is not complete')
        if not _has_training_inputs(path):
            raise ValueError('Training cache is not complete')
        return path

    def _load_jobs(self) -> dict[str, dict[str, RuntimeJob]]:
        if not self._jobs_path.exists():
            return {}
        try:
            with self._jobs_path.open(encoding='utf-8') as stream:
                value = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError('Runtime job map is not a JSON object') from error
        if not isinstance(value, dict):
            raise ValueError('Runtime job map must be an object')

        jobs: dict[str, dict[str, RuntimeJob]] = {}
        for target, fingerprints in value.items():
            if not isinstance(target, str) or not isinstance(fingerprints, dict):
                raise ValueError('Runtime job map must contain target/fingerprint mappings')
            jobs[target] = {}
            for fingerprint, entry in fingerprints.items():
                if (
                    not isinstance(fingerprint, str)
                    or not isinstance(entry, dict)
                    or set(entry)
                    not in ({'task_id', 'job_id', 'sample_ids'}, {'task_id', 'job_id', 'sample_ids', 'frames'})
                ):
                    raise ValueError('Runtime job map contains an invalid JobRef')
                task_id = entry['task_id']
                job_id = entry['job_id']
                sample_ids = entry['sample_ids']
                if (
                    type(task_id) is not int
                    or type(job_id) is not int
                    or not isinstance(sample_ids, list)
                    or any(not isinstance(sample_id, str) for sample_id in sample_ids)
                ):
                    raise ValueError('Runtime job map contains an invalid JobRef')
                ref = JobRef(task_id, job_id, tuple(sample_ids))
                if 'frames' not in entry:
                    jobs[target][fingerprint] = ref
                    continue
                frames = _decode_frames(entry['frames'])
                job = EditJob(ref, frames)
                _validate_edit_job(job)
                jobs[target][fingerprint] = job
        return jobs

    @staticmethod
    def _encode_jobs(jobs: dict[str, dict[str, RuntimeJob]]) -> bytes:
        value = {
            target: {fingerprint: _encode_job(job) for fingerprint, job in fingerprints.items()}
            for target, fingerprints in jobs.items()
        }
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')

    def _replace(self, payload: bytes) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self._root, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary, self._jobs_path)
        finally:
            temporary.unlink(missing_ok=True)


def _safe_component(value: str) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).name == value and value not in {'.', '..'}


def _has_target_manifest(publication: Path, target: str, fingerprint: str) -> bool:
    try:
        with (publication / 'manifest.json').open(encoding='utf-8') as stream:
            manifest = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return manifest == {'fingerprint': fingerprint, 'target': target}


def _has_training_inputs(path: Path) -> bool:
    dataset = path / 'dataset.yaml'
    if not dataset.is_file():
        return False
    try:
        fields = _dataset_fields(dataset.read_text(encoding='utf-8'))
    except (OSError, UnicodeError):
        return False
    publication = path.parent
    root = Path(fields.get('path', ''))
    if not root or root.absolute() != publication.absolute():
        return False
    for split in ('train', 'val'):
        value = fields.get(split)
        if value is None:
            return False
        listed = Path(value)
        list_path = listed if listed.is_absolute() else publication / listed
        if not _is_within(list_path, publication) or not list_path.is_file():
            return False
        try:
            entries = tuple(line.strip() for line in list_path.read_text(encoding='utf-8').splitlines() if line.strip())
        except (OSError, UnicodeError):
            return False
        if not entries:
            return False
        for entry in entries:
            item = Path(entry)
            item_path = item if item.is_absolute() else publication / item
            if not _has_referenced_file(item_path, publication):
                return False
    return True


def _dataset_fields(value: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in value.splitlines():
        key, separator, field_value = line.partition(': ')
        if separator and key.strip() in {'path', 'train', 'val'}:
            cleaned = field_value.strip().strip('\'"')
            if cleaned:
                fields[key.strip()] = cleaned
    return fields


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _has_referenced_file(path: Path, publication: Path) -> bool:
    try:
        path.absolute().relative_to(publication.absolute())
    except ValueError:
        return False
    return path.is_file()


def _decode_frames(value: object) -> tuple[FrameMapping, ...]:
    if not isinstance(value, list):
        raise ValueError('Runtime edit job map frames must be a list')
    frames: list[FrameMapping] = []
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {'frame_id', 'image_id', 'parent_id', 'bounds'}:
            raise ValueError('Runtime edit job map contains an invalid frame mapping')
        frame_id = entry['frame_id']
        image_id = entry['image_id']
        parent_value = entry['parent_id']
        bounds_value = entry['bounds']
        if (
            not isinstance(frame_id, str)
            or not frame_id
            or not isinstance(image_id, str)
            or not image_id
            or (parent_value is not None and not isinstance(parent_value, str))
            or not isinstance(bounds_value, list)
            or len(bounds_value) != 4
            or any(type(value) is not int for value in bounds_value)
        ):
            raise ValueError('Runtime edit job map contains an invalid frame mapping')
        try:
            parent_id = UUID(parent_value) if parent_value is not None else None
        except ValueError as error:
            raise ValueError('Runtime edit job map contains an invalid parent UUID') from error
        frames.append(FrameMapping(frame_id, image_id, parent_id, tuple(bounds_value)))
    return tuple(frames)


def _validate_edit_job(job: EditJob) -> None:
    sample_ids = job.ref.sample_ids
    if (
        type(job.ref.task_id) is not int
        or type(job.ref.job_id) is not int
        or not isinstance(sample_ids, tuple)
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids)
    ):
        raise ValueError('Runtime edit job map contains an invalid JobRef')
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError('Runtime edit job map requires ordered unique original image IDs')
    sample_id_set = set(sample_ids)
    frame_ids: set[str] = set()
    referenced_images: list[str] = []
    referenced_image_set: set[str] = set()
    for frame in job.frames:
        if (
            not isinstance(frame.frame_id, str)
            or not frame.frame_id
            or not isinstance(frame.image_id, str)
            or not frame.image_id
            or (frame.parent_id is not None and not isinstance(frame.parent_id, UUID))
        ):
            raise ValueError('Runtime edit job map contains an invalid frame mapping')
        if frame.frame_id in frame_ids:
            raise ValueError(f'Runtime edit job map contains duplicate frame ID {frame.frame_id!r}')
        frame_ids.add(frame.frame_id)
        if frame.image_id not in sample_id_set:
            raise ValueError(f'Runtime edit job map frame {frame.frame_id!r} references an unknown original image')
        if frame.image_id not in referenced_image_set:
            referenced_images.append(frame.image_id)
            referenced_image_set.add(frame.image_id)
        if (
            not isinstance(frame.bounds, tuple)
            or len(frame.bounds) != 4
            or any(type(value) is not int for value in frame.bounds)
        ):
            raise ValueError(f'Runtime edit job map frame {frame.frame_id!r} has malformed bounds')
        x1, y1, x2, y2 = frame.bounds
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError(f'Runtime edit job map frame {frame.frame_id!r} has malformed bounds')
        expected_frame_id = frame.image_id if frame.parent_id is None else str(frame.parent_id)
        if frame.frame_id != expected_frame_id:
            raise ValueError(f'Runtime edit job map frame {frame.frame_id!r} has mismatched identity')
    if tuple(referenced_images) != sample_ids:
        raise ValueError('Runtime edit job map samples must match ordered unique frame image IDs')


def _encode_job(job: RuntimeJob) -> dict[str, object]:
    ref = job.ref if isinstance(job, EditJob) else job
    value: dict[str, object] = {'task_id': ref.task_id, 'job_id': ref.job_id, 'sample_ids': list(ref.sample_ids)}
    if isinstance(job, EditJob):
        value['frames'] = [
            {
                'frame_id': frame.frame_id,
                'image_id': frame.image_id,
                'parent_id': str(frame.parent_id) if frame.parent_id is not None else None,
                'bounds': list(frame.bounds),
            }
            for frame in job.frames
        ]
    return value
