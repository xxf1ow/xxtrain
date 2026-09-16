import json
import os
import tempfile
from pathlib import Path

from xxtrain.platform.contracts import JobRef


class RuntimeCache:
    """Persist disposable CVAT job references and locate generated detection caches."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._jobs_path = self._root / 'jobs.json'

    def job_for(self, target: str, fingerprint: str) -> JobRef | None:
        """Return the cached job for one target and input fingerprint, if present."""
        jobs = self._load_jobs()
        return jobs.get(target, {}).get(fingerprint)

    def remember_job(self, target: str, fingerprint: str, ref: JobRef) -> None:
        """Atomically record the CVAT job associated with one target and fingerprint."""
        jobs = self._load_jobs()
        jobs.setdefault(target, {})[fingerprint] = ref
        self._replace(self._encode_jobs(jobs))

    def has_detection_cache(self, fingerprint: str) -> bool:
        """Return whether a complete Point detection dataset exists for the fingerprint."""
        return (self._root / 'cache' / fingerprint / 'detect').is_dir()

    def _load_jobs(self) -> dict[str, dict[str, JobRef]]:
        if not self._jobs_path.exists():
            return {}
        try:
            with self._jobs_path.open(encoding='utf-8') as stream:
                value = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError('Runtime job map is not a JSON object') from error
        if not isinstance(value, dict):
            raise ValueError('Runtime job map must be an object')

        jobs: dict[str, dict[str, JobRef]] = {}
        for target, fingerprints in value.items():
            if not isinstance(target, str) or not isinstance(fingerprints, dict):
                raise ValueError('Runtime job map must contain target/fingerprint mappings')
            jobs[target] = {}
            for fingerprint, entry in fingerprints.items():
                if (
                    not isinstance(fingerprint, str)
                    or not isinstance(entry, dict)
                    or set(entry) != {'task_id', 'job_id', 'sample_ids'}
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
                jobs[target][fingerprint] = JobRef(task_id, job_id, tuple(sample_ids))
        return jobs

    @staticmethod
    def _encode_jobs(jobs: dict[str, dict[str, JobRef]]) -> bytes:
        value = {
            target: {
                fingerprint: {'task_id': ref.task_id, 'job_id': ref.job_id, 'sample_ids': list(ref.sample_ids)}
                for fingerprint, ref in fingerprints.items()
            }
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
