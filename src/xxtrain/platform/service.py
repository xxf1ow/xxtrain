from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from xxtrain.business_tasks import POINT_BOX_LABELS
from xxtrain.platform.cache import build_detection_cache
from xxtrain.platform.config import WorkspaceConfig
from xxtrain.platform.contracts import PlatformAccessError, PlatformError, WorkspaceView
from xxtrain.platform.runtime import RuntimeCache
from xxtrain.workspace_data import WorkspaceData

if TYPE_CHECKING:
    from xxtrain.integrations.cvat.client import CvatClient


class AnnotationService:
    """Coordinate one workspace using file-derived views and disposable CVAT/cache mappings.

    All writes share one nonblocking process-local lock. Concurrent writes raise ``PlatformError``;
    other users receive ``PlatformAccessError`` before workspace access.
    """

    def __init__(self, config: WorkspaceConfig, data: WorkspaceData, cvat: CvatClient, runtime: RuntimeCache) -> None:
        self.config = config
        self.data = data
        self.cvat = cvat
        self.runtime = runtime
        self.lock = threading.Lock()

    def view(self, user_id: int) -> WorkspaceView:
        """Derive counts and cache readiness from current files; malformed inputs remain visible."""
        self._require_owner(user_id)
        summary = self.data.detection_summary()
        fingerprint = self.data.detection_fingerprint()
        return WorkspaceView(
            self.config.workspace_id,
            self.config.display_name,
            summary.image_count,
            summary.annotated_image_count,
            summary.boxed_image_count,
            summary.image_count > 0
            and summary.image_count == summary.annotated_image_count
            and summary.boxed_image_count >= 50,
            self.runtime.has_detection_cache(fingerprint),
        )

    def upload(self, user_id: int, staged: tuple[Path, ...]) -> WorkspaceView:
        """Admit staged images under the write lock and return a fresh view.

        Admission consumes the staged files. Ownership or lock rejection leaves them with the caller.
        """
        with self._write(user_id):
            self.data.admit(staged)
            return self.view(user_id)

    def begin_detection(self, user_id: int) -> str:
        """Return an unfinished current-input Job path, preparing a fresh task on a miss.

        Only successful preparation publishes a JobRef. CVAT failures surface as ``PlatformError``.
        """
        with self._write(user_id):
            fingerprint = self.data.detection_fingerprint()
            ref = self.runtime.job_for('detect', fingerprint)
            if ref is not None and self.cvat.job_is_unfinished(ref):
                return self.cvat.job_path(ref)
            images = self.data.images()
            if not images:
                raise PlatformError('Upload images before starting detection annotation')
            task_id = self.cvat.create_task(self.config.display_name, POINT_BOX_LABELS)
            ref = self.cvat.prepare_task(task_id, images, user_id)
            self.runtime.remember_job('detect', fingerprint, ref)
            return self.cvat.job_path(ref)

    def sync_detection(self, user_id: int) -> WorkspaceView:
        """Overwrite detection fields from the current-fingerprint Job and return file-derived counts.

        A missing Job or failed fetch/save raises ``PlatformError``; failed writes restore prior annotations.
        The Job is associated with the predicted resulting fingerprint before saving, so a mapping failure leaves
        annotations unchanged and successful saves remain repeatable.
        """
        with self._write(user_id):
            ref = self.runtime.job_for('detect', self.data.detection_fingerprint())
            if ref is None:
                raise PlatformError('Annotation job is not ready for the current input')
            try:
                results = self.cvat.fetch_detection(ref)
                self.runtime.remember_job('detect', self.data.detection_fingerprint(results), ref)
                self.data.save_detection(results)
            except (OSError, ValueError, PlatformError) as error:
                raise PlatformError('无法取回或保存标注，请重试。') from error
            return self.view(user_id)

    def generate_detection_cache(self, user_id: int) -> WorkspaceView:
        """Publish a fingerprinted dataset when all images are annotated and at least 50 contain boxes.

        Ineligible input raises ``PlatformError``; conversion and filesystem failures remain visible.
        """
        with self._write(user_id):
            view = self.view(user_id)
            if not view.can_generate_detection_cache:
                raise PlatformError('Detection cache requires all images annotated and at least 50 boxed images')
            if not view.detection_cache_ready:
                destination = self.config.runtime_dir / 'cache' / self.data.detection_fingerprint()
                destination.parent.mkdir(parents=True, exist_ok=True)
                build_detection_cache(self.data, destination)
            return self.view(user_id)

    @contextmanager
    def _write(self, user_id: int) -> Iterator[None]:
        self._require_owner(user_id)
        if not self.lock.acquire(blocking=False):
            raise PlatformError('A workspace write is already in progress')
        try:
            yield
        finally:
            self.lock.release()

    def _require_owner(self, user_id: int) -> None:
        if user_id != self.config.owner_user_id:
            raise PlatformAccessError('Workspace access denied')
