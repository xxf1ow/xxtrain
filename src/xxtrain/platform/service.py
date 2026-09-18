from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from xxtrain.business_tasks import POINT_BOX_LABELS, validate_target_annotations
from xxtrain.platform.cache import build_detection_cache
from xxtrain.platform.config import WorkspaceConfig
from xxtrain.platform.contracts import (
    EditFrameResult,
    EditJob,
    JobRef,
    PlatformAccessError,
    PlatformConflictError,
    PlatformError,
    TargetValidationError,
    TargetView,
    WorkspaceCacheRebuildGuard,
    WorkspaceEditGuard,
    WorkspaceView,
)
from xxtrain.platform.runtime import RuntimeCache
from xxtrain.platform.target_cache import build_target_cache
from xxtrain.workspace_data import WorkspaceData

if TYPE_CHECKING:
    from xxtrain.integrations.cvat.client import CvatClient


_TARGETS = frozenset({'detect', 'classify', 'segment'})
_EDIT_TARGETS = {'classify': (('tl', 'tc', 'cl', 'cc'), 'tag'), 'segment': (('1',), 'polyline')}
_EDIT_VALIDATION_REASONS = {
    'Point classification requires at most one annotation': '每张裁剪图只能保留一个分类标签，请删除多余标签。',
    'Point segmentation lines require exactly two points': '每条指针线必须恰好有两个点，请删除错误线并用两点重新绘制。',
    'Point segmentation lines require two distinct points': '指针线的两个端点不能重合，请重新绘制有长度的线。',
    'Line point lies outside the crop': '指针线的端点必须位于裁剪图内，请将越界端点移回图内。',
}


class AnnotationService:
    """Coordinate one SQLite-backed workspace and its disposable CVAT/cache mappings.

    All writes share one nonblocking process-local lock. Concurrent writes raise ``PlatformError``;
    other users receive ``PlatformAccessError`` before workspace access.
    """

    def __init__(
        self,
        config: WorkspaceConfig,
        data: WorkspaceData,
        cvat: CvatClient,
        runtime: RuntimeCache,
        *,
        require_editable: WorkspaceEditGuard | None = None,
        require_cache_rebuild: WorkspaceCacheRebuildGuard | None = None,
    ) -> None:
        self.config = config
        self.data = data
        self.cvat = cvat
        self.runtime = runtime
        self.require_editable = require_editable
        self.require_cache_rebuild = require_cache_rebuild
        self.lock = threading.Lock()

    def view(self, user_id: int) -> WorkspaceView:
        """Derive counts and cache readiness from current registered annotation facts."""
        self._require_owner(user_id)
        summary = self.data.detection_summary()
        fingerprint = self.data.detection_fingerprint()
        detection_complete = (
            summary.image_count > 0
            and summary.image_count == summary.annotated_image_count
            and summary.boxed_image_count >= 50
        )
        classify = self.data.target_summary('classify')
        classify_complete = (
            detection_complete
            and classify.sample_count > 0
            and classify.sample_count == classify.annotated_sample_count
        )
        segment = self.data.target_summary('segment')
        segment_complete = (
            classify_complete and segment.sample_count > 0 and segment.sample_count == segment.annotated_sample_count
        )
        detection_cache_ready = detection_complete and self.runtime.has_detection_cache(fingerprint)
        targets = (
            TargetView(
                'detect',
                summary.image_count,
                summary.annotated_image_count,
                summary.image_count > 0,
                detection_complete,
                detection_cache_ready,
            ),
            TargetView(
                'classify',
                classify.sample_count,
                classify.annotated_sample_count,
                detection_complete,
                classify_complete,
                classify_complete
                and self.runtime.has_target_cache('classify', self.data.target_fingerprint('classify')),
            ),
            TargetView(
                'segment',
                segment.sample_count,
                segment.annotated_sample_count,
                classify_complete,
                segment_complete,
                segment_complete and self.runtime.has_target_cache('segment', self.data.target_fingerprint('segment')),
            ),
        )
        return WorkspaceView(
            self.config.workspace_id,
            self.config.display_name,
            summary.image_count,
            summary.annotated_image_count,
            summary.boxed_image_count,
            detection_complete,
            detection_cache_ready,
            targets,
        )

    def upload(self, user_id: int, staged: tuple[Path, ...]) -> WorkspaceView:
        """Admit staged images under the write lock and return a fresh view.

        Admission consumes the staged files. Ownership or lock rejection leaves them with the caller.
        """
        with self.mutation(user_id):
            self._require_editable()
            self.data.admit(staged)
            return self.view(user_id)

    def begin_detection(self, user_id: int) -> str:
        """Return an unfinished current-input Job path, preparing a fresh task on a miss.

        The runtime publishes a JobRef only after preparation and durable object binding both succeed. CVAT failures
        surface as ``PlatformError``.
        """
        return self.begin_target(user_id, 'detect')

    def begin_target(self, user_id: int, target: str) -> str:
        """Return or prepare the current server-owned annotation Job for one model target."""
        self._validate_target(target)
        with self.mutation(user_id):
            self._require_editable()
            if target == 'detect':
                return self._begin_detection(user_id)
            view = self._target_view(self.view(user_id), target)
            if not view.can_annotate:
                raise PlatformError(f'{target.title()} annotation prerequisites are not complete')
            fingerprint = self.data.target_fingerprint(target)
            job = self.runtime.edit_job_for(target, fingerprint)
            if job is not None and self.cvat.job_is_unfinished(job.ref):
                return self._annotation_path(job.ref, target)
            frames = self.data.target_frames(target, self.config.runtime_dir)
            labels, label_type = _EDIT_TARGETS[target]
            task_id = self.cvat.create_edit_task(f'{self.config.display_name} {target}', labels, label_type)
            prepared = self.cvat.prepare_edit_task(task_id, frames, user_id)
            edit_job = EditJob(prepared.ref, tuple(frame.mapping for frame in frames))
            self.data.bind_job(prepared)
            self.runtime.remember_edit_job(target, fingerprint, edit_job)
            return self._annotation_path(prepared.ref, target)

    def sync_detection(self, user_id: int) -> WorkspaceView:
        """Invalidate dependent Jobs, publish the result fingerprint, then commit one database transaction."""
        return self.sync_target(user_id, 'detect')

    def sync_target(self, user_id: int, target: str) -> WorkspaceView:
        """Validate, prepublish runtime mappings, and atomically commit one target Job."""
        self._validate_target(target)
        with self.mutation(user_id):
            self._require_editable()
            if target == 'detect':
                return self._sync_detection(user_id)
            fingerprint = self.data.target_fingerprint(target)
            job = self.runtime.edit_job_for(target, fingerprint)
            if job is None:
                raise PlatformError('Annotation job is not ready for the current input')
            try:
                current_frames = self.data.target_frames(target, self.config.runtime_dir)
                if tuple(frame.mapping for frame in current_frames) != job.frames:
                    raise ValueError('Edit job does not match current crop sources')
                results = self.cvat.fetch_edit(job)
                self._validate_edit_results(target, job, results)
                sync = self.data.prepare_target_sync(target, job, results)
                self.runtime.forget_targets(sync.changes.invalidated_steps)
                self.runtime.remember_edit_job(target, sync.fingerprint, job)
                self.data.commit_target_sync(job, sync)
            except TargetValidationError:
                raise
            except (OSError, ValueError, PlatformError) as error:
                raise PlatformError('无法取回或保存标注，请重试。') from error
            return self.view(user_id)

    def generate_detection_cache(self, user_id: int) -> WorkspaceView:
        """Publish a fingerprinted dataset when all images are annotated and at least 50 contain boxes.

        Ineligible input raises ``PlatformError``; conversion and filesystem failures remain visible.
        """
        return self.generate_target_cache(user_id, 'detect')

    def generate_target_cache(self, user_id: int, target: str) -> WorkspaceView:
        """Generate the selected target's cache when its fact-derived completeness gate is open."""
        self._validate_target(target)
        with self.mutation(user_id):
            self.ensure_target_cache(user_id, target)
            return self.view(user_id)

    def ensure_target_cache(self, user_id: int, target: str) -> Path:
        """Return a complete current cache while the caller holds ``mutation(user_id)``."""
        self._require_owner(user_id)
        self._validate_target(target)
        view = self.view(user_id)
        target_view = self._target_view(view, target)
        if not target_view.can_generate_cache:
            if target == 'detect':
                raise PlatformError('Detection cache requires all images annotated and at least 50 boxed images')
            raise PlatformError(f'{target.title()} cache requires every crop to be annotated')
        fingerprint = self.data.detection_fingerprint() if target == 'detect' else self.data.target_fingerprint(target)
        if not target_view.cache_ready:
            if self.require_cache_rebuild is not None:
                self.require_cache_rebuild(self.config.workspace_id, target, fingerprint)
            destination = self.config.runtime_dir / 'cache' / fingerprint
            destination.parent.mkdir(parents=True, exist_ok=True)
            if target == 'detect':
                build_detection_cache(self.data, destination)
            else:
                build_target_cache(self.data, target, self.config.runtime_dir, destination)
        return self.runtime.cache_path(target, fingerprint)

    def _begin_detection(self, user_id: int) -> str:
        fingerprint = self.data.detection_fingerprint()
        ref = self.runtime.job_for('detect', fingerprint)
        if ref is not None and self.cvat.job_is_unfinished(ref):
            return self.cvat.job_path(ref)
        images = self.data.images()
        if not images:
            raise PlatformError('Upload images before starting detection annotation')
        task_id = self.cvat.create_task(self.config.display_name, POINT_BOX_LABELS)
        prepared = self.cvat.prepare_task(task_id, images, user_id)
        self.data.bind_job(prepared)
        self.runtime.remember_job('detect', fingerprint, prepared.ref)
        return self.cvat.job_path(prepared.ref)

    def _sync_detection(self, user_id: int) -> WorkspaceView:
        ref = self.runtime.job_for('detect', self.data.detection_fingerprint())
        if ref is None:
            raise PlatformError('Annotation job is not ready for the current input')
        try:
            results = self.cvat.fetch_detection(ref)
            for result in results:
                if result.negative and result.boxes:
                    index = ref.sample_ids.index(result.sample_id)
                    raise TargetValidationError(
                        f'图片 {index + 1}：检测框与“无检测目标”标签不能同时存在，请删除其中一种。',
                        self._annotation_path(ref, 'detect', frame=index),
                    )
            sync = self.data.prepare_detection_sync(ref, results)
            self.runtime.forget_targets(sync.changes.invalidated_steps)
            self.runtime.remember_job('detect', sync.fingerprint, ref)
            self.data.commit_detection_sync(ref, sync)
        except TargetValidationError:
            raise
        except (OSError, ValueError, PlatformError) as error:
            raise PlatformError('无法取回或保存标注，请重试。') from error
        return self.view(user_id)

    def _validate_edit_results(self, target: str, job: EditJob, results: tuple[EditFrameResult, ...]) -> None:
        if tuple(result.frame_id for result in results) != tuple(frame.frame_id for frame in job.frames):
            raise ValueError('Edit results do not match the current frame order')
        for index, (frame, result) in enumerate(zip(job.frames, results, strict=True)):
            try:
                validate_target_annotations(target, result.annotations)
                if target == 'segment':
                    width = frame.bounds[2] - frame.bounds[0]
                    height = frame.bounds[3] - frame.bounds[1]
                    for annotation in result.annotations:
                        for point in annotation.geometry:
                            if not 0 <= float(point[0]) <= width or not 0 <= float(point[1]) <= height:
                                raise ValueError('Line point lies outside the crop')
            except (TypeError, ValueError) as error:
                path = self._annotation_path(job.ref, target, frame=index)
                reason = _EDIT_VALIDATION_REASONS.get(
                    str(error), '标注类型、标签或坐标不符合要求，请检查当前目标的标注规则。'
                )
                raise TargetValidationError(f'裁剪图 {index + 1}：{reason}请返回当前任务修正。', path) from error

    def _annotation_path(self, ref: JobRef, target: str, *, frame: int | None = None) -> str:
        query = {}
        if target == 'classify':
            query['defaultWorkspace'] = 'TAGS'
        if frame is not None:
            query['frame'] = str(frame)
        path = self.cvat.job_path(ref)
        return f'{path}?{urlencode(query)}' if query else path

    @staticmethod
    def _target_view(view: WorkspaceView, target: str) -> TargetView:
        return next(item for item in view.targets if item.id == target)

    @staticmethod
    def _validate_target(target: str) -> None:
        if target not in _TARGETS:
            raise ValueError(f'Unsupported annotation target: {target!r}')

    @contextmanager
    def mutation(self, user_id: int) -> Iterator[None]:
        """Serialize one workspace mutation without permitting recursive acquisition."""
        self._require_owner(user_id)
        if not self.lock.acquire(blocking=False):
            raise PlatformConflictError('A workspace write is already in progress')
        try:
            yield
        finally:
            self.lock.release()

    def _require_editable(self) -> None:
        if self.require_editable is not None:
            self.require_editable(self.config.workspace_id)

    def _require_owner(self, user_id: int) -> None:
        if user_id != self.config.owner_user_id:
            raise PlatformAccessError('Workspace access denied')
