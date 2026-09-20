from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from xxtrain.business_tasks import AnnotationValidationError, validate_step_annotations
from xxtrain.business_tasks.definition import StepDefinition
from xxtrain.platform.config import WorkspaceConfig
from xxtrain.platform.contracts import (
    EditFrameResult,
    EditJob,
    FrameMapping,
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
        summaries = {step.key: self.data.target_summary(step.key) for step in self.data.task.steps}
        complete = {
            step.key: (
                summaries[step.key].sample_count > 0
                and summaries[step.key].sample_count == summaries[step.key].annotated_sample_count
                and summaries[step.key].positive_sample_count >= step.minimum_samples
            )
            for step in self.data.task.steps
        }
        targets = []
        for step in self.data.task.steps:
            summary = summaries[step.key]
            dependencies = self.data.task.input_steps(step.key) - {step.key}
            dependencies_complete = all(complete[dependency] for dependency in dependencies)
            can_annotate = summary.sample_count > 0 and dependencies_complete
            can_generate_cache = can_annotate and complete[step.key] and step.training is not None
            cache_ready = can_generate_cache and self.runtime.has_target_cache(
                step.key, self.data.training_fingerprint(step.key)
            )
            targets.append(
                TargetView(
                    id=step.key,
                    sample_count=summary.sample_count,
                    annotated_sample_count=summary.annotated_sample_count,
                    can_annotate=can_annotate,
                    can_generate_cache=can_generate_cache,
                    cache_ready=cache_ready,
                    display_name=step.display_name,
                    sample_unit=step.sample_unit,
                )
            )
        detection = summaries.get('detect')
        detection_complete = complete.get('detect', False)
        detection_cache_ready = next((target.cache_ready for target in targets if target.id == 'detect'), False)
        return WorkspaceView(
            self.config.workspace_id,
            self.config.display_name,
            detection.sample_count if detection is not None else 0,
            detection.annotated_sample_count if detection is not None else 0,
            detection.positive_sample_count if detection is not None else 0,
            detection_complete,
            detection_cache_ready,
            tuple(targets),
            self.data.task.key,
            self.data.task.display_name,
        )

    def upload(self, user_id: int, staged: tuple[Path, ...]) -> WorkspaceView:
        """Admit staged images under the write lock and return a fresh view.

        Admission consumes the staged files. Ownership or lock rejection leaves them with the caller.
        """
        with self.mutation(user_id):
            self._require_editable(None)
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
            self._require_editable(target)
            view = self._target_view(self.view(user_id), target)
            if not view.can_annotate:
                raise PlatformError(f'{target.title()} annotation prerequisites are not complete')
            step = self.data.task.step(target)
            assert step.annotation is not None
            fingerprint = self.data.target_fingerprint(target)
            job = self.runtime.edit_job_for(target, fingerprint)
            if job is not None and self.cvat.job_is_unfinished(job.ref):
                return self._annotation_path(job.ref, step.annotation.workspace)
            frames = self.data.target_frames(target, self.config.runtime_dir)
            task_id = self.cvat.create_task(
                f'{self.config.display_name} {step.display_name}', tuple(sorted(step.labels)), step.annotation
            )
            prepared = self.cvat.prepare_task(task_id, frames, user_id, step.annotation)
            edit_job = EditJob(prepared.ref, tuple(frame.mapping for frame in frames))
            self.data.bind_job(prepared)
            self.runtime.remember_edit_job(target, fingerprint, edit_job)
            return self._annotation_path(prepared.ref, step.annotation.workspace)

    def sync_detection(self, user_id: int) -> WorkspaceView:
        """Invalidate dependent Jobs, publish the result fingerprint, then commit one database transaction."""
        return self.sync_target(user_id, 'detect')

    def sync_target(self, user_id: int, target: str) -> WorkspaceView:
        """Validate, prepublish runtime mappings, and atomically commit one target Job."""
        self._validate_target(target)
        with self.mutation(user_id):
            self._require_editable(target)
            step = self.data.task.step(target)
            assert step.annotation is not None
            fingerprint = self.data.target_fingerprint(target)
            job = self.runtime.edit_job_for(target, fingerprint)
            if job is None:
                raise PlatformError('Annotation job is not ready for the current input')
            try:
                current_frames = self.data.target_frames(target, self.config.runtime_dir)
                if tuple(frame.mapping for frame in current_frames) != job.frames:
                    raise ValueError('Edit job does not match current crop sources')
                results = self.cvat.fetch_annotations(job, step.annotation)
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
            raise PlatformError(f'{target.title()} cache requires every input sample to be annotated')
        fingerprint = self.data.training_fingerprint(target)
        if not target_view.cache_ready:
            if self.require_cache_rebuild is not None:
                self.require_cache_rebuild(self.config.workspace_id, target, fingerprint)
            destination = self.config.runtime_dir / 'cache' / fingerprint
            destination.parent.mkdir(parents=True, exist_ok=True)
            build_target_cache(self.data, target, self.config.runtime_dir, destination)
        return self.runtime.cache_path(target, fingerprint)

    def _validate_edit_results(self, target: str, job: EditJob, results: tuple[EditFrameResult, ...]) -> None:
        if tuple(result.frame_id for result in results) != tuple(frame.frame_id for frame in job.frames):
            raise ValueError('Edit results do not match the current frame order')
        step = self.data.task.step(target)
        for index, (frame, result) in enumerate(zip(job.frames, results, strict=True)):
            try:
                validate_step_annotations(step, result.annotations)
                width = frame.bounds[2] - frame.bounds[0]
                height = frame.bounds[3] - frame.bounds[1]
                for annotation in result.annotations:
                    if annotation.geometry is not None:
                        for point in annotation.geometry:
                            if not 0 <= float(point[0]) <= width or not 0 <= float(point[1]) <= height:
                                raise AnnotationValidationError('bounds', 'Annotation point lies outside the frame')
            except (TypeError, ValueError) as error:
                assert step.annotation is not None
                path = self._annotation_path(job.ref, step.annotation.workspace, frame=index)
                reason = _safe_validation_reason(step, frame, error)
                raise TargetValidationError(f'样本 {index + 1}：{reason}请返回当前任务修正。', path) from error

    def _annotation_path(self, ref: JobRef, workspace: str, *, frame: int | None = None) -> str:
        query = {}
        if workspace != 'STANDARD':
            query['defaultWorkspace'] = workspace
        if frame is not None:
            query['frame'] = str(frame)
        path = self.cvat.job_path(ref)
        return f'{path}?{urlencode(query)}' if query else path

    @staticmethod
    def _target_view(view: WorkspaceView, target: str) -> TargetView:
        return next(item for item in view.targets if item.id == target)

    def _validate_target(self, target: str) -> None:
        self.data.task.step(target)

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

    def _require_editable(self, target: str | None) -> None:
        if self.require_editable is not None:
            self.require_editable(self.config.workspace_id, target)

    def _require_owner(self, user_id: int) -> None:
        if user_id != self.config.owner_user_id:
            raise PlatformAccessError('Workspace access denied')


def _safe_validation_reason(step: StepDefinition, frame: FrameMapping, error: Exception) -> str:
    """Format one structured task-rule failure without exposing backend diagnostics."""
    if not isinstance(error, AnnotationValidationError):
        return '标注数量、类型、标签或坐标不符合当前任务规则。'
    assert step.annotation is not None
    policy = step.annotation
    if error.reason == 'negative_conflict' and policy.negative_label is not None:
        return f'“{policy.negative_label}”标签不能与其他标注同时存在。'
    if error.reason == 'cardinality':
        if policy.maximum_annotations == 1 and policy.cvat_type == 'tag':
            return f'每{step.sample_unit}只能保留一个{step.display_name}标签，请删除多余标签。'
        return f'每{step.sample_unit}最多允许 {policy.maximum_annotations} 条标注，请删除多余标注。'
    if error.reason == 'point_count':
        if policy.point_count == 2:
            return '每条线必须恰好有两个点，请删除错误线并用两点重新绘制。'
        if policy.point_count is not None:
            return f'每个标注必须恰好有 {policy.point_count} 个点，请重新绘制。'
        return '标注点数量不符合当前任务规则，请重新绘制。'
    if error.reason == 'coincident_points':
        return '线的两个端点不能重合，请重新绘制有长度的线。'
    if error.reason == 'bounds':
        subject = {'polyline': '线的端点', 'rectangle': '矩形端点', 'polygon': '多边形顶点'}.get(
            policy.cvat_type, '标注点'
        )
        image_kind = '裁剪图' if frame.parent_id is not None else '图片'
        return f'{subject}必须位于{image_kind}内，请将越界点移回图内。'
    return '标注数量、类型、标签或坐标不符合当前任务规则。'
