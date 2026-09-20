import re
from dataclasses import dataclass

from xxtrain.training.settings import TrainingSettings


@dataclass(frozen=True, slots=True)
class AnnotationPolicy:
    cvat_type: str
    workspace: str
    minimum_annotations: int = 1
    maximum_annotations: int | None = None
    point_count: int | None = None
    negative_label: str | None = None

    def __post_init__(self) -> None:
        if not self.cvat_type or not self.workspace:
            raise ValueError('Annotation policy type and workspace must be non-empty')
        for name in ('minimum_annotations', 'maximum_annotations', 'point_count'):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f'Annotation policy {name} must be a nonnegative integer')
        if self.maximum_annotations is not None and self.maximum_annotations < self.minimum_annotations:
            raise ValueError('Annotation maximum must not be below its minimum')


@dataclass(frozen=True, slots=True)
class DeliveryDefinition:
    labels: bool
    reference_images: bool


@dataclass(frozen=True, slots=True)
class TargetTrainingDefinition:
    settings: TrainingSettings
    metric_key: str
    metric_name: str
    delivery: DeliveryDefinition


@dataclass(frozen=True)
class StepDefinition:
    """Allowed records, parent steps, and dependency edges for one task step."""

    key: str
    kinds: frozenset[str]
    labels: frozenset[str]
    parent_steps: frozenset[str]
    depends_on: frozenset[str]
    training: TargetTrainingDefinition | None = None
    display_name: str | None = None
    annotation: AnnotationPolicy | None = None
    minimum_samples: int = 0

    def __post_init__(self) -> None:
        if self.display_name is None:
            object.__setattr__(self, 'display_name', self.key)
        if self.annotation is None:
            native = next(iter(self.kinds)) if len(self.kinds) == 1 else 'rectangle'
            if native == 'classification':
                native = 'tag'
            object.__setattr__(self, 'annotation', AnnotationPolicy(native, 'STANDARD'))


@dataclass(frozen=True)
class TaskDefinition:
    """Immutable validation and downstream-dependency rules for a business task."""

    steps: tuple[StepDefinition, ...]
    key: str = 'task'
    display_name: str = 'Task'

    def __post_init__(self) -> None:
        keys = tuple(step.key for step in self.steps)
        safe_key = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
        if not safe_key.fullmatch(self.key) or not self.display_name:
            raise ValueError('Task key must be a safe path component and display name must be non-empty')
        if not keys or any(not isinstance(key, str) or not safe_key.fullmatch(key) for key in keys):
            raise ValueError('Task steps require non-empty string keys')
        if len(set(keys)) != len(keys):
            raise ValueError('Task step keys must be unique')
        known = frozenset(keys)
        for step in self.steps:
            if not step.display_name:
                raise ValueError(f'Task step {step.key!r} requires a display name')
            if (
                isinstance(step.minimum_samples, bool)
                or not isinstance(step.minimum_samples, int)
                or step.minimum_samples < 0
            ):
                raise ValueError(f'Task step {step.key!r} minimum samples must be a nonnegative integer')
            if not step.kinds or any(not isinstance(value, str) or not value for value in step.kinds):
                raise ValueError(f'Task step {step.key!r} requires annotation kinds')
            if not step.labels:
                raise ValueError(f'Task step {step.key!r} requires labels')
            assert step.annotation is not None
            expected_kinds = {
                'tag': frozenset({'classification'}),
                'rectangle': frozenset({'rectangle'}),
                'polyline': frozenset({'polyline'}),
                'polygon': frozenset({'polygon'}),
            }.get(step.annotation.cvat_type)
            if expected_kinds is None:
                raise ValueError(f'Task step {step.key!r} has unsupported native annotation type')
            if step.annotation.negative_label is not None and step.annotation.cvat_type == 'rectangle':
                expected_kinds |= {'negative'}
            if step.kinds != expected_kinds:
                raise ValueError(f'Task step {step.key!r} kinds do not agree with its annotation policy')
            for values, name in (
                (step.labels, 'labels'),
                (step.parent_steps, 'parent steps'),
                (step.depends_on, 'dependencies'),
            ):
                if any(not isinstance(value, str) or not value for value in values):
                    raise ValueError(f'Task step {step.key!r} has invalid {name}')
            unknown = (step.parent_steps | step.depends_on) - known
            if unknown:
                raise ValueError(f'Task step {step.key!r} references unknown steps: {sorted(unknown)}')
        self._validate_acyclic()

    def step(self, key: str) -> StepDefinition:
        """Return a declared step, raising ``ValueError`` when the key is unknown."""
        for step in self.steps:
            if step.key == key:
                return step
        raise ValueError(f'Unknown task step: {key!r}')

    def dependent_steps(self, key: str) -> frozenset[str]:
        """Return every directly or transitively dependent step."""
        self.step(key)
        dependents: set[str] = set()
        pending = [key]
        while pending:
            dependency = pending.pop()
            for step in self.steps:
                if dependency in self.dependencies(step.key) and step.key not in dependents:
                    dependents.add(step.key)
                    pending.append(step.key)
        return frozenset(dependents)

    def dependencies(self, key: str) -> frozenset[str]:
        step = self.step(key)
        return step.parent_steps | step.depends_on

    def input_steps(self, key: str) -> frozenset[str]:
        inputs: set[str] = set()
        pending = [key]
        while pending:
            current = pending.pop()
            if current not in inputs:
                inputs.add(current)
                pending.extend(self.dependencies(current))
        return frozenset(inputs)

    def _validate_acyclic(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(key: str) -> None:
            if key in visiting:
                raise ValueError('Task step dependencies must not contain a cycle')
            if key in visited:
                return
            visiting.add(key)
            for related in self.dependencies(key):
                visit(related)
            visiting.remove(key)
            visited.add(key)

        for step in self.steps:
            visit(step.key)
