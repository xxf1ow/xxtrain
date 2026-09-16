from dataclasses import dataclass


@dataclass(frozen=True)
class StepDefinition:
    """Allowed records, parent steps, and dependency edges for one task step."""

    key: str
    kinds: frozenset[str]
    labels: frozenset[str]
    parent_steps: frozenset[str]
    depends_on: frozenset[str]


@dataclass(frozen=True)
class TaskDefinition:
    """Immutable validation and downstream-dependency rules for a business task."""

    steps: tuple[StepDefinition, ...]

    def __post_init__(self) -> None:
        keys = tuple(step.key for step in self.steps)
        if not keys or any(not isinstance(key, str) or not key for key in keys):
            raise ValueError('Task steps require non-empty string keys')
        if len(set(keys)) != len(keys):
            raise ValueError('Task step keys must be unique')
        known = frozenset(keys)
        for step in self.steps:
            if not step.kinds or any(not isinstance(value, str) or not value for value in step.kinds):
                raise ValueError(f'Task step {step.key!r} requires annotation kinds')
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
        self._validate_acyclic('depends_on')

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
                if dependency in step.depends_on and step.key not in dependents:
                    dependents.add(step.key)
                    pending.append(step.key)
        return frozenset(dependents)

    def _validate_acyclic(self, attribute: str) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(key: str) -> None:
            if key in visiting:
                raise ValueError(f'Task step {attribute} must not contain a cycle')
            if key in visited:
                return
            visiting.add(key)
            for related in getattr(self.step(key), attribute):
                visit(related)
            visiting.remove(key)
            visited.add(key)

        for step in self.steps:
            visit(step.key)
