from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from xxtrain.pipeline import DatasetRecipe


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingScenario:
    dataset: DatasetRecipe
    model_version: str = 'v8'
    model_scale: str = 'n'
    split: int = 10
    reserve_no_label: bool = False
    train_args: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_scale not in {'n', 's', 'm', 'l', 'x'}:
            raise ValueError(f'Unsupported model scale: {self.model_scale}')


def _resolve_paths(value: object, root: Path) -> object:
    if isinstance(value, Path):
        return value if value.is_absolute() else root / value
    if isinstance(value, Mapping):
        return {key: _resolve_paths(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(item, root) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_paths(item, root) for item in value)
    return value


def load_scenario(path: str | Path) -> TrainingScenario:
    scenario_path = Path(path).resolve()
    if not scenario_path.is_file():
        raise FileNotFoundError(f'Scenario file does not exist: {scenario_path}')
    module_name = f'_xxtrain_scenario_{sha256(str(scenario_path).encode()).hexdigest()}'
    spec = spec_from_file_location(module_name, scenario_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load Scenario module: {scenario_path}')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'SCENARIO'):
        raise ValueError(f'Scenario module must export SCENARIO: {scenario_path}')
    scenario = module.SCENARIO
    if not isinstance(scenario, TrainingScenario):
        raise TypeError(f'SCENARIO must be a TrainingScenario: {scenario_path}')
    return replace(scenario, train_args=_resolve_paths(scenario.train_args, scenario_path.parent))
