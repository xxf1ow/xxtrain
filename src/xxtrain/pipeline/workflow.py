from pathlib import Path

from xxtrain.task import TaskType

from .core import Context, ConversionConfig, ConversionReport
from .discovery import DirectorySource, validate_classification_source
from .recipes import DatasetRecipe, _read_labels
from .sinks import print_conversion_report


def _required_output_labels(context: Context) -> tuple[str, ...]:
    names = (
        context.config.labels.names[:1] if context.config.task_type is TaskType.POSE else context.config.labels.names
    )
    return tuple(dict.fromkeys(name for name in names if name))


def _validate_output_labels(recipe: DatasetRecipe, context: Context) -> None:
    if not getattr(recipe.sink, 'tracks_output_labels', False):
        return
    missing = tuple(
        name for name in _required_output_labels(context) if context.report.output_label_counts.get(name, 0) == 0
    )
    context.report.set_missing_output_labels(missing)
    if missing:
        print_conversion_report(context)
        raise ValueError(f'Converted dataset has no output samples for target labels: {", ".join(missing)}')


def convert_dataset(
    recipe: DatasetRecipe, root_path: str | Path, *, split: int = 10, reserve_no_label: bool = False
) -> ConversionReport:
    root = Path(root_path)
    source_catalog = recipe.source.catalog_for(recipe.task_type)
    if recipe.labels is not None and source_catalog is not None and recipe.labels != source_catalog:
        raise ValueError('Recipe labels do not match source catalog')
    if recipe.labels is not None:
        labels = recipe.labels
    elif source_catalog is not None:
        labels = source_catalog
    else:
        labels = _read_labels(root, strict=recipe.task_type is TaskType.CLASSIFY)
    if recipe.labels is None and recipe.task_type is TaskType.CLASSIFY and isinstance(recipe.source, DirectorySource):
        labels = validate_classification_source(root, labels, split)
    context = Context(
        config=ConversionConfig(
            task_name=recipe.name,
            task_type=recipe.task_type,
            root_path=root,
            split=split,
            labels=labels,
            reserve_no_label=reserve_no_label,
        ),
        report=ConversionReport(),
    )
    outputs = recipe.pipeline.run(recipe.source.read(context), context)
    for output in outputs:
        recipe.sink.write(output, context)
    _validate_output_labels(recipe, context)
    recipe.sink.finalize(context)
    print_conversion_report(context)
    return context.report
