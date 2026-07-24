from pathlib import Path

from xxtrain.task import TaskType

from .core import Context, ConversionConfig, ConversionReport
from .discovery import DirectorySource, validate_classification_source
from .recipes import DatasetRecipe, _read_labels
from .sinks import print_conversion_report


def convert_dataset(
    recipe: DatasetRecipe,
    root_path: str | Path,
    *,
    split: int = 10,
    reserve_no_label: bool = False,
) -> ConversionReport:
    root = Path(root_path)
    if recipe.labels is None:
        labels = _read_labels(
            root,
            strict=recipe.task_type is TaskType.CLASSIFY,
        )
        if recipe.task_type is TaskType.CLASSIFY:
            labels = validate_classification_source(root, labels, split)
    else:
        labels = recipe.labels
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
    source = DirectorySource()
    outputs = recipe.pipeline.run(source.read(context), context)
    for output in outputs:
        recipe.sink.write(output, context)
    recipe.sink.finalize(context)
    print_conversion_report(context)
    return context.report
