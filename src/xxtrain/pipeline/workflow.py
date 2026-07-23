from pathlib import Path

from .core import ConversionReport
from .recipes import build_recipe
from .sinks import print_conversion_report


def convert_dataset(
    task_name: str,
    root_path: str | Path,
    *,
    split: int = 10,
    reserve_no_label: bool = True,
) -> ConversionReport:
    recipe, context = build_recipe(
        task_name=task_name,
        root_path=root_path,
        split=split,
        reserve_no_label=reserve_no_label,
    )
    outputs = recipe.pipeline.run(recipe.source.read(context), context)
    for output in outputs:
        recipe.sink.write(output, context)
    recipe.sink.finalize(context)
    print_conversion_report(context)
    return context.report
