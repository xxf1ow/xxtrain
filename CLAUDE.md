# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Repository workflow constraints

- Superpowers-generated artifacts, including specs and implementation plans, are local working files. Never stage or commit them to Git.
- Do not create or use Git worktrees for this repository. Perform all work in the current checkout.
- Source migration is behavior-preserving work. Do not mix algorithm changes, stricter validation, or new conversion features into a migration commit unless the change is explicitly approved.

## What this is

xxtrain is a training harness around [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). It converts LabelImg/LabelMe annotations into YOLO datasets, generates model configuration, downloads pretrained weights, trains, validates, and exports ONNX models.

All Python source lives under `src/`. `src/train.py` is still the executable training entry point. Dataset conversion is implemented by the installable-package-shaped source tree under `src/xxtrain/`; do not recreate or import the removed flat modules `annparser.py`, `annprocessor.py`, or `annconverter.py`.

Do not confuse the repository's `src/` directory with a dataset's `<root_path>/src/` input directory. They are unrelated despite the shared name.

## Current source boundaries

### `xxtrain.task`

`xxtrain.task` defines only the five basic `TaskType` values. Task-name parsing, custom recipe names, label catalogs, and other stable conversion parameters belong to the pipeline recipe layer, not the common task header.

### `xxtrain.data`

The data package owns:

- immutable annotation values and geometry;
- ordered label catalogs;
- one-way LabelImg and LabelMe readers;
- pure YOLO line encoders;
- dataset split and artifact helpers.

Annotation identity uses UUIDs and grouping is independent from labels. Concrete shapes expose immutable tuple geometry. Readers preserve input order and numeric label names remain strings.

Coordinate rules:

- Annotation coordinates are `float`, including `Bbox` fields, point tuples, translation offsets, geometry results, and YOLO normalization inputs.
- LabelImg and LabelMe readers convert coordinates to `float`; do not truncate or round them in the data/model layer.
- `ImageInfo.width` and `ImageInfo.height` are positive `int` values because they describe raster dimensions, not annotation coordinates.
- Quantization is allowed only at an explicit raster boundary such as OpenCV array slicing. Keep the original float geometry available for matching and encoding.
- `CropMatches` currently derives crop `ImageInfo` with `int(x2 - x1)` / `int(y2 - y1)`. This differs from the legacy float geometric extent for fractional parent boxes and is a known migration-parity issue; do not treat the cast as the intended coordinate contract.

### `xxtrain.pipeline`

Conversion uses a typed, streaming `Source -> Pipeline -> Sink` architecture:

- `DirectorySource` discovers samples in deterministic directory/image order.
- Immutable records (`Sample`, `ImageRef`, and stage-specific `*Input` / `*Output` values) carry data between processors.
- `ItemProcessor` maps one input to at most one output; `ExpandProcessor` maps one input to multiple ordered outputs.
- `Pipeline` validates neighboring processor types and streams values without a shared payload dictionary.
- Sinks are the only layer that writes dataset images, labels, lists, and YAML. Crop processors defer image materialization by storing a crop box in `ImageRef`.
- `convert_dataset()` selects one of the supported recipes, executes the stream, finalizes the sink, and returns a `ConversionReport`.

The stable `xxtrain.pipeline` public API is deliberately limited to:

`Context`, `ConversionConfig`, `ConversionReport`, `ExpandProcessor`, `ImageRef`, `ItemProcessor`, `Pipeline`, `Sample`, and `convert_dataset`.

Stage-specific records, discovery classes, concrete processors, recipes, and sinks are internal implementation details and should be imported from their defining modules only when implementing or testing those internals.

## Commands

Production workflows run through `src/train.py`.

```bash
# Full pipeline: convert -> generate model.yaml -> download weights -> train -> export ONNX
python src/train.py --task_type point-detect --root_path data/point

# Standard task; labels come from <root_path>/src/labels.txt
python src/train.py --task_type detect --root_path data/<dataset> --model_version v8 --model_scale n

# Export an existing checkpoint
python src/train.py --mode export --root_path data/point --task_type point-classify --weights runs/classify/train9/weights/best.pt

# Validate a checkpoint; task is inferred from the weights
python src/train.py --mode val --weights runs/classify/train16/weights/best.pt --directory data/light/light-classify

# Canonical test command
python -m unittest discover -s test -t . -p 'test_*.py' -v

# Static verification
ruff check src test
python -m compileall -q src test
git diff --check

# Formatting (configuration is in pyproject.toml)
ruff format src test
```

Test fixtures live in `test/fixtures/` and human-reviewed semantic snapshots live in `test/expected/conversions/`. Tests import the new package directly; there is no `test/support/current_api.py` compatibility adapter.

## Task names

Supported standard recipes:

- `detect`
- `segment`
- `pose`
- `classify`

Supported custom recipes:

- `point-detect`
- `point-classify`
- `point-segment`
- `knob-detect`
- `knob-segment`
- `light1-detect`
- `light2-detect`

`scale-pose` is intentionally unsupported in the new pipeline. Unknown names fail before filesystem access with `ValueError("Unsupported task type: <name>")`.

The final suffix still selects the Ultralytics model family in `src/train.py`: `classify`, `detect`, `obb`, `pose`, or `segment`. OBB has no conversion recipe yet.

## Dataset convention

Input layout:

```text
root_path/
└── src/
    ├── <group>/
    │   ├── imgs/
    │   ├── anns/       # LabelImg XML
    │   └── anns_seg/   # LabelMe JSON
    └── labels.txt      # standard recipes only
```

Non-classification outputs are written under `<root_path>/<task_name>/` as images plus YOLO `.txt` labels, `train.txt`, `val.txt`, and `dataset.yaml`. Whole-image outputs prefer symlinks and fall back to `shutil.copy2`. Crop outputs are materialized by the sink. Classification outputs are written under `train/<class>/` and `val/<class>/`.

`--split N` sends every Nth source image to validation; `N <= 0` includes every image in both splits. `--reserve_no_label` keeps zero-annotation images in split lists. `src/train.py` still skips conversion when its expected output already exists; source-change detection and forced rebuilding remain future work.

## Migration status

- Behavior baseline: complete.
- Data layer migration: complete.
- Typed pipeline and conversion-entry cutover: implemented; the fractional crop-size parity issue above remains to be resolved before declaring the pipeline migration gate closed.
- Training workflow/package entry migration: not started.
- Packaging and installable CLI: not started.
