# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Repository workflow constraints

- Superpowers-generated artifacts, including specs and implementation plans, are local working files. Never stage or commit them to Git.
- Do not create or use Git worktrees for this repository. Perform all work in the current checkout.
- Source migration is behavior-preserving work. Do not mix algorithm changes, stricter validation, or new conversion features into a migration commit unless the change is explicitly approved.

## What this is

xxtrain is a training harness around [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). It converts LabelImg/LabelMe annotations into YOLO datasets, generates model configuration, downloads pretrained weights, trains, inspects prediction results, and exports ONNX models.

All Python source lives under `src/xxtrain/` and is installed as the `xxtrain` package. `xxtrain.cli` is the only command entry and dispatches to the package's conversion and training APIs; do not recreate the removed checkout scripts `src/train.py`, `src/export.py`, or `src/review.py`, and do not recreate or import the removed flat modules `annparser.py`, `annprocessor.py`, or `annconverter.py`.

Do not confuse the repository's `src/` directory with a Scenario's `<scenario_dir>/src/` input directory. They are unrelated despite the shared name.

## Current source boundaries

### `xxtrain.task`

`xxtrain.task` defines only the five basic `TaskType` values. Task-name parsing, custom recipe names, label catalogs, and other stable conversion parameters belong to the pipeline recipe layer, not the common task header.

### `xxtrain.data`

The data package owns:

- immutable annotation values and geometry;
- ordered label catalogs;
- LabelImg, LabelMe, and COCO readers and writers;
- YOLO line encode/decode primitives;
- format- and task-specific validation at I/O boundaries;
- dataset split and artifact helpers.

`Annotation` remains a single annotation value and does not encode file-format distinctions. Annotation identity uses UUIDs and grouping is independent from labels. Concrete shapes expose immutable tuple geometry. Readers preserve input order and numeric label names remain strings.

Coordinate rules:

- Annotation coordinates are `float`, including `Bbox` fields, point tuples, translation offsets, geometry results, and YOLO normalization inputs.
- LabelImg and LabelMe readers convert coordinates to `float`; do not truncate or round them in the data/model layer.
- `ImageInfo.width` and `ImageInfo.height` use a single `float` memory representation. Integer raster dimensions are accepted at construction and normalized to `float`; a virtual crop-local coordinate extent retains its fractional value.
- Do not truncate or round coordinates or coordinate extents in sources, processors, matching, or encoding. Quantization is allowed only at an explicit raster boundary such as OpenCV array slicing.

### `xxtrain.pipeline`

Conversion uses a typed, streaming `Source -> Pipeline -> Sink` architecture:

- `DirectorySource` discovers samples in deterministic directory/image order.
- Immutable records (`Sample`, `ImageRef`, and stage-specific `*Input` / `*Output` values) carry data between processors.
- `ItemProcessor` maps one input to at most one output; `ExpandProcessor` maps one input to multiple ordered outputs.
- `Pipeline` validates neighboring processor types and streams values without a shared payload dictionary.
- Sinks are the only layer that writes dataset images, labels, lists, and YAML. Crop processors defer image materialization by storing a crop box in `ImageRef`.
- `convert_dataset(recipe, root_path, *, split=10, reserve_no_label=False)` executes the supplied `DatasetRecipe`, finalizes its sink, and returns a `ConversionReport`.

The stable `xxtrain.pipeline` public API is deliberately limited to:

`Context`, `ConversionConfig`, `ConversionReport`, `DatasetRecipe`, `ExpandProcessor`, `ImageRef`, `ItemProcessor`, `Pipeline`, `Sample`, `convert_dataset`, and `standard_recipe`.

`standard_recipe()` owns the detect, segment, pose, OBB, and classify pipelines. Special point, knob, and light recipes are composed in their corresponding Scenario files under `data/`, not in a central task-name registry. Stage-specific records and concrete sources, processors, sinks, and helpers are internal implementation details; the tracked preset Scenarios may import them from their defining modules, but they are not part of the stable 11-symbol API or a general third-party plugin API.

### `xxtrain.training`

A Python Scenario file is the composition root for one dataset and training workflow. It exports `SCENARIO: TrainingScenario`, which contains:

- a `DatasetRecipe` (`labels + Pipeline + Sink`);
- model version and scale;
- split settings with `reserve_no_label=False` by default;
- overrides passed to `YOLO.train()`.

Relative `Path` values inside Scenario `train_args` resolve against the Scenario file's directory. Ordinary strings are unchanged. The stable `xxtrain.training` public API is:

`TrainingScenario`, `load_scenario`, `train`, `export`, and `review`.

Tracked classification Scenarios demonstrate three training policies under `data/standard-classify/`: `standard_classify.py` keeps the standard arguments, `direction_sensitive_classify.py` disables horizontal/vertical flips, rotation, and automatic augmentation, and `tuned_classify.py` shows a larger experimentally selected override set. The special `data/point/point_classify.py` Scenario declares the same four direction-sensitive constraints while retaining its crop-based Dataset Recipe. Non-standard arguments belong in each Scenario and must not be added to task-wide classification defaults.

Classification dataset conversion materializes every whole image or deferred crop as a centered `224×224` Letterbox image using OpenCV linear interpolation and padding value 114. The task-wide classification defaults use `imgsz=224` and `scale=0.0`, so Ultralytics does not randomly crop the already-square generated image. Scenario overrides remain allowed and are responsible for staying aligned with deployment.

Model-template handling, pretrained-weight preparation, classification mismatch reporting, and other orchestration details remain internal.

Pretrained model weights are shared through `platformdirs.user_cache_path('xxtrain') / 'weights'`. They are not read from or written to the checkout, installed package, or Scenario directory.

## Commands

Install the project in editable mode before running the installed commands or tests:

```powershell
python -m pip install -e ".[dev]"

# Full pipeline: convert -> generate model.yaml -> download weights -> train -> export ONNX
xxtrain train data/standard-detect/standard_detect.py

# Export an existing checkpoint
xxtrain export data/standard-detect/standard_detect.py --weights runs/detect/train/weights/best.pt

# Inspect prediction results for an existing checkpoint
xxtrain review data/standard-detect/standard_detect.py --weights runs/detect/train/weights/best.pt --directory path/to/images

# Canonical test command
python -m unittest discover -s test -t . -p 'test_*.py' -v

# Static verification
ruff check src test
ruff format --check src test
ruff check --no-respect-gitignore data
ruff format --check --no-respect-gitignore data
python -m compileall -q src test data
git diff --check

# Formatting (configuration is in pyproject.toml)
ruff format src test
```

`xxtrain review` performs prediction result inspection; it does not call `model.val()`. Detect, segment, pose, and OBB checkpoints save prediction visualizations. Classification checkpoints infer the expected class from each image's parent directory, collect mismatches, and write a report.

Test fixtures live in `test/fixtures/` and human-reviewed semantic snapshots live in `test/expected/conversions/`. Tests import the new package directly; there is no `test/support/current_api.py` compatibility adapter.

## Recipe ownership

`standard_recipe(TaskType)` supports five standard recipes:

- `detect`
- `segment`
- `pose`
- `obb`
- `classify`

Seven special recipes are owned by tracked Scenario files:

- `point-detect`
- `point-classify`
- `point-segment`
- `knob-detect`
- `knob-segment`
- `light1-detect`
- `light2-detect`

There is no task-name registry, legacy `Recipe`, or `build_recipe()`. `DatasetRecipe.task_type` directly selects the Ultralytics model family. `scale-pose` remains unsupported.

## Scenario and dataset convention

The Scenario file's parent directory is the dataset root:

```text
<scenario_dir>/
├── <scenario>.py
├── src/
│   ├── <group>/
│   │   ├── imgs/
│   │   ├── anns/       # LabelImg XML
│   │   └── anns_seg/   # LabelMe JSON
│   └── labels.txt      # standard recipes only
├── <recipe.name>/      # generated dataset and model YAML
└── weights/            # exported ONNX and classification references
```

Standard recipes read labels at conversion time from `<scenario_dir>/src/labels.txt`; special recipes carry a fixed `LabelCatalog` in their Scenario. Non-classification outputs are written under `<scenario_dir>/<recipe.name>/` as images plus YOLO `.txt` labels, `train.txt`, `val.txt`, and `dataset.yaml`. Whole-image outputs prefer symlinks and fall back to `shutil.copy2`. Crop outputs are materialized by the sink. Classification outputs are materialized under `train/<class>/` and `val/<class>/` as centered `224×224` Letterbox images; their finalization also writes `train.txt`, `val.txt`, and `dataset.yaml`. Existing generated classification directories from before this contract must be deleted and rebuilt from the sibling `src/` input directory. Generated outputs are disposable, but a Scenario's `src/` directory is immutable source data and must never be deleted or modified during rebuilding.

`TrainingScenario.split=N` sends every Nth source image to validation; `N <= 0` includes every image in both splits. `TrainingScenario.reserve_no_label` defaults to `False`; set it to `True` in the Scenario only when zero-annotation images must remain in split lists. For every task type, training treats `<scenario_dir>/<recipe.name>/dataset.yaml` as the conversion-completion signal and skips conversion only when that file exists. A classification output directory without `dataset.yaml` is incomplete and must be converted again. Source-change detection and forced rebuilding remain future work.

The repository ignores `data/` by default. New Scenario files therefore require forced staging:

```powershell
git add -f data/<dataset>/<scenario>.py
```

Do not force-add raw datasets or generated outputs.

## Migration status

- Behavior baseline: complete.
- Data layer migration: complete.
- Typed pipeline and conversion-entry cutover: complete.
- Scenario-driven training, export, and prediction-review migration: complete.
- Packaging and installed CLI: complete.
