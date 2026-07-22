# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository workflow constraints

- Superpowers-generated artifacts, including specs and implementation plans, are local working files. Never stage or commit them to Git.
- Do not create or use Git worktrees for this repository. Perform all work in the current checkout.

## What this is

A training harness around [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). It takes raw annotation files (labelimg XML for detection boxes, labelme JSON for segment/pose/obb), converts them into YOLO-format datasets, generates a model `.yaml`, downloads pretrained weights, trains, validates, and exports to ONNX. The novel/non-obvious part is the **annotation conversion pipeline** (`src/annconverter.py` + `src/annprocessor.py` + `src/annparser.py`); `src/train.py` is a relatively thin Ultralytics wrapper on top.

All Python source lives under `src/`. The four modules import each other as flat siblings (`import annconverter`, `from annprocessor import ...`), which works because running `python src/train.py` puts `src/` on `sys.path`. **Do not confuse the project's `src/` (code) with a dataset's `root_path/src/` (input images/annotations) — they are unrelated despite the shared name.**

## Commands

There is no test suite, requirements file, or build step. Everything runs through `src/train.py`.

```bash
# Full pipeline: convert annotations -> generate model.yaml -> download weights -> train -> export ONNX
python src/train.py --task_type point-detect --root_path data/point

# Standard tasks (no hyphen) read labels from <root_path>/src/labels.txt; custom tasks hardcode their label list.
python src/train.py --task_type detect --root_path data/<dataset> --model_version v8 --model_scale n

# Export an existing checkpoint to ONNX
python src/train.py --mode export --root_path data/point --task_type point-classify --weights runs/classify/train9/weights/best.pt

# Validate a checkpoint against a directory of images (classify mode reports mismatched samples;
# other tasks run predict+save). Task is inferred from the loaded weights, NOT --task_type.
python src/train.py --mode val --weights runs/classify/train16/weights/best.pt --directory data/light/light-classify

# Lint / format (config in pyproject.toml: line-length 120, single quotes, lf, skip-magic-trailing-comma)
ruff check .
ruff format .
```

Key CLI args: `--split N` (every Nth image goes to validation; `<=0` puts each image in both sets), `--reserve_no_label` (keep images with zero annotations). Training hyperparameters (epochs, batch, imgsz, augmentation) are **hardcoded** in `train_model()` in `src/train.py`, branched by task family.

## Task-type naming convention

`task_type` is the central dispatch key throughout the codebase. Two forms:

- **Standard** (no hyphen): `detect`, `segment`, `pose` — uses `<root_path>/src/labels.txt` for the class list and the `standard_*_pipe` pipelines.
- **Custom** `<family>-<yolotask>`: e.g. `point-detect`, `point-classify`, `point-segment`, `knob-detect`, `knob-segment`, `scale-pose`, `light1-detect`, `light2-detect`. Each is routed in `annconverter.process()` by prefix (`point`/`knob`/`scale`/`light`) to a `task_<family>_process()` builder that returns a custom `(pipeline, label_list)`.

The suffix after the last hyphen must end in one of `classify/detect/obb/pose/segment` — `train.py`'s `suffix_switcher` maps that to the Ultralytics model suffix (`-cls`, ``, `-obb`, `-pose`, `-seg`) and selects training imgsz/epochs/augmentation. Several families (point, knob, scale, light) are *composite* real-world tasks (e.g. gauge reading = detect dial + segment/pose the needle + classify position) split across multiple `task_type` invocations that share one `data/<family>/` root.

## Dataset directory convention

Input layout the converter expects (defined by `GlobalContext` + `DirectoryIterator` in `src/annprocessor.py`):

```
root_path/
└── src/
    ├── <subdir>/
    │   ├── imgs/       # images
    │   ├── anns/       # labelimg XML  (detection boxes)        -> in_det_path
    │   └── anns_seg/   # labelme JSON  (segment / instance)     -> in_seg_path
    └── labels.txt      # class list, one per line (standard tasks only)
```

Output is written to `root_path/<task_type>/` as YOLO `.txt` labels next to **symlinks** of the source images (or cropped JPEGs for crop-based tasks), plus `train.txt`, `val.txt`, and `dataset.yaml`. Classify tasks instead write `train/<NN-label>/` and `val/<NN-label>/` directories of cropped square images. Trained runs land in `runs/<yolotask>/`; exported ONNX in `root_path/weights/`. Pretrained `.pt` weights are cached in `.weights/`.

Note: the `data/example/` dataset uses an older `images/` layout instead of `src/`; trust the code (`get_images_path()` returns `src/`), not that example.

## Pipeline architecture (the core abstraction)

Conversion is a composable pipeline of small **processors**. Understand these three building blocks before editing `src/annprocessor.py`:

- **`TaskPayload`** — a per-image data bag (`set/get/has`) that also records which processor produced each key (`_trace`), and warns on overwrite. Processors communicate *only* through payload keys (`img_size`, `det_anns`, `seg_anns`, `matched_map`, `ann_count`, `out_img_path`, `output_path`, `in_*_path`, etc.).
- **`BaseProcessor`** — every processor declares `required_inputs()` (validated before `process()` runs) and writes outputs via `self.set(payload, ...)`. `Pipeline` is itself a `BaseProcessor` (composite pattern), so pipelines nest.
- **Iterators create fresh sub-payloads.** `DirectoryIterator` walks `src/*/imgs/` and runs the per-image sub-pipeline with a brand-new `TaskPayload` per image (no cross-image state leaks). `DetectBboxCropIterator` crops each matched detection box, translates child annotations into crop-local coordinates, and runs a *sub-pipeline* per crop — this is how detect→classify/segment/pose "drill-down" tasks are built.

A pipeline is assembled in `src/annconverter.py` (e.g. `standard_detect_pipe`, or the `pipe`/`subpipe` lists inside each `task_*_process`), wrapped in `DirectoryIterator`, then driven once by `process()` which builds a `GlobalContext`, runs the pipeline, and calls `ctx.dataset_finalize()` + `ctx.print_summary()`.

Typical processor chain: `ImageSizeParser` → `*AnnsParser` (parse + drop labels not in the label set) → optionally `DetectAndSegAnnsMatcher` (assigns child shapes to parent boxes via `map_parent_child_annotations` / geometric containment) → `*AnnsGenerator` (emit YOLO `.txt`) → `DatasetSplitter` (train/val split + stat counting). Generators set `ann_count`/`out_img_path`, which `DatasetSplitter` consumes — so a generator must run before the splitter.

## Annotation layer (`src/annparser.py`)

- **`Annotation`** dataclass is the universal shape: `label`, `type` (`ShapeType`), `parts` (list of `np.ndarray` point arrays — `parts[0]` via `.points`), `instance` (UUID, or `(label, group_id)` for grouped labelme shapes), with computed `.bbox` and `.translate()`.
- `parse_det_anns_from_labelimg` (XML) and `parse_seg_anns_from_labelme` (JSON) are the two parsers; both assert that the annotation's recorded image size matches the actual image.
- `TaskProcessor.transform` enforces which `ShapeType`s a `TaskType` accepts and converts shapes (e.g. circle→polygon, rectangle 2-pt→4-pt).
- Geometry helpers (`calculate_iou`, `rectangle_include_shape` with tolerance via `calculate_wide`, `map_parent_child_annotations`) implement the parent-box/child-shape matching, with `strict`/`wide` knobs to tolerate sloppy annotations.

Several functions and code paths are explicitly marked `# todo: fix it` (e.g. `calculate_nms`, `create_labelimg`, `shape_to_mask`, mask generation in the labelme parser, OBB conversion in `TaskProcessor.transform`) — treat these as known-incomplete, not as load-bearing.
