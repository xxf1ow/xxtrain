import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

_SRC_PATH = str(Path(__file__).resolve().parents[2] / 'src')
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

import annconverter  # noqa: E402
import annparser  # noqa: E402
import annprocessor  # noqa: E402


@contextmanager
def unavailable_symlinks():
    with patch.object(
        annprocessor.os,
        'symlink',
        side_effect=OSError('symlink privilege unavailable'),
    ):
        yield

convert_dataset = annconverter.process
parse_labelimg = annparser.parse_det_anns_from_labelimg
parse_labelme = annparser.parse_seg_anns_from_labelme
Annotation = annparser.Annotation
ShapeType = annparser.ShapeType
TaskType = annparser.TaskType
TaskProcessor = annparser.TaskProcessor
map_parent_child = annparser.map_parent_child_annotations
ImageSizeParser = annprocessor.ImageSizeParser
Pipeline = annprocessor.Pipeline
TaskPayload = annprocessor.TaskPayload
