from xxtrain.data import Bbox, Polyline
from xxtrain.data.formats import decode_segment
from xxtrain.pipeline import Context
from xxtrain.pipeline.core import CropOutput, EncodeOutput
from xxtrain.pipeline.processors import EncodePointSegment
from xxtrain.platform.cache_builders import encode_rectangles, frame_sample
from xxtrain.platform.contracts import EditFrame


def encode_point_detection(frame: EditFrame, index: int, context: Context) -> EncodeOutput:
    """Map all Point detector rectangle variants to its single training class."""
    return encode_rectangles(frame, index, context, output_label='Point')


def encode_point_segment(frame: EditFrame, index: int, context: Context) -> EncodeOutput:
    """Convert Point pointer lines into the legacy triangular mask encoding."""
    if not frame.annotations:
        raise ValueError(f'Segment frame {frame.mapping.frame_id!r} requires at least one line')
    annotations = tuple(
        Polyline(id=annotation.id, label=annotation.label, points=annotation.geometry)
        for annotation in frame.annotations
    )
    sample = frame_sample(frame, index, annotations)
    parent_id = frame.mapping.parent_id
    if parent_id is None:
        raise ValueError(f'Segment frame {frame.mapping.frame_id!r} has no source box identity')
    crop = CropOutput(
        sample=sample, parent=Bbox(id=parent_id, label='Point', x1=0, y1=0, x2=frame.width, y2=frame.height)
    )
    try:
        output = EncodePointSegment().transform(crop, context)
        for line in output.lines:
            decode_segment(line, sample.image.require_info(), context.config.labels)
    except (AssertionError, ValueError) as error:
        raise ValueError(
            f'Segment frame {frame.mapping.frame_id!r} generates a triangle outside crop bounds'
        ) from error
    return output
