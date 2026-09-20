from xxtrain.data import Annotation, Bbox, ImageInfo
from xxtrain.pipeline import Context, ImageRef, Sample
from xxtrain.pipeline.core import ClassifyOutput, EncodeOutput
from xxtrain.pipeline.processors import EncodeDetection
from xxtrain.platform.contracts import EditFrame


def frame_sample(frame: EditFrame, index: int, annotations: tuple[Annotation, ...] = ()) -> Sample:
    """Build one pipeline sample from a materialized task edit frame."""
    return Sample(
        id=f'workspace/{frame.mapping.frame_id}',
        source_group='workspace',
        source_index=index,
        image=ImageRef(
            path=frame.image_path.absolute(), info=ImageInfo(width=frame.width, height=frame.height), crop_box=None
        ),
        annotations=annotations,
    )


def encode_rectangles(
    frame: EditFrame, index: int, context: Context, *, output_label: str | None = None
) -> EncodeOutput:
    """Encode rectangle annotations, preserving legal negative samples as empty YOLO rows."""
    boxes = []
    for annotation in frame.annotations:
        if annotation.kind == 'negative':
            continue
        if annotation.kind != 'rectangle' or not isinstance(annotation.geometry, list) or len(annotation.geometry) != 2:
            raise ValueError(f'Rectangle frame {frame.mapping.frame_id!r} contains an incompatible annotation')
        first, second = annotation.geometry
        boxes.append(
            Bbox(
                id=annotation.id,
                label=output_label or annotation.label,
                x1=first[0],
                y1=first[1],
                x2=second[0],
                y2=second[1],
            )
        )
    return EncodeDetection().transform(frame_sample(frame, index, tuple(boxes)), context)


def encode_classification(frame: EditFrame, index: int, context: Context) -> ClassifyOutput:
    """Encode one category annotation into the classification directory layout."""
    if len(frame.annotations) != 1 or frame.annotations[0].kind != 'classification':
        raise ValueError(f'Classification frame {frame.mapping.frame_id!r} requires exactly one category')
    label = frame.annotations[0].label
    if not isinstance(label, str):
        raise ValueError(f'Classification frame {frame.mapping.frame_id!r} requires a category label')
    return ClassifyOutput(
        sample=frame_sample(frame, index), class_name=label, output_name=f'{frame.mapping.frame_id}.png'
    )
