from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from xxtrain.business_tasks.definition import StepDefinition
from xxtrain.platform.contracts import AnnotationRecord, EditFrame, FrameMapping, ImageInput, JsonValue

from .crops import _crop_bounds, materialize_crops, to_local, to_original


@dataclass(frozen=True, slots=True)
class OriginalImageInputs:
    """Map each registered image to one full-image edit frame."""

    def mappings(
        self, images: tuple[ImageInput, ...], records: tuple[AnnotationRecord, ...], step: StepDefinition
    ) -> tuple[FrameMapping, ...]:
        del records, step
        return tuple(
            FrameMapping(image.sample_id, image.sample_id, None, (0, 0, image.width, image.height)) for image in images
        )

    def materialize(
        self, images: tuple[ImageInput, ...], mappings: tuple[FrameMapping, ...], runtime_root: Path
    ) -> tuple[EditFrame, ...]:
        del runtime_root
        by_id = {image.sample_id: image for image in images}
        frames = []
        for mapping in mappings:
            image = by_id.get(mapping.image_id)
            if image is None or mapping != FrameMapping(
                image.sample_id, image.sample_id, None, (0, 0, image.width, image.height)
            ):
                raise ValueError('Full-image mapping does not match its registered image')
            frames.append(EditFrame(mapping, image.image_path, image.width, image.height, ()))
        return tuple(frames)

    def to_local(self, mapping: FrameMapping, geometry: JsonValue) -> JsonValue:
        del mapping
        return geometry

    def to_original(self, mapping: FrameMapping, geometry: JsonValue) -> JsonValue:
        del mapping
        return geometry


@dataclass(frozen=True, slots=True)
class AxisAlignedRectangleInputs:
    """Map source rectangles stored in original coordinates to clamped crop frames."""

    def mappings(
        self, images: tuple[ImageInput, ...], records: tuple[AnnotationRecord, ...], step: StepDefinition
    ) -> tuple[FrameMapping, ...]:
        by_image: dict[str, list[AnnotationRecord]] = {}
        for record in records:
            if record.step_key in step.parent_steps and record.kind == 'rectangle':
                by_image.setdefault(record.image_id, []).append(record)
        mappings = []
        frame_ids: set[str] = set()
        for image in images:
            for record in by_image.get(image.sample_id, ()):
                if not isinstance(record.id, UUID):
                    raise ValueError('Crop source annotations require UUID identities')
                geometry = record.geometry
                if not isinstance(geometry, list) or len(geometry) != 2:
                    raise ValueError(f'Crop source {record.id} requires rectangle geometry')
                first, second = geometry
                if not isinstance(first, list) or not isinstance(second, list) or len(first) != 2 or len(second) != 2:
                    raise ValueError(f'Crop source {record.id} requires rectangle geometry')
                frame_id = str(record.id)
                if frame_id in frame_ids:
                    raise ValueError(f'Duplicate crop frame ID {frame_id}')
                frame_ids.add(frame_id)
                bounds = _crop_bounds(
                    (float(first[0]), float(first[1]), float(second[0]), float(second[1])), image.width, image.height
                )
                if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                    raise ValueError(f'Crop frame {frame_id} has no pixels after clamping')
                mappings.append(FrameMapping(frame_id, image.sample_id, record.id, bounds))
        return tuple(mappings)

    def materialize(
        self, images: tuple[ImageInput, ...], mappings: tuple[FrameMapping, ...], runtime_root: Path
    ) -> tuple[EditFrame, ...]:
        return materialize_crops(images, mappings, runtime_root)

    def to_local(self, mapping: FrameMapping, geometry: JsonValue) -> JsonValue:
        return to_local(mapping, geometry)

    def to_original(self, mapping: FrameMapping, geometry: JsonValue) -> JsonValue:
        return to_original(mapping, geometry)
