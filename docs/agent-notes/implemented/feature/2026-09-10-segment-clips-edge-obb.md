# Agent Note: Clip edge-crossing shapes for segment export

Status: implemented

## Problem

A rotated rectangle can extend beyond the image while its visible portion remains a valid training target. YOLO OBB requires all four corners to be inside the image, while a segment polygon can represent the visible clipped shape.

## Decision

The segment preparation processor clips `Bbox`, `Circle`, and `Polygon` shapes to the image rectangle before YOLO segment encoding. The clipping uses polygon-boundary intersection, preserves the annotation identity and metadata, and removes only shapes with no positive visible area. Images are retained. Two-point polylines remain unchanged for their existing downstream handling.

The OBB recipe continues to validate and encode the original four-point polygon without clipping. This keeps OBB semantics separate from segment semantics.

## Alternatives considered

**Allow out-of-range segment coordinates:** Rejected because the YOLO segment format requires normalized coordinates in the image range and downstream loaders reject such coordinates.

**Clamp each vertex independently:** Rejected because independent clamping can change polygon topology and does not compute the visible intersection.

**Clip and refit an OBB:** Rejected for segment export because refitting can change the original rotation angle; the segment polygon preserves the visible geometry without that approximation.

## Consequences

Segment labels for edge-crossing rotated rectangles are valid normalized polygons and no longer fail solely because source vertices are outside the image. The exported mask describes only the visible intersection; it cannot encode or recover geometry hidden outside the image. Shapes completely outside the image produce no segment annotation.
