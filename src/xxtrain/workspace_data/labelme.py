from xxtrain.platform.contracts import AnnotationRecord, JsonObject


def detection_document(
    *, image_path: str, width: int, height: int, annotations: tuple[AnnotationRecord, ...]
) -> JsonObject:
    """Build one disposable LabelMe document from authoritative detection records."""
    shapes = [
        {'label': record.label, 'points': record.geometry, 'shape_type': 'rectangle'}
        for record in annotations
        if record.kind == 'rectangle'
    ]
    return {
        'version': '5.0.0',
        'flags': {},
        'shapes': shapes,
        'imagePath': image_path,
        'imageData': None,
        'imageHeight': height,
        'imageWidth': width,
    }
