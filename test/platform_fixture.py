from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
import uuid
from pathlib import Path

from PIL import Image, ImageDraw

from xxtrain.business_tasks.point import point_task_definition
from xxtrain.platform.contracts import AnnotationRecord
from xxtrain.workspace_data import WorkspaceData
from xxtrain.workspace_data.repository import AnnotationRepository

FIXTURE_MARKER = 'xxtrain-task-7-point-workflow-v1'


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_image(path: Path, index: int) -> None:
    rng = random.Random(index + 7301)
    image = Image.new('RGB', (320, 240))
    image.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(320 * 240)])
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 30, 280, 210), outline='#ffffff', width=4)
    draw.text((48, 38), f'Point acceptance {index:02d}', fill='#ffffff')
    image.save(path, quality=85)


def create_fixture(parent: Path, *, owner_user_id: int, cvat_internal_url: str) -> dict[str, object]:
    """Create an isolated synthetic workspace and return its non-secret acceptance receipt.

    The caller owns the generated temporary directory. The receipt records the database, sample and initial annotation
    identities plus immutable input hashes for the browser harness; it never contains a CVAT password or service token.
    """
    parent = parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='xxtrain-point-acceptance-', dir=parent))
    workspace_dir = root / 'workspace'
    workspace_dir.mkdir()
    images_dir = workspace_dir / 'images'
    staging_dir = root / 'staging'
    baseline_dir = root / 'baseline'
    runtime_dir = root / 'runtime'
    for directory in (images_dir, staging_dir, baseline_dir, runtime_dir):
        directory.mkdir()

    staged_images = []
    for index in range(50):
        image_path = staging_dir / f'point-{index:02d}.jpg'
        _write_image(image_path, index)
        staged_images.append(image_path)

    workspace = WorkspaceData(workspace_dir)
    admission = workspace.admit(tuple(staged_images))
    if admission.accepted_count != 50:
        raise RuntimeError('Synthetic fixture textures must admit as 50 distinct images')
    images = workspace.images()

    original_rectangle_points = [[40.0, 30.0], [280.0, 210.0]]
    primary_detections = tuple(
        AnnotationRecord(uuid.uuid4(), image.sample_id, 'detect', None, 'rectangle', 'tl', original_rectangle_points)
        for image in images
    )
    sibling_detection = AnnotationRecord(
        uuid.uuid4(), images[0].sample_id, 'detect', None, 'rectangle', 'tc', [[60.0, 50.0], [180.0, 150.0]]
    )
    detections = (primary_detections[0], sibling_detection, *primary_detections[1:])
    classifications = tuple(
        AnnotationRecord(
            uuid.uuid4(),
            detection.image_id,
            'classify',
            detection.id,
            'classification',
            ('tl', 'tc', 'cl', 'cc')[index % 4],
            None,
        )
        for index, detection in enumerate(detections)
    )
    segments = tuple(
        AnnotationRecord(
            uuid.uuid4(), detection.image_id, 'segment', detection.id, 'polyline', '1', [[80.0, 70.0], [220.0, 170.0]]
        )
        for detection in detections
    )
    extra_segment = AnnotationRecord(
        uuid.uuid4(),
        primary_detections[0].image_id,
        'segment',
        primary_detections[0].id,
        'polyline',
        '1',
        [[90.0, 160.0], [210.0, 80.0]],
    )
    database_path = workspace_dir / 'annotations.db'
    AnnotationRepository(database_path, point_task_definition()).save_annotations(
        (*detections, *classifications, *segments, extra_segment)
    )

    baseline_path = baseline_dir / 'reference.json'
    _write_json(baseline_path, {'kind': 'synthetic-baseline', 'immutable': True})

    suffix = uuid.uuid4().hex[:10]
    workspace_id = f'task-7-point-workflow-{suffix}'
    display_name = f'Point 三模型验收 {suffix}'
    config_path = root / 'workspace.json'
    _write_json(
        config_path,
        {
            'workspace_id': workspace_id,
            'display_name': display_name,
            'owner_user_id': owner_user_id,
            'workspace_dir': str(workspace_dir),
            'runtime_dir': str(runtime_dir),
            'cvat_internal_url': cvat_internal_url,
        },
    )

    return {
        'marker': FIXTURE_MARKER,
        'root': str(root),
        'workspace_id': workspace_id,
        'display_name': display_name,
        'config_path': str(config_path),
        'database_path': str(database_path),
        'runtime_dir': str(runtime_dir),
        'initial_annotation_ids': {
            'detect': [str(record.id) for record in detections],
            'classify': [str(record.id) for record in classifications],
            'segment': [str(record.id) for record in (*segments, extra_segment)],
        },
        'original_rectangle_points': original_rectangle_points,
        'images': [
            {'path': str(image.image_path), 'sha256': _sha256(image.image_path), 'sample_id': image.sample_id}
            for image in images
        ],
        'baseline': {'path': str(baseline_path), 'sha256': _sha256(baseline_path)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Create the dedicated synthetic Point browser-acceptance workspace.')
    parser.add_argument('parent', type=Path, help='Writable parent for a newly generated temporary directory.')
    parser.add_argument('--owner-user-id', type=int, required=True, help='Dedicated CVAT test user ID.')
    parser.add_argument('--cvat-internal-url', default='http://cvat_server:8080')
    parser.add_argument(
        '--receipt',
        type=Path,
        default=Path('.superpowers/platform-acceptance/fixture.json'),
        help='Ignored local receipt path read by the browser acceptance test.',
    )
    args = parser.parse_args(argv)
    if args.owner_user_id <= 0:
        parser.error('--owner-user-id must be a positive integer')

    receipt = create_fixture(args.parent, owner_user_id=args.owner_user_id, cvat_internal_url=args.cvat_internal_url)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.receipt, receipt)
    print(args.receipt.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
