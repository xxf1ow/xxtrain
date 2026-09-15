from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import uuid
from pathlib import Path

from PIL import Image, ImageDraw

FIXTURE_MARKER = 'xxtrain-task-6-synthetic-workspace-v1'


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_image(path: Path, accent: str) -> None:
    image = Image.new('RGB', (800, 600), '#eef2f2')
    draw = ImageDraw.Draw(image)
    for x in range(0, 801, 100):
        draw.line((x, 0, x, 600), fill='#c5d2d2', width=1)
    for y in range(0, 601, 100):
        draw.line((0, y, 800, y), fill='#c5d2d2', width=1)
    draw.rectangle((120, 100, 300, 260), outline=accent, width=5)
    draw.text((130, 110), 'synthetic Point acceptance', fill='#172c32')
    image.save(path, quality=95)


def create_fixture(parent: Path, *, owner_user_id: int, cvat_internal_url: str) -> dict[str, object]:
    """Create an isolated synthetic workspace and return its non-secret acceptance receipt.

    The caller owns the generated temporary directory. The receipt records immutable input hashes and paths for the
    browser harness; it never contains a CVAT password or service token.
    """
    parent = parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='xxtrain-point-acceptance-', dir=parent))
    workspace_dir = root / 'workspace'
    workspace_dir.mkdir()
    images_dir = workspace_dir / 'images'
    annotations_dir = workspace_dir / 'annotations'
    baseline_dir = root / 'baseline'
    runtime_dir = root / 'runtime'
    for directory in (images_dir, annotations_dir, baseline_dir, runtime_dir):
        directory.mkdir()

    first_image = images_dir / 'point-a.jpg'
    second_image = images_dir / 'point-b.jpg'
    _write_image(first_image, '#007d8a')
    _write_image(second_image, '#d56a30')

    original_rectangle_points = [[120.0, 100.0], [300.0, 260.0]]
    annotation_path = annotations_dir / 'point-a.json'
    _write_json(
        annotation_path,
        {
            'version': '5.4.1',
            'flags': {'fixture': 'task-6', 'preserve': True},
            'description': 'synthetic mixed annotation',
            'shapes': [
                {
                    'label': 'tl',
                    'points': original_rectangle_points,
                    'group_id': 11,
                    'description': 'classification-metadata',
                    'shape_type': 'rectangle',
                    'flags': {'reviewed': True},
                },
                {
                    'label': 'wire',
                    'points': [[40.0, 40.0], [90.0, 70.0]],
                    'group_id': None,
                    'description': 'preserve-line',
                    'shape_type': 'line',
                    'flags': {'preserve': True},
                },
                {
                    'label': 'mask',
                    'points': [[500.0, 100.0], [620.0, 130.0], [560.0, 250.0]],
                    'group_id': 23,
                    'description': 'preserve-polygon',
                    'shape_type': 'polygon',
                    'flags': {'preserve': True},
                },
            ],
            'imagePath': '../images/point-a.jpg',
            'imageData': None,
            'imageHeight': 600,
            'imageWidth': 800,
            'custom': {'preserve': ['root', 'value']},
        },
    )

    baseline_path = baseline_dir / 'reference.json'
    _write_json(baseline_path, {'kind': 'synthetic-baseline', 'immutable': True})

    suffix = uuid.uuid4().hex[:10]
    workspace_id = f'task-6-synthetic-{suffix}'
    display_name = f'Task 6 合成验收 {suffix}'
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
        'annotation_path': str(annotation_path),
        'original_rectangle_points': original_rectangle_points,
        'images': [
            {'path': str(first_image), 'sha256': _sha256(first_image)},
            {'path': str(second_image), 'sha256': _sha256(second_image)},
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
