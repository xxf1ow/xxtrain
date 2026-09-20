import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xxtrain.platform.training_config import load_training_config


class TrainingConfigTest(unittest.TestCase):
    def test_loads_exact_fields_and_resolves_paths_relative_to_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'training.json'
            path.write_text(
                json.dumps(
                    {
                        'project': 'xxtrain',
                        'queue': 'training',
                        'shared_root': 'shared',
                        'metadata_dir': 'metadata',
                        'worker_script': 'installed/worker.py',
                        'run_root': 'runs',
                    }
                ),
                encoding='utf-8',
            )

            config = load_training_config(path)

            self.assertEqual('xxtrain', config.project)
            self.assertEqual('training', config.queue)
            self.assertEqual((root / 'shared').resolve(), config.shared_root)
            self.assertEqual((root / 'metadata').resolve(), config.metadata_dir)
            self.assertEqual((root / 'installed/worker.py').resolve(), config.worker_script)
            self.assertEqual((root / 'runs').resolve(), config.run_root)

    def test_rejects_missing_extra_and_invalid_fields(self) -> None:
        valid = {
            'project': 'xxtrain',
            'queue': 'training',
            'shared_root': 'shared',
            'metadata_dir': 'metadata',
            'worker_script': 'worker.py',
            'run_root': 'runs',
        }
        cases = (
            {**valid, 'secret': 'must-not-be-configured'},
            {key: value for key, value in valid.items() if key != 'metadata_dir'},
            {**valid, 'queue': ''},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'training.json'
            for payload in cases:
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        load_training_config(path)


class TrainingEntrypointTest(unittest.TestCase):
    def test_training_config_builds_service_and_binds_edit_guard(self) -> None:
        from xxtrain.platform.__main__ import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace_dir = root / 'workspace'
            runtime_dir = root / 'shared/runtime'
            worker = root / 'worker.py'
            workspace_dir.mkdir()
            runtime_dir.mkdir(parents=True)
            worker.write_text('pass\n', encoding='utf-8')
            workspace_path = root / 'workspace.json'
            workspace_path.write_text(
                json.dumps(
                    {
                        'workspace_id': 'line-3',
                        'display_name': 'Line 3',
                        'owner_user_id': 17,
                        'workspace_dir': 'workspace',
                        'runtime_dir': 'shared/runtime',
                        'cvat_internal_url': 'http://cvat.test',
                    }
                ),
                encoding='utf-8',
            )
            training_path = root / 'training.json'
            training_path.write_text(
                json.dumps(
                    {
                        'project': 'xxtrain',
                        'queue': 'training',
                        'shared_root': 'shared',
                        'metadata_dir': 'metadata',
                        'worker_script': 'worker.py',
                        'run_root': 'runs',
                    }
                ),
                encoding='utf-8',
            )
            events: list[tuple[object, ...]] = []

            class Resource:
                def __init__(self, *args: object, **kwargs: object) -> None:
                    events.append((type(self).__name__, *args, kwargs))

            class Annotation(Resource):
                require_editable = None
                require_cache_rebuild = None

            class Training(Resource):
                def require_editable(self, workspace_id: str, target: str | None = None) -> None:
                    pass

                def require_cache_rebuild(self, workspace_id: str, target: str, fingerprint: str) -> None:
                    pass

            class Http:
                def __enter__(self) -> 'Http':
                    return self

                def __exit__(self, *args: object) -> None:
                    pass

            captured: dict[str, object] = {}

            def initialize(training: Training) -> None:
                events.append(('initialize_input_compatibility', training))

            def make_app(
                config: object, annotations: Annotation, cvat: object, *, training_service: Training | None = None
            ) -> object:
                events.append(('create_app',))
                captured['guard'] = annotations.require_editable
                captured['cache_guard'] = annotations.require_cache_rebuild
                captured['training'] = training_service
                return object()

            environment = {
                'XXTRAIN_CVAT_SERVICE_TOKEN': 'cvat',
                'CLEARML_API_ACCESS_KEY': 'access',
                'CLEARML_API_SECRET_KEY': 'secret',
            }
            with (
                patch.dict(os.environ, environment, clear=True),
                patch('xxtrain.platform.__main__.WorkspaceData', Resource),
                patch('xxtrain.platform.__main__.RuntimeCache', Resource),
                patch('xxtrain.platform.__main__.CvatClient', Resource),
                patch('xxtrain.platform.__main__.AnnotationService', Annotation),
                patch('xxtrain.platform.__main__.TrainingRunStore', Resource),
                patch('xxtrain.platform.__main__.ClearMLClient', Resource),
                patch('xxtrain.platform.__main__.TrainingService', Training),
                patch('xxtrain.platform.__main__.initialize_input_compatibility', side_effect=initialize),
                patch('xxtrain.platform.__main__.create_app', side_effect=make_app),
                patch('xxtrain.platform.__main__.httpx.Client', return_value=Http()),
                patch('xxtrain.platform.__main__.uvicorn.run'),
            ):
                main(['--config', str(workspace_path), '--training-config', str(training_path)])

            self.assertIsNotNone(captured['training'])
            self.assertEqual(captured['training'].require_editable, captured['guard'])
            self.assertEqual(captured['training'].require_cache_rebuild, captured['cache_guard'])
            self.assertLess(
                events.index(('initialize_input_compatibility', captured['training'])), events.index(('create_app',))
            )
            clearml_event = next(
                event for event in events if event[0] == 'Resource' and event[1:3] == ('xxtrain', 'training')
            )
            self.assertEqual((root / 'runs').resolve(), clearml_event[-1]['run_root'])

    def test_training_startup_rejects_unsafe_paths_and_missing_clearml_keys(self) -> None:
        from xxtrain.platform.__main__ import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'workspace').mkdir()
            workspace = root / 'workspace.json'
            workspace.write_text(
                json.dumps(
                    {
                        'workspace_id': 'line-3',
                        'display_name': 'Line 3',
                        'owner_user_id': 17,
                        'workspace_dir': 'workspace',
                        'runtime_dir': 'runtime',
                        'cvat_internal_url': 'http://cvat.test',
                    }
                ),
                encoding='utf-8',
            )
            training = root / 'training.json'
            training.write_text(
                json.dumps(
                    {
                        'project': 'xxtrain',
                        'queue': 'training',
                        'shared_root': 'other',
                        'metadata_dir': 'runtime/metadata',
                        'worker_script': 'missing.py',
                        'run_root': 'runs',
                    }
                ),
                encoding='utf-8',
            )
            with patch.dict(os.environ, {'XXTRAIN_CVAT_SERVICE_TOKEN': 'cvat'}, clear=True):
                with self.assertRaises(SystemExit):
                    main(['--config', str(workspace), '--training-config', str(training)])


if __name__ == '__main__':
    unittest.main()
