import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from xxtrain.platform.runtime import RuntimeCache
from xxtrain.platform.training_contracts import TrainingRun
from xxtrain.platform.training_store import TrainingRunStore


class PlatformTrainingStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix='xxtrain-training-store-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / 'metadata' / 'training-runs.db'
        self.run = TrainingRun(
            str(uuid4()),
            17,
            'line-3',
            'Line 3',
            'detect',
            'a' * 64,
            'cache/a/detect',
            '2026-09-17T10:00:00Z',
            None,
            None,
        )

    def test_round_trip_survives_reopen_and_keeps_user_and_workspace_indexes(self) -> None:
        store = TrainingRunStore(self.path)
        store.create(self.run)
        store.mark_create_attempted(self.run.id, '2026-09-17T10:01:00Z')
        store.bind_task(self.run.id, 'clearml-1')

        reopened = TrainingRunStore(self.path)

        expected = replace(self.run, create_attempted_at='2026-09-17T10:01:00Z', clearml_task_id='clearml-1')
        self.assertEqual(expected, reopened.get(17, self.run.id))
        self.assertEqual((expected,), reopened.list_user(17))
        self.assertEqual((expected,), reopened.list_workspace('line-3'))

    def test_create_is_idempotent_only_for_the_same_facts(self) -> None:
        store = TrainingRunStore(self.path)

        self.assertEqual(self.run, store.create(self.run))
        self.assertEqual(self.run, store.create(self.run))
        with self.assertRaisesRegex(ValueError, 'different facts'):
            store.create(replace(self.run, fingerprint='b' * 64))

    def test_get_rejects_a_run_owned_by_another_user(self) -> None:
        store = TrainingRunStore(self.path)
        store.create(self.run)

        with self.assertRaisesRegex(ValueError, 'access denied'):
            store.get(18, self.run.id)

    def test_binding_is_idempotent_but_cannot_change_the_task_identity(self) -> None:
        store = TrainingRunStore(self.path)
        store.create(self.run)

        store.bind_task(self.run.id, 'clearml-1')
        store.bind_task(self.run.id, 'clearml-1')
        with self.assertRaisesRegex(ValueError, 'different task'):
            store.bind_task(self.run.id, 'clearml-2')

    def test_run_table_contains_no_business_stage(self) -> None:
        TrainingRunStore(self.path)

        with closing(sqlite3.connect(self.path)) as connection:
            columns = {row[1] for row in connection.execute('PRAGMA table_info(training_runs)')}

        self.assertIn('clearml_task_id', columns)
        self.assertTrue(columns.isdisjoint({'status', 'stage', 'active_count', 'completed'}))

    def test_cache_path_requires_complete_published_training_inputs(self) -> None:
        runtime = RuntimeCache(self.root / 'runtime')
        fingerprint = 'a' * 64
        for target in ('detect', 'classify', 'segment'):
            publication = self.root / 'runtime' / 'cache' / fingerprint / target
            publication.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'complete'):
                runtime.cache_path(target, fingerprint)
            (publication / 'dataset.yaml').write_text(
                f'path: {publication.parent}\ntrain: {target}/train.txt\nval: {target}/val.txt\n', encoding='utf-8'
            )
            (publication / 'train.txt').write_text(f'{target}/workspace/train.png\n', encoding='utf-8')
            (publication / 'val.txt').write_text(f'{target}/workspace/val.png\n', encoding='utf-8')
            (publication / 'workspace').mkdir()
            (publication / 'workspace' / 'train.png').touch()
            (publication / 'workspace' / 'val.png').touch()
            if target != 'detect':
                manifest = {'fingerprint': fingerprint, 'target': target}
                (publication.parent / 'manifest.json').write_text(str(manifest).replace("'", '"'), encoding='utf-8')
            self.assertEqual(publication, runtime.cache_path(target, fingerprint))

    def test_cache_path_rejects_paths_outside_the_runtime_cache(self) -> None:
        runtime = RuntimeCache(self.root / 'runtime')

        with self.assertRaisesRegex(ValueError, 'Unsupported target'):
            runtime.cache_path('../detect', 'a' * 64)
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            runtime.cache_path('detect', '../outside')


if __name__ == '__main__':
    unittest.main()
