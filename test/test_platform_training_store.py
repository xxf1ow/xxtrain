import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from uuid import uuid4

from xxtrain.platform.contracts import PlatformConflictError
from xxtrain.platform.runtime import RuntimeCache
from xxtrain.platform.training_contracts import DownloadFile, ExecutionView, TrainingRun, TrainingRunView
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

    def test_cancel_intent_survives_reopen_and_duplicate_create(self) -> None:
        store = TrainingRunStore(self.path)
        candidate = replace(self.run, desired_action='execute')
        store.create(candidate)
        cancelled = store.request_cancel(candidate.user_id, candidate.id)
        self.assertEqual('cancel', cancelled.desired_action)
        self.assertEqual(cancelled, store.create(candidate))
        self.assertEqual(cancelled, TrainingRunStore(self.path).get(candidate.user_id, candidate.id))
        with self.assertRaisesRegex(ValueError, 'different facts'):
            store.create(replace(candidate, fingerprint='b' * 64))
        with self.assertRaises(ValueError):
            store.request_cancel(18, candidate.id)

    def test_cancel_is_idempotent_and_invalid_intent_is_rejected(self) -> None:
        store = TrainingRunStore(self.path)
        candidate = replace(self.run, desired_action='execute')
        store.create(candidate)

        cancelled = store.request_cancel(candidate.user_id, candidate.id)

        self.assertEqual(cancelled, store.request_cancel(candidate.user_id, candidate.id))
        with self.assertRaisesRegex(ValueError, 'intent'):
            store.create(replace(self.run, id=str(uuid4()), desired_action='pause'))

    def test_cancelled_input_remains_canonical_for_a_different_candidate_id(self) -> None:
        store = TrainingRunStore(self.path)
        candidate = replace(self.run, desired_action='execute')
        store.create(candidate)
        cancelled = store.request_cancel(candidate.user_id, candidate.id)

        duplicate = replace(candidate, id=str(uuid4()), submitted_at='2026-09-17T13:00:00+00:00')

        self.assertEqual(cancelled, store.create(duplicate))

    def test_same_input_returns_original_identity(self) -> None:
        store = TrainingRunStore(self.path)
        first = store.create(self.run)
        candidate = replace(self.run, id=str(uuid4()), submitted_at='2026-09-17T13:00:00+00:00')

        self.assertEqual(first, store.create(candidate))
        self.assertEqual((first,), store.list_user(self.run.user_id))
        self.assertEqual(first, store.find_input(first.user_id, first.workspace_id, first.target, first.fingerprint))

    def test_input_key_dimensions_create_distinct_runs(self) -> None:
        store = TrainingRunStore(self.path)
        variants = (
            replace(self.run, id=str(uuid4()), user_id=18),
            replace(self.run, id=str(uuid4()), workspace_id='line-4'),
            replace(self.run, id=str(uuid4()), target='segment'),
            replace(self.run, id=str(uuid4()), fingerprint='b' * 64),
        )

        store.create(self.run)
        for variant in variants:
            self.assertEqual(variant, store.create(variant))

        self.assertEqual(5, sum(len(store.list_user(user_id)) for user_id in (17, 18)))

    def test_same_input_rejects_a_different_cache_association(self) -> None:
        store = TrainingRunStore(self.path)
        store.create(self.run)

        with self.assertRaisesRegex(ValueError, 'cache'):
            store.create(replace(self.run, id=str(uuid4()), cache_relative_path='cache/b/detect'))

    def test_find_input_uses_the_full_owned_key(self) -> None:
        store = TrainingRunStore(self.path)
        store.create(self.run)

        self.assertIsNone(store.find_input(18, self.run.workspace_id, self.run.target, self.run.fingerprint))
        self.assertIsNone(store.find_input(17, 'line-4', self.run.target, self.run.fingerprint))
        self.assertIsNone(store.find_input(17, self.run.workspace_id, 'segment', self.run.fingerprint))
        self.assertIsNone(store.find_input(17, self.run.workspace_id, self.run.target, 'b' * 64))

    def test_concurrent_same_id_creates_return_the_same_persisted_run(self) -> None:
        TrainingRunStore(self.path)
        barrier = Barrier(2)

        def create() -> TrainingRun:
            barrier.wait()
            return TrainingRunStore(self.path).create(self.run)

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(executor.map(lambda _: create(), range(2)))

        self.assertEqual((self.run, self.run), results)

    def test_concurrent_same_input_candidates_create_one_canonical_run(self) -> None:
        TrainingRunStore(self.path)
        barrier = Barrier(2)
        first = replace(self.run, desired_action='execute')
        candidates = (first, replace(first, id=str(uuid4()), submitted_at='2026-09-17T13:00:00+00:00'))

        def create(candidate: TrainingRun) -> TrainingRun:
            store = TrainingRunStore(self.path)
            barrier.wait()
            return store.create(candidate)

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(executor.map(create, candidates))

        reopened = TrainingRunStore(self.path)
        self.assertEqual(results[0], results[1])
        self.assertEqual((results[0],), reopened.list_user(self.run.user_id))

    def test_canonical_reuse_preserves_attempt_and_task_binding(self) -> None:
        store = TrainingRunStore(self.path)
        store.create(self.run)
        store.mark_create_attempted(self.run.id, '2026-09-17T10:01:00Z')
        store.mark_create_attempted(self.run.id, '2026-09-17T10:02:00Z')
        store.bind_task(self.run.id, 'clearml-1')

        canonical = store.create(replace(self.run, id=str(uuid4()), submitted_at='2026-09-17T13:00:00+00:00'))

        self.assertEqual(self.run.id, canonical.id)
        self.assertEqual('2026-09-17T10:01:00Z', canonical.create_attempted_at)
        self.assertEqual('clearml-1', canonical.clearml_task_id)

    def test_database_lock_is_normalized_to_a_typed_conflict(self) -> None:
        store = TrainingRunStore(self.path)
        with closing(sqlite3.connect(self.path)) as locked, locked:
            locked.execute('BEGIN EXCLUSIVE')
            with self.assertRaises(PlatformConflictError):
                store.create(self.run)

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

    def test_legacy_duplicate_inputs_refuse_startup_without_changing_rows(self) -> None:
        self.path.parent.mkdir(parents=True)
        duplicate = replace(self.run, id=str(uuid4()), clearml_task_id='clearml-2')
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE training_runs (
                  id TEXT PRIMARY KEY,
                  user_id INTEGER NOT NULL,
                  workspace_id TEXT NOT NULL,
                  workspace_name TEXT NOT NULL,
                  target TEXT NOT NULL,
                  fingerprint TEXT NOT NULL,
                  cache_relative_path TEXT NOT NULL,
                  submitted_at TEXT NOT NULL,
                  create_attempted_at TEXT,
                  clearml_task_id TEXT UNIQUE
                );
                CREATE INDEX training_runs_by_user ON training_runs(user_id);
                CREATE INDEX training_runs_by_workspace ON training_runs(workspace_id);
                """
            )
            connection.executemany(
                'INSERT INTO training_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (self._run_values(self.run), self._run_values(duplicate)),
            )
            before = tuple(connection.execute('SELECT * FROM training_runs ORDER BY rowid'))

        with self.assertRaisesRegex(PlatformConflictError, 'duplicate.*user_id=17.*line-3.*detect'):
            TrainingRunStore(self.path)

        with closing(sqlite3.connect(self.path)) as connection:
            after = tuple(connection.execute('SELECT * FROM training_runs ORDER BY rowid'))
            indexes = {row[1] for row in connection.execute('PRAGMA index_list(training_runs)')}
        self.assertEqual(before, after)
        self.assertNotIn('training_runs_by_input', indexes)

    def test_legacy_nonduplicate_database_acquires_input_index_without_changing_facts(self) -> None:
        self.path.parent.mkdir(parents=True)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE training_runs (
                  id TEXT PRIMARY KEY,
                  user_id INTEGER NOT NULL,
                  workspace_id TEXT NOT NULL,
                  workspace_name TEXT NOT NULL,
                  target TEXT NOT NULL,
                  fingerprint TEXT NOT NULL,
                  cache_relative_path TEXT NOT NULL,
                  submitted_at TEXT NOT NULL,
                  create_attempted_at TEXT,
                  clearml_task_id TEXT UNIQUE
                );
                """
            )
            connection.execute(
                'INSERT INTO training_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', self._run_values(self.run)
            )

        store = TrainingRunStore(self.path)

        self.assertEqual(self.run, store.get(self.run.user_id, self.run.id))
        with closing(sqlite3.connect(self.path)) as connection:
            indexes = {row[1] for row in connection.execute('PRAGMA index_list(training_runs)')}
        self.assertIn('training_runs_by_input', indexes)

    def test_legacy_database_adds_nullable_intent_without_changing_bound_run(self) -> None:
        self.path.parent.mkdir(parents=True)
        bound = replace(self.run, create_attempted_at='2026-09-17T10:01:00Z', clearml_task_id='clearml-1')
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE training_runs (
                  id TEXT PRIMARY KEY,
                  user_id INTEGER NOT NULL,
                  workspace_id TEXT NOT NULL,
                  workspace_name TEXT NOT NULL,
                  target TEXT NOT NULL,
                  fingerprint TEXT NOT NULL,
                  cache_relative_path TEXT NOT NULL,
                  submitted_at TEXT NOT NULL,
                  create_attempted_at TEXT,
                  clearml_task_id TEXT UNIQUE
                );
                """
            )
            connection.execute(
                'INSERT INTO training_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', self._run_values(bound)
            )

        first = TrainingRunStore(self.path).get(bound.user_id, bound.id)
        second = TrainingRunStore(self.path).get(bound.user_id, bound.id)

        self.assertEqual(bound, first)
        self.assertEqual(bound, second)
        self.assertIsNone(first.desired_action)
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute('SELECT * FROM training_runs WHERE id = ?', (bound.id,)).fetchone()
            columns = tuple(column[1] for column in connection.execute('PRAGMA table_info(training_runs)'))
        self.assertEqual(self._run_values(bound) + (None,), row)
        self.assertEqual('desired_action', columns[-1])

    def test_run_table_contains_no_business_stage(self) -> None:
        TrainingRunStore(self.path)

        with closing(sqlite3.connect(self.path)) as connection:
            columns = {row[1] for row in connection.execute('PRAGMA table_info(training_runs)')}

        self.assertIn('clearml_task_id', columns)
        self.assertIn('desired_action', columns)
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

    def test_cache_path_rejects_an_escaping_publication_symlink(self) -> None:
        runtime = RuntimeCache(self.root / 'runtime')
        outside = self.root / 'outside'
        target = outside / 'detect'
        target.mkdir(parents=True)
        cache = self.root / 'runtime' / 'cache'
        cache.mkdir(parents=True)
        fingerprint = 'a' * 64
        publication = cache / fingerprint
        (publication).symlink_to(outside, target_is_directory=True)
        (target / 'dataset.yaml').write_text(
            f'path: {publication}\ntrain: detect/train.txt\nval: detect/val.txt\n', encoding='utf-8'
        )
        (target / 'train.txt').write_text('detect/workspace/train.png\n', encoding='utf-8')
        (target / 'val.txt').write_text('detect/workspace/val.png\n', encoding='utf-8')
        (target / 'workspace').mkdir()
        (target / 'workspace' / 'train.png').touch()
        (target / 'workspace' / 'val.png').touch()

        with self.assertRaisesRegex(ValueError, 'complete'):
            runtime.cache_path('detect', fingerprint)

    def test_cache_path_rejects_a_listed_file_symlink_escaping_its_publication(self) -> None:
        runtime = RuntimeCache(self.root / 'runtime')
        target = self._complete_detection_cache('b' * 64)
        outside = self.root / 'outside.png'
        outside.touch()
        (target / 'workspace' / 'train.png').unlink()
        (target / 'workspace' / 'train.png').symlink_to(outside)

        with self.assertRaisesRegex(ValueError, 'complete'):
            runtime.cache_path('detect', 'b' * 64)

    def test_cache_path_accepts_a_listed_detection_file_symlink_inside_its_publication(self) -> None:
        runtime = RuntimeCache(self.root / 'runtime')
        target = self._complete_detection_cache('c' * 64)
        workspace = target / 'workspace'
        (workspace / 'source.png').touch()
        (workspace / 'train.png').unlink()
        (workspace / 'train.png').symlink_to('source.png')

        self.assertEqual(target, runtime.cache_path('detect', 'c' * 64))

    def test_execution_contracts_carry_only_view_data(self) -> None:
        execution = ExecutionView('task-1', 'unknown', True, None, None, None, None, False, 'Unavailable')

        self.assertEqual(TrainingRunView(self.run, execution), TrainingRunView(self.run, execution))
        self.assertEqual(
            DownloadFile(self.root / 'model.onnx', 'model.onnx', 'application/octet-stream').filename, 'model.onnx'
        )

    def _complete_detection_cache(self, fingerprint: str) -> Path:
        publication = self.root / 'runtime' / 'cache' / fingerprint
        target = publication / 'detect'
        workspace = target / 'workspace'
        workspace.mkdir(parents=True)
        (target / 'dataset.yaml').write_text(
            f'path: {publication}\ntrain: detect/train.txt\nval: detect/val.txt\n', encoding='utf-8'
        )
        (target / 'train.txt').write_text('detect/workspace/train.png\n', encoding='utf-8')
        (target / 'val.txt').write_text('detect/workspace/val.png\n', encoding='utf-8')
        (workspace / 'train.png').touch()
        (workspace / 'val.png').touch()
        return target

    @staticmethod
    def _run_values(run: TrainingRun) -> tuple[object, ...]:
        return (
            run.id,
            run.user_id,
            run.workspace_id,
            run.workspace_name,
            run.target,
            run.fingerprint,
            run.cache_relative_path,
            run.submitted_at,
            run.create_attempted_at,
            run.clearml_task_id,
        )


if __name__ == '__main__':
    unittest.main()
