import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import pandas as pd

from scripts.build_career_supplement import assemble, build_bundle, digest, prior_history


A, B, C, D, CURRENT = 'a' * 16, 'b' * 16, 'c' * 16, 'd' * 16, 'f' * 16


class CareerSupplementTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.addCleanup(self.db.close)
        self.db.executescript('''
            CREATE TABLE fighters(fighter_id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE fighter_aliases(fighter_id INTEGER, source TEXT, source_id TEXT, name TEXT);
            CREATE TABLE events(event_id INTEGER PRIMARY KEY, date TEXT, name TEXT, promotion TEXT);
            CREATE TABLE fights(fight_id INTEGER PRIMARY KEY, event_id INTEGER, bout_type TEXT,
                                overturned INTEGER, no_contest INTEGER, method TEXT);
            CREATE TABLE fight_participants(fight_id INTEGER, fighter_id INTEGER, result TEXT);
            CREATE TABLE fight_aliases(fight_id INTEGER, source TEXT, source_id TEXT);
        ''')
        self.db.executemany('INSERT INTO fighters VALUES (?, ?)', [(1, 'One'), (2, 'Two'), (3, 'Regional'), (4, 'Extra')])
        self.db.executemany('INSERT INTO fighter_aliases VALUES (?, ?, ?, ?)',
                            [(1, 'ufcstats', A, 'One'), (2, 'ufcstats', B, 'Two'),
                             (4, 'ufcstats', D, 'Extra'), (1, 'ufcstats', A, 'One Alias')])
        self.registry = pd.DataFrame({'fighter_id': [A, B], 'name': ['One', 'Two']})
        self.current = pd.DataFrame([{'fight_id': CURRENT, 'red_fighter_id': A, 'blue_fighter_id': B,
                                      'date': pd.Timestamp('2020-01-01'), 'winner': 'red'}])
        self.add_bout(10, '2020-01-01', participants=((1, 'W'), (2, 'L')), alias=('ufcstats', CURRENT))
        self.add_bout(20, '2019-01-01')
        self.add_bout(21, '2021-01-01', participants=((2, 'L'), (3, 'W')))
        self.add_bout(22, '2024-01-01', participants=((1, 'D'), (2, 'D')))

    def add_bout(self, fight_id, day, participants=((1, 'W'), (3, 'L')), bout_type='pro', overturned=0,
                 no_contest=0, alias=None):
        self.db.execute('INSERT INTO events VALUES (?, ?, ?, ?)', (fight_id, day, f'Event {fight_id}', 'Regional'))
        self.db.execute('INSERT INTO fights VALUES (?, ?, ?, ?, ?, ?)',
                        (fight_id, fight_id, bout_type, overturned, no_contest, 'Decision'))
        self.db.executemany('INSERT INTO fight_participants VALUES (?, ?, ?)',
                            [(fight_id, fighter_id, result) for fighter_id, result in participants])
        source, source_id = alias or ('sherdog', str(fight_id))
        self.db.execute('INSERT INTO fight_aliases VALUES (?, ?, ?)', (fight_id, source, source_id))

    def build(self, current=None, registry=None, extra=None):
        return assemble(self.db, self.current if current is None else current,
                        self.registry if registry is None else registry, extra or {}, '2024-12-31', '2026-11-04T12:00:00Z')

    def test_exact_ids_project_each_fighter_once_and_exclude_existing_ufc(self):
        tables, counts = self.build()
        history = tables['fighter_bouts.parquet']
        self.assertEqual(set(history.canonical_fight_id), {20, 21, 22})
        self.assertEqual((counts['bouts'], counts['fighterBouts']), (3, 4))
        self.assertFalse(history.duplicated(['fighter_id', 'canonical_fight_id']).any())
        self.assertTrue(pd.isna(history[history.canonical_fight_id == 20].iloc[0].opponent_id))
        self.assertEqual(history[history.canonical_fight_id == 22].iloc[0].opponent_id, B)
        self.assertNotIn('wins', history.columns)

    def test_future_same_day_and_current_fights_cannot_enter_prior_history(self):
        self.add_bout(30, '2025-01-01')
        tables, _ = self.build()
        history = tables['fighter_bouts.parquet']
        self.assertEqual(set(prior_history(history, A, '2024-01-01').canonical_fight_id), {20})
        self.assertEqual(set(prior_history(history, A, '2019-01-01').canonical_fight_id), set())
        self.assertNotIn(30, set(history.canonical_fight_id))
        with self.assertRaises(ValueError):
            prior_history(history, A, '2024-1-1')

    def test_quarantines_unknown_pro_status_invalid_outcomes_and_revisions(self):
        self.add_bout(30, '2010-01-01', bout_type=None)
        self.add_bout(31, '2011-01-01', overturned=1)
        self.add_bout(32, '2012-01-01', participants=((1, 'NC'), (3, 'NC')))
        self.add_bout(33, '2013-01-01', participants=((1, 'W'), (3, 'W')))
        self.add_bout(34, None)
        self.add_bout(35, '2015-01-01', participants=((1, 'W'),))
        self.add_bout(36, '2016-01-01')
        self.db.execute('DELETE FROM fight_aliases WHERE fight_id=36')
        self.add_bout(37, '2017-01-01', participants=((1, 'W'), (999, 'L')))
        tables, _ = self.build()
        excluded = tables['exclusions.parquet'].set_index('canonical_fight_id')
        for fight_id, reason in [(30, 'professional_status_not_explicit'), (31, 'overturned'),
                                 (32, 'no_contest_flag_conflict'), (33, 'invalid_completed_results'),
                                 (34, 'missing_or_invalid_date'), (35, 'invalid_participant_pair'),
                                 (36, 'missing_source_alias'), (37, 'invalid_participant_pair')]:
            self.assertIn(reason, json.loads(excluded.loc[fight_id, 'exclusion_reasons_json']))

    def test_rejects_both_duplicate_pair_dates_and_alias_free_current_overlap(self):
        self.add_bout(30, '2010-01-01')
        self.add_bout(31, '2010-01-01')
        self.add_bout(32, '2020-01-01', participants=((2, 'L'), (1, 'W')))
        tables, _ = self.build()
        excluded = tables['exclusions.parquet'].set_index('canonical_fight_id')
        for fight_id in (30, 31):
            self.assertIn('ambiguous_pair_date', json.loads(excluded.loc[fight_id, 'exclusion_reasons_json']))
        self.assertIn('current_bundle_pair_date_overlap', json.loads(excluded.loc[32, 'exclusion_reasons_json']))

    def test_reviewed_holds_are_excluded_even_when_provider_says_pro(self):
        self.add_bout(151925, '2013-12-27')
        self.add_bout(323526, '2023-11-03')
        tables, _ = self.build()
        self.assertFalse(tables['fighter_bouts.parquet'].canonical_fight_id.isin([151925, 323526]).any())

    def test_ambiguous_source_alias_or_unproved_extra_identity_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'reviewed proof'):
            self.build(extra={D: 3})
        self.db.execute('INSERT INTO fighter_aliases VALUES (?, ?, ?, ?)', (3, 'ufcstats', A, 'One'))
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.build()

    def test_known_current_identity_conflict_must_be_repaired_first(self):
        self.db.execute('INSERT INTO fighter_aliases VALUES (?, ?, ?, ?)', (3, 'ufcstats', C, 'Regional'))
        current = self.current.assign(blue_fighter_id=C)
        registry = pd.concat([self.registry, pd.DataFrame([{'fighter_id': C, 'name': 'Regional'}])])
        with self.assertRaisesRegex(ValueError, 'Repair known'):
            self.build(current=current, registry=registry)

    def test_unresolved_current_identity_still_excludes_canonical_pair_overlap(self):
        self.add_bout(30, '2020-01-01', participants=((2, 'L'), (1, 'W')))
        tables, _ = self.build(current=self.current.assign(blue_fighter_id=None))
        self.assertNotIn(30, set(tables['fighter_bouts.parquet'].canonical_fight_id))
        self.assertIn('current_identity_unresolved', tables['overlap-audit.parquet'].iloc[0].reasons_json)

    def test_duplicate_ufc_fight_alias_fails_closed(self):
        self.db.execute('INSERT INTO fight_aliases VALUES (?, ?, ?)', (20, 'ufcstats', CURRENT))
        with self.assertRaisesRegex(ValueError, 'multiple bouts'):
            self.build()

    def bundle_fixture(self, root):
        source, current = root / 'source', root / 'current'
        source.mkdir()
        current.mkdir()
        database = sqlite3.connect(source / 'canon.db')
        self.db.commit()
        self.db.backup(database)
        database.close()
        database_sha = digest(source / 'canon.db')
        (source / 'download-receipt.json').write_text(json.dumps({'sha256': database_sha, 'retrievedAt': '2026-11-04T12:00:00Z'}))
        (source / 'upstream-license-receipt.json').write_text(json.dumps({'content': 'MIT License\nTest copyright notice\n'}))
        (source / 'manifest.json').write_text(json.dumps({'files': {path.name: digest(path) for path in source.iterdir()}}))
        self.current.to_parquet(current / 'fights.parquet', index=False)
        self.registry.to_parquet(current / 'fighters.parquet', index=False)
        (current / 'manifest.json').write_text(json.dumps({'bundleType': 'identity-corrected-research',
            'dataThrough': '2024-12-31', 'files': {path.name: digest(path) for path in current.iterdir()}}))
        proof = root / 'proof.json'
        proof.write_text(json.dumps({'database': {'sha256': database_sha}, 'fighters': []}))
        return source, current, proof

    def test_new_immutable_bundle_preserves_sources_and_contains_license_without_db(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, current, proof = self.bundle_fixture(root)
            before = {path: digest(path) for folder in (source, current) for path in folder.iterdir()}
            target = root / 'supplement-v1'
            manifest = build_bundle(source, current, proof, target, digest(source / 'manifest.json'), digest(proof))
            self.assertEqual(manifest['bundleType'], 'career-supplement')
            self.assertNotIn('canon.db', manifest['files'])
            self.assertEqual((target / 'LICENSE').read_text(), 'MIT License\nTest copyright notice\n')
            for name, checksum in manifest['files'].items():
                self.assertEqual(digest(target / name), checksum)
            self.assertEqual(before, {path: digest(path) for path in before})
            with self.assertRaisesRegex(ValueError, 'new directory'):
                build_bundle(source, current, proof, target, digest(source / 'manifest.json'), digest(proof))

    def test_changed_source_and_proof_abort_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, current, proof = self.bundle_fixture(root)
            target = root / 'supplement-v1'
            with self.assertRaisesRegex(ValueError, 'Identity proof checksum'):
                build_bundle(source, current, proof, target, digest(source / 'manifest.json'), '0' * 64)
            (source / 'upstream-license-receipt.json').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Input checksum'):
                build_bundle(source, current, proof, target, digest(source / 'manifest.json'), digest(proof))
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
