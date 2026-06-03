from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

try:
    from .repair_identities import apply_corrections, digest, repair_bundle, resolve_fighter_id
except ImportError:
    from repair_identities import apply_corrections, digest, repair_bundle, resolve_fighter_id


OLD = '12ebd7d157e91701'
NEW = '294aa73dbf37d281'
OPPONENT = '814e5233e2acf2ee'
FIRST, SECOND = 'fc9168fd102e2e66', '6ecfa439ef900bcc'


class IdentityRepairTests(unittest.TestCase):
    def setUp(self):
        self.fighters = pd.DataFrame({'fighter_id': [OLD, NEW, OPPONENT],
                                      'name': ['Bruno Silva', 'Bruno Silva', 'Charles Johnson']})
        self.fights = pd.DataFrame([
            {'fight_id': FIRST, 'red_fighter_id': OPPONENT, 'blue_fighter_id': OLD,
             'red_name': 'Charles Johnson', 'blue_name': 'Bruno Silva', 'date': pd.Timestamp('2026-03-14'),
             'weightclass': 'Flyweight', 'winner': 'red'},
            {'fight_id': SECOND, 'red_fighter_id': OLD, 'blue_fighter_id': OPPONENT,
             'red_name': 'Bruno Silva', 'blue_name': 'Charles Johnson', 'date': pd.Timestamp('2025-05-10'),
             'weightclass': 'Middleweight', 'winner': 'blue'},
        ])
        self.rounds = pd.DataFrame([{'fight_id': bout, 'fighter_id': fighter, 'round': 1, 'sig_str_landed': 10}
                                    for bout in (FIRST, SECOND) for fighter in (OLD, OPPONENT)])
        self.document = {'type': 'proposed_fight_specific_identity_corrections_not_applied',
                         'proposals': [{'fight_id': FIRST, 'side': 'blue', 'old_fighter_id': OLD,
                                        'proposed_fighter_id': NEW, 'name': 'Bruno Silva', 'date': '2026-03-14',
                                        'weightclass': 'Flyweight', 'affected_round_rows': 1,
                                        'affected_round_numbers': [1]}],
                         'roundRows': [{'fight_id': FIRST, 'round': 1, 'old_fighter_id': OLD, 'proposed_fighter_id': NEW}]}

    def test_ambiguous_name_never_uses_registry_order(self):
        for fighters in (self.fighters, self.fighters.iloc[::-1]):
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                resolve_fighter_id(fighters, name='Bruno Silva')
            self.assertEqual(resolve_fighter_id(fighters, NEW, 'Bruno Silva'), NEW)

    def test_unknown_exact_id_never_falls_back_to_name(self):
        with self.assertRaisesRegex(ValueError, 'Unknown explicit'):
            resolve_fighter_id(self.fighters, 'a' * 16, 'Charles Johnson')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            resolve_fighter_id(self.fighters, OPPONENT, 'Bruno Silva')

    def test_repairs_only_the_requested_fight_and_its_round_owner(self):
        fights_before, rounds_before = self.fights.copy(), self.rounds.copy()
        fights, rounds, report = apply_corrections(self.fights, self.rounds, self.fighters, self.document)
        self.assertEqual(fights.loc[0, 'blue_fighter_id'], NEW)
        self.assertEqual(fights.loc[1, 'red_fighter_id'], OLD)
        self.assertEqual(set(rounds[rounds.fight_id == FIRST].fighter_id), {NEW, OPPONENT})
        pd.testing.assert_frame_equal(rounds[rounds.fight_id == SECOND], self.rounds[self.rounds.fight_id == SECOND])
        pd.testing.assert_frame_equal(self.fights, fights_before)
        pd.testing.assert_frame_equal(self.rounds, rounds_before)
        pd.testing.assert_series_equal(fights.winner, self.fights.winner)
        self.assertEqual((report['fightSideCorrections'], report['roundOwnershipCorrections']), (1, 1))

    def test_stale_side_and_wrong_date_are_rejected(self):
        for field, value in [('old_fighter_id', NEW), ('side', 'red'), ('date', '2026-03-15'),
                             ('proposed_fighter_id', OPPONENT), ('weightclass', 'Middleweight')]:
            document = deepcopy(self.document)
            document['proposals'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                apply_corrections(self.fights, self.rounds, self.fighters, document)

    def test_exact_round_inventory_and_ownership_are_required(self):
        for mutation in ('missing_inventory', 'wrong_round', 'wrong_owner', 'duplicate_round', 'duplicate_proposal'):
            document, rounds = deepcopy(self.document), self.rounds.copy()
            if mutation == 'missing_inventory':
                document['roundRows'] = []
            elif mutation == 'wrong_round':
                document['roundRows'][0]['round'] = 2
            elif mutation == 'wrong_owner':
                rounds.loc[0, 'fighter_id'] = NEW
            elif mutation == 'duplicate_round':
                rounds = pd.concat([rounds, rounds.iloc[[0]]], ignore_index=True)
            else:
                document['proposals'] *= 2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                apply_corrections(self.fights, rounds, self.fighters, document)

    def bundle_fixture(self, root):
        source = root / 'source'
        source.mkdir()
        for name, frame in [('fights', self.fights), ('rounds', self.rounds), ('fighters', self.fighters)]:
            frame.to_parquet(source / f'{name}.parquet', index=False)
        source_manifest = {'schemaVersion': 1, 'dataThrough': '2026-03-14',
                           'files': {p.name: digest(p) for p in source.iterdir()}}
        (source / 'manifest.json').write_text(json.dumps(source_manifest))
        document = deepcopy(self.document)
        document['activeInputs'] = {str(p): digest(p) for p in source.iterdir()}
        correction = root / 'corrections.json'
        correction.write_text(json.dumps(document))
        return source, correction

    def test_new_bundle_verifies_hashes_and_keeps_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, correction = self.bundle_fixture(root)
            hashes = {p.name: digest(p) for p in source.iterdir()}
            target = root / 'corrected-v1'
            repair_bundle(source, correction, digest(correction), target)
            self.assertEqual(hashes, {p.name: digest(p) for p in source.iterdir()})
            self.assertEqual(pd.read_parquet(target / 'fights.parquet').loc[0, 'blue_fighter_id'], NEW)
            manifest = json.loads((target / 'manifest.json').read_text())
            self.assertTrue(manifest['requiresRetraining'])
            for name, checksum in manifest['files'].items():
                self.assertEqual(digest(target / name), checksum)
            with self.assertRaisesRegex(ValueError, 'new directory'):
                repair_bundle(source, correction, digest(correction), target)

    def test_changed_plan_and_source_are_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, correction = self.bundle_fixture(root)
            target = root / 'corrected-v1'
            with self.assertRaisesRegex(ValueError, 'checksum'):
                repair_bundle(source, correction, '0' * 64, target)
            (source / 'fighters.parquet').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Source bundle differs'):
                repair_bundle(source, correction, digest(correction), target)
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
