import hashlib
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from research import ResearchData


BULLDOG = "294aa73dbf37d281"
BLINDADO = "12ebd7d157e91701"
KO_KING = "0778f94eb5d588a5"
OTHER_JOEY = "3a28e1e641366308"
OPPONENT = "aaaaaaaaaaaaaaaa"


class ResearchIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "corrected"
        self.bundle.mkdir()
        self.registry = pd.DataFrame([
            {"fighter_id": BULLDOG, "name": "Bruno Silva", "nickname": "Bulldog", "height_cm": 163.0, "reach_cm": 165.0},
            {"fighter_id": BLINDADO, "name": "Bruno Silva", "nickname": "Blindado", "height_cm": 183.0, "reach_cm": 188.0},
            {"fighter_id": KO_KING, "name": "Joey Gomez", "nickname": "KO King", "height_cm": 178.0},
            {"fighter_id": OTHER_JOEY, "name": "Joey Gomez", "nickname": None, "height_cm": 175.0, "weight_lbs": 155.0},
            {"fighter_id": OPPONENT, "name": "Test Opponent", "nickname": None},
        ])
        fights, rounds = [], []
        for fighter_id, count, division in (
            (BULLDOG, 12, "Flyweight"), (BLINDADO, 11, "Middleweight"),
            (KO_KING, 2, "Bantamweight"), (OTHER_JOEY, 1, "Lightweight"),
        ):
            name = "Bruno Silva" if fighter_id in (BULLDOG, BLINDADO) else "Joey Gomez"
            for number in range(count):
                fight_id = f"{len(fights) + 1:016x}"
                fights.append({
                    "fight_id": fight_id, "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=len(fights)),
                    "red_fighter_id": fighter_id, "blue_fighter_id": OPPONENT,
                    "red_name": "Bruno A Silva" if fighter_id == BULLDOG and number == 0 else name,
                    "blue_name": "Test Opponent", "winner": "red", "method": "Decision - Unanimous",
                    "finish_round": 1, "finish_time": "5:00", "weightclass": division,
                })
                for participant in (fighter_id, OPPONENT):
                    rounds.append({"fight_id": fight_id, "fighter_id": participant, "round": 1,
                                   "time_format": "1 Rnd (5)", "sig_str_landed": 10,
                                   "sig_str_attempted": 20, "td_landed": 1, "td_attempted": 2})
        self.fights = pd.DataFrame(fights)
        self.rounds = pd.DataFrame(rounds)
        legacy_fights = self.fights.copy()
        legacy_fights.loc[legacy_fights.red_fighter_id.eq(BULLDOG), "red_fighter_id"] = BLINDADO
        legacy_snapshots = self.registry.copy()
        legacy_snapshots["height_cm"] = 999.0
        self.predictor = SimpleNamespace(norm_fights=legacy_fights, fights_df=None,
                                         snapshots=legacy_snapshots, is_ready=lambda: True)
        self.write_bundle()

    def write_bundle(self):
        hashes = {}
        for name, frame in (("fights", self.fights), ("rounds", self.rounds), ("fighters", self.registry)):
            path = self.bundle / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {"schemaVersion": 1, "dataThrough": str(self.fights.date.max().date()),
                    "source": "Verified identity correction fixture", "sourceUrl": "https://github.com/example/source",
                    "retrievedAt": datetime.now(timezone.utc).isoformat(), "files": hashes}
        (self.bundle / "manifest.json").write_text(json.dumps(manifest))

    def load(self, directory=None):
        return ResearchData(self.predictor, str(self.root / "missing-model.pkl"),
                            bundle_directory=directory or self.bundle)

    def test_recent_fights_use_unique_canonical_fighters_in_date_order(self):
        research = self.load()
        payload = research.recent_fights()
        fights = payload["fights"]
        self.assertEqual(len(fights), 8)
        self.assertEqual(len({fight["fightId"] for fight in fights}), 8)
        self.assertEqual([fight["date"] for fight in fights], sorted((fight["date"] for fight in fights), reverse=True))
        self.assertEqual(fights[0]["firstFighter"], "Joey Gomez (Lightweight)")
        names = research.get_fighter_names()
        for fight in fights:
            self.assertIn(fight["firstFighter"], names)
            self.assertIn(fight["secondFighter"], names)
            self.assertEqual(fight["winner"], fight["firstFighter"])
            self.assertTrue(fight["url"].startswith("http://ufcstats.com/fight-details/"))
        self.assertEqual(payload["dataThrough"], research.data_through)
        research.fighter_labels_by_id.clear()
        self.assertEqual(research.recent_fights()["fights"], [])

    def test_duplicate_names_keep_separate_histories_and_physical_profiles(self):
        research = self.load()
        labels = research.get_fighter_names()
        self.assertIn("Bruno Silva (Bulldog)", labels)
        self.assertIn("Bruno Silva (Blindado)", labels)
        self.assertIn("Joey Gomez (KO King)", labels)
        self.assertIn("Joey Gomez (Lightweight)", labels)
        first, second = research.matchup_evidence("Bruno Silva (Bulldog)", "Bruno Silva (Blindado)")["fighters"]
        self.assertEqual((first["fighterId"], first["totalRecordedBouts"]), (BULLDOG, 12))
        self.assertEqual((second["fighterId"], second["totalRecordedBouts"]), (BLINDADO, 11))
        self.assertEqual((first["physical"]["heightCm"], second["physical"]["heightCm"]), (163.0, 183.0))
        self.assertEqual(first["recentMetrics"]["coveredBouts"], 5)
        self.assertTrue(all(fight["weightClass"] == "Flyweight" for fight in first["recentFive"]["fights"]))
        self.assertEqual(research.history("Joey Gomez (KO King)")["totalFights"], 2)
        self.assertEqual(research.history("Joey Gomez (Lightweight)")["totalFights"], 1)
        self.assertEqual(self.predictor.norm_fights.red_fighter_id.eq(BLINDADO).sum(), 23)

    def test_aliases_resolve_by_id_and_bare_duplicate_names_fail_closed(self):
        research = self.load()
        self.assertEqual(research.history("  BRUNO A SILVA  ")["fighterId"], BULLDOG)
        self.assertIsNone(research.history("Bruno Silva"))
        self.assertIsNone(research.history("Joey Gomez"))
        with self.assertRaisesRegex(ValueError, "multiple fighters"):
            research.matchup_evidence("Bruno Silva", "Test Opponent")
        with self.assertRaisesRegex(ValueError, "different fighters"):
            research.matchup_evidence("Bruno A Silva", "Bruno Silva (Bulldog)")

    def test_row_order_cannot_change_labels_histories_or_id_maps(self):
        original = self.load()
        self.fights = self.fights.sample(frac=1, random_state=12)
        self.rounds = self.rounds.sample(frac=1, random_state=13)
        self.registry = self.registry.sample(frac=1, random_state=14)
        self.write_bundle()
        shuffled = self.load()
        self.assertEqual(original.fighters, shuffled.fighters)
        self.assertEqual(original.histories, shuffled.histories)
        self.assertEqual(original.fighter_labels_by_id, shuffled.fighter_labels_by_id)

    def test_nickname_collision_uses_exact_id_and_missing_profile_stays_missing(self):
        self.registry.loc[self.registry.name.eq("Bruno Silva"), "nickname"] = "Same nickname"
        self.registry.loc[self.registry.fighter_id.eq(BULLDOG), "height_cm"] = None
        self.write_bundle()
        research = self.load()
        labels = research.fighter_labels_by_id
        self.assertIn(BULLDOG, labels[BULLDOG])
        self.assertIn(BLINDADO, labels[BLINDADO])
        self.assertIsNone(research.matchup_evidence(labels[BULLDOG], labels[BLINDADO])["fighters"][0]["physical"]["heightCm"])

    def test_registry_weight_distinguishes_duplicate_without_recorded_history(self):
        self.fights = self.fights[self.fights.red_fighter_id.ne(OTHER_JOEY)]
        self.rounds = self.rounds[self.rounds.fight_id.isin(self.fights.fight_id)]
        self.write_bundle()
        research = self.load()
        self.assertEqual(research.fighter_labels_by_id[OTHER_JOEY], "Joey Gomez (155 lb)")
        self.assertEqual(research.history("Joey Gomez (155 lb)")["totalFights"], 0)
        self.registry.loc[self.registry.fighter_id.eq(OTHER_JOEY), "weight_lbs"] = None
        self.write_bundle()
        self.assertEqual(self.load().fighter_labels_by_id[OTHER_JOEY], f"Joey Gomez ({OTHER_JOEY})")

    def test_validated_manifest_identity_is_exposed(self):
        research = self.load()
        self.assertEqual(research.research_bundle_directory, self.bundle.resolve())
        self.assertEqual(research.research_manifest_sha256, hashlib.sha256((self.bundle / "manifest.json").read_bytes()).hexdigest())
        self.assertTrue(research.research_status()["available"])

    def test_bout_time_formats_can_be_joined_from_exact_fight_ids(self):
        original = self.load()
        self.fights["time_format"] = "1 Rnd (5)"
        self.rounds = self.rounds.drop(columns="time_format").sample(frac=1, random_state=17)
        self.write_bundle()
        source_hash = hashlib.sha256((self.bundle / "rounds.parquet").read_bytes()).hexdigest()
        joined = self.load()
        self.assertEqual(joined.refresh_status, "loaded")
        self.assertEqual(joined.round_stats, original.round_stats)
        self.assertEqual(source_hash, hashlib.sha256((self.bundle / "rounds.parquet").read_bytes()).hexdigest())
        self.fights = self.fights.drop(columns="time_format")
        self.write_bundle()
        missing = self.load()
        self.assertEqual(missing.refresh_status, "invalid")
        self.assertIsNone(missing.research_fights)

    def test_rounds_cannot_borrow_an_unrelated_fights_time_format(self):
        self.fights["time_format"] = "1 Rnd (5)"
        self.rounds = self.rounds.drop(columns="time_format")
        self.rounds.loc[0, "fight_id"] = "ffffffffffffffff"
        self.write_bundle()
        research = self.load()
        self.assertEqual(research.refresh_status, "invalid")
        self.assertEqual(research.round_stats, {})

    def test_explicit_missing_or_corrupted_bundle_never_loads_legacy_data(self):
        for directory in (self.root / "absent", self.bundle):
            if directory == self.bundle:
                (self.bundle / "fighters.parquet").write_bytes(b"invalid checksum")
            with self.subTest(directory=directory):
                research = self.load(directory)
                self.assertEqual(research.refresh_status, "invalid")
                self.assertFalse(research.research_status()["available"])
                self.assertIsNone(research.research_fights)
                self.assertIsNone(research.data_through)
                self.assertIsNone(research.research_manifest_sha256)
                self.assertIsNone(research.research_bundle_directory)
                self.assertEqual(research.get_fighter_names(), [])
                self.assertEqual(research.snapshot_by_id, {})
                self.assertEqual(research.histories, {})

    def test_duplicate_registry_ids_and_unknown_participants_are_rejected(self):
        original = self.registry.copy()
        for malformed in (pd.concat([original, original.iloc[:1]], ignore_index=True),
                          original[original.fighter_id.ne(BULLDOG)]):
            self.registry = malformed
            self.write_bundle()
            research = self.load()
            self.assertEqual(research.refresh_status, "invalid")
            self.assertEqual(research.get_fighter_names(), [])


if __name__ == "__main__":
    unittest.main()
