import hashlib
import json
import unittest
from datetime import datetime, timezone

import pandas as pd

from tests import test_research_identity as fixtures
from research import ResearchData


class ResearchCareerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ResearchIdentityTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.directory = self.fixture.root / "career"
        self.directory.mkdir()
        self.retrieved_at = datetime.now(timezone.utc).isoformat()
        self.rows = []
        for index in range(15):
            self.rows.append(self.row(index + 1, f"2025-01-{index + 1:02d}"))
        self.rows.extend([
            self.row(16, "2025-12-31", result="L", method="Submission (Armbar)"),
            self.row(17, "2026-01-01", result="D", method="Decision - Draw"),
            self.row(18, "2026-01-26", result="NC", method="No Contest"),
            self.row(19, "2026-01-27"),
        ])
        self.frame = pd.DataFrame(self.rows)
        self.write_bundle()

    def row(self, number, date, *, result="W", method="TKO (Punches)"):
        return {"fighter_id": fixtures.BULLDOG, "opponent_id": f"{1000 + number:016x}",
                "canonical_fighter_id": 1, "canonical_opponent_id": 1000 + number,
                "canonical_fight_id": number, "canonical_event_id": number,
                "date": date, "result": result, "opponent_result": {"W": "L", "L": "W", "D": "D", "NC": "NC"}[result],
                "fighter_name": "Bruno Silva", "opponent_name": f"Regional opponent {number}",
                "promotion": "Regional Promotion", "event_name": f"Regional Event {number}", "method": method,
                "source_aliases_json": json.dumps([{"source": "sherdog", "sourceID": str(number),
                                                    "sourceURL": f"https://www.sherdog.com/events/Event-{number}"}]),
                "source_database_retrieved_at": self.retrieved_at}

    def write_bundle(self):
        path = self.directory / "fighter_bouts.parquet"
        self.frame.to_parquet(path, index=False)
        self.manifest = {"schemaVersion": 1, "bundleType": "career-supplement", "cutoffInclusive": "2026-10-31",
                         "knownAt": "2026-11-03T22:06:00Z",
                         "source": {"name": "Fight Canon", "url": "https://github.com/Jadevit/fight-canon-db/releases/tag/db-2026-11-03-2206",
                                    "retrievedAt": self.retrieved_at, "databaseSha256": "a" * 64},
                         "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}}
        content = json.dumps(self.manifest).encode()
        (self.directory / "manifest.json").write_bytes(content)
        self.digest = hashlib.sha256(content).hexdigest()

    def load(self, expected_hash=None):
        return ResearchData(self.fixture.predictor, str(self.fixture.root / "missing-model.pkl"),
                            bundle_directory=self.fixture.bundle, career_directory=self.directory,
                            career_manifest_sha256=self.digest if expected_hash is None else expected_hash)

    def test_supplement_is_separate_capped_and_exposed_in_both_research_responses(self):
        baseline = self.fixture.load()
        research = self.load()
        self.assertEqual(research.histories, baseline.histories)
        self.assertEqual(research.round_stats, baseline.round_stats)
        career = research.career_history(fixtures.BULLDOG)
        self.assertTrue(career["available"])
        self.assertEqual(career["scope"], "additional_professional_results")
        self.assertEqual(career["asOf"], "2026-01-26")
        self.assertEqual(career["totalRecordedBouts"], 18)
        self.assertEqual(career["record"], {"wins": 15, "losses": 1, "draws": 1, "noContests": 1, "unknown": 0})
        self.assertEqual(career["winMethods"]["koTko"], 15)
        self.assertEqual(career["promotions"], [{"name": "Regional Promotion", "bouts": 18}])
        self.assertEqual(len(career["recentBouts"]), 12)
        self.assertFalse(career["coverage"]["completeCareer"])
        self.assertFalse(career["coverage"]["historicalAvailabilityVerified"])
        self.assertEqual(career["source"]["manifestSha256"], self.digest)
        self.assertEqual(research.history("Bruno Silva (Bulldog)")["careerHistory"], career)
        evidence = research.matchup_evidence("Bruno Silva (Bulldog)", "Bruno Silva (Blindado)")
        self.assertEqual(evidence["fighters"][0]["careerHistory"], career)
        self.assertEqual(evidence["fighters"][0]["totalRecordedBouts"], 12)

    def test_explicit_event_date_excludes_same_day_and_later_results(self):
        career = self.load().career_history(fixtures.BULLDOG, as_of="2026-01-01")
        self.assertEqual(career["asOf"], "2025-12-31")
        self.assertEqual(career["beforeEventDate"], "2026-01-01")
        self.assertEqual(career["totalRecordedBouts"], 16)
        self.assertEqual(career["record"]["draws"], 0)
        self.assertTrue(all(row["date"] < "2026-01-01" for row in career["recentBouts"]))
        with self.assertRaises(ValueError):
            self.load().career_history(fixtures.BULLDOG, as_of="2026-02-30")

    def test_absent_or_bad_hash_disables_only_supplement(self):
        self.assertFalse(self.fixture.load().career_history(fixtures.BULLDOG)["available"])
        for digest in ("0" * 64, ""):
            with self.subTest(digest=digest):
                research = self.load(expected_hash=digest)
                self.assertFalse(research.career_history(fixtures.BULLDOG)["available"])
                self.assertTrue(research.research_status()["available"])
                self.assertIsNotNone(research.research_status()["career"]["loadError"])
                self.assertEqual(research.history("Bruno Silva (Bulldog)")["totalFights"], 12)

    def test_corrupt_career_file_is_rejected_without_falling_back(self):
        (self.directory / "fighter_bouts.parquet").write_bytes(b"corrupt")
        research = self.load()
        self.assertFalse(research.career_history(fixtures.BULLDOG)["available"])
        self.assertEqual(research.career_histories, {})

    def test_duplicate_and_ambiguous_same_day_bouts_are_rejected(self):
        original = self.frame.copy()
        for duplicate in (self.frame.iloc[0].to_dict(), {**self.frame.iloc[0].to_dict(), "canonical_fight_id": 999}):
            self.frame = pd.concat([original, pd.DataFrame([duplicate])], ignore_index=True)
            self.write_bundle()
            self.assertFalse(self.load().career_history(fixtures.BULLDOG)["available"])

    def test_conflicting_outcomes_and_canonical_identity_are_rejected(self):
        original = self.frame.copy()
        for field, value in (("opponent_result", "W"), ("canonical_fighter_id", 2)):
            self.frame = original.copy()
            self.frame.loc[0, field] = value
            self.write_bundle()
            self.assertFalse(self.load().career_history(fixtures.BULLDOG)["available"])

    def test_both_participant_rows_must_agree_on_winner(self):
        row = self.frame.iloc[0].to_dict()
        mirror = {**row, "fighter_id": row["opponent_id"], "opponent_id": row["fighter_id"],
                  "canonical_fighter_id": row["canonical_opponent_id"], "canonical_opponent_id": row["canonical_fighter_id"],
                  "fighter_name": row["opponent_name"], "opponent_name": row["fighter_name"], "result": "L", "opponent_result": "W"}
        self.frame = pd.concat([self.frame, pd.DataFrame([mirror])], ignore_index=True)
        self.write_bundle()
        self.assertTrue(self.load().career_history(fixtures.BULLDOG)["available"])
        self.frame.loc[self.frame.index[-1], ["result", "opponent_result"]] = ["W", "L"]
        self.write_bundle()
        self.assertFalse(self.load().career_history(fixtures.BULLDOG)["available"])

    def test_ufc_source_alias_and_exact_pair_overlap_are_rejected(self):
        original = self.frame.copy()
        known = self.fixture.fights.iloc[0]
        self.frame.loc[0, "source_aliases_json"] = json.dumps([{"source": "ufcstats", "sourceID": known.fight_id}])
        self.write_bundle()
        self.assertFalse(self.load().career_history(fixtures.BULLDOG)["available"])
        self.frame = original
        self.frame.loc[0, "date"] = str(known.date.date())
        self.frame.loc[0, "opponent_id"] = known.blue_fighter_id
        self.write_bundle()
        self.assertFalse(self.load().career_history(fixtures.BULLDOG)["available"])

    def test_order_invariance_and_untrusted_source_urls(self):
        self.frame.loc[0, "source_aliases_json"] = json.dumps([{"source": "sherdog", "sourceID": "1", "sourceURL": "javascript:alert(1)"}])
        self.write_bundle()
        original = self.load()
        self.assertIsNone(original.career_histories[fixtures.BULLDOG][-1]["sourceUrl"])
        self.frame = self.frame.sample(frac=1, random_state=42)
        self.write_bundle()
        self.assertEqual(original.career_histories, self.load().career_histories)


if __name__ == "__main__":
    unittest.main()
