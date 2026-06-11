import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import career_features as career
from scripts import evaluate_challenger as challenger
from scripts import evaluate_research as base


A, B, C = "a" * 16, "b" * 16, "c" * 16
FIGHT = "1" * 16


def fights() -> pd.DataFrame:
    return pd.DataFrame([{"fight_id": FIGHT, "date": pd.Timestamp("2020-01-01"),
                          "red_fighter_id": A, "blue_fighter_id": B, "winner": "red", "method": "KO/TKO"}])


def bout(fighter=A, canonical=1, opponent=99, fight=100, date="2020-02-01", result="W", method="Submission") -> dict:
    return {"fighter_id": fighter, "opponent_id": None, "canonical_fighter_id": canonical,
            "canonical_opponent_id": opponent, "canonical_fight_id": fight, "date": date,
            "result": result, "opponent_result": {"W": "L", "L": "W", "D": "D", "NC": "NC"}[result],
            "method": method, "source_aliases_json": json.dumps([{"source": "sherdog", "sourceID": str(fight)}])}


def request(date="2020-06-01", red=A, blue=B) -> pd.DataFrame:
    return pd.DataFrame([{"date": pd.Timestamp(date), "red_fighter_id": red, "blue_fighter_id": blue,
                          "diff_career_wins": 1.0, "diff_career_losses": -1.0, "diff_elo": 32.0,
                          "ctx_min_career": 1.0, "weightclass": "Lightweight"}])


class CareerFeatureTests(unittest.TestCase):
    def test_future_and_same_date_bouts_never_change_features(self):
        prior = pd.DataFrame([bout()])
        full = pd.DataFrame([bout(), bout(fight=101, date="2020-06-01", result="L"),
                             bout(fight=102, date="2025-01-01", result="L")])
        expected = career.add_career_features(request(), fights(), prior)
        actual = career.add_career_features(request(), fights(), full)
        pd.testing.assert_frame_equal(expected, actual)
        source = pd.concat([fights(), fights().assign(fight_id="2" * 16, date=pd.Timestamp("2020-06-01"), winner="blue")])
        pd.testing.assert_frame_equal(expected, career.add_career_features(request(), source, prior))

    def test_original_features_and_corner_symmetry_are_preserved(self):
        original = request()
        supplement = pd.DataFrame([bout(), bout(fighter=B, canonical=2, fight=101, result="L", method="Decision")])
        result = career.add_career_features(original, fights(), supplement)
        self.assertFalse(set(career.ADDED_FEATURES) & set(original))
        pd.testing.assert_frame_equal(result[original.columns], original)
        reversed_request = base.swapped(original).assign(red_fighter_id=B, blue_fighter_id=A)
        reversed_result = career.add_career_features(reversed_request, fights(), supplement)
        np.testing.assert_allclose(reversed_result[career.ADDED_FEATURES], base.swapped(result)[career.ADDED_FEATURES], equal_nan=True)
        self.assertEqual(len(career.ADDED_FEATURES), 12)

    def test_draws_nc_and_unknown_methods_have_distinct_denominators(self):
        supplement = pd.DataFrame([bout(result="D", fight=101), bout(result="NC", fight=102, date="2020-03-01"),
                                   bout(result="W", method=None, fight=103, date="2020-04-01"),
                                   bout(result="L", method="Decision", fight=104, date="2020-05-01")])
        state = career._state(career.prepare_history(fights(), supplement)[A], pd.Timestamp("2020-06-01"))
        self.assertEqual((state["bouts"], state["wins"], state["losses"], state["supplement_bouts"]), (5, 2, 1, 4))
        self.assertAlmostEqual(state["smoothed_score"], 3.5 / 6)
        self.assertEqual(state["finish_win_rate"], 1.0)
        self.assertEqual(state["finish_loss_rate"], 0.0)
        self.assertEqual(state["recent365_bouts"], 5)

    def test_known_side_retained_when_opponent_id_is_missing(self):
        source = fights().assign(blue_fighter_id=None)
        empty = pd.DataFrame([bout()]).iloc[:0]
        history = career.prepare_history(source, empty)
        state = career._state(history[A], pd.Timestamp("2020-02-01"))
        self.assertEqual((state["bouts"], state["wins"]), (1, 1))
        self.assertEqual(set(history), {A})

    def test_no_observed_history_is_not_a_zero_finish_rate(self):
        state = career._state(None, pd.Timestamp("2020-01-01"))
        self.assertEqual(state["bouts"], 0)
        self.assertEqual(state["smoothed_score"], 0.5)
        self.assertTrue(np.isnan(state["finish_win_rate"]))
        self.assertTrue(np.isnan(state["finish_loss_rate"]))
        self.assertTrue(np.isnan(state["layoff_days"]))

    def test_dynamic_wrapper_preserves_all48_features_and_quality(self):
        date = pd.Timestamp("2020-06-01")
        state = base.fighter_state([], 1500.0, date)
        context = challenger.context_state([], date, pd.NaT)
        original = pd.DataFrame([{**base.matchup_features(state, state, "Lightweight", 3, False),
                                  **challenger.matchup_context(context, context, 1500.0, 1500.0)}])
        self.assertEqual(len(original.columns), 48)
        quality = {"fighter1": {"priorBouts": 0}, "fighter2": {"priorBouts": 0}}
        with patch.object(challenger, "build_inference_features", return_value=(original, quality)):
            result, actual_quality = career.build_inference_features(fights(), pd.DataFrame(), pd.DataFrame(),
                pd.DataFrame([bout()]), A, B, str(date.date()), "Lightweight", 3, False)
        self.assertEqual(len(result.columns), 60)
        pd.testing.assert_frame_equal(result[original.columns], original)
        self.assertIs(actual_quality, quality)

    def test_recent_window_includes_boundary_but_excludes_request_day(self):
        date = pd.Timestamp("2021-06-01")
        supplement = pd.DataFrame([bout(date=str((date - pd.Timedelta(days=365)).date())),
                                   bout(fight=101, date=str((date - pd.Timedelta(days=366)).date()), result="L"),
                                   bout(fight=102, date=str(date.date()), result="L")])
        state = career._state(career.prepare_history(fights(), supplement)[A], date)
        self.assertEqual(state["recent365_bouts"], 1)
        self.assertAlmostEqual(state["recent365_score"], 2 / 3)
        self.assertEqual(state["layoff_days"], 365)

    def test_history_row_order_does_not_change_features(self):
        supplement = pd.DataFrame([bout(fight=i, date=f"2020-0{i - 98}-01") for i in range(100, 104)])
        expected = career.add_career_features(request(), fights(), supplement)
        actual = career.add_career_features(request(), fights().iloc[::-1], supplement.sample(frac=1, random_state=42))
        pd.testing.assert_frame_equal(expected, actual)

    def test_duplicates_and_existing_alias_or_pair_overlap_are_rejected(self):
        for rows, message in [([bout(), bout()], "Duplicate"),
                              ([bout(), bout(fight=101)], "same-pair/date"),
                              ([dict(bout(), source_aliases_json=json.dumps([{"source": "ufcstats", "sourceID": FIGHT}]))], "UFCStats"),
                              ([dict(bout(date="2020-01-01"), opponent_id=B, canonical_opponent_id=2)], "pair/date")]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                career.prepare_history(fights(), pd.DataFrame(rows))

    def test_conflicting_crosswalk_outcome_and_empty_provenance_are_rejected(self):
        cases = [([dict(bout(), opponent_result="W")], "outcomes"),
                 ([dict(bout(), source_aliases_json="[]")], "aliases"),
                 ([bout(), bout(fight=101, canonical=3, date="2020-03-01")], "crosswalk"),
                 ([dict(bout(), opponent_id=B), bout(fighter=B, canonical=2, fight=101)], "opponent crosswalk"),
                 ([bout(), bout(fighter=B, canonical=2, opponent=1, fight=100, result="W")], "participants or outcomes"),
                 ([bout(opponent=2), bout(fighter=B, canonical=2, opponent=1, result="L", method="Decision")], "methods")]
        for rows, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                career.prepare_history(fights(), pd.DataFrame(rows))

    def test_invalid_dates_and_noninteger_canonical_ids_are_rejected(self):
        for changed in [dict(date="not-a-date"), dict(date="2020-02-01T01:00:00"), dict(canonical_fight_id=True)]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                career.prepare_history(fights(), pd.DataFrame([dict(bout(), **changed)]))

    def test_source_checksum_is_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "fighter_bouts.parquet"
            pd.DataFrame([bout()]).to_parquet(path, index=False)
            manifest = {"schemaVersion": 1, "bundleType": "career-supplement", "cutoffInclusive": "2020-12-31",
                        "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}}
            (directory / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(len(career.load_supplement(directory)[0]), 1)
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                career.load_supplement(directory)


if __name__ == "__main__":
    unittest.main()
