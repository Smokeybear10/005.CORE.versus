import unittest

import numpy as np
import pandas as pd

from scripts import evaluate_context_normalization as candidate
from scripts import evaluate_research as base


def fixtures() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fights, rounds = [], []
    for index, date in enumerate(("2020-01-01", "2020-02-01", "2020-03-01")):
        fights.append({"fight_id": str(index), "event_id": f"event{index}", "date": pd.Timestamp(date),
                       "red_fighter_id": "a", "blue_fighter_id": "b", "red_name": "A", "blue_name": "B",
                       "winner": "red", "weightclass": "Ultimate Fighter 33 Flyweight Tournament" if index == 0 else "Flyweight",
                       "is_title_bout": index == 0, "finish_time": "3:00", "finish_round": 2, "num_rounds": 3,
                       "time_format": candidate.OVERTIME_FORMAT if index == 0 else "3 Rnd (5-5-5)"})
        for fid in ("a", "b"):
            for rnd in (1, 2):
                stats = {name: 0 for name in base.STAT_COLS}
                stats.update(sig_str_landed=3 if fid == "a" else 1, sig_str_attempted=5, td_landed=0, td_attempted=1)
                rounds.append({"fight_id": str(index), "fighter_id": fid, "round": rnd, **stats})
    fighters = pd.DataFrame({"fighter_id": ["a", "b"], "dob": pd.to_datetime(["1990-01-01", "1992-01-01"])})
    return pd.DataFrame(fights), pd.DataFrame(rounds), fighters


class ContextNormalizationTests(unittest.TestCase):
    def test_exact_standard_and_interim_labels(self) -> None:
        for division in candidate.DIVISIONS:
            self.assertEqual(candidate.normalized_division(division), (division, "standard"))
            self.assertEqual(candidate.normalized_division(f"Interim {division}"), (division, "interim"))

    def test_supported_tournament_grammar_preserves_gender_and_weight(self) -> None:
        examples = {
            "Ultimate Fighter 33 Flyweight Tournament": "Flyweight",
            "Ultimate Fighter Brazil 3 Heavyweight Tournament": "Heavyweight",
            "Ultimate Fighter Latin America 2 Lightweight Tournament": "Lightweight",
            "Ultimate Fighter China Featherweight Tournament": "Featherweight",
            "Ultimate Fighter Australia vs. UK Welterweight Tournament": "Welterweight",
            "TUF Nations Canada vs. Australia Middleweight Tournament": "Middleweight",
            "Road to 3 Women's Strawweight Tournament TitleBout": "Women's Strawweight",
            "Road To 1 Bantamweight Tournament": "Bantamweight",
        }
        for value, division in examples.items():
            self.assertEqual(candidate.normalized_division(value), (division, "tournament"))

    def test_unknown_and_malformed_labels_are_not_guessed(self) -> None:
        for value in (None, np.nan, 155, "", "Flyweight ", "flyweight", "Interim Interim Flyweight",
                      "Super Heavyweight", "13 Lightweight Tournament", "Catch Weight", "Open Weight",
                      "Unknown Flyweight", "Ultimate Fighter Flyweight Tournament", "Ultimate Fighter 0 Flyweight Tournament",
                      "Ultimate Fighter 33 Flyweight Tournament extra", "prefix Ultimate Fighter 33 Flyweight Tournament",
                      "Road to 3 Mens Strawweight Tournament", "Road To 1  Bantamweight Tournament"):
            self.assertEqual(candidate.normalized_division(value), ("Unknown", "unknown"))

    def test_original_source_title_and_audit_labels_preserved(self) -> None:
        fights, _, _ = fixtures()
        original = fights.copy(deep=True)
        view, metadata = candidate.context_view(fights)
        pd.testing.assert_frame_equal(fights, original)
        pd.testing.assert_series_equal(view.is_title_bout, fights.is_title_bout)
        self.assertEqual(view.loc[0, "weightclass"], "Flyweight")
        self.assertEqual(metadata.loc[0, "raw_weightclass"], original.loc[0, "weightclass"])
        self.assertEqual(metadata.loc[0, "competition_kind"], "tournament")

    def test_overtime_duration_uses_only_completed_five_minute_rounds(self) -> None:
        fights, rounds, fighters = fixtures()
        old = candidate.features(fights, rounds, fighters, normalize=False)
        new = candidate.features(fights, rounds, fighters)
        self.assertEqual(old.loc[1, "ctx_min_stats_last5"], 0)
        self.assertEqual(new.loc[1, "ctx_min_stats_last5"], 1)
        self.assertAlmostEqual(new.loc[1, "diff_sig_landed_pm_last3"], (6 - 2) / 8)
        candidate.assert_compatible(old, new)

    def test_fourth_round_other_format_or_wrong_schedule_not_admitted(self) -> None:
        fights, _, _ = fixtures()
        for field, value in (("finish_round", 4), ("finish_round", 1.5), ("num_rounds", 5),
                             ("time_format", "3 Rnd + OT (5-5-5-10)"), ("time_format", "3 Rnd + OT (5-5-5-5) ")):
            altered = fights.copy()
            if isinstance(value, float):
                altered[field] = altered[field].astype(float)
            altered.loc[0, field] = value
            view, metadata = candidate.context_view(altered)
            self.assertEqual(view.loc[0, "time_format"], altered.loc[0, "time_format"])
            self.assertFalse(metadata.loc[0, "completed_before_optional_overtime"])

    def test_invalid_or_incomplete_round_stats_still_rejected(self) -> None:
        fights, rounds, fighters = fixtures()
        examples = []
        examples.append(rounds[~((rounds.fight_id == "0") & (rounds.fighter_id == "a") & (rounds["round"] == 2))])
        examples.append(pd.concat([rounds, rounds.iloc[[0]]], ignore_index=True))
        invalid = rounds.copy()
        invalid.loc[invalid.fight_id == "0", "td_landed"] = 100
        examples.append(invalid)
        for altered in examples:
            result = candidate.features(fights, altered, fighters)
            self.assertEqual(result.loc[1, "ctx_min_stats_last5"], 0)

    def test_prefix_and_same_day_no_future_leakage(self) -> None:
        fights, rounds, fighters = fixtures()
        full = candidate.features(fights, rounds, fighters)
        prefix = candidate.features(fights, rounds, fighters, "2020-03-01")
        pd.testing.assert_frame_equal(prefix, full.iloc[:2].reset_index(drop=True))
        fights.loc[1, "date"] = fights.loc[2, "date"]
        before = candidate.features(fights, rounds, fighters)
        changed = rounds.copy()
        changed.loc[changed.fight_id.isin(["1", "2"]), ["sig_str_landed", "sig_str_attempted"]] = [10, 20]
        pd.testing.assert_frame_equal(before, candidate.features(fights, changed, fighters))
        selected = [column for column in before if column.startswith(("diff_", "ctx_"))]
        pd.testing.assert_series_equal(before.loc[1, selected], before.loc[2, selected], check_names=False)

    def test_corner_reversal_preserves_context_and_negates_differences(self) -> None:
        fights, rounds, fighters = fixtures()
        forward = candidate.features(fights, rounds, fighters)
        reverse = fights.rename(columns={"red_fighter_id": "blue_fighter_id", "blue_fighter_id": "red_fighter_id",
                                         "red_name": "blue_name", "blue_name": "red_name"}).copy()
        reverse["winner"] = reverse.winner.map({"red": "blue", "blue": "red", "draw": "draw", "nc": "nc"})
        backward = candidate.features(reverse, rounds, fighters)
        columns = [name for name in forward if name.startswith(("diff_", "ctx_"))] + ["weightclass"]
        pd.testing.assert_frame_equal(base.swapped(forward[columns]), backward[columns])


if __name__ == "__main__":
    unittest.main()
