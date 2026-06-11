import unittest

import numpy as np
import pandas as pd

from scripts import evaluate_grappling as candidate
from scripts import evaluate_research as base


def fixtures() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fights, rounds = [], []
    for index, (date, landed, attempts) in enumerate((("2020-01-01", 1, 1), ("2020-02-01", 0, 9), ("2020-03-01", 0, 0))):
        fight_id = str(index)
        fights.append({"fight_id": fight_id, "date": pd.Timestamp(date), "red_fighter_id": "a", "blue_fighter_id": "b",
                       "finish_time": "5:00", "finish_round": 1, "num_rounds": 3, "time_format": "3 Rnd (5-5-5)"})
        for fid in ("a", "b"):
            stats = {column: 0 for column in base.STAT_COLS}
            stats.update(sig_str_landed=3, sig_str_attempted=5, td_landed=landed if fid == "a" else 0,
                         td_attempted=attempts if fid == "a" else 0)
            rounds.append({"fight_id": fight_id, "fighter_id": fid, "round": 1, **stats})
    frame = pd.DataFrame({"fight_id": ["0", "1", "2"], **{name: [np.nan] * 3 for name in candidate.REPLACED},
                          "diff_unchanged": [1.0, 2.0, 3.0], "ctx_unchanged": [3, 3, 3]})
    return frame, pd.DataFrame(fights), pd.DataFrame(rounds)


class GrapplingFeatureTests(unittest.TestCase):
    def test_pooled_counts_instead_of_mean_ratios(self) -> None:
        frame, fights, rounds = fixtures()
        result = candidate.add_features(frame, fights, rounds)
        self.assertAlmostEqual(result.loc[2, "diff_td_accuracy_last3"], 2 / 12 - .5)
        self.assertAlmostEqual(result.loc[2, "diff_td_defense_last3"], .5 - 10 / 12)
        self.assertAlmostEqual(result.loc[2, "diff_td_attack_exposure_last3"], np.log1p(10))
        self.assertEqual(result.loc[2, "ctx_min_td_attack_exposure_last3"], 0)

    def test_no_history_remains_missing(self) -> None:
        frame, fights, rounds = fixtures()
        result = candidate.add_features(frame, fights, rounds)
        self.assertTrue(result.loc[0, candidate.REPLACED].isna().all())
        self.assertEqual(candidate.state([None], 3)["attack_exposure"], 0)
        self.assertTrue(np.isnan(candidate.state([None], 3)["attack"]))

    def test_valid_zero_attempt_is_prior_with_zero_exposure(self) -> None:
        zero = {"attack_attempted": 0, "attack_successes": 0, "defense_attempted": 0, "defense_successes": 0}
        state = candidate.state([zero], 3)
        self.assertEqual(state["attack"], .5)
        self.assertEqual(state["defense"], .5)
        self.assertEqual(state["attack_exposure"], 0)

    def test_same_day_and_future_do_not_enter_history(self) -> None:
        frame, fights, rounds = fixtures()
        fights.loc[1, "date"] = fights.loc[2, "date"]
        result = candidate.add_features(frame, fights, rounds)
        pd.testing.assert_series_equal(result.loc[1, candidate.REPLACED + candidate.ADDED], result.loc[2, candidate.REPLACED + candidate.ADDED], check_names=False)
        changed = rounds.copy()
        changed.loc[changed.fight_id.isin(["1", "2"]), ["td_landed", "td_attempted"]] = [10, 15]
        pd.testing.assert_frame_equal(result, candidate.add_features(frame, fights, changed))

    def test_invalid_stats_are_not_zero_attempts(self) -> None:
        frame, fights, rounds = fixtures()
        rounds.loc[rounds.fight_id == "0", "td_landed"] = 100
        result = candidate.add_features(frame, fights, rounds)
        self.assertTrue(result.loc[1, candidate.REPLACED].isna().all())

    def test_unrelated_columns_and_row_order_preserved(self) -> None:
        frame, fights, rounds = fixtures()
        result = candidate.add_features(frame, fights, rounds)
        unchanged = [name for name in frame if name not in candidate.REPLACED]
        pd.testing.assert_frame_equal(frame[unchanged], result[unchanged])
        shuffled = frame.sample(frac=1, random_state=4).reset_index(drop=True)
        rebuilt = candidate.add_features(shuffled, fights.sample(frac=1, random_state=3), rounds.sample(frac=1, random_state=2))
        pd.testing.assert_frame_equal(result.set_index("fight_id").sort_index(), rebuilt.set_index("fight_id").sort_index())

    def test_corner_swap_negates_only_differences(self) -> None:
        frame, fights, rounds = fixtures()
        expected = candidate.add_features(frame, fights, rounds)
        reverse = fights.rename(columns={"red_fighter_id": "blue_fighter_id", "blue_fighter_id": "red_fighter_id"})
        actual = candidate.add_features(base.swapped(frame), reverse, rounds)
        pd.testing.assert_frame_equal(base.swapped(expected), actual)

    def test_window_counts_include_invalid_bouts(self) -> None:
        known = {"attack_attempted": 10, "attack_successes": 5, "defense_attempted": 5, "defense_successes": 3}
        self.assertTrue(np.isnan(candidate.state([known, None, None, None], 3)["attack"]))
        self.assertEqual(candidate.state([known, None, None, None], 5)["attack"], .5)


if __name__ == "__main__":
    unittest.main()
