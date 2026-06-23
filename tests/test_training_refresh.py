import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import evaluate_training_refresh as refresh


class TrainingRefreshTests(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        dates = ["2009-12-31", "2010-01-01", "2025-12-31", "2026-01-01", "2026-05-09", "2026-05-10", "2026-10-31", "2026-11-01"]
        return pd.DataFrame({"fight_id": [str(i) for i in range(len(dates))],
                             "event_id": [str(i) for i in range(len(dates))], "date": pd.to_datetime(dates)})

    def test_fixed_exclusive_boundaries(self) -> None:
        parts = refresh.split(self.frame())
        self.assertEqual(parts["fit"].fight_id.tolist(), ["1", "2"])
        self.assertEqual(parts["calibration"].fight_id.tolist(), ["3", "4"])
        self.assertEqual(parts["test"].fight_id.tolist(), ["5", "6"])

    def test_duplicate_identity_and_cross_split_events_fail(self) -> None:
        frame = self.frame()
        frame.loc[3, "fight_id"] = frame.loc[2, "fight_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            refresh.split(frame)
        frame = self.frame()
        frame.loc[3, "event_id"] = frame.loc[2, "event_id"]
        with self.assertRaisesRegex(ValueError, "crosses"):
            refresh.split(frame)

    def test_temperature_control_cannot_change_winners(self) -> None:
        raw = np.array([.1, .2, .4, .5, .6, .8, .9])
        target = np.array([0, 1, 0, 1, 1, 0, 1])
        temperature = refresh.calibrate(raw, target)
        self.assertGreater(temperature, 0)
        calibrated = refresh.base.apply_temperature(raw, temperature)
        np.testing.assert_array_equal(calibrated >= .5, raw >= .5)
        np.testing.assert_allclose(calibrated + refresh.base.apply_temperature(1 - raw, temperature), 1, atol=1e-12)

    def test_invalid_calibration_probability_rejected(self) -> None:
        for bad in [np.nan, np.inf, 0., 1.]:
            with self.subTest(value=bad), self.assertRaisesRegex(ValueError, "Invalid"):
                refresh.calibrate(np.array([.4, bad]), np.array([0, 1]))

    def test_actual_quote_day_excludes_intervening_and_same_day_fights(self) -> None:
        frame = pd.DataFrame({"fight_id": ["prior", "quote_day", "later", "event"],
                              "date": pd.to_datetime(["2025-08-20", "2025-08-21", "2025-08-22", "2025-08-23"])})
        prior, cutoff = refresh.quote_history(frame, pd.Series(["2025-08-22T00:00:00Z", "2025-08-21T23:30:00Z"]))
        self.assertEqual(cutoff, pd.Timestamp("2025-08-21"))
        self.assertEqual(prior.fight_id.tolist(), ["prior"])
        with self.assertRaisesRegex(ValueError, "timezone"):
            refresh.quote_history(frame, pd.Series(["2025-08-21T23:30:00"]))

    def test_existing_archive_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            sentinel = path / "original"
            sentinel.write_text("unchanged")
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                refresh.run(path)
            self.assertEqual(sentinel.read_text(), "unchanged")


if __name__ == "__main__":
    unittest.main()
