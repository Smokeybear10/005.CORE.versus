import unittest

import numpy as np
import pandas as pd
from scipy.special import logit

from scripts import evaluate_market_history as history


def candle(timestamp: int, midpoint: float, volume: object = "1.00", spread: float = .02) -> dict:
    return {"end_period_ts": timestamp, "volume": volume,
            "yes_bid": {"close": f"{midpoint - spread / 2:.4f}"},
            "yes_ask": {"close": f"{midpoint + spread / 2:.4f}"}}


class MarketHistoryTests(unittest.TestCase):
    def test_future_and_boundary_bars_cannot_change_features(self) -> None:
        red = [candle(60, .4), candle(120, .5)]
        blue = [candle(60, .6), candle(120, .5)]
        expected = history.history_features(red, blue, 0, 120, .5)
        actual = history.history_features([candle(0, .1, "9000"), *red, candle(180, .9, "9999")],
                                          [candle(0, .9, "1"), *blue, candle(180, .1, "1")], 0, 120, .5)
        self.assertEqual(expected, actual)
        self.assertAlmostEqual(expected["price_drift"], -logit(.4))

    def test_no_forward_fill_or_previous_trade_proxy(self) -> None:
        red = [candle(60, .4), candle(120, .5)]
        blue = [candle(120, .5)]
        red[0]["price"] = {"previous": ".99"}
        values = history.history_features(red, blue, 0, 120, .5)
        self.assertEqual(values["common_observations"], 1)
        self.assertEqual(values["price_drift"], 0)
        self.assertFalse(values["drift_available"])
        self.assertAlmostEqual(values["volume_imbalance"], 1 / 4)

    def test_reverse_corners_negates_all_model_features(self) -> None:
        red = [candle(60, .4, "2"), candle(120, .6, "8")]
        blue = [candle(60, .6, "1"), candle(120, .4, "0")]
        a = history.history_features(red, blue, 0, 120, .6)
        b = history.history_features(blue, red, 0, 120, .4)
        np.testing.assert_allclose([a[k] for k in history.FEATURES], [-b[k] for k in history.FEATURES], atol=1e-12)

    def test_invalid_or_missing_volume_does_not_become_directional_signal(self) -> None:
        for invalid in (None, "bad", "-1", "nan", True):
            result = history.history_features([candle(60, .5, invalid)], [candle(60, .5, "9")], 0, 60, .5)
            self.assertFalse(result["volume_available"])
            self.assertEqual(result["volume_imbalance"], 0)
        result = history.history_features([candle(60, .5, "0")], [candle(60, .5, "0")], 0, 60, .5)
        self.assertTrue(result["volume_available"])
        self.assertEqual(result["volume_imbalance"], 0)

    def test_empty_history_keeps_neutral_features(self) -> None:
        result = history.history_features([candle(0, .6)], [candle(0, .4)], 0, 0, .6)
        self.assertEqual(result["price_drift"], 0)
        self.assertEqual(result["volume_imbalance"], 0)
        self.assertFalse(result["volume_available"])
        self.assertAlmostEqual(result["market_logit"], logit(.6))

    def test_five_cent_spreads_match_frozen_quote_rule(self) -> None:
        self.assertAlmostEqual(history.midpoint(candle(60, .5, spread=.05)), .5)
        self.assertIsNone(history.midpoint(candle(60, .5, spread=.051)))
        crossed = candle(60, .5)
        crossed["yes_bid"]["close"] = ".7"
        self.assertIsNone(history.midpoint(crossed))

    def test_duplicate_timestamps_and_endpoint_mismatch_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            history.history_features([candle(60, .5), candle(60, .6)], [candle(60, .5)], 0, 60, .5)
        with self.assertRaisesRegex(ValueError, "probability differs"):
            history.history_features([candle(60, .6)], [candle(60, .4)], 0, 60, .5)

    def test_fitted_model_scale_uses_only_fit_rows_and_is_symmetric(self) -> None:
        train = pd.DataFrame({"market_logit": [-1., .5, 1., -.5], "price_drift": [0., .2, -.1, .3],
                              "volume_imbalance": [0., 0., 0., 0.], "target": [0, 1, 1, 0]})
        calibration = train.copy()
        calibration[history.FEATURES] *= 10
        model = history.fit(train, calibration, history.FEATURES)
        np.testing.assert_allclose(model["scales"], [np.sqrt(.625), np.sqrt(.035), 1])
        reverse = calibration.copy()
        reverse[history.FEATURES] *= -1
        np.testing.assert_allclose(history.predict(calibration, model) + history.predict(reverse, model), 1., atol=1e-12)


if __name__ == "__main__":
    unittest.main()
