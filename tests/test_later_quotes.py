import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import collect_later_quotes as later


class LaterQuoteTests(unittest.TestCase):
    def setUp(self):
        self.pair = {'cutoffTimestamp': 7200, 'red_ticker': 'RED', 'blue_ticker': 'BLUE',
                     'red_fighter_id': 'r', 'blue_fighter_id': 'b', 'red_name': 'Red', 'blue_name': 'Blue',
                     'red_rawFile': 'raw/RED.body', 'blue_rawFile': 'raw/BLUE.body'}

    def candle(self, timestamp, bid='.45', ask='.48', volume='0.00'):
        return {'end_period_ts': timestamp, 'yes_bid': {'close': bid}, 'yes_ask': {'close': ask}, 'volume': volume}

    def payload(self, red, blue):
        return {'RED': {'ticker': 'RED', 'candlesticks': red}, 'BLUE': {'ticker': 'BLUE', 'candlesticks': blue}}

    def test_latest_simultaneous_valid_minute_and_no_forward_fill(self):
        result = later.select(self.pair, self.payload([self.candle(7140), self.candle(7200)], [self.candle(7140), self.candle(7180)]))
        self.assertEqual(result['quoteEndTimestamp'], 7140)
        self.assertEqual(result['candleObservationAgeSeconds'], 60)
        self.assertAlmostEqual(sum(m['normalizedMidpoint'] for m in result['markets']), 1)
        result = later.select(self.pair, self.payload([self.candle(7200)], [self.candle(7140)]))
        self.assertEqual(result['reason'], 'no_simultaneous_candle_within_hour')

    def test_inclusive_hour_and_no_future_candles(self):
        rows = [self.candle(3599), self.candle(3600), self.candle(7201)]
        self.assertEqual(later.select(self.pair, self.payload(rows, rows))['quoteEndTimestamp'], 3600)
        rows = [self.candle(3599), self.candle(7201)]
        self.assertEqual(later.select(self.pair, self.payload(rows, rows))['reason'], 'no_simultaneous_candle_within_hour')

    def test_spread_and_nonfinite_prices(self):
        for bid, ask in [('0', '.01'), ('.99', '1'), ('.6', '.5'), ('nan', '.5'), ('.4', '.46'), (True, '.5')]:
            rows = [self.candle(7200, bid, ask)]
            self.assertEqual(later.select(self.pair, self.payload(rows, rows))['reason'], 'no_simultaneous_valid_price_and_spread')
        rows = [self.candle(7200, '.4', '.45')]
        self.assertEqual(later.select(self.pair, self.payload(rows, rows))['reason'], 'qualified')

    def test_legacy_dollar_fallback_and_primary_null(self):
        candle = self.candle(7200)
        self.assertEqual(later.dollar(candle, 'yes_bid'), .45)
        candle['yes_bid']['close_dollars'] = None
        self.assertIsNone(later.dollar(candle, 'yes_bid'))
        candle['yes_bid']['close_dollars'] = '.44'
        self.assertEqual(later.dollar(candle, 'yes_bid'), .44)

    def test_duplicate_wrong_ticker_and_timestamp_fail_closed(self):
        row = self.candle(7200)
        self.assertEqual(later.select(self.pair, self.payload([row, row], [row]))['reason'], 'duplicate_candle_timestamp')
        payload = self.payload([row], [row]); payload['BLUE']['ticker'] = 'OTHER'
        self.assertEqual(later.select(self.pair, payload)['reason'], 'invalid_response_schema')
        row = self.candle(True)
        self.assertEqual(later.select(self.pair, self.payload([row], [row]))['reason'], 'invalid_candle_timestamp')

    def test_zero_volume_and_outcome_fields_do_not_affect_eligibility(self):
        row = self.candle(7200)
        expected = later.select(self.pair, self.payload([row], [row]))
        row.update({'outcome': 'yes', 'target': 1, 'result': 'win', 'price': {'previous': '.99'}})
        self.assertEqual(later.select(self.pair, self.payload([row], [row])), expected)
        self.assertEqual(expected['reason'], 'qualified')

    def test_fixed_period_edges_and_partition_no_fallback(self):
        self.assertEqual(later.cohort('2025-12-31'), 'fit2025')
        self.assertEqual(later.cohort('2026-05-15'), 'calibration2026')
        self.assertEqual(later.cohort('2026-05-16'), 'test2026')
        self.assertIn('/historical/', later.candle_url('ABC', 'historical', 7200))
        self.assertIn('/series/KXUFCFIGHT/', later.candle_url('ABC', 'current', 7200))
        with self.assertRaises(ValueError): later.candle_url('ABC', 'guess', 7200)

    def test_no_retry_and_exhausted_budget_prevent_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ['intents', 'receipts', 'raw']: (root / folder).mkdir()
            acquisition = later.Acquisition(root, cap=1)
            acquisition.used = 1
            with self.assertRaises(ValueError): acquisition.request({'ticker': 'ABC'})
            acquisition.used = 0
            (root / 'intents/ABC.json').write_text('{}')
            with self.assertRaises(ValueError): acquisition.request({'ticker': 'ABC'})

    def test_denial_stops_and_preserves_response_once(self):
        class Response:
            status = 429
            headers = {'Content-Length': '2', 'Date': 'today'}
            def __init__(self): self.parts = [b'{}', b'']
            def read(self, size): return self.parts.pop(0)
            def close(self): pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ['intents', 'receipts', 'raw']: (root / folder).mkdir()
            acquisition = later.Acquisition(root)
            with patch.object(acquisition.opener, 'open', return_value=Response()) as request:
                record = acquisition.request({'ticker': 'ABC', 'url': 'https://example.test', 'rawFile': 'raw/ABC.body'})
                self.assertEqual(request.call_count, 1)
            self.assertEqual(acquisition.stopped, 'HTTP429')
            self.assertEqual(record['status'], 429)
            self.assertTrue((root / 'receipts/ABC.json').exists())
            with self.assertRaises(ValueError): acquisition.request({'ticker': 'XYZ'})

    def test_terminal_stop_persists_across_stages_before_any_write(self):
        for status, reason in [(401, None), (403, None), (429, None), (451, None),
                               (302, None), (None, 'transport_error'), (200, 'response_budget')]:
            with self.subTest(status=status, reason=reason), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'receipts').mkdir()
                (root / 'receipts/old.json').write_text(json.dumps({'status': status, 'stopReason': reason, 'accountedBytes': 20}))
                before = set(root.rglob('*'))
                with patch.object(later.urllib.request.OpenerDirector, 'open') as request:
                    with self.assertRaisesRegex(ValueError, 'Prior terminal'):
                        later.collect(root, 'calibration2026')
                    request.assert_not_called()
                self.assertEqual(set(root.rglob('*')), before)

    def test_existing_stage_cannot_resume_even_with_remaining_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'receipts').mkdir()
            (root / 'acquisition-started-fit2025.json').write_text('{}')
            with patch.object(later.urllib.request.OpenerDirector, 'open') as request:
                with self.assertRaisesRegex(ValueError, 'Stage already attempted'):
                    later.collect(root, 'fit2025')
                request.assert_not_called()

    def test_nonterminal_contract_failure_does_not_change_cohort_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'receipts').mkdir()
            (root / 'receipts/old.json').write_text(json.dumps({'status': 404, 'stopReason': None, 'accountedBytes': 20}))
            acquisition = later.Acquisition(root)
            self.assertIsNone(acquisition.stopped)
            later.assert_stage_available(root, 'calibration2026', acquisition)


if __name__ == '__main__':
    unittest.main()
