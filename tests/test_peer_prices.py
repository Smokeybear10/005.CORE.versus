import json
import tempfile
import unittest
import hashlib
from email.message import Message
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import collect_peer_prices as peer


class PeerPriceTests(unittest.TestCase):
    def setUp(self):
        self.identity = {'conditionId': '0x' + 'a' * 64, 'quoteObservedAt': '2026-10-29T23:27:00Z',
                         'tokens': [{'tokenId': '11', 'outcomeIndex': 0, 'side': 'blue'},
                                    {'tokenId': '22', 'outcomeIndex': 1, 'side': 'red'}]}
        self.observed = 1793316420

    def trade(self, **changes):
        return {'condition_id': self.identity['conditionId'], 'asset_id': '11', 'outcome_seq': 1,
                'block_timestamp': self.observed-10, 'price': .3, 'usdc_amount': 5, **changes}

    def test_actual_observation_and_fixed_age_boundaries(self):
        records = [self.trade(block_timestamp=self.observed-3601), self.trade(block_timestamp=self.observed+1)]
        self.assertEqual(peer.select_execution(records, self.identity)['status'], 'no_execution_in_window')
        result = peer.select_execution(records+[self.trade(block_timestamp=self.observed-3600)], self.identity)
        self.assertEqual(result['ageSeconds'], 3600)
        self.assertEqual(result['redProbabilityProxy'], .7)
        self.assertEqual(peer.select_execution([self.trade(block_timestamp=self.observed)], self.identity)['ageSeconds'], 0)

    def test_target_fields_cannot_change_selection(self):
        baseline = peer.select_execution([self.trade()], self.identity)
        toxic = self.trade(p_event=.999, D=-1, winning_outcome_label='red', resolution_status='resolved', resolved_at='2026-01-01')
        self.assertEqual(peer.select_execution([toxic], self.identity), baseline)

    def test_exact_tokens_orientation_and_finite_values(self):
        bad = [self.trade(asset_id='unknown'), self.trade(outcome_seq=2), self.trade(outcome_seq=True), self.trade(price=float('nan')),
               self.trade(price=True), self.trade(usdc_amount=0), self.trade(block_timestamp=True),
               self.trade(condition_id='wrong'), self.trade(price=1)]
        result = peer.select_execution(bad, self.identity)
        self.assertEqual(result['invalidRecords'], len(bad))
        result = peer.select_execution([self.trade(asset_id='22', outcome_seq=2, price=.7)], self.identity)
        self.assertEqual(result['redProbabilityProxy'], .7)

    def test_latest_second_ambiguity_and_truncation_fail_closed(self):
        rows = [self.trade(price=.3), self.trade(price=.4)]
        self.assertEqual(peer.select_execution(rows, self.identity)['status'], 'ambiguous_latest_second')
        self.assertEqual(peer.select_execution([self.trade()], self.identity, response_truncated=True)['status'], 'ambiguous_latest_second')
        self.assertEqual(peer.select_execution([self.trade(), self.trade(price=float('nan'))], self.identity,
                                              response_truncated=True)['status'], 'ambiguous_latest_second')
        result = peer.select_execution([self.trade(), self.trade(asset_id='22', outcome_seq=2, price=.7)], self.identity)
        self.assertEqual(result['status'], 'candidate_execution')
        self.assertTrue(result['bothTokensObserved'])

    def test_date_is_explicit_and_unique(self):
        self.assertTrue(peer.explicit_date('bout scheduled for May 31, 2025.', '2025-05-31'))
        self.assertFalse(peer.explicit_date('slug2025-05-31 ends2025-06-01', '2025-05-31'))
        self.assertFalse(peer.explicit_date('scheduled for May 31, 2025; scheduled for June 1, 2025', '2025-05-31'))
        self.assertTrue(peer.explicit_date('scheduled for October 18.', '2025-10-18', 'ufc-a-b-2025-10-18', '2025-10-18'))
        self.assertFalse(peer.explicit_date('scheduled for October 18.', '2025-10-18', 'ufc-a-b-2025-10-18', '2026-10-18'))

    def test_fighter_alias_is_specific_to_reviewed_bout(self):
        row = {'fight_id': 'ad68450307a21ab1', 'date': '2025-05-31', 'red_name': 'Zach Reese',
               'blue_name': 'Dusko Todorovic', 'red_fighter_id': '23dec7c47cb418f8',
               'blue_fighter_id': '866fd7b1a6c90e7f', 'quote_observed_at': '2025-05-29T23:28:00Z'}
        market = {'id': '1', 'conditionId': '0x'+'a'*64, 'sportsMarketType': 'moneyline',
                  'description': 'scheduled for May 31, 2025. If Zachary Reese is declared the winner of this bout, this market will resolve to "Reese." If Dusko Todorovic is declared the winner of this bout, this market will resolve to "Todorovic."',
                  'outcomes': json.dumps(['Reese', 'Todorovic']), 'clobTokenIds': json.dumps(['11', '22']),
                  'createdAt': '2025-05-20T00:00:00Z'}
        self.assertIsNotNone(peer.market_identity(market, row))
        reverse = peer.market_identity({**market, 'outcomes': json.dumps(['Todorovic', 'Reese'])}, row)
        self.assertEqual([t['side'] for t in reverse['tokens']], ['blue', 'red'])
        self.assertEqual([t['tokenId'] for t in reverse['tokens']], ['11', '22'])
        self.assertIsNone(peer.market_identity(market, {**row, 'fight_id': 'other'}))
        self.assertIsNone(peer.market_identity({**market, 'createdAt': '2025-05-30T00:00:00Z'}, row))
        reverse_style = {**market, 'description': 'scheduled for May 31, 2025. This market will resolve to "Reese" if Zachary Reese is officially declared the winner. It will resolve to "Todorovic" if Dusko Todorovic is officially declared the winner.'}
        self.assertEqual(peer.market_identity(reverse_style, row)['tokens'], peer.market_identity(market, row)['tokens'])
        period = peer.market_identity({**market, 'outcomes': json.dumps(['Reese.', 'Todorovic'])}, row)
        self.assertEqual([t['side'] for t in period['tokens']], ['red', 'blue'])
        self.assertIsNone(peer.market_identity({**market, 'outcomes': json.dumps(['Reese', 'Reese.'])}, row))

    def test_budget_stops_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            store = peer.HttpStore(Path(directory), limit=10)
            with self.assertRaisesRegex(peer.AcquisitionError, 'budget'):
                store.fetch('no-request', 'https://example.invalid', cap=11)

    def test_cached_ranges_and_actual_revision_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'chunk'
            path.write_bytes(b'ab')
            receipt = {'url': 'https://example.invalid/file', 'status': 206, 'bytes': 2,
                       'sha256': hashlib.sha256(b'ab').hexdigest(), 'range': [0, 1, 10],
                       'expectedRevision': 'abc', 'headers': {'Content-Range': 'bytes 0-1/10', 'x-revision': 'abc'}}
            peer.save(Path(directory)/'chunk.receipt.json', receipt)
            store = peer.HttpStore(Path(directory))
            self.assertEqual(store.fetch('chunk', receipt['url'], 2, (0, 1, 10), 'abc'), b'ab')
            with self.assertRaisesRegex(peer.AcquisitionError, 'range or revision'):
                store.fetch('chunk', receipt['url'], 2, (2, 3, 10), 'abc')
            receipt['headers']['x-revision'] = 'wrong'
            (Path(directory)/'chunk.receipt.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(peer.AcquisitionError, 'actual revision'):
                store.fetch('chunk', receipt['url'], 2, (0, 1, 10), 'abc')

    def test_truncated_range_body_rejected_and_preserved(self):
        class Response:
            status = 206
            headers = Message()
            headers['Content-Range'] = 'bytes 0-3/10'
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, size): return b'ab'
        with tempfile.TemporaryDirectory() as directory, patch.object(peer, 'urlopen', return_value=Response()), patch.object(peer.time, 'sleep'):
            store = peer.HttpStore(Path(directory))
            with self.assertRaisesRegex(peer.AcquisitionError, 'range body'):
                store.fetch('chunk', 'https://example.invalid/file', 4, (0, 3, 10))
            receipt = json.loads((Path(directory)/'chunk.receipt.json').read_text())
            self.assertIn('range body', receipt['error'])
            self.assertEqual(receipt['bytes'], 2)

    def test_partition_date_boundaries_and_integrity(self):
        self.assertEqual(peer.required_dates('2026-01-01T00:00:00Z'), ['2025-12-31', '2026-01-01'])
        self.assertEqual(peer.required_dates('2026-01-01T01:00:00+02:00'), ['2025-12-30', '2025-12-31'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'sample.parquet'
            frame = pd.DataFrame([self.trade()])[peer.COLUMNS]
            frame.to_parquet(path, engine='fastparquet', index=False)
            item = {'projectedFile': path.name, 'projectedSHA256': peer.digest(path), 'sourceRevision': peer.REVISIONS['v1'],
                    'sourceComplete': True, 'fullSourceHashVerified': True, 'rows': 1,
                    'minTimestamp': self.observed-10, 'maxTimestamp': self.observed-10, 'partitionDate': '2026-10-29'}
            self.assertEqual(len(peer.load_partition(Path(directory), item)), 1)
            with self.assertRaisesRegex(peer.AcquisitionError, 'recorded UTC day'):
                peer.load_partition(Path(directory), {**item, 'partitionDate': '2026-10-30'})
            with self.assertRaisesRegex(peer.AcquisitionError, 'Incomplete'):
                peer.load_partition(Path(directory), {**item, 'sourceComplete': False})
            frame.assign(winning_outcome_label='red').to_parquet(path, engine='fastparquet', index=False)
            with self.assertRaisesRegex(peer.AcquisitionError, 'unexpected columns'):
                peer.load_partition(Path(directory), {**item, 'projectedSHA256': peer.digest(path)})

    def test_later_overlap_and_midnight_require_complete_sources(self):
        self.assertEqual(peer.required_source_keys('2026-04-03T00:00:00Z'),
                         ['v1:2026-04-02', 'v1:2026-04-03', 'v2:2026-04-03'])
        self.assertEqual(peer.required_source_keys('2026-04-29T00:00:00Z'),
                         ['v1:2026-04-28', 'v2:2026-04-28', 'v2:2026-04-29'])
        self.assertEqual(peer.required_source_keys('2026-10-30T00:00:00Z'),
                         ['v2:2026-10-29', 'v2:2026-10-30'])
        self.assertEqual(peer.required_source_keys('2026-10-29T23:30:00Z'), ['v2:2026-10-29'])

    def test_verified_empty_partition_is_complete_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'empty.parquet'
            pd.DataFrame([self.trade()])[peer.COLUMNS].iloc[:0].to_parquet(path, engine='fastparquet', index=False)
            item = {'projectedFile': path.name, 'projectedSHA256': peer.digest(path),
                    'sourceVersion': 'v2', 'sourceRevision': peer.REVISIONS['v2'],
                    'sourceComplete': True, 'fullSourceHashVerified': True, 'rows': 0,
                    'minTimestamp': None, 'maxTimestamp': None, 'partitionDate': '2026-04-09'}
            self.assertEqual(len(peer.load_partition(Path(directory), item)), 0)


if __name__ == '__main__':
    unittest.main()
