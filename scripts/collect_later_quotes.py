"""Collect fixed-cohort Kalshi candle availability without outcome scoring."""
import argparse
import hashlib
import json
import math
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
QUOTE_SOURCE = ROOT / 'data/evaluation/market-edge-v2/verified-quotes.csv'
PILOT = ROOT / 'data/evaluation/later-quote-feasibility-v1'
SOURCE_SHA = '88520cca5cbfa51277d130d8ad1fceada6f976d97f8723462ed44b2e2a107a86'
COLUMNS = ['fight_id', 'event_id', 'date', 'red_fighter_id', 'blue_fighter_id', 'red_name', 'blue_name', 'event_ticker', 'red_ticker', 'blue_ticker']
CAP = 50_000_000
STAGES = ['fit2025', 'calibration2026', 'test2026']


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value: object) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def cohort(date: str) -> str:
    if date < '2026-01-01':
        return 'fit2025'
    return 'calibration2026' if date < '2026-05-16' else 'test2026'


def candle_url(ticker: str, partition: str, cutoff: int) -> str:
    if partition not in ['historical', 'current']:
        raise ValueError('Unknown frozen API partition')
    prefix = f'/historical/markets/{ticker}' if partition == 'historical' else f'/series/KXUFCFIGHT/markets/{ticker}'
    query = urllib.parse.urlencode({'start_ts': cutoff - 3600, 'end_ts': cutoff, 'period_interval': 1})
    return 'https://external-api.kalshi.com/trade-api/v2' + prefix + '/candlesticks?' + query


def dollar(candle: dict, kind: str) -> float | None:
    fields = candle.get(kind)
    if not isinstance(fields, dict):
        return None
    value = fields.get('close_dollars', fields.get('close'))
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def select(pair: dict, payloads: dict) -> dict:
    cutoff = pair['cutoffTimestamp']
    indexed = []
    for corner in ['red', 'blue']:
        ticker = pair[corner + '_ticker']
        if ticker not in payloads:
            return {'reason': 'response_unavailable'}
        payload = payloads[ticker]
        if not isinstance(payload, dict) or not isinstance(payload.get('candlesticks'), list) or payload.get('ticker', ticker) != ticker:
            return {'reason': 'invalid_response_schema'}
        within = {}
        for candle in payload['candlesticks']:
            if not isinstance(candle, dict) or type(candle.get('end_period_ts')) is not int:
                return {'reason': 'invalid_candle_timestamp'}
            timestamp = candle['end_period_ts']
            if cutoff - 3600 <= timestamp <= cutoff:
                if timestamp in within:
                    return {'reason': 'duplicate_candle_timestamp'}
                within[timestamp] = candle
        indexed.append(within)
    common = sorted(set(indexed[0]) & set(indexed[1]), reverse=True)
    for timestamp in common:
        prices = [(dollar(side[timestamp], 'yes_bid'), dollar(side[timestamp], 'yes_ask')) for side in indexed]
        if not all(bid is not None and ask is not None and 0 < bid <= ask < 1 and ask - bid <= .05 + 1e-9 for bid, ask in prices):
            continue
        mids = [(bid + ask) / 2 for bid, ask in prices]
        markets = []
        for corner, (bid, ask), mid, candles in zip(['red', 'blue'], prices, mids, indexed):
            candle = candles[timestamp]
            markets.append({'ticker': pair[corner + '_ticker'], 'fighterId': pair[corner + '_fighter_id'], 'fighterName': pair[corner + '_name'], 'bid': bid, 'ask': ask, 'midpoint': mid, 'normalizedMidpoint': mid / sum(mids), 'selectedMinuteVolume': candle.get('volume_fp', candle.get('volume')), 'rawFile': pair[corner + '_rawFile']})
        return {'reason': 'qualified', 'quoteEndTimestamp': timestamp, 'quoteObservedAt': datetime.fromtimestamp(timestamp, timezone.utc).isoformat(), 'candleObservationAgeSeconds': cutoff - timestamp, 'actualQuoteUpdateAgeKnown': False, 'midpointSum': sum(mids), 'markets': markets}
    return {'reason': 'no_simultaneous_valid_price_and_spread' if common else 'no_simultaneous_candle_within_hour'}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Acquisition:
    def __init__(self, output: Path, cap: int = CAP):
        self.output = output
        self.cap = cap
        receipts = [json.loads(p.read_text()) for p in sorted((output / 'receipts').glob('*.json'))]
        self.used = sum(record['accountedBytes'] for record in receipts)
        self.last = 0.0
        self.opener = urllib.request.build_opener(NoRedirect)
        self.stopped = None
        for record in receipts:
            status = record.get('status')
            if record.get('stopReason'):
                self.stopped = record['stopReason']
                break
            if status in [401, 403, 429, 451] or isinstance(status, int) and 300 <= status < 400:
                self.stopped = f'prior_HTTP{status}'
                break

    def request(self, item: dict) -> dict:
        key = item['ticker']
        intent = self.output / 'intents' / f'{key}.json'
        if self.stopped or self.used >= self.cap:
            raise ValueError('Acquisition stopped or budget exhausted')
        if intent.exists() or (self.output / 'receipts' / f'{key}.json').exists():
            raise ValueError('URL already attempted; no retry permitted')
        time.sleep(max(0, self.last + 1 - time.monotonic()))
        self.last = time.monotonic()
        save(intent, {**item, 'requestedAt': now(), 'attempt': 1})
        record = {**item, 'attempt': 1, 'requestedAt': json.loads(intent.read_text())['requestedAt'], 'status': None, 'accountedBytes': 0, 'bodyComplete': False}
        raw = bytearray()
        header_bytes = b''
        response = None
        try:
            request = urllib.request.Request(item['url'], headers={'Accept': 'application/json', 'Accept-Encoding': 'identity', 'User-Agent': 'Versus-research-feasibility/1.0'})
            try:
                response = self.opener.open(request, timeout=30)
            except urllib.error.HTTPError as error:
                response = error
            record['status'] = response.status
            header_bytes = json.dumps(dict(response.headers.items()), indent=2).encode()
            record.update({'headersFile': f'raw/{key}.headers.json', 'httpDate': response.headers.get('Date')})
            (self.output / record['headersFile']).write_bytes(header_bytes)
            remaining = self.cap - self.used - len(header_bytes)
            length = response.headers.get('Content-Length')
            if remaining <= 0 or (length is not None and int(length) > remaining):
                self.stopped = 'response_budget'
                record['error'] = 'Response header/body length exceeds remaining budget; body not read'
            else:
                while len(raw) < remaining:
                    chunk = response.read(min(65536, remaining - len(raw)))
                    if not chunk:
                        record['bodyComplete'] = True
                        break
                    raw.extend(chunk)
                if not record['bodyComplete']:
                    self.stopped = 'response_budget'
                    record['error'] = 'Hard response budget reached; body may be truncated'
            if response.status in [401, 403, 429, 451]:
                self.stopped = f'HTTP{response.status}'
                record['error'] = 'Stop entire acquisition; no retry, route fallback, or bypass'
            elif 300 <= response.status < 400:
                self.stopped = f'HTTP{response.status}_redirect'
                record['error'] = 'Redirect not followed'
            elif response.status != 200:
                record['error'] = f'HTTP{response.status}; this contract unavailable; no retry'
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            record['error'] = str(error)
            self.stopped = 'transport_error'
        finally:
            if response is not None:
                response.close()
            (self.output / item['rawFile']).write_bytes(raw)
            record.update({'bodySha256': hashlib.sha256(raw).hexdigest(), 'bodyBytes': len(raw), 'headerBytes': len(header_bytes), 'headersSha256': hashlib.sha256(header_bytes).hexdigest(), 'accountedBytes': len(raw) + len(header_bytes), 'finishedAt': now()})
            self.used += record['accountedBytes']
            record.update({'cumulativeNewAccountedBytes': self.used, 'stopReason': self.stopped})
            save(self.output / 'receipts' / f'{key}.json', record)
        return record


def freeze(output: Path) -> None:
    if output.exists():
        raise ValueError('Output already exists; preserve frozen files')
    if sha(QUOTE_SOURCE) != SOURCE_SHA:
        raise ValueError('Fixed499 source changed')
    pilot_inventory = json.loads((PILOT / 'final-inventory.json').read_text())
    for name, expected in pilot_inventory.items():
        if sha(PILOT / name) != expected:
            raise ValueError('Pilot source changed')
    rows = pd.read_csv(QUOTE_SOURCE, usecols=COLUMNS).sort_values(['date', 'event_ticker']).to_dict('records')
    if len(rows) != 499 or len({r['fight_id'] for r in rows}) != 499 or len({r['date'] for r in rows}) != 58:
        raise ValueError('Fixed cohort shape changed')
    routing = {}
    source_names = ['market-benchmark-v1/source/qualified-quotes.json', 'market-coverage-v2/qualified-quotes.json']
    for name in source_names:
        for row in json.loads((ROOT / 'data/evaluation' / name).read_text()):
            for market in row['markets']:
                partition = market['apiPartition']
                if market['ticker'] in routing and routing[market['ticker']] != partition:
                    raise ValueError('Conflicting frozen API partition')
                routing[market['ticker']] = partition
    pilot_pairs = {r['fight_id']: r for r in json.loads((PILOT / 'selected-denominator.json').read_text())}
    output.mkdir(parents=True)
    for name in ['raw', 'receipts', 'intents', 'checkpoints']:
        (output / name).mkdir()
    shutil.copytree(PILOT, output / 'pilot')
    requests = []
    for row in rows:
        row.update({'cohort': cohort(row['date']), 'cutoff': row['date'] + 'T00:00:00+00:00'})
        row['cutoffTimestamp'] = int(datetime.fromisoformat(row['cutoff']).timestamp())
        reused = row['fight_id'] in pilot_pairs
        if reused and any(row[k] != pilot_pairs[row['fight_id']][k] for k in COLUMNS):
            raise ValueError('Pilot identity mismatch')
        row['reusedPilot'] = reused
        for corner in ['red', 'blue']:
            ticker = row[corner + '_ticker']
            row[corner + '_apiPartition'] = routing[ticker]
            row[corner + '_rawFile'] = ('pilot/' if reused else '') + 'raw/' + ticker + '.body'
            row[corner + '_receiptFile'] = ('pilot/' if reused else '') + 'receipts/' + ticker + '.json'
            request = {'ticker': ticker, 'fightId': row['fight_id'], 'cohort': row['cohort'], 'date': row['date'], 'apiPartition': routing[ticker], 'url': candle_url(ticker, routing[ticker], row['cutoffTimestamp']), 'rawFile': row[corner + '_rawFile'], 'receiptFile': row[corner + '_receiptFile'], 'reusedPilot': reused}
            if reused and json.loads((output / request['receiptFile']).read_text())['url'] != request['url']:
                raise ValueError('Pilot request window mismatch')
            requests.append(request)
    if Counter(r['cohort'] for r in rows) != {'fit2025': 127, 'calibration2026': 155, 'test2026': 217}:
        raise ValueError('Chronological cohorts changed')
    if sum(not r['reusedPilot'] for r in requests) != 982:
        raise ValueError('New request denominator changed')
    save(output / 'selected-denominator.json', rows)
    save(output / 'request-plan.json', requests)
    shutil.copy2(Path(__file__), output / 'collector-frozen.py')
    shutil.copy2(ROOT / 'tests/test_later_quotes.py', output / 'test_later_quotes.py')
    docs = ROOT / 'data/evaluation/market-coverage-v2/evidence/quote-endpoint-docs.json'
    shutil.copy2(docs, output / 'endpoint-docs-original.json')
    sources = [QUOTE_SOURCE, PILOT / 'completion.json', PILOT / 'final-inventory.json', docs] + [ROOT / 'data/evaluation' / n for n in source_names]
    protocol = {'createdAt': now(), 'schemaVersion': 1, 'purpose': 'Fixed499 later-horizon quote availability only; no outcome join, model prediction or scoring', 'expectedPairs': 499, 'eventDates': 58, 'newPairs': 491, 'newContracts': 982, 'reusedPilotPairs': 8, 'cohorts': {'fit2025': {'from': '2025-01-01', 'before': '2026-01-01', 'count': 127}, 'calibration2026': {'from': '2026-01-01', 'before': '2026-05-16', 'count': 155}, 'test2026': {'from': '2026-05-16', 'before': '2026-11-01', 'count': 217}}, 'stageGate': 'Collect fit2025 first, then stop. Calibration/test require explicit root review authorization saved before their acquisition; all499 remain in each status report.', 'sources': {str(p.relative_to(ROOT)): sha(p) for p in sources}, 'selectedDenominatorSHA256': sha(output / 'selected-denominator.json'), 'requestPlanSHA256': sha(output / 'request-plan.json'), 'collectorSHA256': sha(output / 'collector-frozen.py'), 'testsSHA256': sha(output / 'test_later_quotes.py'), 'rule': {'cutoff': 'Event-date00:00:00UTC', 'window': 'Inclusive[cutoff-3600,cutoff]', 'periodInterval': 1, 'selection': 'Latest simultaneous returned endpoint; both sides0<bid<=ask<1 andspread<=.05+1e-9; normalize pairedYESmidpoints. No client forward-fill; no use of trade price.previous. Zero trade volume does not alter eligibility.', 'priceFields': 'yes_bid/yes_ask close_dollars preferred, else documented dollarclose', 'endpointRouting': 'Use each contract historical/current partition from frozen successful original quote metadata. No alternate route or retry.', 'updateAgeKnown': False}, 'acquisition': {'newResponseBytesCap': CAP, 'accounting': 'New response bodies plus serialized response headers. Reused pilot bytes excluded; TLS overhead not measurable.', 'requestsPerSecond': 1, 'concurrency': 1, 'attemptsPerURL': 1, 'stopStatuses': [401, 403, 429, 451], 'redirects': False, 'credentials': False, 'payments': False}, 'limitations': ['Selected499 are an existing earlier-horizon subset, not a fullUFCcoverage denominator.', 'Candles do not certify quote-update age, executable depth or liquidity; selected zero-volume minutes explicitly counted.', 'Actual historical start times and source availability by each observed minute require separate validation before any performance evaluation.', 'No outcome labels, model probabilities or performance metrics are included.']}
    save(output / 'protocol.json', protocol)
    save(output / 'freeze.json', {'createdAt': now(), 'protocolSHA256': sha(output / 'protocol.json')})
    print(json.dumps({'frozen': str(output), 'protocolSHA256': sha(output / 'protocol.json'), 'newRequests': 982, 'firstStageRequests': sum(r['cohort'] == 'fit2025' and not r['reusedPilot'] for r in requests)}), flush=True)


def summarize(output: Path, stage: str, stopped: str | None, final: bool = False) -> dict:
    rows = json.loads((output / 'selected-denominator.json').read_text())
    decisions = []
    for pair in rows:
        payloads = {}
        statuses = []
        for corner in ['red', 'blue']:
            path = output / pair[corner + '_receiptFile']
            if not path.exists():
                statuses.append('pending')
                continue
            receipt = json.loads(path.read_text())
            raw = output / pair[corner + '_rawFile']
            if not raw.exists() or sha(raw) != receipt['bodySha256']:
                raise ValueError('Raw response hash mismatch')
            if receipt['status'] != 200 or not receipt.get('bodyComplete') or receipt.get('error'):
                statuses.append('response_failed')
                continue
            try:
                payloads[pair[corner + '_ticker']] = json.loads(raw.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                statuses.append('invalid_response_json')
        result = select(pair, payloads)
        if statuses:
            reason = 'response_failed' if 'response_failed' in statuses else 'invalid_response_json' if 'invalid_response_json' in statuses else 'pending_unrequested'
            if reason == 'pending_unrequested' and stopped and pair['cohort'] == stage:
                reason = 'unrequested_after_stop'
            result = {'reason': reason}
        decisions.append({**pair, **result})
    qualified = [r for r in decisions if r['reason'] == 'qualified']
    receipts = [json.loads(p.read_text()) for p in (output / 'receipts').glob('*.json')]
    groups = {}
    for name in STAGES:
        subset = [r for r in decisions if r['cohort'] == name]
        ready = [r for r in subset if r['reason'] == 'qualified']
        groups[name] = {'denominator': len(subset), 'qualified': len(ready), 'qualifiedEvents': len({r['event_id'] for r in ready}), 'reasons': dict(Counter(r['reason'] for r in subset)), 'bothSelectedMinuteVolumesZero': sum(all(m['selectedMinuteVolume'] is not None and float(m['selectedMinuteVolume']) == 0 for m in r['markets']) for r in ready)}
    report = {'createdAt': now(), 'stage': stage, 'status': 'stopped_partial' if stopped else 'stage_complete_waiting_review' if final and stage == 'fit2025' else 'complete' if final else 'running', 'stopReason': stopped, 'denominator': len(rows), 'groups': groups, 'newRequestsAttempted': len(receipts), 'newResponseBytes': sum(r['accountedBytes'] for r in receipts), 'reusedPilotContracts': 16, 'scoringPerformed': False, 'historicalScoringApproved': False, 'protocolSHA256': sha(output / 'protocol.json')}
    name = f'{stage}-final' if final else f'{stage}-{len(receipts):04d}'
    save(output / 'checkpoints' / f'{name}.json', report)
    if final:
        save(output / f'decisions-{stage}.json', decisions)
        save(output / f'qualified-quotes-{stage}.json', qualified)
        save(output / f'report-{stage}.json', report)
        save(output / f'inventory-{stage}.json', {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    print(json.dumps(report), flush=True)
    return report


def assert_stage_available(output: Path, stage: str, acquisition: Acquisition) -> None:
    if (output / f'acquisition-started-{stage}.json').exists():
        raise ValueError('Stage already attempted; no resume or repeated URLs')
    if acquisition.stopped:
        raise ValueError(f'Prior terminal acquisition stop: {acquisition.stopped}')
    if acquisition.used >= acquisition.cap:
        raise ValueError('Response budget exhausted before stage start')


def collect(output: Path, stage: str) -> None:
    acquisition = Acquisition(output)
    assert_stage_available(output, stage, acquisition)
    protocol = json.loads((output / 'protocol.json').read_text())
    frozen = json.loads((output / 'freeze.json').read_text())
    expected_code = protocol['collectorSHA256']
    amendment_path = output / 'collector-amendment-stage-guard.json'
    if stage != 'fit2025' and amendment_path.exists():
        amendment = json.loads(amendment_path.read_text())
        if amendment['protocolSHA256'] != frozen['protocolSHA256'] or stage not in amendment['appliesToStages']:
            raise ValueError('Collector amendment does not match the frozen stage')
        expected_code = amendment['collectorSHA256']
        if sha(output / amendment['collectorSnapshot']) != expected_code:
            raise ValueError('Amended collector snapshot changed')
    if sha(output / 'protocol.json') != frozen['protocolSHA256'] or sha(Path(__file__)) != expected_code:
        raise ValueError('Frozen protocol or collector changed')
    for name, field in [('selected-denominator.json', 'selectedDenominatorSHA256'), ('request-plan.json', 'requestPlanSHA256')]:
        if sha(output / name) != protocol[field]:
            raise ValueError('Frozen input changed')
    if stage != 'fit2025':
        authorization = json.loads((output / f'authorization-{stage}.json').read_text())
        if authorization.get('authorized') is not True or authorization.get('protocolSHA256') != frozen['protocolSHA256']:
            raise ValueError('Root stage authorization required')
    save(output / f'acquisition-started-{stage}.json', {'startedAt': now(), 'protocolSHA256': frozen['protocolSHA256']})
    plan = [r for r in json.loads((output / 'request-plan.json').read_text()) if r['cohort'] == stage and not r['reusedPilot']]
    for index, item in enumerate(plan, 1):
        if acquisition.stopped:
            break
        acquisition.request(item)
        if index % 25 == 0:
            summarize(output, stage, acquisition.stopped)
    summarize(output, stage, acquisition.stopped, final=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'collect'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=STAGES, default='fit2025')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.output.resolve())
    else:
        collect(args.output.resolve(), args.stage)


if __name__ == '__main__':
    main()
