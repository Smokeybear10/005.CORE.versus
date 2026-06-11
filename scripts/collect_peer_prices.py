import argparse
import hashlib
import json
import math
import re
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from fastparquet import ParquetFile


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'data/evaluation/peer-market-discovery-v1'
DEFAULT_OUTPUT = ROOT / 'data/evaluation/peer-market-discovery-v2'
REVISIONS = {'v1': '5aa1b9d52316a8b2e789e81c8ae42c7ed532e8aa',
             'v2': '8fd92ca5f2c08c8e619c2eb294fd06beeb3bc4ae'}
COLUMNS = ['asset_id', 'block_timestamp', 'price', 'usdc_amount', 'condition_id', 'outcome_seq']
BYTE_LIMIT = 500_000_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def norm(name: str) -> str:
    return ' '.join(''.join(c for c in unicodedata.normalize('NFKD', name)
                            if not unicodedata.combining(c)).casefold().split())


class AcquisitionError(RuntimeError):
    pass


class HttpStore:
    def __init__(self, directory: Path, limit: int = BYTE_LIMIT):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.used = sum(json.loads(p.read_text()).get('bytes', 0)
                        for p in directory.glob('*.receipt.json'))

    def fetch(self, name: str, url: str, cap: int, byte_range: tuple[int, int, int] | None = None,
              revision: str | None = None) -> bytes:
        path = self.directory / name
        receipt_path = self.directory / f'{name}.receipt.json'
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get('error') or receipt.get('status') not in (200, 206):
                raise AcquisitionError(f'Prior failed request remains stopped: {name}')
            if receipt['url'] != url or digest(path) != receipt['sha256']:
                raise AcquisitionError('Cached request provenance mismatch')
            if receipt.get('expectedRevision') != revision or receipt.get('range') != (list(byte_range) if byte_range else None):
                raise AcquisitionError('Cached range or revision does not match request')
            response_headers = {key.lower(): value for key, value in receipt['headers'].items()}
            if revision and response_headers.get('x-revision') != revision:
                raise AcquisitionError('Cached actual revision does not match request')
            if path.stat().st_size != receipt['bytes']:
                raise AcquisitionError('Cached response size does not match receipt')
            if byte_range:
                expected = f'bytes {byte_range[0]}-{byte_range[1]}/{byte_range[2]}'
                if (receipt['status'] != 206 or response_headers.get('content-range') != expected
                        or path.stat().st_size != byte_range[1]-byte_range[0]+1):
                    raise AcquisitionError('Cached range response is not exact')
            return path.read_bytes()
        if self.used + cap > self.limit:
            raise AcquisitionError('Predeclared additional download budget exhausted')
        headers = {'Accept': 'application/json'}
        if byte_range:
            start, end, _ = byte_range
            headers['Range'] = f'bytes={start}-{end}'
        receipt = {'url': url, 'requestedAt': utc_now(), 'expectedRevision': revision,
                   'range': byte_range, 'maxBytes': cap, 'bytes': 0}
        body = b''
        try:
            try:
                response = urlopen(Request(url, headers=headers), timeout=45)
            except HTTPError as error:
                response = error
            with response:
                receipt.update({'status': response.status, 'headers': dict(response.headers)})
                if byte_range and response.status == 206:
                    expected = f'bytes {byte_range[0]}-{byte_range[1]}/{byte_range[2]}'
                    if response.headers.get('Content-Range') != expected:
                        raise AcquisitionError('Unexpected Content-Range')
                elif byte_range and response.status == 200:
                    raise AcquisitionError('Server ignored range; refusing whole file')
                body = response.read(cap + 1)
                self.used += len(body)
                receipt['bytes'] = len(body)
                if len(body) > cap:
                    raise AcquisitionError('Response exceeded per-request budget')
                if byte_range and response.status == 206 and len(body) != byte_range[1]-byte_range[0]+1:
                    raise AcquisitionError('Truncated or oversized range body')
                if response.status not in (200, 206):
                    raise AcquisitionError(f'HTTP {response.status}; stopped without retry')
                if revision and response.headers.get('x-revision') != revision:
                    raise AcquisitionError('Dataset server revision changed or absent')
        except Exception as error:
            receipt['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            receipt.update({'retrievedAt': utc_now(), 'sha256': hashlib.sha256(body).hexdigest()})
            with path.open('xb') as handle:
                handle.write(body)
            save(receipt_path, receipt)
            time.sleep(1)
        return body


def explicit_date(description: str, date: str, market_slug: str = '', event_date: str = '') -> bool:
    found = re.findall(r'scheduled (?:for|to take place on) ([A-Za-z]+ \d{1,2}, \d{4})', description)
    dates = set()
    for value in found:
        try:
            dates.add(datetime.strptime(value, '%B %d, %Y').date().isoformat())
        except ValueError:
            continue
    if dates:
        return dates == {date}
    month_days = re.findall(r'scheduled (?:for|to take place on) ([A-Za-z]+ \d{1,2})(?!\d|, \d{4})', description)
    expected = datetime.fromisoformat(date).strftime('%B %d').replace(' 0', ' ')
    return month_days == [expected] and market_slug.endswith(date) and event_date == date


def market_identity(market: dict, row: dict, event: dict | None = None) -> dict | None:
    if market.get('sportsMarketType') != 'moneyline':
        return None
    description = market.get('description', '').replace('“', '"').replace('”', '"')
    if not explicit_date(description, row['date'], market.get('slug', ''), (event or {}).get('eventDate', '')):
        return None
    labels = json.loads(market.get('outcomes', '[]'))
    tokens = json.loads(market.get('clobTokenIds', '[]'))
    if len(labels) != 2 or len(tokens) != 2 or len(set(tokens)) != 2:
        return None
    if not all(isinstance(t, str) and t.isdigit() for t in tokens):
        return None
    label_keys = [norm(label).rstrip('.') for label in labels]
    if len(set(label_keys)) != 2:
        return None
    rules = re.findall(r'If (.*?) is declared the winner of this bout, this market will resolve to "([^"]+)"', description)
    rules.extend((name, label) for label, name in re.findall(
        r'(?:This market|It) will resolve to "([^"]+)" if ([^\n]+?) is (?:officially )?declared the winner', description))
    names = labels
    if rules:
        by_label = {}
        for name, label in rules:
            key = norm(label).rstrip('.')
            if key in by_label:
                return None
            by_label[key] = name
        if len(by_label) != 2 or any(key not in by_label for key in label_keys):
            return None
        names = [by_label[key] for key in label_keys]
    sides = []
    for index, name in enumerate(names):
        matches = [side for side in ['red', 'blue'] if norm(name) == norm(row[f'{side}_name'])]
        reviewed_alias = (row['fight_id'] == 'ad68450307a21ab1' and row['date'] == '2025-05-31'
                          and name == 'Zachary Reese' and row['red_fighter_id'] == '23dec7c47cb418f8'
                          and row['blue_fighter_id'] == '866fd7b1a6c90e7f')
        if not matches and reviewed_alias:
            matches = ['red']
        if len(matches) != 1:
            return None
        side = matches[0]
        sides.append({'tokenId': tokens[index], 'outcomeIndex': index, 'outcomeLabel': labels[index],
                      'sourceName': name, 'side': side, 'fighterId': row[f'{side}_fighter_id'],
                      'reviewedAlias': reviewed_alias})
    if {s['side'] for s in sides} != {'red', 'blue'}:
        return None
    condition = market.get('conditionId', '')
    if not re.fullmatch(r'0x[0-9a-fA-F]{64}', condition):
        return None
    created = pd.Timestamp(market['createdAt'])
    if created.tzinfo is None or created >= pd.Timestamp(row['quote_observed_at']):
        return None
    return {'fightId': row['fight_id'], 'fightDate': row['date'],
            'redFighterId': row['red_fighter_id'], 'blueFighterId': row['blue_fighter_id'],
            'conditionId': condition, 'marketId': market['id'], 'marketSlug': market.get('slug'),
            'createdAt': market['createdAt'], 'tokens': sides, 'settlementRules': description,
            'resolutionSource': market.get('resolutionSource'), 'historicalRulesRevisionVerified': False,
            'quoteObservedAt': row['quote_observed_at'], 'eventDateMetadata': (event or {}).get('eventDate')}


def select_execution(records: list[dict], identity: dict, max_age: int = 3600,
                     response_truncated: bool = False) -> dict:
    observed = int(pd.Timestamp(identity['quoteObservedAt']).timestamp())
    tokens = {item['tokenId']: item for item in identity['tokens']}
    valid = []
    rejected = 0
    for record in records:
        token = tokens.get(str(record.get('asset_id')))
        timestamp = record.get('block_timestamp')
        price, amount = record.get('price'), record.get('usdc_amount')
        sequence = record.get('outcome_seq')
        if (record.get('condition_id') != identity['conditionId'] or token is None
                or isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp) or int(timestamp) != timestamp
                or isinstance(sequence, bool) or sequence != token['outcomeIndex'] + 1
                or isinstance(price, bool) or not isinstance(price, (int, float))
                or not math.isfinite(price) or not 0 < price < 1
                or isinstance(amount, bool) or not isinstance(amount, (int, float))
                or not math.isfinite(amount) or amount <= 0):
            rejected += 1
            continue
        age = observed - int(timestamp)
        if 0 <= age <= max_age:
            valid.append({'timestamp': int(timestamp), 'ageSeconds': age, 'tokenId': str(record['asset_id']),
                          'tokenPrice': price, 'usdcAmount': amount,
                          'redProbabilityProxy': price if token['side'] == 'red' else 1-price})
    result = {'recordsRead': len(records), 'invalidRecords': rejected, 'eligibleRecords': len(valid),
              'bothTokensObserved': len({r['tokenId'] for r in valid}) == 2,
              'activityCountIsLowerBound': response_truncated, 'maxAgeSeconds': max_age}
    if not valid:
        return {**result, 'status': 'no_execution_in_window'}
    latest = max(r['timestamp'] for r in valid)
    last = [r for r in valid if r['timestamp'] == latest]
    prices = {round(r['redProbabilityProxy'], 12) for r in last}
    page_timestamps = [r.get('block_timestamp') for r in records]
    page_times_valid = all(isinstance(t, (int, float)) and not isinstance(t, bool)
                           and math.isfinite(t) and int(t) == t for t in page_timestamps)
    incomplete_latest = response_truncated and (not page_times_valid or min(page_timestamps) >= latest)
    if len(prices) != 1 or incomplete_latest:
        return {**result, 'status': 'ambiguous_latest_second', 'latestTimestamp': latest}
    return {**result, 'status': 'candidate_execution', 'latestTimestamp': latest,
            'ageSeconds': observed-latest, 'redProbabilityProxy': next(iter(prices)),
            'latestRecords': last}


def initialize(output: Path) -> list[dict]:
    roster_path = output / 'candidate-roster.csv'
    if not roster_path.exists():
        roster_path = OLD / 'candidate-roster.csv'
    roster = pd.read_csv(roster_path).sort_values(['date', 'fight_id']).to_dict('records')
    path = output / 'protocol.json'
    if not path.exists():
        save(path, {'createdAt': utc_now(), 'purpose': 'Source acquisition and coverage only; no fitting or outcome scoring',
                    'candidateBouts': len(roster), 'candidateSHA256': digest(OLD / 'candidate-roster.csv'),
                    'priorReportSHA256': digest(OLD / 'report-final.json'), 'codeSHA256AtFreeze': digest(Path(__file__)),
                    'primaryWindowSeconds': 3600, 'diagnosticWindowSeconds': 86400,
                    'diagnosticMeaning': 'Older prior execution plus age, never a current quote or primary replacement',
                    'sampleExposure': 'Nine prior discovery bouts and earlier outcomes are already exposed; no future validation claim',
                    'timeRule': 'Execution timestamp<=actual Kalshi observed timestamp, not nominal cutoff',
                    'identity': 'Exact pair/date and token IDs; case/diacritic normalization only, plus previously reviewed Reese exact bout alias',
                    'selection': 'Latest valid execution; conflicting red prices at same latest second excluded',
                    'columns': COLUMNS, 'excludedColumns': ['p_event', 'D', 'winning_outcome_label', 'resolved_at', 'resolution_status'],
                    'pinnedRevisions': REVISIONS, 'additionalByteLimit': BYTE_LIMIT,
                    'acquisitionOrder': 'All499 fixed fights in chronological date/fightID order; stop at budget or access error, never at favorable outcomes',
                    'probe': 'One v1 filter positive-control query after v2 prior temporaryHTTP500; no denied403/451 route retried',
                    'rateLimit': 'At least1 second between requests; no retries in this collector',
                    'metadataHistoricalAvailability': 'Retrospective rules/metadata revisions not proven',
                    'tradeSemantics': 'Executed token price only; no orderbook or current fillability',
                    'noOutcomeFieldsInNormalizedOutputs': True})
    protocol = json.loads(path.read_text())
    if protocol['candidateSHA256'] != digest(roster_path):
        raise AcquisitionError('Frozen candidate source changed')
    return roster


def probe(store: HttpStore) -> None:
    params = {'dataset': 'TimeSeventeen/Polymarket-v1', 'config': 'daily_aligned', 'split': 'train',
              'where': '"condition_id"=\'0x7b6289e51a7e241f601689f84e8770a5a9ebb27eb04c90f8e8115e52744b7990\' AND "block_timestamp">=1748557680 AND "block_timestamp"<=1748561280',
              'orderby': '"block_timestamp" DESC', 'offset': 0, 'length': 100}
    raw = store.fetch('filter-positive-control.raw.json', 'https://datasets-server.huggingface.co/filter?' + urlencode(params),
                      2_000_000, revision=REVISIONS['v1'])
    data = json.loads(raw)
    print(json.dumps({'stage': 'probe', 'returnedRows': len(data.get('rows', [])),
                      'numRowsTotal': data.get('num_rows_total')}), flush=True)


def metadata(store: HttpStore, roster: list[dict], output: Path, cohort: str) -> None:
    decisions = []
    for date in sorted({r['date'] for r in roster}):
        params = {'tag_slug': 'ufc', 'end_date_min': f'{date}T00:00:00Z',
                  'end_date_max': f'{(datetime.fromisoformat(date)+timedelta(days=2)).date()}T00:00:00Z', 'limit': 100}
        events = json.loads(store.fetch(f'gamma-{date}.raw.json', 'https://gamma-api.polymarket.com/events?' + urlencode(params), 5_000_000))
        if not isinstance(events, list) or len(events) == 100:
            raise AcquisitionError('Gamma response malformed or possibly paginated; stop')
        for row in [r for r in roster if r['date'] == date]:
            matches = []
            for event in events:
                for market in event.get('markets', []):
                    identity = market_identity(market, row, event)
                    if identity:
                        matches.append({**identity, 'eventId': event['id'], 'metadataFile': f'http/gamma-{date}.raw.json',
                                        'metadataSHA256': digest(store.directory / f'gamma-{date}.raw.json')})
            unique = {r['conditionId']: r for r in matches}
            inconsistent = any(previous['tokens'] != current['tokens']
                               for previous in matches for current in matches
                               if previous['conditionId'] == current['conditionId'])
            decisions.append({'fightId': row['fight_id'], 'date': date,
                              'status': 'exact_pair' if len(unique) == 1 and not inconsistent else 'unmatched_or_ambiguous_pair_date',
                              'identities': list(unique.values())})
        print(json.dumps({'stage': 'metadata', 'date': date, 'boutsProcessed': len(decisions),
                          'matched': sum(d['status'] == 'exact_pair' for d in decisions), 'downloadedBytes': store.used}), flush=True)
    save(output / f'identity-decisions-{cohort}.json', decisions)


def normalize_metadata(store: HttpStore, roster: list[dict], output: Path, cohort: str) -> None:
    sources = []
    for path in sorted(store.directory.glob('gamma*.raw.json')):
        receipt = json.loads(path.with_name(path.name+'.receipt.json').read_text())
        if receipt.get('status') != 200 or receipt.get('error') or digest(path) != receipt['sha256']:
            raise AcquisitionError('Gamma source receipt invalid')
        events = json.loads(path.read_text())
        if not isinstance(events, list) or len(events) >= 100:
            raise AcquisitionError('Gamma response malformed or possibly paginated')
        sources.append((path, events))
    decisions = []
    for row in roster:
        matches = []
        for path, events in sources:
            for event in events:
                for market in event.get('markets', []):
                    identity = market_identity(market, row, event)
                    if identity:
                        matches.append({**identity, 'eventId': event['id'],
                                        'metadataFile': str(path.relative_to(output)), 'metadataSHA256': digest(path)})
        unique = {m['conditionId']: m for m in matches}
        inconsistent = any(a['tokens'] != b['tokens'] for a in matches for b in matches
                           if a['conditionId'] == b['conditionId'])
        decisions.append({'fightId': row['fight_id'], 'date': row['date'],
                          'status': 'exact_pair' if len(unique) == 1 and not inconsistent else 'unmatched_or_ambiguous_pair_date',
                          'identities': list(unique.values())})
    save(output / f'identity-decisions-{cohort}-final.json', decisions)
    print(json.dumps({'stage': 'normalize', 'rows': len(decisions),
                      'matched': sum(d['status'] == 'exact_pair' for d in decisions),
                      'unmatched': [d['fightId'] for d in decisions if d['status'] != 'exact_pair']}), flush=True)


def required_dates(observed_at: str, seconds: int = 86400) -> list[str]:
    observed = pd.Timestamp(observed_at)
    if observed.tzinfo is None:
        raise AcquisitionError('Observation timestamp must have timezone')
    observed = observed.tz_convert('UTC')
    return [str(t.date()) for t in pd.date_range((observed-pd.Timedelta(seconds=seconds)).normalize(), observed.normalize())]


def load_partition(directory: Path, item: dict) -> pd.DataFrame:
    path = directory / item['projectedFile']
    if not path.resolve().is_relative_to(directory.resolve()):
        raise AcquisitionError('Projected source escapes source directory')
    if not item.get('sourceComplete') or not item.get('fullSourceHashVerified'):
        raise AcquisitionError('Incomplete source partition')
    version = item.get('sourceVersion', 'v1')
    if digest(path) != item['projectedSHA256'] or item['sourceRevision'] != REVISIONS.get(version):
        raise AcquisitionError('Projected source hash or revision mismatch')
    with path.open('rb') as handle:
        parquet = ParquetFile(handle)
        if set(parquet.columns) != set(COLUMNS):
            raise AcquisitionError('Projected source has unexpected columns')
        frame = parquet.to_pandas(columns=COLUMNS)
    if len(frame) != item['rows']:
        raise AcquisitionError('Projected source row count mismatch')
    if not frame.empty:
        minimum, maximum = int(frame.block_timestamp.min()), int(frame.block_timestamp.max())
        start = int(pd.Timestamp(item['partitionDate'], tz='UTC').timestamp())
        if (minimum != item['minTimestamp'] or maximum != item['maxTimestamp']
                or not start <= minimum <= maximum < start+86400):
            raise AcquisitionError('Partition epoch range does not match recorded UTC day')
    return frame


def coverage(roster: list[dict], output: Path, cohort: str) -> None:
    identity_path = output / f'identity-decisions-{cohort}-final.json'
    identities = {row['fightId']: row for row in json.loads(identity_path.read_text())}
    if set(identities) != {row['fight_id'] for row in roster}:
        raise AcquisitionError('Identity manifest differs from fixed cohort')
    directory = output / 'partitions'
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('completed') is not True or manifest.get('status') != 'complete' or manifest.get('failedFiles'):
        raise AcquisitionError('Source collection is not complete')
    files = {item['partitionDate']: item for item in manifest['files']}
    needed = {date for row in roster for date in required_dates(row['quote_observed_at'])}
    if set(files) != needed or len(files) != len(manifest['files']):
        raise AcquisitionError('Source partition set differs from fixed windows')
    save(output / 'coverage-input-freeze.json', {
        'createdAt': utc_now(), 'identitySHA256': digest(identity_path), 'partitionManifestSHA256': digest(manifest_path),
        'collectorSHA256': digest(Path(__file__)), 'testsSHA256': digest(ROOT/'tests/test_peer_prices.py'),
        'primaryWindowSeconds': 3600, 'diagnosticWindowSeconds': 86400,
        'futureFitAvailabilityGate': 'Parent declared >=40 distinct strict1h training bouts before price coverage was inspected; this collector fits/scores nothing',
        'timestampStatus': 'Publisher block timestamp assertion and raw-cleaned consistency; independent node/explorer proof unavailable, provisional research only'})
    frames = {}
    decisions, normalized = [], []
    for row in roster:
        decision = identities[row['fight_id']]
        base = {key: row[key] for key in ['fight_id', 'event_id', 'date', 'red_fighter_id', 'blue_fighter_id',
                                        'red_name', 'blue_name', 'quote_observed_at', 'cutoff']}
        if decision['status'] != 'exact_pair' or len(decision['identities']) != 1:
            decisions.append({**base, 'status': 'identity_unresolved'})
            normalized.append({**base, 'status': 'identity_unresolved', 'peer_1h': None, 'peer_24h_diagnostic': None})
            continue
        identity = decision['identities'][0]
        if (identity['redFighterId'] != row['red_fighter_id'] or identity['blueFighterId'] != row['blue_fighter_id']
                or identity['fightDate'] != row['date'] or identity['quoteObservedAt'] != row['quote_observed_at']):
            raise AcquisitionError('Identity orientation or observation changed')
        dates = required_dates(row['quote_observed_at'])
        frames = {date: frame for date, frame in frames.items() if date in dates}
        records = []
        for date in dates:
            if date not in frames:
                frames[date] = load_partition(directory, files[date])
            frame = frames[date]
            records.extend(frame.loc[frame.condition_id == identity['conditionId'], COLUMNS].to_dict('records'))
        primary = select_execution(records, identity, 3600)
        diagnostic = select_execution(records, identity, 86400)
        decisions.append({**base, 'status': 'source_available', 'conditionId': identity['conditionId'],
                          'metadataFile': identity['metadataFile'], 'metadataSHA256': identity['metadataSHA256'],
                          'partitionDates': dates, 'primary': primary, 'diagnostic': diagnostic})
        normalized.append({**base, 'status': primary['status'], 'condition_id': identity['conditionId'],
                           'peer_1h': primary.get('redProbabilityProxy'), 'peer_1h_age_seconds': primary.get('ageSeconds'),
                           'peer_24h_diagnostic': diagnostic.get('redProbabilityProxy'),
                           'peer_24h_age_seconds': diagnostic.get('ageSeconds'),
                           'peer_timestamp_verification': 'publisher_asserted_provisional'})
    save(output / f'coverage-decisions-{cohort}.json', decisions)
    frame = pd.DataFrame(normalized)
    csv_path = output / f'peer-prices-{cohort}.csv'
    with csv_path.open('x') as handle:
        frame.to_csv(handle, index=False)
    primary_count = int(frame.peer_1h.notna().sum())
    report = {'createdAt': utc_now(), 'cohort': cohort, 'candidateBouts': len(roster),
              'exactIdentityPairs': sum(d['status'] == 'source_available' for d in decisions),
              'strict1hCandidateBouts': primary_count, 'diagnostic24hCandidateBouts': int(frame.peer_24h_diagnostic.notna().sum()),
              'primaryBothTokensObserved': sum(d.get('primary', {}).get('bothTokensObserved', False) for d in decisions),
              'identityUnresolved': int((frame.status == 'identity_unresolved').sum()),
              'primaryAmbiguousLatestSecond': int((frame.status == 'ambiguous_latest_second').sum()),
              'primaryNoExecutionInWindow': int((frame.status == 'no_execution_in_window').sum()),
              'atLeast40StrictTrainingBouts': primary_count >= 40,
              'pricesCSV': csv_path.name, 'pricesSHA256': digest(csv_path),
              'sourceManifestSHA256': digest(manifest_path), 'identityManifestSHA256': digest(identity_path),
              'codeSHA256': digest(Path(__file__)), 'downloadedPartitionBytes': manifest['downloadedBytes'],
              'noOutcomeScoring': True, 'noModelFitting': True,
              'limitations': ['Execution price proxy, not contemporaneous executable quote',
                             'One-hour primary unchanged;24h age-bearing diagnostic is separate',
                             'Retrospective metadata/rule revisions and independently unverified block timestamps',
                             'No bid/ask, depth, fill guarantee, or net trading-cost claim']}
    save(output / f'coverage-report-{cohort}.json', report)
    print(json.dumps(report), flush=True)


def required_source_keys(observed_at: str) -> list[str]:
    keys = []
    for date in required_dates(observed_at, 3600):
        if date <= '2026-04-28':
            keys.append(f'v1:{date}')
        if date >= '2026-04-03':
            keys.append(f'v2:{date}')
    return keys


def coverage_later(roster: list[dict], output: Path, cohort: str) -> None:
    identity_path = output / f'identity-decisions-{cohort}-final.json'
    identity_rows = json.loads(identity_path.read_text())
    identities = {row['fightId']: row for row in identity_rows}
    if len(identity_rows) != len(roster) or set(identities) != {row['fight_id'] for row in roster}:
        raise AcquisitionError('Later identity manifest differs from fixed cohort')
    directory = output / 'partitions'
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('completed') is not True or manifest.get('status') != 'complete_with_declared_missing'
            or manifest.get('failedFiles')):
        raise AcquisitionError('Later source collection is incomplete')
    files = {item['sourceKey']: item for item in manifest['files']}
    missing = {item['sourceKey']: item for item in manifest['missingSources']}
    required = {item if isinstance(item, str) else item['sourceKey'] for item in manifest['requiredSources']}
    if (len(files) != len(manifest['files']) or set(missing) != {'v2:2026-10-29'}
            or set(files) & set(missing) or set(files) | set(missing) != required or len(required) != 68):
        raise AcquisitionError('Unexpected later source partition set')
    expected_all = pd.read_csv(output/'candidate-roster.csv').to_dict('records')
    expected_map = {r['fight_id']: required_source_keys(r['quote_observed_at']) for r in expected_all}
    received_map = manifest['requiredSourceKeysByFightId']
    if set(received_map) != set(expected_map) or any(set(received_map[k]) != set(v) for k, v in expected_map.items()):
        raise AcquisitionError('Per-bout source windows differ from fixed one-hour rule')
    if {key for keys in expected_map.values() for key in keys} != required:
        raise AcquisitionError('Declared source set is not exact fixed cohort union')
    snapshot_name = f'source-snapshots/collector-coverage-frozen-{cohort}.py'
    snapshot = output / snapshot_name
    snapshot.parent.mkdir(exist_ok=True)
    with snapshot.open('xb') as handle:
        handle.write(Path(__file__).read_bytes())
    freeze_name = f'coverage-input-freeze-{cohort}.json'
    save(output/freeze_name, {'createdAt': utc_now(), 'identitySHA256': digest(identity_path),
                             'partitionManifestSHA256': digest(manifest_path), 'collectorSHA256': digest(snapshot),
                             'collectorSnapshotFile': snapshot_name, 'testsSHA256': digest(ROOT/'tests/test_peer_prices.py'),
                             'primaryWindowSeconds': 3600, 'diagnostic24hAcquired': False,
                             'timestampStatus': 'Publisher asserted, provisional; no independent node/explorer timestamp proof',
                             'sourceRawPolicy': 'New temporary full files removed only after checksum/projection verification; receipts attest raw hashes, retained projections can be rehashed'})
    frames, decisions, normalized = {}, [], []
    for row in roster:
        base = {key: row[key] for key in ['fight_id', 'event_id', 'date', 'red_fighter_id', 'blue_fighter_id',
                                        'red_name', 'blue_name', 'quote_observed_at', 'cutoff']}
        identity_decision = identities[row['fight_id']]
        keys = required_source_keys(row['quote_observed_at'])
        missing_records = [missing[key] for key in keys if key in missing]
        decision = {**base, 'requiredSourceKeys': keys, 'missingRequiredSources': missing_records}
        normalized_row = {**base, 'peer_1h': None, 'peer_1h_age_seconds': None,
                          'peer_24h_diagnostic': None, 'peer_24h_age_seconds': None,
                          'peer_timestamp_verification': 'publisher_asserted_provisional'}
        if identity_decision['status'] != 'exact_pair' or len(identity_decision['identities']) != 1:
            decision['status'] = normalized_row['status'] = 'identity_unresolved'
        else:
            identity = identity_decision['identities'][0]
            if (identity['redFighterId'] != row['red_fighter_id'] or identity['blueFighterId'] != row['blue_fighter_id']
                    or identity['fightDate'] != row['date'] or identity['quoteObservedAt'] != row['quote_observed_at']):
                raise AcquisitionError('Later identity orientation/date/observation changed')
            decision.update({'identity': identity, 'conditionId': identity['conditionId'],
                             'metadataFile': identity['metadataFile'], 'metadataSHA256': identity['metadataSHA256']})
            normalized_row['condition_id'] = identity['conditionId']
            if missing_records:
                decision['status'] = normalized_row['status'] = 'source_unavailable'
            else:
                frames = {key: frame for key, frame in frames.items() if key in keys}
                records = []
                for key in keys:
                    item = files[key]
                    if item['sourceVersion']+':'+item['partitionDate'] != key:
                        raise AcquisitionError('Source version/date key mismatch')
                    if key not in frames:
                        frames[key] = load_partition(directory, item)
                    frame = frames[key]
                    records.extend(frame.loc[frame.condition_id == identity['conditionId'], COLUMNS].to_dict('records'))
                primary = select_execution(records, identity, 3600)
                decision.update({'status': 'source_available', 'primary': primary,
                                 'sourceRecords': [{'sourceKey': key, 'sourceRevision': files[key]['sourceRevision'],
                                                    'sourcePath': files[key]['sourcePath'],
                                                    'projectedSHA256': files[key]['projectedSHA256']} for key in keys]})
                normalized_row.update({'status': primary['status'], 'peer_1h': primary.get('redProbabilityProxy'),
                                       'peer_1h_age_seconds': primary.get('ageSeconds')})
        decisions.append(decision)
        normalized.append(normalized_row)
    save(output/f'coverage-decisions-{cohort}.json', decisions)
    csv_path = output/f'peer-prices-{cohort}.csv'
    frame = pd.DataFrame(normalized)
    with csv_path.open('x') as handle:
        frame.to_csv(handle, index=False)
    report = {'createdAt': utc_now(), 'cohort': cohort, 'candidateBouts': len(roster),
              'exactIdentityPairs': sum('identity' in row for row in decisions),
              'strict1hCandidateBouts': int(frame.peer_1h.notna().sum()),
              'strict1hDistinctEvents': int(frame.loc[frame.peer_1h.notna(), 'event_id'].nunique()),
              'identityUnresolved': int((frame.status == 'identity_unresolved').sum()),
              'sourceUnavailable': int((frame.status == 'source_unavailable').sum()),
              'primaryAmbiguousLatestSecond': int((frame.status == 'ambiguous_latest_second').sum()),
              'primaryNoExecutionInWindow': int((frame.status == 'no_execution_in_window').sum()),
              'diagnostic24hAcquired': False, 'pricesCSV': csv_path.name, 'pricesSHA256': digest(csv_path),
              'sourceManifestSHA256': digest(manifest_path), 'identityManifestSHA256': digest(identity_path),
              'codeSHA256': digest(snapshot), 'inputFreezeFile': freeze_name, 'collectorSnapshotFile': snapshot_name,
              'inputFreezeSHA256': digest(output/freeze_name), 'noOutcomeScoring': True, 'noModelFitting': True,
              'limitations': ['Full fixed denominator retained; unavailable peer is null for market fallback',
                             'Six Oct31 bouts require missing published Oct29 source; never backfill earlier prices',
                             'Execution proxy, not bid/ask midpoint, available liquidity, or profitable trading evidence',
                             'Publisher timestamp assertion and retrospective metadata remain provisional',
                             'Deleted temporary raw files have pinned URL/LFS/full-download receipts; independent rehash is limited to retained projections']}
    save(output/f'coverage-report-{cohort}.json', report)
    print(json.dumps(report), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--stage', choices=['probe', 'metadata', 'normalize', 'metadata-fallback', 'coverage'], required=True)
    parser.add_argument('--fallback-date')
    parser.add_argument('--cohort', choices=['training2025', 'all499', '2026', 'calibration2026', 'test2026'], default='training2025')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    roster = initialize(args.output)
    if args.cohort == 'training2025':
        roster = [row for row in roster if row['date'] < '2026-01-01']
        if len(roster) != 127:
            raise AcquisitionError('Fixed2025 training source cohort changed')
    elif args.cohort in ['2026', 'calibration2026', 'test2026']:
        roster = [row for row in roster if row['date'] >= '2026-01-01']
        if args.cohort == 'calibration2026':
            roster = [row for row in roster if row['date'] < '2026-05-16']
        elif args.cohort == 'test2026':
            roster = [row for row in roster if row['date'] >= '2026-05-16']
        if len(roster) != {'2026': 372, 'calibration2026': 155, 'test2026': 217}[args.cohort]:
            raise AcquisitionError('Fixed2026 cohort changed')
    store = HttpStore(args.output / 'http', limit=100_000_000 if args.cohort in ['2026', 'calibration2026', 'test2026'] else BYTE_LIMIT)
    if args.stage == 'probe':
        probe(store)
    elif args.stage == 'metadata':
        metadata(store, roster, args.output, args.cohort)
    elif args.stage == 'normalize':
        normalize_metadata(store, roster, args.output, args.cohort)
    elif args.stage == 'coverage':
        if args.cohort in ['calibration2026', 'test2026']:
            coverage_later(roster, args.output, args.cohort)
        else:
            coverage(roster, args.output, args.cohort)
    else:
        if args.fallback_date not in {r['date'] for r in roster}:
            raise AcquisitionError('Fallback date must belong to frozen cohort')
        date = args.fallback_date
        params = {'tag_slug': 'ufc', 'end_date_min': f'{date}T00:00:00Z',
                  'end_date_max': f'{(datetime.fromisoformat(date)+timedelta(days=9)).date()}T00:00:00Z', 'limit': 100}
        store.fetch(f'gamma-supplement-{date}.raw.json', 'https://gamma-api.polymarket.com/events?'+urlencode(params), 5_000_000)


if __name__ == '__main__':
    main()
