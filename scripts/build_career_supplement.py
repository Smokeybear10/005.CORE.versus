import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / 'data/evaluation'
SOURCE_MANIFEST_SHA256 = '9bf5b026af39c533af39ed674b2bd5b385587869116e9732f6ec929a2fab0148'
IDENTITY_PROOF_SHA256 = '69d0f556503fc4155a6585a86b25c6d471f6ad7ebd5fea7af94731c69fefa9f4'
RELEASE_URL = 'https://github.com/Jadevit/fight-canon-db/releases/tag/db-2026-11-03-2206'
HOLDS = {323526: 'primary_method_disagreement', 151925: 'mma_discipline_unverified'}
HISTORY_COLUMNS = ['fighter_id', 'opponent_id', 'canonical_fighter_id', 'canonical_opponent_id',
                   'canonical_fight_id', 'canonical_event_id', 'date', 'result', 'opponent_result',
                   'fighter_name', 'opponent_name', 'promotion', 'event_name', 'method',
                   'source_aliases_json', 'source_database_retrieved_at']


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def valid_date(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def prior_history(history: pd.DataFrame, fighter_id: str, event_date: str) -> pd.DataFrame:
    require(valid_date(event_date), 'Use an exact YYYY-MM-DD event date')
    return history[(history.fighter_id == fighter_id) & (history.date < event_date)].copy()


def assemble(connection: sqlite3.Connection, current: pd.DataFrame, registry: pd.DataFrame,
             extra_identities: dict[str, int], cutoff: str, retrieved_at: str) -> tuple[dict[str, pd.DataFrame], dict]:
    require(valid_date(cutoff), 'Invalid inclusive cutoff')
    require(current.fight_id.is_unique and registry.fighter_id.is_unique, 'Current fight and fighter IDs must be unique')
    aliases = pd.read_sql_query('SELECT fighter_id, source, source_id, name FROM fighter_aliases', connection)
    target_ids = set(registry.fighter_id) | set(extra_identities)
    crosswalk = aliases[(aliases.source == 'ufcstats') & aliases.source_id.isin(target_ids)][
        ['source_id', 'fighter_id']].drop_duplicates().rename(columns={
            'source_id': 'fighter_id', 'fighter_id': 'canonical_fighter_id'})
    require(set(crosswalk.fighter_id) == target_ids and crosswalk.fighter_id.is_unique
            and crosswalk.canonical_fighter_id.is_unique, 'Missing or ambiguous exact UFCStats identity mapping')
    to_canonical = dict(zip(crosswalk.fighter_id, crosswalk.canonical_fighter_id))
    require(all(to_canonical[fighter_id] == canonical_id for fighter_id, canonical_id in extra_identities.items()),
            'Extra identity differs from its reviewed proof')
    to_fighter = {value: key for key, value in to_canonical.items()}
    names = pd.read_sql_query('SELECT fighter_id, name FROM fighters', connection).set_index('fighter_id').name.to_dict()
    crosswalk['name'] = crosswalk.canonical_fighter_id.map(names)
    crosswalk['extra_verified_identity'] = crosswalk.fighter_id.isin(extra_identities)
    placeholders = ','.join('?' for _ in to_fighter)
    source = pd.read_sql_query(f'''SELECT DISTINCT f.fight_id canonical_fight_id, f.event_id canonical_event_id,
        f.bout_type, f.overturned, f.no_contest, f.method, e.date, e.name event_name, e.promotion
        FROM fights f JOIN events e USING(event_id) JOIN fight_participants p USING(fight_id)
        WHERE p.fighter_id IN ({placeholders})''', connection, params=sorted(to_fighter))
    require(source.canonical_fight_id.is_unique, 'Duplicate canonical fight records')
    people = pd.read_sql_query('SELECT fight_id, fighter_id, result FROM fight_participants', connection)
    people = people[people.fight_id.isin(source.canonical_fight_id)].sort_values(['fight_id', 'fighter_id'])
    participants = {int(key): group.to_dict('records') for key, group in people.groupby('fight_id', sort=False)}
    fight_aliases = pd.read_sql_query('SELECT fight_id, source, source_id FROM fight_aliases', connection).drop_duplicates()
    fight_aliases = fight_aliases.sort_values(['fight_id', 'source', 'source_id'])
    ufc_aliases = fight_aliases[fight_aliases.source == 'ufcstats']
    require(not ufc_aliases.source_id.duplicated().any(), 'An exact UFCStats fight alias maps to multiple bouts')
    existing_aliases = dict(zip(ufc_aliases.source_id, ufc_aliases.fight_id))
    by_alias = defaultdict(list)
    for row in fight_aliases[fight_aliases.fight_id.isin(source.canonical_fight_id)].itertuples():
        by_alias[row.fight_id].append({'source': row.source, 'sourceID': row.source_id,
                                      'sourceURL': f'http://ufcstats.com/fight-details/{row.source_id}'
                                      if row.source == 'ufcstats' else None, 'datasetURL': RELEASE_URL})
    source_by_id = source.set_index('canonical_fight_id').to_dict('index')
    current_pairs, existing_canonical, overlap = set(), set(), []
    for row in current.itertuples():
        canonical = existing_aliases.get(row.fight_id)
        require(canonical in source_by_id, f'Cannot prove current-bout overlap exclusion: {row.fight_id}')
        existing_canonical.add(canonical)
        day = str(pd.Timestamp(row.date).date())
        actual = source_by_id[canonical]
        ps = participants.get(canonical, [])
        reasons = []
        if len(ps) == 2 and len({person['fighter_id'] for person in ps}) == 2:
            pair = sorted(person['fighter_id'] for person in ps)
            current_pairs.add((day, *pair))
            if valid_date(actual['date']):
                current_pairs.add((actual['date'], *pair))
        ids = (row.red_fighter_id, row.blue_fighter_id)
        if not all(fighter_id in to_canonical for fighter_id in ids) or ids[0] == ids[1]:
            reasons.append('current_identity_unresolved')
        else:
            pair = sorted(to_canonical[fighter_id] for fighter_id in ids)
            current_pairs.add((day, *pair))
            if sorted(person['fighter_id'] for person in ps) != pair:
                reasons.append('participant_conflict')
            results = {'red': ('W', 'L'), 'blue': ('L', 'W'), 'draw': ('D', 'D'), 'nc': ('NC', 'NC')}.get(row.winner)
            expected = dict(zip([to_canonical[fighter_id] for fighter_id in ids], results)) if results else {}
            if {person['fighter_id']: person['result'] for person in ps} != expected:
                reasons.append('outcome_conflict')
        if actual['date'] != day:
            reasons.append('date_conflict')
        overlap.append({'fight_id': row.fight_id, 'canonical_fight_id': canonical, 'current_date': day,
                        'source_date': actual['date'], 'reasons_json': json_text(reasons), 'passes': not reasons})
    require(not any('participant_conflict' in row['reasons_json'] for row in overlap),
            'Repair known current fighter-identity conflicts before integrating histories')
    pairs = defaultdict(list)
    for row in source.itertuples():
        ps = participants.get(row.canonical_fight_id, [])
        if valid_date(row.date) and len(ps) == 2 and len({person['fighter_id'] for person in ps}) == 2:
            pairs[(row.date, *sorted(person['fighter_id'] for person in ps))].append(row.canonical_fight_id)
    accepted, excluded, history = [], [], []
    for row in source.sort_values(['date', 'canonical_fight_id'], na_position='last').itertuples(index=False):
        ps = participants.get(row.canonical_fight_id, [])
        reasons = []
        pair_valid = (len(ps) == 2 and len({person['fighter_id'] for person in ps}) == 2
                      and all(person['fighter_id'] in names for person in ps))
        results = sorted(str(person['result']) for person in ps)
        results_valid = results in [['L', 'W'], ['D', 'D'], ['NC', 'NC']]
        if not valid_date(row.date):
            reasons.append('missing_or_invalid_date')
        elif row.date > cutoff:
            reasons.append('after_cutoff')
        if row.bout_type != 'pro':
            reasons.append('professional_status_not_explicit')
        if not pair_valid:
            reasons.append('invalid_participant_pair')
        if not results_valid:
            reasons.append('invalid_completed_results')
        if row.overturned != 0:
            reasons.append('overturned')
        if row.no_contest not in (0, 1) or results_valid and bool(row.no_contest) != (results == ['NC', 'NC']):
            reasons.append('no_contest_flag_conflict')
        if row.canonical_fight_id in existing_canonical:
            reasons.append('already_in_current_bundle')
        key = (row.date, *sorted(person['fighter_id'] for person in ps)) if pair_valid and valid_date(row.date) else None
        if key is not None and len(pairs[key]) > 1:
            reasons.append('ambiguous_pair_date')
        if key in current_pairs:
            reasons.append('current_bundle_pair_date_overlap')
        if row.canonical_fight_id in HOLDS:
            reasons.append(HOLDS[row.canonical_fight_id])
        if not by_alias[row.canonical_fight_id]:
            reasons.append('missing_source_alias')
        record = {**row._asdict(), 'participants_json': json_text(ps),
                  'source_aliases_json': json_text(by_alias[row.canonical_fight_id]),
                  'source_database_retrieved_at': retrieved_at, 'exclusion_reasons_json': json_text(reasons)}
        if reasons:
            excluded.append(record)
            continue
        accepted.append(record)
        for person in ps:
            if person['fighter_id'] not in to_fighter:
                continue
            opponent = next(other for other in ps if other['fighter_id'] != person['fighter_id'])
            history.append({'fighter_id': to_fighter[person['fighter_id']], 'opponent_id': to_fighter.get(opponent['fighter_id']),
                            'canonical_fighter_id': person['fighter_id'], 'canonical_opponent_id': opponent['fighter_id'],
                            'canonical_fight_id': row.canonical_fight_id, 'canonical_event_id': row.canonical_event_id,
                            'date': row.date, 'result': person['result'], 'opponent_result': opponent['result'],
                            'fighter_name': names[person['fighter_id']], 'opponent_name': names[opponent['fighter_id']],
                            'promotion': row.promotion, 'event_name': row.event_name, 'method': row.method,
                            'source_aliases_json': record['source_aliases_json'], 'source_database_retrieved_at': retrieved_at})
    history = pd.DataFrame(history, columns=HISTORY_COLUMNS).sort_values(['fighter_id', 'date', 'canonical_fight_id'])
    require(not history.duplicated(['fighter_id', 'canonical_fight_id']).any(), 'Duplicate fighter/bout history')
    tables = {'fighter_bouts.parquet': history, 'bouts.parquet': pd.DataFrame(accepted),
              'exclusions.parquet': pd.DataFrame(excluded), 'overlap-audit.parquet': pd.DataFrame(overlap),
              'crosswalk.csv': crosswalk.sort_values('fighter_id')}
    counts = {'bouts': len(accepted), 'fighterBouts': len(history), 'fightersWithHistory': history.fighter_id.nunique(),
              'matchedFighters': len(crosswalk), 'excludedBouts': len(excluded), 'currentBoutsExcluded': len(current),
              'overlapConflicts': sum(not item['passes'] for item in overlap),
              'exclusionReasons': dict(Counter(reason for item in excluded for reason in json.loads(item['exclusion_reasons_json'])))}
    return tables, counts


def verified_manifest(directory: Path, expected_sha256: str | None = None) -> tuple[dict, dict]:
    manifest_path = directory / 'manifest.json'
    if expected_sha256:
        require(digest(manifest_path) == expected_sha256, 'Source manifest checksum mismatch')
    manifest = json.loads(manifest_path.read_text())
    hashes = {str(manifest_path): digest(manifest_path)}
    for name, checksum in manifest['files'].items():
        require(Path(name).name == name and not (directory / name).is_symlink(), 'Unsafe manifest filename')
        hashes[str(directory / name)] = digest(directory / name)
        require(hashes[str(directory / name)] == checksum, f'Input checksum mismatch: {name}')
    return manifest, hashes


def build_bundle(source_directory: Path, input_bundle: Path, identity_proof: Path, output_bundle: Path,
                 source_manifest_sha256: str = SOURCE_MANIFEST_SHA256,
                 identity_proof_sha256: str = IDENTITY_PROOF_SHA256) -> dict:
    source_directory, input_bundle, identity_proof, output_bundle = [path.resolve() for path in
        (source_directory, input_bundle, identity_proof, output_bundle)]
    require(not output_bundle.exists() and not any(output_bundle.is_relative_to(path) for path in (source_directory, input_bundle)),
            'Output must be a new directory outside all source bundles')
    _, source_hashes = verified_manifest(source_directory, source_manifest_sha256)
    input_manifest, input_hashes = verified_manifest(input_bundle)
    require(input_manifest.get('bundleType') == 'identity-corrected-research', 'Use the identity-corrected source bundle')
    require(digest(identity_proof) == identity_proof_sha256, 'Identity proof checksum mismatch')
    proof = json.loads(identity_proof.read_text())
    database = source_directory / 'canon.db'
    require(proof['database']['sha256'] == digest(database), 'Identity proof refers to a different database')
    extra_ids = {row['ufcstatsID']: row['canonicalFighterID'] for row in proof['fighters']}
    require(len(extra_ids) == len(proof['fighters']), 'Duplicate extra identity proof')
    receipt = json.loads((source_directory / 'download-receipt.json').read_text())
    require(receipt['sha256'] == digest(database), 'Database download checksum mismatch')
    license_receipt = json.loads((source_directory / 'upstream-license-receipt.json').read_text())
    require(isinstance(license_receipt.get('content'), str) and 'MIT License' in license_receipt['content'], 'Missing source license')
    before = {**source_hashes, **input_hashes, str(identity_proof): identity_proof_sha256, str(Path(__file__)): digest(Path(__file__))}
    current = pd.read_parquet(input_bundle / 'fights.parquet')
    registry = pd.read_parquet(input_bundle / 'fighters.parquet')
    connection = sqlite3.connect(f'file:{database}?mode=ro&immutable=1', uri=True)
    connection.execute('PRAGMA trusted_schema=OFF')
    connection.execute('PRAGMA query_only=ON')
    try:
        tables, counts = assemble(connection, current, registry, extra_ids, input_manifest['dataThrough'], receipt['retrievedAt'])
    finally:
        connection.close()
    policy = {'history': 'Additional observed professional fight histories only. Current-bundle aliases and pair/date overlaps excluded.',
              'temporal': 'Consumers must use date strictly earlier than the requested event date. Same-day bouts are never prior history.',
              'holds': {str(key): value for key, value in HOLDS.items()},
              'excluded': 'Unknown professional status, invalid dates/results/participants, overturned results, NC flag conflicts, ambiguous pair/date groups and missing source aliases.',
              'identity': 'Exact unique UFCStats source IDs only, including separately reviewed extra identities. No fuzzy names.',
              'availability': 'This retrospective snapshot is known at retrieval; historical publication/revision times are unavailable. Do not call it a point-in-time archive.',
              'coverage': 'Partial, provider-classified professional histories; not independently verified complete career totals.',
              'notIncluded': ['current profile career totals', 'physical measurements', 'rankings', 'round statistics', 'odds']}
    require(all(digest(Path(name)) == checksum for name, checksum in before.items()), 'Inputs changed during ingestion')
    output_bundle.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.career-supplement-', dir=output_bundle.parent) as temporary:
        stage = Path(temporary) / 'bundle'
        stage.mkdir()
        for name, frame in tables.items():
            if name.endswith('.csv'):
                frame.to_csv(stage / name, index=False)
            else:
                frame.to_parquet(stage / name, index=False)
        (stage / 'LICENSE').write_text(license_receipt['content'])
        (stage / 'identity-proof.json').write_bytes(identity_proof.read_bytes())
        manifest = {'schemaVersion': 1, 'bundleType': 'career-supplement', 'createdAt': datetime.now(timezone.utc).isoformat(),
                    'cutoffInclusive': input_manifest['dataThrough'], 'knownAt': receipt['retrievedAt'],
                    'source': {'name': 'Fight Canon', 'url': RELEASE_URL, 'retrievedAt': receipt['retrievedAt'],
                               'databaseSha256': receipt['sha256']},
                    'inputHashes': before, 'inputIntegrityVerified': True, 'policy': policy, 'counts': counts,
                    'files': {path.name: digest(path) for path in stage.iterdir()}, 'modelsTrained': False}
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        require(all(digest(Path(name)) == checksum for name, checksum in before.items()), 'Inputs changed before publication')
        output_bundle.mkdir(exist_ok=False)
        for path in sorted(stage.iterdir(), key=lambda path: (path.name == 'manifest.json', path.name)):
            path.rename(output_bundle / path.name)
        for path in output_bundle.iterdir():
            path.chmod(0o444)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description='Build additional dated career histories from a verified Fight Canon snapshot')
    parser.add_argument('--source-directory', type=Path, default=EVALUATION / 'fight-canon-discovery-20261104')
    parser.add_argument('--input-bundle', type=Path, default=EVALUATION / 'identity-repaired-source-v1')
    parser.add_argument('--identity-proof', type=Path, default=EVALUATION / 'identity-context-discovery-20261104/canonical-id-corroboration.json')
    parser.add_argument('--output-bundle', type=Path, default=EVALUATION / 'career-supplement-v1')
    args = parser.parse_args()
    manifest = build_bundle(args.source_directory, args.input_bundle, args.identity_proof, args.output_bundle)
    print(json.dumps({'outputBundle': str(args.output_bundle), 'manifestSHA256': digest(args.output_bundle / 'manifest.json'),
                      'counts': manifest['counts']}, indent=2))


if __name__ == '__main__':
    main()
