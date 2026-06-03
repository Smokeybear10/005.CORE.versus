import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import pandas as pd


BUNDLE_FILES = ("fights.parquet", "rounds.parquet", "fighters.parquet")
ID_PATTERN = re.compile(r"[a-f0-9]{16}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_fighter_id(fighters: pd.DataFrame, source_id: str | None = None,
                       name: str | None = None) -> str:
    require(fighters.fighter_id.is_unique, "Fighter registry IDs must be unique")
    if source_id is not None:
        require(isinstance(source_id, str) and ID_PATTERN.fullmatch(source_id) is not None,
                "Invalid explicit fighter ID")
        match = fighters[fighters.fighter_id == source_id]
        require(len(match) == 1, f"Unknown explicit fighter ID: {source_id}")
        if name is not None:
            require(isinstance(name, str) and match.iloc[0]['name'].strip().casefold() == name.strip().casefold(),
                    f"Explicit fighter ID does not match the supplied name: {source_id}")
        return source_id
    require(isinstance(name, str) and bool(name.strip()), "An exact ID or name is required")
    matches = fighters.loc[fighters.name.str.strip().str.casefold() == name.strip().casefold(), 'fighter_id']
    require(len(matches) == 1, f"Unknown or ambiguous fighter name: {name}; provide an exact source ID")
    return matches.iloc[0]


def apply_corrections(fights: pd.DataFrame, rounds: pd.DataFrame, fighters: pd.DataFrame,
                      document: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    require(fights.fight_id.is_unique and fighters.fighter_id.is_unique, "Fight and registry IDs must be unique")
    proposals, round_rows = document.get('proposals'), document.get('roundRows')
    require(isinstance(proposals, list) and bool(proposals) and isinstance(round_rows, list),
            "Nonempty fight corrections and an explicit roundRows inventory are required")
    corrected_fights, corrected_rounds = fights.copy(), rounds.copy()
    seen, expected_round_rows, changes = set(), [], []
    for item in proposals:
        require(isinstance(item, dict), "Invalid fight correction")
        fight_id, side = item.get('fight_id'), item.get('side')
        old_id, new_id = item.get('old_fighter_id'), item.get('proposed_fighter_id')
        require(isinstance(fight_id, str) and ID_PATTERN.fullmatch(fight_id) is not None,
                "Invalid correction fight ID")
        require(side in ('red', 'blue') and (fight_id, side) not in seen, "Duplicate or invalid fight-side correction")
        seen.add((fight_id, side))
        require(old_id != new_id, "A correction must change the fighter ID")
        name = item.get('name')
        require(isinstance(name, str) and bool(name.strip()), "Correction must include the expected fighter name")
        resolve_fighter_id(fighters, old_id, name)
        resolve_fighter_id(fighters, new_id, name)
        match = fights[fights.fight_id == fight_id]
        require(len(match) == 1, f"Unknown correction fight: {fight_id}")
        row = match.iloc[0]
        require(row[f'{side}_fighter_id'] == old_id, f"Old fighter ID mismatch: {fight_id}/{side}")
        require(row[f'{side}_name'] == name and str(pd.Timestamp(row.date).date()) == item.get('date'),
                f"Fight name/date mismatch: {fight_id}")
        require(row.weightclass == item.get('weightclass'), f"Fight division mismatch: {fight_id}")
        other_id = row['blue_fighter_id' if side == 'red' else 'red_fighter_id']
        require(new_id != other_id, f"Correction would put one fighter in both corners: {fight_id}")
        bout_rounds = rounds[rounds.fight_id == fight_id]
        require(set(bout_rounds.fighter_id) == {row.red_fighter_id, row.blue_fighter_id},
                f"Round ownership does not match the original fight: {fight_id}")
        owned = bout_rounds[bout_rounds.fighter_id == old_id]
        require(not owned.duplicated(['fight_id', 'fighter_id', 'round']).any(), f"Ambiguous affected rounds: {fight_id}")
        expected_numbers = item.get('affected_round_numbers')
        require(type(item.get('affected_round_rows')) is int and isinstance(expected_numbers, list)
                and all(type(number) is int and number > 0 for number in expected_numbers),
                "Correction must include exact round counts and positive round numbers")
        require(len(owned) == item['affected_round_rows'] == len(expected_numbers)
                and sorted(owned['round'].tolist()) == sorted(expected_numbers),
                f"Affected round inventory mismatch: {fight_id}")
        for number in expected_numbers:
            expected_round_rows.append((fight_id, number, old_id, new_id))
        corrected_fights.loc[match.index, f'{side}_fighter_id'] = new_id
        corrected_rounds.loc[owned.index, 'fighter_id'] = new_id
        changes.append({'fight_id': fight_id, 'side': side, 'old_fighter_id': old_id,
                        'fighter_id': new_id, 'round_rows': len(owned)})
    require(all(isinstance(item, dict) and type(item.get('round')) is int for item in round_rows),
            "Invalid round correction inventory")
    supplied = [(item.get('fight_id'), item['round'], item.get('old_fighter_id'), item.get('proposed_fighter_id'))
                for item in round_rows]
    require(Counter(supplied) == Counter(expected_round_rows) and len(set(supplied)) == len(supplied),
            "Round corrections do not match the fight-side corrections exactly")
    affected_ids = {fight_id for fight_id, _ in seen}
    for row in corrected_fights[corrected_fights.fight_id.isin(affected_ids)].itertuples():
        group = corrected_rounds[corrected_rounds.fight_id == row.fight_id]
        require(row.red_fighter_id != row.blue_fighter_id and set(group.fighter_id) == {row.red_fighter_id, row.blue_fighter_id},
                f"Invalid corrected fight/round ownership: {row.fight_id}")
        require(not group.duplicated(['fight_id', 'fighter_id', 'round']).any(), f"New duplicate round key: {row.fight_id}")
    return corrected_fights, corrected_rounds, {'fightSideCorrections': len(changes),
                                               'roundOwnershipCorrections': len(expected_round_rows), 'changes': changes}


def repair_bundle(input_bundle: Path, corrections: Path, corrections_sha256: str, output_bundle: Path) -> dict:
    input_bundle, corrections, output_bundle = input_bundle.resolve(), corrections.resolve(), output_bundle.resolve()
    require(not output_bundle.exists() and not output_bundle.is_relative_to(input_bundle),
            "Output must be a new directory outside the source bundle")
    require(re.fullmatch(r'[a-f0-9]{64}', corrections_sha256) is not None, "Provide the reviewed correction SHA256")
    require(digest(corrections) == corrections_sha256, "Correction manifest checksum mismatch")
    document = json.loads(corrections.read_text())
    require(document.get('type') == 'proposed_fight_specific_identity_corrections_not_applied',
            "Unsupported correction manifest")
    expected = {}
    for name, checksum in document.get('activeInputs', {}).items():
        filename = Path(name).name
        if filename in (*BUNDLE_FILES, 'manifest.json'):
            require(filename not in expected, "Ambiguous source fingerprints in correction manifest")
            expected[filename] = checksum
    require(set(expected) == {*BUNDLE_FILES, 'manifest.json'}, "Correction manifest must pin all source bundle files")
    before = {name: digest(input_bundle / name) for name in expected}
    require(before == expected, "Source bundle differs from the reviewed correction inputs")
    source_manifest = json.loads((input_bundle / 'manifest.json').read_text())
    require(all(source_manifest.get('files', {}).get(name) == before[name] for name in BUNDLE_FILES),
            "Source bundle manifest does not match its tables")
    fights, rounds, fighters = [pd.read_parquet(input_bundle / name) for name in BUNDLE_FILES]
    fights, rounds, report = apply_corrections(fights, rounds, fighters, document)
    require({name: digest(input_bundle / name) for name in before} == before
            and digest(corrections) == corrections_sha256, "Inputs changed during repair")
    output_bundle.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.identity-repair-', dir=output_bundle.parent) as temporary:
        stage = Path(temporary) / 'bundle'
        stage.mkdir()
        fights.to_parquet(stage / 'fights.parquet', index=False)
        rounds.to_parquet(stage / 'rounds.parquet', index=False)
        shutil.copyfile(input_bundle / 'fighters.parquet', stage / 'fighters.parquet')
        shutil.copyfile(corrections, stage / 'corrections.json')
        report.update({'schemaVersion': 1, 'createdAt': datetime.now(timezone.utc).isoformat(),
                       'inputBundle': str(input_bundle), 'inputFiles': before, 'correctionsSHA256': corrections_sha256,
                       'inputIntegrityVerified': True, 'modelsUpdated': False, 'runtimeActivated': False,
                       'fieldsChanged': ['fight-side fighter IDs', 'round fighter IDs']})
        (stage / 'repair-report.json').write_text(json.dumps(report, indent=2) + '\n')
        manifest = {'schemaVersion': 1, 'bundleType': 'identity-corrected-research',
                    'dataThrough': source_manifest['dataThrough'], 'createdAt': report['createdAt'],
                    'parentManifestSHA256': before['manifest.json'], 'correctionsSHA256': corrections_sha256,
                    'repairCodeSHA256': digest(Path(__file__)), 'researchOnly': True, 'requiresRetraining': True,
                    'runtimeActivated': False, 'files': {path.name: digest(path) for path in stage.iterdir()},
                    'quarantinedStatFightIds': source_manifest.get('quarantinedStatFightIds', []),
                    'unresolvedIdentityFightIds': source_manifest.get('unresolvedIdentityFightIds', [])}
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        require({name: digest(input_bundle / name) for name in before} == before
                and digest(corrections) == corrections_sha256, "Inputs changed before output publication")
        output_bundle.mkdir(exist_ok=False)
        for path in sorted(stage.iterdir(), key=lambda path: (path.name == 'manifest.json', path.name)):
            path.rename(output_bundle / path.name)
        for path in output_bundle.iterdir():
            path.chmod(0o444)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Apply reviewed fight-specific identity repairs to a new research bundle')
    parser.add_argument('--input-bundle', type=Path, required=True)
    parser.add_argument('--corrections', type=Path, required=True)
    parser.add_argument('--corrections-sha256', required=True)
    parser.add_argument('--output-bundle', type=Path, required=True)
    args = parser.parse_args()
    report = repair_bundle(args.input_bundle, args.corrections, args.corrections_sha256, args.output_bundle)
    print(json.dumps({'outputBundle': str(args.output_bundle), **report}, indent=2))


if __name__ == '__main__':
    main()
