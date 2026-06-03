import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit


if __package__:
    from . import evaluate_research as base
    from . import evaluate_challenger as challenger
else:
    import evaluate_research as base
    import evaluate_challenger as challenger


QUARANTINE_STATS = ['f9bda28454d49c78']


def cluster_bootstrap(predictions: pd.DataFrame) -> dict:
    rng = np.random.default_rng(942)
    labels = predictions.target.to_numpy()
    first = predictions.p_red_baseline_temperature.to_numpy()
    second = predictions.p_red_challenger_temperature.to_numpy()
    groups = [indices.to_numpy() for _, indices in predictions.groupby('event_id').groups.items()]
    accuracy_first, accuracy_second, brier_difference = [], [], []
    for _ in range(2000):
        rows = np.concatenate([groups[index] for index in rng.integers(0, len(groups), len(groups))])
        accuracy_first.append(float(((first[rows] >= 0.5) == labels[rows]).mean()))
        accuracy_second.append(float(((second[rows] >= 0.5) == labels[rows]).mean()))
        brier_difference.append(float(np.mean((first[rows] - labels[rows]) ** 2 - (second[rows] - labels[rows]) ** 2)))
    return {'method': '2000 event-cluster percentile bootstrap replicates; seed942; descriptive uncertainty',
            'baseline_accuracy_95': np.quantile(accuracy_first, [0.025, 0.975]).tolist(),
            'challenger_accuracy_95': np.quantile(accuracy_second, [0.025, 0.975]).tolist(),
            'brier_improvement_challenger_vs_baseline_95': np.quantile(brier_difference, [0.025, 0.975]).tolist()}


def main(args: argparse.Namespace) -> None:
    data_dir = args.data_dir.resolve()
    out = args.output_dir.resolve()
    source = args.source_dir.resolve()
    audit_slice = source / 'post_2026-05-09'
    baseline_dir = args.baseline_dir.resolve()
    challenger_dir = args.challenger_dir.resolve()
    baseline_artifact_path = baseline_dir / 'provisional_baseline.joblib'
    challenger_artifact_path = challenger_dir / 'provisional_challenger.joblib'
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    files = [baseline_artifact_path, challenger_artifact_path, source / 'manifest.json', source / 'normalization_report.json',
             *[audit_slice / f'{name}.parquet' for name in ['events', 'fights', 'rounds', 'fighters']]]
    for folder in [baseline_artifact_path.parent, challenger_artifact_path.parent]:
        hashes = json.loads((folder / 'artifact_hashes.json').read_text())
        for filename, expected in hashes.items():
            assert base.digest(folder / filename) == expected, (folder, filename)
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': base.digest(Path(__file__)),
                'source_and_artifact_files': {str(path): base.digest(path) for path in files},
                'protocol': {
                    'model_updates': 'NONE: load frozen artifacts, no fitting, selection, calibration, or threshold changes.',
                    'audit_dates': ['2026-05-10', '2026-10-31'], 'thresholds': [0.8, 0.85, 0.9],
                    'splice': 'Original frozen canonical history throughMay9, append newmirror slice only. No original historical rows replaced.',
                    'uncertain_statistics_quarantined': QUARANTINE_STATS,
                    'identity_gap': 'Drop bouts lacking either exact canonical fighter ID; do not infer aliases.',
                    'DOB': 'Preserve original fighter DOBs; append DOBs only for previously unknown fighter IDs.',
                    'confidence_intervals': 'Wilson95 for independent-bout interpretation plus eventcluster bootstrap with2000 replicates/seed942.',
                }}
    base.write_json(out / 'predeclared_manifest.json', manifest)
    original_manifest = json.loads((baseline_dir / 'predeclared_manifest.json').read_text())
    for item in original_manifest['source_files'].values():
        assert base.digest(Path(item['path'])) == item['sha256']
    old_fights = pd.read_parquet(data_dir / 'processed/fights.parquet')
    old_events = pd.read_parquet(data_dir / 'processed/events.parquet')
    old_rounds = pd.read_parquet(data_dir / 'processed/rounds.parquet')
    old_fighters = pd.read_parquet(data_dir / 'processed/fighters.parquet')
    formats = pd.read_csv(data_dir / 'raw/ufcstats/ufc_fight_results.csv', usecols=['URL', 'TIME FORMAT'])
    formats['fight_id'] = formats.URL.str.rstrip('/').str.rsplit('/', n=1).str[-1]
    formats = formats[['fight_id', 'TIME FORMAT']].drop_duplicates().rename(columns={'TIME FORMAT': 'time_format'})
    old_fights = old_fights.merge(formats, on='fight_id', how='left', validate='one_to_one').merge(old_events[['event_id', 'date']], on='event_id', validate='many_to_one')
    new_fights = pd.read_parquet(audit_slice / 'fights.parquet')
    new_events = pd.read_parquet(audit_slice / 'events.parquet')
    new_rounds = pd.read_parquet(audit_slice / 'rounds.parquet')
    new_fighters = pd.read_parquet(audit_slice / 'fighters.parquet')
    new_fights = new_fights.merge(new_events[['event_id', 'date']], on='event_id', validate='many_to_one')
    assert old_fights.date.max() <= pd.Timestamp('2026-05-09')
    assert (new_fights.date > pd.Timestamp('2026-05-09')).all()
    assert not (set(old_fights.fight_id) & set(new_fights.fight_id))
    fights = pd.concat([old_fights, new_fights[old_fights.columns]], ignore_index=True)
    rounds = pd.concat([old_rounds, new_rounds[old_rounds.columns]], ignore_index=True)
    rounds = rounds[~rounds.fight_id.isin(QUARANTINE_STATS)]
    totals, round_quality = base.prepare_round_stats(rounds)
    data = base.reconstruct(fights, totals)
    old_features = pd.read_parquet(baseline_dir / 'reconstructed_features.parquet')
    prefix = data[data.date <= pd.Timestamp('2026-05-09')].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix, old_features)
    fighters = pd.concat([old_fighters, new_fighters[~new_fighters.fighter_id.isin(old_fighters.fighter_id)][old_fighters.columns]], ignore_index=True)
    enriched = challenger.add_context(data, fights, fighters)
    old_challenger_features = pd.read_parquet(challenger_dir / 'development_and_calibration_features.parquet')
    pd.testing.assert_frame_equal(enriched[enriched.date < pd.Timestamp('2025-01-01')].reset_index(drop=True), old_challenger_features)
    audit = enriched[(enriched.date > pd.Timestamp('2026-05-09')) & (enriched.date <= pd.Timestamp('2026-10-31'))].copy()
    assert not audit.fight_id.duplicated().any()
    for side in ('red', 'blue'):
        through = audit[f'{side}_history_through']
        assert (through.isna() | (through < audit.date)).all()
    baseline_artifact = joblib.load(baseline_artifact_path)
    challenger_artifact = joblib.load(challenger_artifact_path)
    baseline_raw = np.mean([base.symmetric_probability(model, audit[baseline_artifact['feature_columns']]) for model in baseline_artifact['models'].values()], axis=0)
    challenger_raw = base.symmetric_probability(challenger_artifact['model'], audit[challenger_artifact['feature_columns']])
    training_prior = old_features.loc[old_features.date < pd.Timestamp('2023-01-01'), 'target'].mean()
    probabilities = {
        'baseline_raw': baseline_raw,
        'baseline_temperature': base.apply_temperature(baseline_raw, baseline_artifact['temperature']),
        'challenger_raw': challenger_raw,
        'challenger_temperature': base.apply_temperature(challenger_raw, challenger_artifact['temperature']),
        'training_red_prior': np.full(len(audit), training_prior),
        'constant_half': np.full(len(audit), 0.5),
        'elo_unfitted': expit(audit.diff_elo.to_numpy() * np.log(10) / 400),
    }
    predictions = audit[['fight_id', 'event_id', 'date', 'red_name', 'blue_name', 'target', 'red_history_through', 'blue_history_through']].reset_index(drop=True)
    for name, probability in probabilities.items():
        predictions[f'p_red_{name}'] = probability
    report = {
        'audit_dates': [str(audit.date.min().date()), str(audit.date.max().date())],
        'audit_bouts': len(audit), 'audit_events': audit.event_id.nunique(),
        'new_source_bouts': len(new_fights), 'new_decisive_bouts': int(new_fights.winner.isin(['red', 'blue']).sum()),
        'new_identity_exclusions': new_fights.loc[new_fights[['red_fighter_id', 'blue_fighter_id']].isna().any(axis=1), 'fight_id'].tolist(),
        'new_draws_excluded': int((new_fights.winner == 'draw').sum()), 'new_no_contests_excluded': int((new_fights.winner == 'nc').sum()),
        'quarantined_duration_statistics': QUARANTINE_STATS,
        'original_history_prefix_identical': True, 'challenger_training_calibration_prefix_identical': True,
        'strict_prior_date_checks': True, 'no_model_updates': True,
        'baseline_artifact_sha256': base.digest(baseline_artifact_path), 'challenger_artifact_sha256': base.digest(challenger_artifact_path),
        'baseline_temperature': baseline_artifact['temperature'], 'challenger_temperature': challenger_artifact['temperature'],
        'challenger_model': challenger_artifact['selected_model'],
        'results': {name: base.metrics(audit.target.to_numpy(), probability) for name, probability in probabilities.items()},
        'event_cluster_bootstrap': cluster_bootstrap(predictions),
        'limitations': ['Source is a versioned third-party mirror of UFCStats; not a time-stamped original pre-fight archive.',
                        'Old histories preserved verbatim; uncertain new identity not resolved, ambiguous duration statistics excluded.',
                        'These results validate only the exact frozen temporary artifacts, not the deployed V3RSUS model.',
                        'The new audit slice is now inspected; it must not be reused as untouched evidence after model changes.',
                        'No observed betting prices were evaluated; predictive accuracy is not profitability.',
                        'Constant-half classification ties choose red deterministically; its Brier/logloss are the neutral-probability reference.'],
    }
    predictions.to_csv(out / 'predictions.csv', index=False)
    audit.to_parquet(out / 'features.parquet', index=False)
    report['elapsed_seconds'] = time.monotonic() - started
    base.write_json(out / 'report.json', report)
    base.write_json(out / 'artifact_hashes.json', {path.name: base.digest(path) for path in out.iterdir() if path.is_file()})
    print(json.dumps({key: report[key] for key in ['audit_dates', 'audit_bouts', 'audit_events', 'new_identity_exclusions', 'event_cluster_bootstrap', 'elapsed_seconds']}, indent=2), flush=True)
    for name, result in report['results'].items():
        print(json.dumps({'model': name, **{key: value for key, value in result.items() if key not in ['calibration_bins']}}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audit frozen baseline and challenger on the untouched May–October 2026 source slice; no fitting.')
    parser.add_argument('--data-dir', type=Path, required=True, help='Original frozen MMA pipeline root.')
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--challenger-dir', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True, help='Verified mirror package with manifest.json and post_2026-05-09/.')
    parser.add_argument('--output-dir', type=Path, required=True, help='New audit output directory.')
    main(parser.parse_args())
