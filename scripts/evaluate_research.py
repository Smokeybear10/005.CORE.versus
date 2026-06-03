import argparse
import hashlib
import json
import math
import platform
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


STAT_COLS = ['sig_str_landed', 'sig_str_attempted', 'td_landed', 'td_attempted', 'sub_att', 'kd', 'ctrl_seconds']
RATE_COLS = ['sig_landed_pm', 'sig_attempted_pm', 'sig_absorbed_pm', 'sig_accuracy', 'sig_defense',
             'td_landed_pm', 'td_attempted_pm', 'td_accuracy', 'td_defense', 'sub_att_pm', 'kd_pm', 'control_share']
CONFIG = {
    'train_start': '2010-01-01', 'train_end_exclusive': '2023-01-01',
    'calibration_start': '2023-01-01', 'calibration_end_exclusive': '2025-01-01',
    'holdout_start': '2025-01-01', 'holdout_end_inclusive': '2026-05-09',
    'primary_model': 'equal_weight_ensemble_temperature',
    'logistic_regression': {'C': 1.0, 'max_iter': 2000, 'random_state': 42},
    'random_forest': {'n_estimators': 200, 'max_depth': 8, 'min_samples_leaf': 10, 'n_jobs': 2, 'random_state': 42},
    'gradient_boosting': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'random_state': 42},
    'calibration': 'one positive temperature, log(T) bounded [-3,3], calibration-period log-loss only',
    'thresholds_predeclared': [0.8, 0.85, 0.9], 'confidence_bin_edges': [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
    'elo': {'initial': 1500.0, 'k': 32.0, 'denominator': 400.0},
    'feature_policy': 'Only bouts with date strictly before prediction date; all same-day feature rows built before any same-day updates. Historical results, rolling per-minute bout stats, pre-bout Elo, layoff, and known division/title/round schedule. No current static profiles, retrospective pedigree, rankings, or odds.',
    'training_policy': 'One row per decisive bout with both fighter IDs. Each training bout augmented with corner-swapped row; preprocessing fitted only to augmented training. Each prediction averaged over both corner orders. No parameter updates from calibration or holdout labels except the declared temperature on calibration.',
    'outcome_policy': 'Binary outcome conditional on a decisive result. Draws and no-contests excluded from fitting/scoring, retained in prior-fight histories; draws update Elo as 0.5, NC does not update Elo.',
    'future_policy': 'Earlier holdout bouts may supply history for later holdout bouts, exactly as online prediction; no same-day history permitted. No fit or calibration updates in holdout.',
    'audit_policy': 'Source files are retrospective snapshots without per-record retrieval/revision timestamps. Computational temporal separation is audited; original historical availability and revision provenance cannot be certified.',
}


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    valid = array[np.isfinite(array)]
    return float(valid.mean()) if len(valid) else np.nan


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if np.isfinite(numerator) and np.isfinite(denominator) and denominator > 0 else np.nan


def prepare_round_stats(rounds: pd.DataFrame) -> tuple[dict, dict]:
    duplicate_mask = rounds.duplicated(['fight_id', 'fighter_id', 'round'], keep=False)
    ambiguous = set(rounds.loc[duplicate_mask, 'fight_id'])
    usable = rounds[~rounds.fight_id.isin(ambiguous)].copy()
    grouped = usable.groupby(['fight_id', 'fighter_id'])
    sums = grouped[STAT_COLS].sum(min_count=1)
    # Partial control-time records are missing, not a sum of only observed rounds.
    sums['ctrl_seconds'] = sums['ctrl_seconds'].where(grouped.ctrl_seconds.count() == grouped.size())
    totals = {key: row.to_dict() for key, row in sums.iterrows()}
    for key, group in grouped:
        totals[key]['rounds_present'] = sorted(group['round'].dropna().astype(int).tolist())
    return totals, {'ambiguous_bout_ids': sorted(ambiguous), 'ambiguous_round_rows': int(duplicate_mask.sum()), 'total_round_rows': len(rounds)}


def bout_rates(row: object, fighter_id: str, opponent_id: str, totals: dict) -> dict:
    missing = {key: np.nan for key in RATE_COLS}
    own = totals.get((row.fight_id, fighter_id))
    opp = totals.get((row.fight_id, opponent_id))
    minutes, seconds = map(int, row.finish_time.split(':'))
    # Early tournament round formats are not assumed to be five-minute rounds.
    if row.date < pd.Timestamp('2000-01-01') or pd.isna(row.num_rounds) or row.num_rounds not in (3, 5):
        return missing
    time_format = row.time_format if isinstance(row.time_format, str) else ''
    expected_format = f"{row.num_rounds} Rnd ({'-'.join(['5'] * row.num_rounds)})"
    if time_format.strip() != expected_format:
        return missing
    duration = (row.finish_round - 1) * 300 + minutes * 60 + seconds
    if own is None or opp is None or not 0 < duration <= row.num_rounds * 300 or minutes * 60 + seconds > 300:
        return missing
    expected_rounds = list(range(1, row.finish_round + 1))
    if own['rounds_present'] != expected_rounds or opp['rounds_present'] != expected_rounds:
        return missing
    for stats in (own, opp):
        if any(not np.isfinite(stats[column]) or stats[column] < 0 for column in STAT_COLS if column != 'ctrl_seconds'):
            return missing
        if stats['sig_str_landed'] > stats['sig_str_attempted'] or stats['td_landed'] > stats['td_attempted']:
            return missing
    elapsed_minutes = duration / 60
    return {
        'sig_landed_pm': own['sig_str_landed'] / elapsed_minutes,
        'sig_attempted_pm': own['sig_str_attempted'] / elapsed_minutes,
        'sig_absorbed_pm': opp['sig_str_landed'] / elapsed_minutes,
        'sig_accuracy': safe_ratio(own['sig_str_landed'], own['sig_str_attempted']),
        'sig_defense': 1 - safe_ratio(opp['sig_str_landed'], opp['sig_str_attempted']),
        'td_landed_pm': own['td_landed'] / elapsed_minutes,
        'td_attempted_pm': own['td_attempted'] / elapsed_minutes,
        'td_accuracy': safe_ratio(own['td_landed'], own['td_attempted']),
        'td_defense': 1 - safe_ratio(opp['td_landed'], opp['td_attempted']),
        'sub_att_pm': own['sub_att'] / elapsed_minutes,
        'kd_pm': own['kd'] / elapsed_minutes,
        'control_share': own['ctrl_seconds'] / duration if 0 <= own['ctrl_seconds'] <= duration else np.nan,
    }


def fighter_state(history: list[dict], elo: float, date: pd.Timestamp) -> dict:
    state = {
        'elo': elo, 'career_fights': len(history),
        'career_wins': sum(bout['result'] == 1 for bout in history),
        'career_losses': sum(bout['result'] == 0 for bout in history),
        'layoff_days': (date - history[-1]['date']).days if history else np.nan,
    }
    for window in (3, 5):
        recent = history[-window:]
        state[f'win_score_last{window}'] = finite_mean([bout['result'] for bout in recent])
        state[f'stat_bouts_last{window}'] = sum(np.isfinite(bout['sig_landed_pm']) for bout in recent)
        for key in RATE_COLS:
            state[f'{key}_last{window}'] = finite_mean([bout[key] for bout in recent])
    return state


def matchup_features(red: dict, blue: dict, weightclass: str, scheduled_rounds: float, is_title: bool) -> dict:
    return {
        'weightclass': weightclass,
        'ctx_title': int(is_title),
        'ctx_rounds': float(scheduled_rounds) if pd.notna(scheduled_rounds) else np.nan,
        'ctx_min_career': min(red['career_fights'], blue['career_fights']),
        'ctx_max_career': max(red['career_fights'], blue['career_fights']),
        'ctx_min_stats_last5': min(red['stat_bouts_last5'], blue['stat_bouts_last5']),
        **{f'diff_{key}': red[key] - blue[key] for key in red},
    }


def advance_history(day: pd.DataFrame, history: dict, elo: dict, totals: dict | None = None) -> None:
    updates = []
    elo_deltas = defaultdict(float)
    for row in day.itertuples(index=False):
        rid, bid = row.red_fighter_id, row.blue_fighter_id
        score = 1.0 if row.winner == 'red' else 0.0 if row.winner == 'blue' else 0.5 if row.winner == 'draw' else np.nan
        expected = 1 / (1 + 10 ** ((elo[bid] - elo[rid]) / 400))
        for fid, opponent_id, result in ((rid, bid, score), (bid, rid, 1 - score)):
            rates = bout_rates(row, fid, opponent_id, totals) if totals is not None else {}
            updates.append((fid, {
                'date': row.date, 'result': result, 'weightclass': row.weightclass,
                'opponent_elo': elo[opponent_id],
                'result_surprise': score - expected if fid == rid else expected - score,
                **rates,
            }))
        if np.isfinite(score):
            elo_deltas[rid] += 32 * (score - expected)
            elo_deltas[bid] -= 32 * (score - expected)
    for fighter_id, bout in updates:
        history[fighter_id].append(bout)
    for fighter_id, delta in elo_deltas.items():
        elo[fighter_id] += delta


def history_before(fights: pd.DataFrame, totals: dict, date: pd.Timestamp) -> tuple[dict, dict]:
    history = defaultdict(list)
    elo = defaultdict(lambda: 1500.0)
    usable = fights.dropna(subset=['red_fighter_id', 'blue_fighter_id', 'date'])
    usable = usable[usable.date < date].sort_values(['date', 'fight_id'])
    for _, day in usable.groupby('date', sort=True):
        advance_history(day, history, elo, totals)
    return history, elo


def reconstruct(fights: pd.DataFrame, totals: dict) -> pd.DataFrame:
    history = defaultdict(list)
    elo = defaultdict(lambda: 1500.0)
    rows = []
    usable = fights.dropna(subset=['red_fighter_id', 'blue_fighter_id', 'date'])
    for date, day in usable.sort_values(['date', 'fight_id']).groupby('date', sort=True):
        for row in day.itertuples(index=False):
            rid, bid = row.red_fighter_id, row.blue_fighter_id
            red = fighter_state(history[rid], elo[rid], date)
            blue = fighter_state(history[bid], elo[bid], date)
            if row.winner in ('red', 'blue') and date >= pd.Timestamp(CONFIG['train_start']):
                features = matchup_features(red, blue, row.weightclass, row.num_rounds, row.is_title_bout)
                sample = {
                    'fight_id': row.fight_id, 'event_id': row.event_id, 'date': date,
                    'red_name': row.red_name, 'blue_name': row.blue_name, 'red_fighter_id': rid, 'blue_fighter_id': bid,
                    'target': int(row.winner == 'red'),
                    **{key: value for key, value in features.items() if not key.startswith('diff_')},
                    'red_history_through': history[rid][-1]['date'] if history[rid] else pd.NaT,
                    'blue_history_through': history[bid][-1]['date'] if history[bid] else pd.NaT,
                    **{key: value for key, value in features.items() if key.startswith('diff_')},
                }
                rows.append(sample)
        advance_history(day, history, elo, totals)
    return pd.DataFrame(rows).sort_values(['date', 'fight_id']).reset_index(drop=True)


def swapped(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    columns = [column for column in frame if column.startswith('diff_')]
    result[columns] = -result[columns]
    return result


def symmetric_probability(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    return (model.predict_proba(frame)[:, 1] + 1 - model.predict_proba(swapped(frame))[:, 1]) / 2


def apply_temperature(probability: np.ndarray, temperature: float) -> np.ndarray:
    return expit(logit(np.clip(probability, 1e-9, 1 - 1e-9)) / temperature)


def wilson(successes: int, count: int) -> list[float] | None:
    if not count:
        return None
    z = 1.959963984540054
    mean = successes / count
    denominator = 1 + z * z / count
    center = (mean + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(mean * (1 - mean) / count + z * z / (4 * count * count)) / denominator
    return [center - radius, center + radius]


def metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    confidence = np.maximum(probability, 1 - probability)
    correct = (probability >= 0.5) == labels
    bands = []
    ece = 0.0
    edges = CONFIG['confidence_bin_edges']
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence >= lower) & ((confidence < upper) if index < len(edges) - 2 else (confidence <= upper))
        count = int(mask.sum())
        hitrate = float(correct[mask].mean()) if count else None
        mean_confidence = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(labels) * abs(hitrate - mean_confidence)
        bands.append({'lower': lower, 'upper': upper, 'count': count, 'mean_confidence': mean_confidence,
                      'hit_rate': hitrate, 'wilson_95': wilson(int(correct[mask].sum()), count)})
    subsets = {}
    for threshold in CONFIG['thresholds_predeclared']:
        mask = confidence >= threshold
        count = int(mask.sum())
        subsets[str(threshold)] = {'count': count, 'coverage': count / len(labels),
                                  'wins': int(correct[mask].sum()), 'hit_rate': float(correct[mask].mean()) if count else None,
                                  'mean_confidence': float(confidence[mask].mean()) if count else None,
                                  'wilson_95': wilson(int(correct[mask].sum()), count)}
    return {'count': len(labels), 'accuracy': float(accuracy_score(labels, probability >= 0.5)),
            'accuracy_wilson_95': wilson(int(correct.sum()), len(labels)),
            'brier': float(brier_score_loss(labels, probability)), 'log_loss': float(log_loss(labels, probability)),
            'auc': float(roc_auc_score(labels, probability)), 'confidence_ece': ece,
            'calibration_bins': bands, 'threshold_subsets': subsets}


def main(args: argparse.Namespace) -> None:
    root = args.data_dir.resolve()
    out = args.output_dir.resolve()
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    sources = {name: root / name for name in ['processed/fights.parquet', 'processed/events.parquet', 'processed/rounds.parquet', 'raw/ufcstats/ufc_fight_results.csv']}
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config': CONFIG,
                'script_sha256': digest(Path(__file__)),
                'source_files': {name: {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size} for name, path in sources.items()},
                'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__}}
    write_json(out / 'predeclared_manifest.json', manifest)
    fights = pd.read_parquet(sources['processed/fights.parquet'])
    events = pd.read_parquet(sources['processed/events.parquet'])
    rounds = pd.read_parquet(sources['processed/rounds.parquet'])
    fights = fights.merge(events[['event_id', 'date']], on='event_id', validate='many_to_one')
    formats = pd.read_csv(sources['raw/ufcstats/ufc_fight_results.csv'], usecols=['URL', 'TIME FORMAT'])
    formats['fight_id'] = formats.URL.str.rstrip('/').str.rsplit('/', n=1).str[-1]
    formats = formats[['fight_id', 'TIME FORMAT']].drop_duplicates().rename(columns={'TIME FORMAT': 'time_format'})
    fights = fights.merge(formats, on='fight_id', how='left', validate='one_to_one')
    assert not fights.fight_id.duplicated().any()
    totals, round_quality = prepare_round_stats(rounds)
    data = reconstruct(fights, totals)
    for side in ('red', 'blue'):
        through = data[f'{side}_history_through']
        assert (through.isna() | (through < data.date)).all()
    prefix = reconstruct(fights[fights.date < pd.Timestamp(CONFIG['train_end_exclusive'])], totals)
    pd.testing.assert_frame_equal(data[data.date < pd.Timestamp(CONFIG['train_end_exclusive'])].reset_index(drop=True), prefix)
    data.to_parquet(out / 'reconstructed_features.parquet', index=False)
    train = data[data.date < pd.Timestamp(CONFIG['train_end_exclusive'])]
    calibration = data[(data.date >= pd.Timestamp(CONFIG['calibration_start'])) & (data.date < pd.Timestamp(CONFIG['calibration_end_exclusive']))]
    holdout = data[(data.date >= pd.Timestamp(CONFIG['holdout_start'])) & (data.date <= pd.Timestamp(CONFIG['holdout_end_inclusive']))]
    assert train.date.max() < calibration.date.min() < holdout.date.min()
    assert calibration.date.max() < holdout.date.min()
    assert not (set(train.event_id) & set(calibration.event_id) | set(train.event_id) & set(holdout.event_id) | set(calibration.event_id) & set(holdout.event_id))
    columns = [column for column in data if column.startswith(('diff_', 'ctx_'))] + ['weightclass']
    numeric = [column for column in columns if column != 'weightclass']
    X = pd.concat([train[columns], swapped(train[columns])], ignore_index=True)
    y = np.concatenate([train.target.to_numpy(), 1 - train.target.to_numpy()])
    preprocessing = lambda: ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)), ('scale', StandardScaler())]), numeric),
        ('division', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['weightclass']),
    ])
    constructors = {'logistic_regression': LogisticRegression, 'random_forest': RandomForestClassifier, 'gradient_boosting': GradientBoostingClassifier}
    models = {}
    calibration_probabilities = {}
    holdout_probabilities = {}
    timings = {}
    for name, constructor in constructors.items():
        fit_start = time.monotonic()
        model = Pipeline([('preprocess', preprocessing()), ('model', constructor(**CONFIG[name]))])
        model.fit(X, y)
        models[name] = model
        calibration_probabilities[name] = symmetric_probability(model, calibration[columns])
        holdout_probabilities[name] = symmetric_probability(model, holdout[columns])
        reverse = symmetric_probability(model, swapped(holdout[columns]))
        assert np.allclose(holdout_probabilities[name] + reverse, 1, atol=1e-12)
        timings[name] = time.monotonic() - fit_start
        print(f'Fitted {name} in {timings[name]:.2f}s', flush=True)
    calibration_probabilities['equal_weight_ensemble'] = np.mean(list(calibration_probabilities.values()), axis=0)
    holdout_probabilities['equal_weight_ensemble'] = np.mean(list(holdout_probabilities.values()), axis=0)
    fit = minimize_scalar(lambda log_temperature: log_loss(calibration.target, apply_temperature(calibration_probabilities['equal_weight_ensemble'], math.exp(log_temperature))),
                          bounds=(-3, 3), method='bounded')
    temperature = math.exp(fit.x)
    calibration_probabilities['equal_weight_ensemble_temperature'] = apply_temperature(calibration_probabilities['equal_weight_ensemble'], temperature)
    holdout_probabilities['equal_weight_ensemble_temperature'] = apply_temperature(holdout_probabilities['equal_weight_ensemble'], temperature)
    for destination, frame in [(calibration_probabilities, calibration), (holdout_probabilities, holdout)]:
        destination['coin_flip'] = np.full(len(frame), 0.5)
        destination['training_red_prior'] = np.full(len(frame), float(train.target.mean()))
        destination['elo_unfitted'] = expit(frame.diff_elo.to_numpy() * np.log(10) / 400)
    report = {'manifest': str(out / 'predeclared_manifest.json'), 'primary_model': CONFIG['primary_model'],
              'data_through': str(fights.date.max().date()), 'temperature': temperature,
              'splits': {name: {'count': len(frame), 'events': frame.event_id.nunique(), 'start': str(frame.date.min().date()), 'end': str(frame.date.max().date())}
                         for name, frame in [('train', train), ('calibration', calibration), ('holdout', holdout)]},
              'quality': {**round_quality, 'total_bouts': len(fights), 'draws_excluded': int((fights.winner == 'draw').sum()),
                          'no_contests_excluded': int((fights.winner == 'nc').sum()),
                          'unresolved_fighter_bouts_excluded': int(fights[['red_fighter_id', 'blue_fighter_id']].isna().any(axis=1).sum()),
                          'strict_prior_date_check': True, 'training_prefix_invariance_check': True,
                          'holdout_bouts_with_both_fighters_three_prior_stat_bouts': int((holdout.ctx_min_stats_last5 >= 3).sum())},
              'calibration': {name: metrics(calibration.target.to_numpy(), probability) for name, probability in calibration_probabilities.items()},
              'holdout': {name: metrics(holdout.target.to_numpy(), probability) for name, probability in holdout_probabilities.items()},
              'timings_seconds': timings,
              'limitations': ['Retrospective raw sources lack archived observation/revision timestamps.',
                              'This is an independently rebuilt baseline, not validation of the deployed v2 artifact or its snapshot inference.',
                              'Final-period predictions are now inspected; future model changes require new untouched holdout evidence.',
                              'Wilson intervals treat bouts as independent; repeated fighters/events mean these are descriptive, not a guarantee.',
                              'Calibration-period metrics are in-sample for the temperature fit; only final holdout is independent of that calibration.',
                              'No causal betting return or profit claim; historical prices were excluded.']}
    predictions = holdout[['fight_id', 'event_id', 'date', 'red_name', 'blue_name', 'target', 'red_history_through', 'blue_history_through']].copy()
    for name, probability in holdout_probabilities.items():
        predictions[f'p_red_{name}'] = probability
    predictions.to_csv(out / 'holdout_predictions.csv', index=False)
    calibration_predictions = calibration[['fight_id', 'date', 'target']].copy()
    for name, probability in calibration_probabilities.items():
        calibration_predictions[f'p_red_{name}'] = probability
    calibration_predictions.to_csv(out / 'calibration_predictions.csv', index=False)
    joblib.dump({'models': models, 'temperature': temperature, 'feature_columns': columns,
                 'config': CONFIG, 'manifest': manifest, 'validated_for_betting': False}, out / 'provisional_baseline.joblib', compress=3)
    report['elapsed_seconds'] = time.monotonic() - start
    write_json(out / 'report.json', report)
    write_json(out / 'artifact_hashes.json', {path.name: digest(path) for path in out.iterdir() if path.is_file()})
    print(json.dumps({'splits': report['splits'], 'temperature': temperature,
                      'holdout': {name: {key: value for key, value in result.items() if key not in ('calibration_bins',)} for name, result in report['holdout'].items()},
                      'elapsed_seconds': report['elapsed_seconds'], 'output': str(out)}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the frozen chronological research baseline; never overwrite an existing run.')
    parser.add_argument('--data-dir', type=Path, required=True, help='MMA pipeline root containing processed/ and raw/ufcstats/.')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory for manifests, predictions, model, and metrics.')
    main(parser.parse_args())
