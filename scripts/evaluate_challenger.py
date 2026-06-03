import argparse
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


if __package__:
    from . import evaluate_research as base
else:
    import evaluate_research as base


CONFIG = {
    'fit_start': '2010-01-01', 'development_fit_end_exclusive': '2021-01-01',
    'development_start': '2021-01-01', 'development_end_exclusive': '2023-01-01',
    'refit_end_exclusive': '2023-01-01', 'calibration_start': '2023-01-01', 'calibration_end_exclusive': '2025-01-01',
    'fresh_audit_start_exclusive': '2026-05-09', 'fresh_audit_end_inclusive': '2026-10-31',
    'model_selection': 'Lowest Brier loss on 2021-2022 development; exact tie breaks by insertion order. No old final-holdout predictions computed. Fresh audit remains sealed until selected model and temperature are saved.',
    'models': {
        'logistic_regression': {'C': 0.5, 'max_iter': 2000, 'random_state': 42},
        'gradient_boosting': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'random_state': 42},
        'hist_gradient_boosting': {'max_iter': 300, 'max_leaf_nodes': 15, 'min_samples_leaf': 30, 'l2_regularization': 5.0,
                                  'learning_rate': 0.05, 'early_stopping': False, 'random_state': 42},
    },
    'features_added': ['age_at_bout_from_DOB', 'squared_age_at_bout', 'minimum_and_maximum_age', 'mean_pair_pre_bout_Elo',
                       'difference_in_prior_opponent_pre_bout_Elo_last3_and_last5',
                       'difference_in_prior_result_minus_Elo_expectation_last3_and_last5'],
    'history_rule': 'Draws and NC remain in prior-bout history; draws score 0.5, NC is missing outcome and does not change Elo. All features use only dates strictly before the matchup date, except immutable DOB and known bout schedule/context.',
    'calibration': 'Single positive temperature learned only on 2023-2024 after final model refit on 2010-2022. No refit on calibration labels.',
    'thresholds_predeclared': [0.8, 0.85, 0.9],
    'source_qualification': 'DOB is treated as immutable identity data, with source fingerprint. Historical observation/revision timestamps remain unavailable.',
}


def context_state(history: list[dict], date: pd.Timestamp, dob: pd.Timestamp) -> dict:
    age = (date - dob).days / 365.25 if pd.notna(dob) else np.nan
    if not 16 <= age <= 60:
        age = np.nan
    state = {'age': age, 'age_squared': age * age}
    for window in (3, 5):
        recent = history[-window:]
        state[f'opponent_elo_last{window}'] = base.finite_mean([bout['opponent_elo'] for bout in recent])
        state[f'result_surprise_last{window}'] = base.finite_mean([bout['result_surprise'] for bout in recent])
    return state


def matchup_context(red: dict, blue: dict, red_elo: float, blue_elo: float) -> dict:
    return {
        'ctx_min_age': min(red['age'], blue['age']) if np.isfinite(red['age']) and np.isfinite(blue['age']) else np.nan,
        'ctx_max_age': max(red['age'], blue['age']) if np.isfinite(red['age']) and np.isfinite(blue['age']) else np.nan,
        'ctx_mean_elo': (red_elo + blue_elo) / 2,
        **{f'diff_{key}': red[key] - blue[key] for key in red},
    }


def add_context(data: pd.DataFrame, fights: pd.DataFrame, fighters: pd.DataFrame) -> pd.DataFrame:
    dobs = pd.to_datetime(fighters.set_index('fighter_id').dob, errors='coerce').to_dict()
    history = defaultdict(list)
    elo = defaultdict(lambda: 1500.0)
    records = []
    usable = fights.dropna(subset=['red_fighter_id', 'blue_fighter_id', 'date'])
    for date, day in usable.sort_values(['date', 'fight_id']).groupby('date', sort=True):
        for row in day.itertuples(index=False):
            rid, bid = row.red_fighter_id, row.blue_fighter_id
            red = context_state(history[rid], date, dobs.get(rid, pd.NaT))
            blue = context_state(history[bid], date, dobs.get(bid, pd.NaT))
            if row.winner in ('red', 'blue') and date >= pd.Timestamp(CONFIG['fit_start']):
                records.append({'fight_id': row.fight_id, **matchup_context(red, blue, elo[rid], elo[bid])})
        base.advance_history(day, history, elo)
    return data.merge(pd.DataFrame(records), on='fight_id', how='left', validate='one_to_one')


def build_inference_features(
    fights: pd.DataFrame, rounds: pd.DataFrame, fighters: pd.DataFrame,
    fighter1_id: str, fighter2_id: str, event_date: str,
    weightclass: str, scheduled_rounds: int, is_title: bool,
    prepared_totals: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    date = pd.Timestamp(event_date).normalize()
    totals = prepared_totals if prepared_totals is not None else base.prepare_round_stats(rounds)[0]
    history, elo = base.history_before(fights, totals, date)
    dobs = pd.to_datetime(fighters.set_index('fighter_id').dob, errors='coerce').to_dict()
    red = base.fighter_state(history[fighter1_id], elo[fighter1_id], date)
    blue = base.fighter_state(history[fighter2_id], elo[fighter2_id], date)
    red_context = context_state(history[fighter1_id], date, dobs.get(fighter1_id, pd.NaT))
    blue_context = context_state(history[fighter2_id], date, dobs.get(fighter2_id, pd.NaT))
    features = {
        **base.matchup_features(red, blue, weightclass, scheduled_rounds, is_title),
        **matchup_context(red_context, blue_context, elo[fighter1_id], elo[fighter2_id]),
    }
    quality = {}
    for label, fighter_id in [('fighter1', fighter1_id), ('fighter2', fighter2_id)]:
        bouts = history[fighter_id]
        classes = sorted({bout['weightclass'] for bout in bouts if isinstance(bout['weightclass'], str)})
        quality[label] = {
            'historyThrough': str(bouts[-1]['date'].date()) if bouts else None,
            'priorBouts': len(bouts),
            'validStatBouts': int(sum(np.isfinite(bout['sig_landed_pm']) for bout in bouts)),
            'last5StatBouts': int(sum(np.isfinite(bout['sig_landed_pm']) for bout in bouts[-5:])),
            'lastWeightClass': bouts[-1]['weightclass'] if bouts else None,
            'weightClasses': classes,
        }
    return pd.DataFrame([features]), quality


def make_pipeline(name: str, columns: list[str]) -> Pipeline:
    numeric = [column for column in columns if column != 'weightclass']
    preprocessing = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)), ('scale', StandardScaler())]), numeric),
        ('division', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['weightclass']),
    ])
    constructors = {'logistic_regression': LogisticRegression, 'gradient_boosting': GradientBoostingClassifier,
                    'hist_gradient_boosting': HistGradientBoostingClassifier}
    return Pipeline([('preprocess', preprocessing), ('model', constructors[name](**CONFIG['models'][name]))])


def fit_model(name: str, frame: pd.DataFrame, columns: list[str]) -> Pipeline:
    X = pd.concat([frame[columns], base.swapped(frame[columns])], ignore_index=True)
    y = np.concatenate([frame.target.to_numpy(), 1 - frame.target.to_numpy()])
    return make_pipeline(name, columns).fit(X, y)


def main(args: argparse.Namespace) -> None:
    root = args.data_dir.resolve()
    out = args.output_dir.resolve()
    baseline_dir = args.baseline_dir.resolve()
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    fighters_path = root / 'processed/fighters.parquet'
    features_path = baseline_dir / 'reconstructed_features.parquet'
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config': CONFIG,
                'script_sha256': base.digest(Path(__file__)), 'frozen_baseline_script_sha256': base.digest(Path(base.__file__)),
                'sources': {'DOB_fighters': {'path': str(fighters_path), 'sha256': base.digest(fighters_path)},
                            'baseline_features': {'path': str(features_path), 'sha256': base.digest(features_path)},
                            'original_manifest': str(baseline_dir / 'predeclared_manifest.json')}}
    base.write_json(out / 'predeclared_manifest.json', manifest)
    fights = pd.read_parquet(root / 'processed/fights.parquet').merge(
        pd.read_parquet(root / 'processed/events.parquet')[['event_id', 'date']], on='event_id', validate='many_to_one')
    fights = fights[fights.date < pd.Timestamp(CONFIG['calibration_end_exclusive'])]
    data = pd.read_parquet(features_path)
    data = data[data.date < pd.Timestamp(CONFIG['calibration_end_exclusive'])].copy()
    fighters = pd.read_parquet(fighters_path)
    data = add_context(data, fights, fighters)
    fit = data[data.date < pd.Timestamp(CONFIG['development_fit_end_exclusive'])]
    development = data[(data.date >= pd.Timestamp(CONFIG['development_start'])) & (data.date < pd.Timestamp(CONFIG['development_end_exclusive']))]
    refit = data[data.date < pd.Timestamp(CONFIG['refit_end_exclusive'])]
    calibration = data[(data.date >= pd.Timestamp(CONFIG['calibration_start'])) & (data.date < pd.Timestamp(CONFIG['calibration_end_exclusive']))]
    columns = [column for column in data if column.startswith(('diff_', 'ctx_'))] + ['weightclass']
    prefix = add_context(fit.drop(columns=[column for column in fit if column in ['ctx_min_age', 'ctx_max_age', 'ctx_mean_elo'] or column in ['diff_age', 'diff_age_squared', 'diff_opponent_elo_last3', 'diff_opponent_elo_last5', 'diff_result_surprise_last3', 'diff_result_surprise_last5']]),
                         fights[fights.date < pd.Timestamp(CONFIG['development_fit_end_exclusive'])], fighters)
    pd.testing.assert_frame_equal(fit.reset_index(drop=True), prefix.reset_index(drop=True))
    scores = {}
    development_predictions = development[['fight_id', 'date', 'target']].copy()
    for name in CONFIG['models']:
        fit_start = time.monotonic()
        model = fit_model(name, fit, columns)
        probability = base.symmetric_probability(model, development[columns])
        scores[name] = {**base.metrics(development.target.to_numpy(), probability), 'elapsed_seconds': time.monotonic() - fit_start}
        development_predictions[f'p_red_{name}'] = probability
        print(f'{name}: development Brier={scores[name]["brier"]:.6f}, elapsed={scores[name]["elapsed_seconds"]:.2f}s', flush=True)
    winner = min(scores, key=lambda name: scores[name]['brier'])
    selection = {'selected_model': winner, 'selected_by': CONFIG['model_selection'],
                 'development_scores': scores, 'selected_at_utc': datetime.now(timezone.utc).isoformat()}
    base.write_json(out / 'selection.json', selection)
    model = fit_model(winner, refit, columns)
    calibration_probability = base.symmetric_probability(model, calibration[columns])
    temperature_fit = minimize_scalar(lambda log_temperature: log_loss(calibration.target, base.apply_temperature(calibration_probability, math.exp(log_temperature))),
                                      bounds=(-3, 3), method='bounded')
    temperature = math.exp(temperature_fit.x)
    calibrated = base.apply_temperature(calibration_probability, temperature)
    reverse = base.symmetric_probability(model, base.swapped(calibration[columns]))
    assert np.allclose(calibration_probability + reverse, 1, atol=1e-12)
    artifact = {'model': model, 'selected_model': winner, 'temperature': temperature,
                'feature_columns': columns, 'config': CONFIG, 'manifest': manifest, 'validated_for_betting': False}
    joblib.dump(artifact, out / 'provisional_challenger.joblib', compress=3)
    development_predictions.to_csv(out / 'development_predictions.csv', index=False)
    calibration_predictions = calibration[['fight_id', 'date', 'target']].copy()
    calibration_predictions['p_red_uncalibrated'] = calibration_probability
    calibration_predictions['p_red_temperature'] = calibrated
    calibration_predictions.to_csv(out / 'calibration_predictions.csv', index=False)
    data.to_parquet(out / 'development_and_calibration_features.parquet', index=False)
    report = {'selected_model': winner, 'selection': selection, 'temperature': temperature, 'feature_count': len(columns),
              'splits': {name: {'count': len(frame), 'start': str(frame.date.min().date()), 'end': str(frame.date.max().date())}
                         for name, frame in [('development_fit', fit), ('development', development), ('final_fit', refit), ('calibration', calibration)]},
              'calibration_raw': base.metrics(calibration.target.to_numpy(), calibration_probability),
              'calibration_temperature': base.metrics(calibration.target.to_numpy(), calibrated),
              'elapsed_seconds': time.monotonic() - start,
              'no_old_holdout_scored': True, 'fresh_audit_not_read': True, 'prefix_invariance_passed': True,
              'limitations': ['Calibration metrics are in-sample for temperature fitting.',
                              'This model has not yet been evaluated on its independent new-source audit slice.',
                              'Retrospective DOB lacks observation history; treated as immutable, not asserted historically observed.',
                              'No claim of 90% confidence calibration or profitability.']}
    base.write_json(out / 'report.json', report)
    base.write_json(out / 'artifact_hashes.json', {path.name: base.digest(path) for path in out.iterdir() if path.is_file()})
    print(json.dumps({'selected_model': winner, 'temperature': temperature, 'splits': report['splits'],
                      'calibration_brier': report['calibration_temperature']['brier'], 'elapsed_seconds': report['elapsed_seconds'], 'output': str(out)}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select the bounded challenger on development data only, then calibrate on a disjoint period.')
    parser.add_argument('--data-dir', type=Path, required=True, help='MMA pipeline root.')
    parser.add_argument('--baseline-dir', type=Path, required=True, help='Completed evaluate_research.py output directory.')
    parser.add_argument('--output-dir', type=Path, required=True, help='New output directory; existing runs are never overwritten.')
    main(parser.parse_args())
