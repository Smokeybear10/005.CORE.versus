import math
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

if __package__:
    from . import evaluate_research as base
else:
    import evaluate_research as base


HALF_LIFE_DAYS = 365.0
PRIOR_MINUTES = 15.0
PRIOR_BOUTS = 2.0
RATE_NAMES = (
    'sig_landed_pm', 'sig_absorbed_pm', 'sig_accuracy', 'sig_defense',
    'td_landed_pm', 'td_absorbed_pm', 'td_accuracy', 'td_defense',
    'control_share', 'control_conceded_share', 'kd_pm', 'kd_absorbed_pm',
    'sub_att_pm', 'sub_att_faced_pm',
)
ADJUSTED = {'sig_landed_pm': 'sig_absorbed_pm', 'sig_absorbed_pm': 'sig_landed_pm',
            'td_landed_pm': 'td_absorbed_pm', 'td_absorbed_pm': 'td_landed_pm',
            'control_share': 'control_conceded_share', 'control_conceded_share': 'control_share',
            'kd_absorbed_pm': 'kd_pm', 'sub_att_faced_pm': 'sub_att_pm'}
PARTS = ('head', 'body', 'leg', 'distance', 'clinch', 'ground')
FINISH_NAMES = ('ko_win', 'ko_loss', 'submission_win', 'submission_loss')
ADDED_FEATURES = (
    [f'diff_ew_{name}' for name in RATE_NAMES]
    + [f'diff_oppadj_{name}' for name in ADJUSTED]
    + [f'diff_ew_{name}_rate' for name in FINISH_NAMES]
    + ['diff_elo_decay365', 'ctx_mean_elo_decay365', 'diff_ew_result_surprise']
    + ['diff_trend_sig_net', 'diff_trend_td_net', 'diff_trend_control_net']
    + ['diff_height_cm', 'diff_reach_cm']
    + ['ctx_min_ew_stat_minutes', 'ctx_max_ew_stat_minutes', 'ctx_min_ew_adjusted_minutes', 'ctx_measurements_known']
    + [f'diff_ew_{part}_net_pm' for part in PARTS]
)
CONFIG = {
    'version': 1, 'added_features': ADDED_FEATURES, 'half_life_days': HALF_LIFE_DAYS,
    'prior_minutes': PRIOR_MINUTES, 'prior_bouts': PRIOR_BOUTS,
    'elo': {'initial': 1500.0, 'k': 32.0, 'denominator': 400.0, 'reversion_half_life_days': HALF_LIFE_DAYS},
    'rates': 'All prior bouts, exponentially weighted by elapsed days. Counts and denominators are pooled before division; 15 minutes of expanding strictly prior league counts/exposure supplies the prior, including attempt denominators for accuracy/defense.',
    'opponent_adjustment': 'Observed rate minus opponent corresponding pre-bout produced/allowed rate. Opponent state is frozen before the event date. Residuals are exposure/time weighted and shrunk toward zero with 15 minutes; absent residual history remains missing.',
    'finishing': 'Time-weighted KO/TKO and submission wins/losses divided by prior decisive bouts with known methods, with two bouts of strictly prior league rates.',
    'trend': 'Shrunk exposure-weighted net rate in latest three recorded bouts minus preceding three; both windows need valid exposure. Missing-stat bouts retain their history positions.',
    'style': 'Six target/position net landed-minus-absorbed rates; per-round target and position sums must equal parent significant strikes.',
    'time_policy': 'Date-only batches: all requested features and opponent baselines are frozen before any same-date outcome/stat/prior update. Same-day ordering is never used as pre-fight evidence.',
    'missing_policy': 'Invalid or partial counts are missing, never zero. Valid zero attempts remain zero exposure trials, not perfect defense. Control/style missingness does not erase otherwise valid parent stats.',
    'profile_policy': 'Only positive finite height_cm and reach_cm are read. Retrospective snapshots are treated as approximately fixed measurements; original historical observation/revision times are unknown. No stance, weight, rankings, or outcome-derived profile values.',
    'source_limitations': 'Computational prior-date separation does not certify when retrospective source records were first available or revised.',
}


def _prepare(fights: pd.DataFrame, rounds: pd.DataFrame) -> tuple[dict, dict]:
    totals, base_quality = base.prepare_round_stats(rounds)
    groups = {key: group for key, group in rounds.groupby('fight_id', sort=False)}
    prepared = {}
    quality = Counter()
    count_columns = [name for name in base.STAT_COLS if name != 'ctrl_seconds']
    style_columns = [f'{part}_{kind}' for part in PARTS for kind in ('landed', 'attempted')]
    for row in fights.itertuples(index=False):
        group = groups.get(row.fight_id)
        valid_time = (isinstance(row.finish_time, str) and re.fullmatch(r'\d+:[0-5]\d', row.finish_time)
                      and pd.notna(row.finish_round) and row.finish_round >= 1
                      and pd.notna(row.num_rounds) and row.num_rounds in (3, 5))
        if not valid_time or group is None or set(group.fighter_id) != {row.red_fighter_id, row.blue_fighter_id}:
            quality['missing_or_invalid_identity_time_stats'] += 1
            continue
        rates = base.bout_rates(row, row.red_fighter_id, row.blue_fighter_id, totals)
        counts = group[count_columns].to_numpy(dtype=float, na_value=np.nan)
        if (not np.isfinite(rates['sig_landed_pm']) or not np.isfinite(counts).all() or (counts < 0).any()
                or not np.equal(counts, np.floor(counts)).all()
                or (group.sig_str_landed > group.sig_str_attempted).any() or (group.td_landed > group.td_attempted).any()):
            quality['invalid_or_partial_parent_stats'] += 1
            continue
        minute, second = map(int, row.finish_time.split(':'))
        seconds = (row.finish_round - 1) * 300 + minute * 60 + second
        minutes = seconds / 60
        controls = {}
        for fid in (row.red_fighter_id, row.blue_fighter_id):
            own = group[group.fighter_id == fid]
            control = own.ctrl_seconds.to_numpy(dtype=float, na_value=np.nan)
            duration = np.where(own['round'].to_numpy() == row.finish_round, minute * 60 + second, 300)
            controls[fid] = float(control.sum()) if np.isfinite(control).all() and (control >= 0).all() and (control <= duration).all() else np.nan
        if sum(controls.values()) > seconds:
            controls = {fid: np.nan for fid in controls}
        style = None
        if all(name in group for name in style_columns):
            values = group[style_columns].to_numpy(dtype=float, na_value=np.nan)
            style_valid = (np.isfinite(values).all() and (values >= 0).all() and np.equal(values, np.floor(values)).all()
                           and all((group[f'{part}_landed'] <= group[f'{part}_attempted']).all() for part in PARTS)
                           and all(np.array_equal(group[[f'{part}_{kind}' for part in parts]].sum(axis=1).to_numpy(), group[f'sig_str_{kind}'].to_numpy())
                                   for parts in (PARTS[:3], PARTS[3:]) for kind in ('landed', 'attempted')))
            if style_valid:
                sums = group.groupby('fighter_id')[[f'{part}_landed' for part in PARTS]].sum()
                style = {fid: sums.loc[fid].to_numpy(dtype=float) for fid in controls}
        quality['valid_parent_bouts'] += 1
        quality['valid_style_bouts' if style is not None else 'missing_or_invalid_style_bouts'] += 1
        for fid, opp in ((row.red_fighter_id, row.blue_fighter_id), (row.blue_fighter_id, row.red_fighter_id)):
            own, other = totals[(row.fight_id, fid)], totals[(row.fight_id, opp)]
            numerator = np.array([own['sig_str_landed'], other['sig_str_landed'], own['sig_str_landed'], other['sig_str_attempted'] - other['sig_str_landed'],
                                  own['td_landed'], other['td_landed'], own['td_landed'], other['td_attempted'] - other['td_landed'],
                                  controls[fid], controls[opp], own['kd'], other['kd'], own['sub_att'], other['sub_att']], dtype=float)
            denominator = np.array([minutes, minutes, own['sig_str_attempted'], other['sig_str_attempted'],
                                    minutes, minutes, own['td_attempted'], other['td_attempted'], seconds, seconds,
                                    minutes, minutes, minutes, minutes], dtype=float)
            denominator[~np.isfinite(numerator)] = np.nan
            prepared[(row.fight_id, fid)] = {'minutes': minutes, 'numerator': numerator, 'denominator': denominator,
                                            'style': style[fid] - style[opp] if style is not None else None}
            quality['missing_control_fighter_bouts'] += int(not np.isfinite(controls[fid]))
    return prepared, {'counts': dict(quality), 'base_round_quality': base_quality}


def _weights(history: list[dict], date: pd.Timestamp) -> np.ndarray:
    return np.array([math.exp2(-(date - bout['date']).days / HALF_LIFE_DAYS) for bout in history])


def _rates(history: list[dict], date: pd.Timestamp, league: dict) -> np.ndarray:
    numerator = np.zeros(len(RATE_NAMES))
    denominator = np.zeros(len(RATE_NAMES))
    for bout, weight in zip(history, _weights(history, date)):
        stats = bout['stats']
        if stats is not None:
            numerator += weight * np.nan_to_num(stats['numerator'], nan=0)
            denominator += weight * np.nan_to_num(stats['denominator'], nan=0)
    exposure = league['exposure']
    with np.errstate(invalid='ignore', divide='ignore'):
        prior_num = PRIOR_MINUTES * league['numerator'] / exposure
        prior_den = PRIOR_MINUTES * league['denominator'] / exposure
        result = (numerator + prior_num) / (denominator + prior_den)
    for index in (2, 3, 6, 7, 8, 9):
        result[index] = np.clip(result[index], 0, 1)
    return result


def _state(history: list[dict], date: pd.Timestamp, league: dict) -> dict:
    rates = _rates(history, date, league)
    weights = _weights(history, date)
    residual_counts = np.zeros(len(ADJUSTED))
    residual_minutes = np.zeros(len(ADJUSTED))
    style_counts = np.zeros(len(PARTS))
    style_minutes = 0.0
    finish_counts = np.zeros(len(FINISH_NAMES))
    finish_bouts = surprise_sum = surprise_bouts = stat_minutes = 0.0
    for bout, weight in zip(history, weights):
        stats = bout['stats']
        if stats is not None:
            stat_minutes += weight * stats['minutes']
            valid = np.isfinite(bout['residual'])
            residual_counts[valid] += weight * stats['minutes'] * bout['residual'][valid]
            residual_minutes[valid] += weight * stats['minutes']
            if stats['style'] is not None:
                style_counts += weight * stats['style']
                style_minutes += weight * stats['minutes']
        if bout['finish'] is not None:
            finish_counts += weight * bout['finish']
            finish_bouts += weight
        if np.isfinite(bout['surprise']):
            surprise_sum += weight * bout['surprise']
            surprise_bouts += weight
    adjusted = np.divide(residual_counts, residual_minutes + PRIOR_MINUTES)
    adjusted[residual_minutes == 0] = np.nan
    finish = ((finish_counts + PRIOR_BOUTS * league['finish'] / league['finish_bouts']) / (finish_bouts + PRIOR_BOUTS)
              if league['finish_bouts'] else np.full(len(FINISH_NAMES), np.nan))
    style = style_counts / (style_minutes + PRIOR_MINUTES) if league['style_minutes'] else np.full(len(PARTS), np.nan)
    trend = np.full(3, np.nan)
    recent, previous = history[-3:], history[-6:-3]
    if any(bout['stats'] is not None for bout in recent) and any(bout['stats'] is not None for bout in previous):
        change = _rates(recent, date, league) - _rates(previous, date, league)
        trend = np.array([change[0] - change[1], change[4] - change[5], change[8] - change[9]])
    return {'rates': rates, 'adjusted': adjusted, 'finish': finish, 'style': style, 'trend': trend,
            'surprise': surprise_sum / (surprise_bouts + PRIOR_BOUTS),
            'stat_minutes': stat_minutes, 'adjusted_minutes': residual_minutes[0]}


def _elo(state: tuple[float, pd.Timestamp | None], date: pd.Timestamp) -> float:
    rating, through = state
    return 1500.0 + (rating - 1500.0) * math.exp2(-(date - through).days / HALF_LIFE_DAYS) if through is not None else rating


def add_predictors(frame: pd.DataFrame, fights: pd.DataFrame, rounds: pd.DataFrame, fighters: pd.DataFrame) -> pd.DataFrame:
    if set(ADDED_FEATURES) & set(frame.columns):
        raise ValueError('Predictor columns already exist; existing features will not be overwritten')
    needed = {'date', 'red_fighter_id', 'blue_fighter_id'}
    if not needed <= set(frame.columns):
        raise ValueError('Feature rows require date and both exact fighter IDs')
    if fights.fight_id.duplicated().any() or fighters.fighter_id.duplicated().any():
        raise ValueError('Fight and fighter identities must be unique')
    result = frame.copy()
    if frame.empty:
        for name in ADDED_FEATURES:
            result[name] = pd.Series(index=frame.index, dtype=float)
        result.attrs['predictor_features'] = {'config': CONFIG, 'quality': {}}
        return result
    requests = frame[list(needed)].copy()
    requests['date'] = pd.to_datetime(requests.date)
    requests['_position'] = np.arange(len(requests))
    usable = fights.dropna(subset=['date', 'red_fighter_id', 'blue_fighter_id']).copy()
    usable['date'] = pd.to_datetime(usable.date)
    for dates in (requests.date, usable.date):
        if dates.isna().any() or dates.dt.tz is not None or not dates.equals(dates.dt.normalize()):
            raise ValueError('Use complete timezone-naive calendar dates; intraday timestamps are unsupported')
    usable = usable[usable.date <= requests.date.max()].sort_values(['date', 'fight_id'])
    prepared, quality = _prepare(usable, rounds[rounds.fight_id.isin(usable.fight_id)])
    profiles = {}
    for row in fighters.itertuples(index=False):
        values = [getattr(row, key, np.nan) for key in ('height_cm', 'reach_cm')]
        profiles[row.fighter_id] = np.array([float(value) if pd.notna(value) and np.isfinite(value) and value > 0 else np.nan for value in values])
    history = defaultdict(list)
    elo = defaultdict(lambda: (1500.0, None))
    league = {'numerator': np.zeros(len(RATE_NAMES)), 'denominator': np.zeros(len(RATE_NAMES)),
              'exposure': np.zeros(len(RATE_NAMES)), 'finish': np.zeros(len(FINISH_NAMES)),
              'finish_bouts': 0, 'style_minutes': 0.0}
    feature_rows = np.full((len(frame), len(ADDED_FEATURES)), np.nan)
    fight_days = dict(tuple(usable.groupby('date', sort=True)))
    request_days = dict(tuple(requests.groupby('date', sort=True)))
    adjusted_indices = [RATE_NAMES.index(name) for name in ADJUSTED]
    opponent_indices = [RATE_NAMES.index(name) for name in ADJUSTED.values()]
    for date in sorted(set(fight_days) | set(request_days)):
        day = fight_days.get(date)
        wanted = request_days.get(date)
        identities = set()
        for part in (day, wanted):
            if part is not None:
                identities.update(part.red_fighter_id)
                identities.update(part.blue_fighter_id)
        states = {fid: _state(history[fid], date, league) for fid in identities}
        ratings = {fid: _elo(elo[fid], date) for fid in identities}
        if wanted is not None:
            for row in wanted.itertuples(index=False, name=None):
                request = dict(zip(wanted.columns, row))
                rid, bid = request['red_fighter_id'], request['blue_fighter_id']
                red, blue = states[rid], states[bid]
                first, second = profiles.get(rid, np.full(2, np.nan)), profiles.get(bid, np.full(2, np.nan))
                feature_rows[request['_position']] = np.concatenate([
                    red['rates'] - blue['rates'], red['adjusted'] - blue['adjusted'], red['finish'] - blue['finish'],
                    [ratings[rid] - ratings[bid], (ratings[rid] + ratings[bid]) / 2, red['surprise'] - blue['surprise']],
                    red['trend'] - blue['trend'], first - second,
                    [min(red['stat_minutes'], blue['stat_minutes']), max(red['stat_minutes'], blue['stat_minutes']),
                     min(red['adjusted_minutes'], blue['adjusted_minutes']), np.isfinite(np.concatenate([first, second])).sum()],
                    red['style'] - blue['style'],
                ])
        if day is None:
            continue
        updates = []
        deltas = defaultdict(float)
        for row in day.itertuples(index=False):
            rid, bid = row.red_fighter_id, row.blue_fighter_id
            score = 1.0 if row.winner == 'red' else 0.0 if row.winner == 'blue' else 0.5 if row.winner == 'draw' else np.nan
            expected = 1 / (1 + 10 ** ((ratings[bid] - ratings[rid]) / 400))
            method = row.method.casefold().strip() if isinstance(row.method, str) else ''
            for fid, opp, outcome, expectation in ((rid, bid, score, expected), (bid, rid, 1 - score, 1 - expected)):
                stats = prepared.get((row.fight_id, fid))
                residual = np.full(len(ADJUSTED), np.nan)
                if stats is not None:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        observed = stats['numerator'] / stats['denominator']
                    residual = observed[adjusted_indices] - states[opp]['rates'][opponent_indices]
                finish = (np.array([outcome == 1 and 'ko' in method, outcome == 0 and 'ko' in method,
                                    outcome == 1 and 'submission' in method, outcome == 0 and 'submission' in method], dtype=float)
                          if outcome in (0, 1) and method not in ('', '--', 'unknown') else None)
                updates.append((fid, {'date': date, 'stats': stats, 'residual': residual, 'finish': finish, 'surprise': outcome - expectation}))
            if np.isfinite(score):
                deltas[rid] += 32 * (score - expected)
                deltas[bid] -= 32 * (score - expected)
        for fid, bout in updates:
            history[fid].append(bout)
            stats = bout['stats']
            if stats is not None:
                valid = np.isfinite(stats['numerator']) & np.isfinite(stats['denominator'])
                league['numerator'][valid] += stats['numerator'][valid]
                league['denominator'][valid] += stats['denominator'][valid]
                league['exposure'][valid] += stats['minutes']
                if stats['style'] is not None:
                    league['style_minutes'] += stats['minutes']
            if bout['finish'] is not None:
                league['finish'] += bout['finish']
                league['finish_bouts'] += 1
        for fid in set(day.red_fighter_id) | set(day.blue_fighter_id):
            elo[fid] = (ratings[fid] + deltas[fid], date)
    for index, name in enumerate(ADDED_FEATURES):
        result[name] = feature_rows[:, index]
    result.attrs['predictor_features'] = {'config': CONFIG, 'quality': quality}
    return result
