import math
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import expit


INITIAL_RATING = 1500.0
INITIAL_RD = MAX_RD = 350.0
MIN_RD = 30.0
RD_VARIANCE_PER_DAY = (350.0 ** 2 - 50.0 ** 2) / (5 * 365.25)
Q = math.log(10) / 400.0
ADDED_FEATURES = [
    'diff_glicko_rating', 'ctx_mean_glicko_rating', 'diff_glicko_rd',
    'ctx_glicko_pair_rd', 'diff_glicko_expected_logit',
]
CONFIG = {
    'version': 1, 'features': ADDED_FEATURES, 'system': 'Glicko-1 with fixed calendar-time uncertainty growth',
    'initial_rating': INITIAL_RATING, 'initial_rd': INITIAL_RD, 'minimum_rd': MIN_RD, 'maximum_rd': MAX_RD,
    'rd_variance_per_day': RD_VARIANCE_PER_DAY,
    'calendar_adaptation': 'RD grows by elapsed calendar days: sqrt(previous_RD^2 + variance_per_day * days), capped at350. Fixed assumption: RD50 grows to350 after5*365.25 inactive days. This is an MMA experiment design choice, not a fitted or validated MMA parameter. Inactivity does not change the rating mean.',
    'batch_policy': 'Freeze all participant states before a calendar date; accumulate all same-date outcomes against those frozen opponents, then update simultaneously. No future/current-date outcomes enter feature rows.',
    'outcome_policy': 'Win1, loss0, draw0.5. No-contest/unknown outcomes supply no rating information; elapsed-time uncertainty still applies.',
    'pair_feature': 'q*g(sqrt(RD_first^2+RD_second^2))*(rating_first-rating_second), the logit of the author-described approximate expected score accounting for both rating deviations. It is antisymmetric and is not a calibrated MMA win probability.',
    'source_paper': 'https://glicko.net/ratings/glicko.pdf',
    'author_formula_and_worked_examples': 'https://www.glicko.net/glicko/glicko.pdf',
    'source_limitations': 'Reconstructed from retrospective outcome records without original observation/revision timestamps. No existing rating snapshots, profiles, rankings, current records or future outcomes are used.',
    'tuning_policy': 'Constants fixed before the paired experiment; no search over floors, inflation, scale or starting ratings.',
}


def _g(rd: float) -> float:
    return 1 / math.sqrt(1 + 3 * Q * Q * rd * rd / math.pi ** 2)


def _inflate(state: tuple[float, float, pd.Timestamp | None], date: pd.Timestamp) -> tuple[float, float]:
    rating, rd, through = state
    elapsed = (date - through).days if through is not None else 0
    if elapsed < 0:
        raise ValueError('Rating state cannot come from a future date')
    return rating, min(MAX_RD, math.sqrt(rd * rd + RD_VARIANCE_PER_DAY * elapsed))


def _update(rating: float, rd: float, opponents: list[tuple[float, float, float]]) -> tuple[float, float]:
    if not opponents:
        return rating, rd
    information, residuals = [], []
    for opponent_rating, opponent_rd, score in opponents:
        attenuation = _g(opponent_rd)
        expected = float(expit(Q * attenuation * (rating - opponent_rating)))
        information.append(Q * Q * attenuation * attenuation * expected * (1 - expected))
        residuals.append(attenuation * (score - expected))
    precision = 1 / (rd * rd) + math.fsum(information)
    return rating + Q * math.fsum(residuals) / precision, max(MIN_RD, math.sqrt(1 / precision))


def _pair(first: tuple[float, float], second: tuple[float, float]) -> tuple[float, ...]:
    first_rating, first_rd = first
    second_rating, second_rd = second
    combined_rd = math.hypot(first_rd, second_rd)
    difference = first_rating - second_rating
    return (difference, (first_rating + second_rating) / 2, first_rd - second_rd,
            combined_rd, Q * _g(combined_rd) * difference)


def add_ratings(frame: pd.DataFrame, fights: pd.DataFrame) -> pd.DataFrame:
    required = ['date', 'red_fighter_id', 'blue_fighter_id']
    if not set(required) <= set(frame.columns) or not {*required, 'fight_id', 'winner'} <= set(fights.columns):
        raise ValueError('Features require dates/fighter IDs; fight history also requires fight_id and winner')
    if set(ADDED_FEATURES) & set(frame.columns):
        raise ValueError('Rating columns already exist; existing features will not be overwritten')
    if fights.fight_id.duplicated().any():
        raise ValueError('Fight identities must be unique')
    result = frame.copy()
    if frame.empty:
        for name in ADDED_FEATURES:
            result[name] = pd.Series(index=frame.index, dtype=float)
        result.attrs['rating_features'] = {'config': CONFIG, 'history_bouts': 0}
        return result
    requests = frame[required].copy()
    requests['position'] = np.arange(len(requests))
    usable = fights.dropna(subset=required).copy()
    for part in (requests, usable):
        part['date'] = pd.to_datetime(part.date)
        if part.date.isna().any() or part.date.dt.tz is not None or not part.date.equals(part.date.dt.normalize()):
            raise ValueError('Use complete timezone-naive calendar dates')
        if part[required[1:]].isna().any().any() or (part.red_fighter_id == part.blue_fighter_id).any():
            raise ValueError('Rating features require two distinct known fighter IDs')
    usable = usable[usable.date <= requests.date.max()].sort_values(['date', 'fight_id'])
    fight_days = dict(tuple(usable.groupby('date', sort=True)))
    request_days = dict(tuple(requests.groupby('date', sort=True)))
    states = defaultdict(lambda: (INITIAL_RATING, INITIAL_RD, None))
    values = np.empty((len(frame), len(ADDED_FEATURES)))
    for date in sorted(set(fight_days) | set(request_days)):
        day, wanted = fight_days.get(date), request_days.get(date)
        identities = set()
        for part in (day, wanted):
            if part is not None:
                identities.update(part.red_fighter_id)
                identities.update(part.blue_fighter_id)
        before = {fid: _inflate(states[fid], date) for fid in identities}
        if wanted is not None:
            for row in wanted.itertuples(index=False):
                values[row.position] = _pair(before[row.red_fighter_id], before[row.blue_fighter_id])
        if day is None:
            continue
        opponents = defaultdict(list)
        for row in day.itertuples(index=False):
            score = {'red': 1.0, 'blue': 0.0, 'draw': 0.5}.get(row.winner)
            if score is None:
                continue
            rid, bid = row.red_fighter_id, row.blue_fighter_id
            opponents[rid].append((*before[bid], score))
            opponents[bid].append((*before[rid], 1 - score))
        for fid in set(day.red_fighter_id) | set(day.blue_fighter_id):
            rating, rd = _update(*before[fid], opponents[fid])
            states[fid] = (rating, rd, date)
    for index, name in enumerate(ADDED_FEATURES):
        result[name] = values[:, index]
    result.attrs['rating_features'] = {'config': CONFIG, 'history_bouts': len(usable)}
    return result
