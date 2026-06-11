import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate_challenger as challenger


STATE_NAMES = (
    "bouts", "wins", "losses", "smoothed_score", "finish_win_rate", "finish_loss_rate",
    "recent365_score", "recent365_bouts", "layoff_days", "supplement_bouts",
)
ADDED_FEATURES = [f"diff_observed_career_{name}" for name in STATE_NAMES] + [
    "ctx_min_observed_career_bouts", "ctx_min_observed_career_supplement_bouts",
]
CONFIG = {
    "version": 1, "addedFeatures": ADDED_FEATURES, "recentWindowDays": 365,
    "score": "(wins + 0.5*draws +1)/(wins+losses+draws+2); fixed neutral Beta(1,1) smoothing",
    "experience": "Observed W/L/D/NC bouts; never an assertion of complete career coverage",
    "finishes": "KO/TKO or submission per observed win/loss with recognized method; absent denominator remains missing",
    "recent": "All bouts in [eventDate-365days,eventDate); avoids arbitrary ordering within a date",
    "history": "Corrected UFC result history plus approved nonoverlapping supplement; date strictly before request",
    "identity": "Retain each individually known UFC fighter side even if the opponent ID is missing; never infer the missing identity",
    "nc": "NC contributes experience/activity/layoff only, never score or finish-rate denominators",
    "unchanged": "Original48, UFC round metrics, Elo, training labels and fighter physical profiles are untouched",
    "availability": "Retrospective event dates do not establish historical publication or revision availability",
}


def _dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise")
    if parsed.isna().any() or parsed.dt.tz is not None or not parsed.equals(parsed.dt.normalize()):
        raise ValueError("Career histories require complete timezone-naive calendar dates")
    return parsed


def _identity(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{16}", value) is not None


def load_supplement(directory: Path) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schemaVersion") != 1 or manifest.get("bundleType") != "career-supplement":
        raise ValueError("Unsupported career supplement manifest")
    files = manifest.get("files", {})
    if "fighter_bouts.parquet" not in files:
        raise ValueError("Career supplement history file is missing")
    for name, expected in files.items():
        path = directory / name
        if path.resolve().parent != directory.resolve() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Career supplement checksum mismatch: {name}")
    frame = pd.read_parquet(directory / "fighter_bouts.parquet")
    frame["date"] = _dates(frame.date)
    if not frame.empty and frame.date.max() > pd.Timestamp(manifest["cutoffInclusive"]):
        raise ValueError("Career supplement extends beyond its declared cutoff")
    return frame, manifest


def prepare_history(fights: pd.DataFrame, supplement: pd.DataFrame) -> dict[str, pd.DataFrame]:
    required = {"fighter_id", "opponent_id", "canonical_fighter_id", "canonical_opponent_id", "canonical_fight_id",
                "date", "result", "opponent_result", "method", "source_aliases_json"}
    if not required <= set(supplement):
        raise ValueError("Career supplement columns are incomplete")
    extra = supplement.copy()
    extra["date"] = _dates(extra.date)
    if not extra.fighter_id.map(_identity).all() or not extra.opponent_id.map(lambda x: pd.isna(x) or _identity(x)).all():
        raise ValueError("Career supplement fighter identities are invalid")
    if extra.duplicated(["fighter_id", "canonical_fight_id"]).any():
        raise ValueError("Duplicate fighter/bout career records")
    for column in ("canonical_fighter_id", "canonical_opponent_id", "canonical_fight_id"):
        if not extra[column].map(lambda x: isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))).all():
            raise ValueError("Canonical career identities must be positive integers")
        values = pd.to_numeric(extra[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Canonical career identities must be positive integers")
    if (extra.canonical_fighter_id == extra.canonical_opponent_id).any() or (extra.fighter_id == extra.opponent_id).any():
        raise ValueError("A career opponent cannot be the same fighter")
    complement = {"W": "L", "L": "W", "D": "D", "NC": "NC"}
    if not extra.result.isin(complement).all() or not (extra.result.map(complement) == extra.opponent_result).all():
        raise ValueError("Career outcomes are inconsistent")
    if (extra.groupby("fighter_id").canonical_fighter_id.nunique() > 1).any() or (extra.groupby("canonical_fighter_id").fighter_id.nunique() > 1).any():
        raise ValueError("Career identity crosswalk is ambiguous")
    canonical_ids = dict(zip(extra.fighter_id, extra.canonical_fighter_id))
    for row in extra.itertuples():
        if pd.notna(row.opponent_id) and row.opponent_id in canonical_ids and canonical_ids[row.opponent_id] != row.canonical_opponent_id:
            raise ValueError("Career opponent crosswalk is inconsistent")
    for _, group in extra.groupby("canonical_fight_id", sort=False):
        if len(group) > 2 or group.date.nunique() != 1:
            raise ValueError("Career bout participants or dates are inconsistent")
        if len(group) == 2:
            first, second = list(group.itertuples(index=False))
            if (first.canonical_fighter_id != second.canonical_opponent_id or first.canonical_opponent_id != second.canonical_fighter_id
                    or first.result != second.opponent_result or first.opponent_result != second.result):
                raise ValueError("Career bout participants or outcomes are inconsistent")
            methods = group.method.fillna("").str.casefold().str.strip()
            if methods.nunique() != 1:
                raise ValueError("Career bout methods are inconsistent")
    if extra.duplicated(["fighter_id", "canonical_opponent_id", "date"]).any():
        raise ValueError("Ambiguous same-pair/date career bouts")
    source = fights.copy()
    source["date"] = _dates(source.date)
    valid_side = lambda x: pd.isna(x) or _identity(x)
    if source.fight_id.duplicated().any() or not source.red_fighter_id.map(valid_side).all() or not source.blue_fighter_id.map(valid_side).all():
        raise ValueError("Existing source fight identities are invalid")
    if (source.red_fighter_id == source.blue_fighter_id).any():
        raise ValueError("Existing source has a self matchup")
    fight_ids = set(fights.fight_id)
    existing_pairs = {(row.date, *sorted((row.red_fighter_id, row.blue_fighter_id)))
                      for row in source.dropna(subset=["red_fighter_id", "blue_fighter_id"]).itertuples()}
    for row in extra.itertuples():
        aliases = json.loads(row.source_aliases_json)
        if not isinstance(aliases, list) or not aliases or any(not isinstance(item, dict) or not item.get("source") or not item.get("sourceID") for item in aliases):
            raise ValueError("Career source aliases are malformed")
        if any(item.get("source") == "ufcstats" and item.get("sourceID") in fight_ids for item in aliases):
            raise ValueError("Career supplement overlaps an existing UFCStats fight")
        if pd.notna(row.opponent_id) and (row.date, *sorted((row.fighter_id, row.opponent_id))) in existing_pairs:
            raise ValueError("Career supplement overlaps an existing pair/date")
    red_result = source.winner.map({"red": "W", "blue": "L", "draw": "D", "nc": "NC"})
    if red_result.isna().any():
        raise ValueError("Existing source contains an unsupported outcome")
    records = []
    for side, results in (("red", red_result), ("blue", red_result.map(complement))):
        records.append(pd.DataFrame({"fighter_id": source[f"{side}_fighter_id"], "bout_id": "ufcstats:" + source.fight_id,
                                     "date": source.date, "result": results, "method": source.method,
                                     "supplement": False}).dropna(subset=["fighter_id"]))
    records.append(extra.assign(bout_id="canon:" + extra.canonical_fight_id.astype(str), supplement=True)[
        ["fighter_id", "bout_id", "date", "result", "method", "supplement"]])
    combined = pd.concat(records, ignore_index=True)
    method = combined.method.fillna("").str.casefold().str.strip()
    combined["finish"] = np.where(method.str.contains(r"\b(?:ko|tko)\b|submission", regex=True), 1.0,
                                   np.where(method.str.contains(r"decision|\bdq\b|disqualification|forfeit", regex=True), 0.0, np.nan))
    return {fighter_id: frame.sort_values(["date", "bout_id"]).reset_index(drop=True)
            for fighter_id, frame in combined.groupby("fighter_id", sort=False)}


def _state(history: pd.DataFrame | None, date: pd.Timestamp) -> dict:
    if history is None:
        return dict(zip(STATE_NAMES, [0, 0, 0, 0.5, np.nan, np.nan, 0.5, 0, np.nan, 0]))
    stop = history.date.searchsorted(date, side="left")
    past = history.iloc[:stop]
    start = history.date.searchsorted(date - pd.Timedelta(days=365), side="left")
    recent = history.iloc[min(start, stop):stop]
    wins, losses, draws = (int(past.result.eq(value).sum()) for value in ("W", "L", "D"))
    recent_wins, recent_losses, recent_draws = (int(recent.result.eq(value).sum()) for value in ("W", "L", "D"))
    return {"bouts": len(past), "wins": wins, "losses": losses,
            "smoothed_score": (wins + 0.5 * draws + 1) / (wins + losses + draws + 2),
            "finish_win_rate": float(past.loc[past.result.eq("W"), "finish"].mean()),
            "finish_loss_rate": float(past.loc[past.result.eq("L"), "finish"].mean()),
            "recent365_score": (recent_wins + 0.5 * recent_draws + 1) / (recent_wins + recent_losses + recent_draws + 2),
            "recent365_bouts": len(recent), "layoff_days": (date - past.date.iloc[-1]).days if len(past) else np.nan,
            "supplement_bouts": int(past.supplement.sum())}


def add_career_features(frame: pd.DataFrame, fights: pd.DataFrame, supplement: pd.DataFrame,
                        prepared_history: dict | None = None) -> pd.DataFrame:
    if set(ADDED_FEATURES) & set(frame):
        raise ValueError("Career feature columns already exist")
    if not {"date", "red_fighter_id", "blue_fighter_id"} <= set(frame):
        raise ValueError("Career feature requests need a date and both fighter IDs")
    requests = frame[["date", "red_fighter_id", "blue_fighter_id"]].copy()
    requests["date"] = _dates(requests.date)
    if not requests.red_fighter_id.map(_identity).all() or not requests.blue_fighter_id.map(_identity).all() or (requests.red_fighter_id == requests.blue_fighter_id).any():
        raise ValueError("Career feature request identities are invalid")
    history = prepared_history if prepared_history is not None else prepare_history(fights, supplement)
    states, values = {}, []
    for row in requests.itertuples(index=False):
        for fighter_id in (row.red_fighter_id, row.blue_fighter_id):
            key = (fighter_id, row.date)
            if key not in states:
                states[key] = _state(history.get(fighter_id), row.date)
        first, second = states[(row.red_fighter_id, row.date)], states[(row.blue_fighter_id, row.date)]
        values.append([first[name] - second[name] for name in STATE_NAMES] +
                      [min(first["bouts"], second["bouts"]), min(first["supplement_bouts"], second["supplement_bouts"])])
    result = frame.copy()
    for index, name in enumerate(ADDED_FEATURES):
        result[name] = [row[index] for row in values]
    result.attrs["career_features"] = {"config": CONFIG, "requestedRows": len(frame),
        "fighterDateStatesWithPriorSupplement": sum(state["supplement_bouts"] > 0 for state in states.values()),
        "uniqueFighterDateStates": len(states)}
    return result


def build_inference_features(fights: pd.DataFrame, rounds: pd.DataFrame, fighters: pd.DataFrame, supplement: pd.DataFrame,
                             fighter1_id: str, fighter2_id: str, event_date: str, weightclass: str,
                             scheduled_rounds: int, is_title: bool, prepared_totals: dict | None = None,
                             prepared_history: dict | None = None) -> tuple[pd.DataFrame, dict]:
    original, quality = challenger.build_inference_features(fights, rounds, fighters, fighter1_id, fighter2_id,
        event_date, weightclass, scheduled_rounds, is_title, prepared_totals=prepared_totals)
    request = original.assign(date=pd.Timestamp(event_date), red_fighter_id=fighter1_id, blue_fighter_id=fighter2_id)
    augmented = add_career_features(request, fights, supplement, prepared_history)
    return augmented.drop(columns=["date", "red_fighter_id", "blue_fighter_id"]), quality
