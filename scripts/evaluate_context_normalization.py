import argparse
import importlib.metadata
import json
import math
import platform
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_career_supplement as source_tools
from scripts import evaluate_challenger as challenger
from scripts import evaluate_identity_repair as audit
from scripts import evaluate_research as base

DIVISIONS = (
    "Flyweight", "Bantamweight", "Featherweight", "Lightweight", "Welterweight",
    "Middleweight", "Light Heavyweight", "Heavyweight", "Women's Strawweight",
    "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight",
)
DIVISION_PATTERN = "(?:" + "|".join(re.escape(value) for value in DIVISIONS) + ")"
TOURNAMENT_PATTERN = re.compile(
    r"^(?:Ultimate Fighter (?:[1-9][0-9]*|Brazil [1-9][0-9]*|Latin America(?: [1-9][0-9]*)?|China|Australia vs\. UK)"
    r"|TUF Nations Canada vs\. Australia|Road [Tt]o [1-9][0-9]*) "
    + f"(?P<division>{DIVISION_PATTERN})" + r" Tournament(?: TitleBout)?$"
)
OVERTIME_FORMAT = "3 Rnd + OT (5-5-5-5)"
CONFIG = {
    "candidate": "context_normalized48", "featureCount": 48,
    "developmentYears": list(range(2018, 2023)), "trainingStart": "2010-01-01",
    "fitEndExclusive": "2023-01-01", "calibrationEndExclusive": "2025-01-01",
    "division": "Exact standard labels, Interim <standard>, and anchored named TUF/Road-to-UFC tournament grammar map to the standard division. Every other label maps to Unknown; no substring or physical-weight guesses.",
    "divisionParser": TOURNAMENT_PATTERN.pattern, "standardDivisions": list(DIVISIONS),
    "title": "Original is_title_bout is unchanged, including tournament finals. Raw competition kind is retained in audit metadata only; no claim that a tournament title is a UFC championship.",
    "rounds": "Only exact 3 Rnd + OT (5-5-5-5), num_rounds=3 and finish_round in1..3 receives the existing 3 Rnd (5-5-5) duration path. All original count, round-completeness, duplicate, date and duration validation remains required. No actual fourth round is admitted.",
    "learner": challenger.CONFIG["models"]["gradient_boosting"],
    "development": "For yearY, fit2010..Y-1 and scoreY with raw probabilities, for both fixed models. No selection or tuning.",
    "calibration": "Final candidate fits2010..2022; one positive temperature fits2023..2024 logloss with log(T) bounds[-3,3].",
    "comparison": "Same985 decisive replay rows, plus285 after2026-05-09 and49 on the last four cards. Original corrected48 predictions must match the sealed archive to1e-12.",
    "selection": "One fixed paired comparison; report accuracy, Brier, logloss and paired event bootstrap intervals. No pass threshold, model selection or automatic promotion.",
    "bootstrap": audit.BOOTSTRAP, "tieRule": "Red when p>=0.5", "promotionAllowed": False,
    "status": "Previously inspected historical data, not untouched validation. No claim of beating Kalshi or of prospective accuracy.",
}


def normalized_division(value: object) -> tuple[str, str]:
    if not isinstance(value, str):
        return "Unknown", "unknown"
    if value in DIVISIONS:
        return value, "standard"
    if value.startswith("Interim ") and value[8:] in DIVISIONS:
        return value[8:], "interim"
    match = TOURNAMENT_PATTERN.fullmatch(value)
    if match:
        return match["division"], "tournament"
    return "Unknown", "unknown"


def context_view(fights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fights.fight_id.duplicated().any():
        raise ValueError("Duplicate fight identities")
    result = fights.copy(deep=True)
    normalized = [normalized_division(value) for value in fights.weightclass]
    result["weightclass"] = [value[0] for value in normalized]
    overtime = (fights.time_format.eq(OVERTIME_FORMAT) & fights.num_rounds.eq(3)
                & fights.finish_round.isin([1, 2, 3]))
    result.loc[overtime, "time_format"] = "3 Rnd (5-5-5)"
    metadata = fights[["fight_id", "date", "weightclass", "time_format", "is_title_bout"]].copy()
    metadata = metadata.rename(columns={"weightclass": "raw_weightclass", "time_format": "raw_time_format"})
    metadata["canonical_weightclass"] = result.weightclass
    metadata["competition_kind"] = [value[1] for value in normalized]
    metadata["completed_before_optional_overtime"] = overtime
    return result, metadata


def features(fights: pd.DataFrame, rounds: pd.DataFrame, fighters: pd.DataFrame,
             cutoff: str | None = None, normalize: bool = True) -> pd.DataFrame:
    prior = fights if cutoff is None else fights[fights.date < pd.Timestamp(cutoff)]
    prior_rounds = rounds[rounds.fight_id.isin(prior.fight_id)]
    view = context_view(prior)[0] if normalize else prior
    totals, _ = base.prepare_round_stats(prior_rounds)
    return challenger.add_context(base.reconstruct(view, totals), view, fighters)


def assert_compatible(original: pd.DataFrame, candidate: pd.DataFrame) -> None:
    allowed = {"weightclass", "ctx_min_stats_last5"}
    for window in (3, 5):
        allowed.add(f"diff_stat_bouts_last{window}")
        allowed.update(f"diff_{rate}_last{window}" for rate in base.RATE_COLS)
    if list(original.columns) != list(candidate.columns):
        raise ValueError("Feature schema changed")
    unchanged = [column for column in original if column not in allowed]
    pd.testing.assert_frame_equal(original[unchanged], candidate[unchanged], check_exact=True)


def run(source: Path, output: Path) -> None:
    baseline_path = source_tools.BASELINE / "provisional_challenger.joblib"
    if base.digest(baseline_path) != source_tools.MODEL_SHA:
        raise ValueError("Corrected48 model changed")
    code = [Path(__file__), Path(base.__file__), Path(challenger.__file__), Path(audit.__file__),
            Path(source_tools.__file__), ROOT / "tests/test_context_normalization.py"]
    inputs = [*code, baseline_path, source_tools.BASELINE / "development_and_calibration_features.parquet",
              source_tools.BASELINE / "predictions.csv", audit.RECENT, source / "manifest.json"]
    inputs += [source / name for name in json.loads((source / "manifest.json").read_text())["files"]]
    hashes = {str(path): base.digest(path) for path in inputs}
    output.mkdir(parents=True, exist_ok=False)
    audit.write(output / "protocol.json", {"createdAt": audit.now(), "config": CONFIG, "inputHashes": hashes,
                "environment": {"python": platform.python_version(), "packages": {
                    name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "fastparquet")}}})
    for path in code:
        (output / path.name).write_bytes(path.read_bytes())
    print(json.dumps({"stage": "protocol_frozen", "output": str(output)}), flush=True)
    fights, rounds, fighters = source_tools.source_frames(source)
    view, inventory = context_view(fights)
    unchanged_source = [column for column in fights if column not in ("weightclass", "time_format")]
    pd.testing.assert_frame_equal(fights[unchanged_source], view[unchanged_source], check_exact=True)
    inventory.to_csv(output / "context_inventory.csv", index=False)
    original = pd.read_parquet(source_tools.BASELINE / "development_and_calibration_features.parquet")
    rebuilt = features(fights, rounds, fighters, CONFIG["calibrationEndExclusive"], normalize=False)
    pd.testing.assert_frame_equal(original, rebuilt, check_exact=True)
    candidate = features(fights, rounds, fighters, CONFIG["calibrationEndExclusive"])
    assert_compatible(original, candidate)
    model48 = joblib.load(baseline_path)
    columns = model48["feature_columns"]
    if len(columns) != 48 or len(set(columns)) != 48:
        raise ValueError("Unexpected original feature schema")
    prefix = []
    for cutoff in ("2018-01-01", "2021-01-01", "2023-01-01"):
        rebuilt = features(fights, rounds, fighters, cutoff)
        expected = candidate[candidate.date < pd.Timestamp(cutoff)].reset_index(drop=True)
        pd.testing.assert_frame_equal(rebuilt, expected, check_exact=True)
        prefix.append({"cutoffExclusive": cutoff, "rows": len(expected), "passed": True})
    audit.write(output / "prefix_checks.json", prefix)
    candidate.to_parquet(output / "development_and_calibration_features.parquet", index=False)
    audit.write(output / "feature_columns.json", columns)
    folds = {}
    for year in CONFIG["developmentYears"]:
        folds[year] = {}
        for name, data in (("corrected48", original), ("normalized48", candidate)):
            train = data[(data.date >= pd.Timestamp(CONFIG["trainingStart"])) & (data.date < pd.Timestamp(f"{year}-01-01"))]
            model = challenger.fit_model("gradient_boosting", train, columns)
            filename = f"{year}_{name}.joblib"
            joblib.dump(model, output / filename, compress=3)
            folds[year][name] = {"file": filename, "sha256": base.digest(output / filename),
                                 "trainRows": len(train), "trainIds": audit.id_hash(train.fight_id)}
        if folds[year]["corrected48"]["trainIds"] != folds[year]["normalized48"]["trainIds"]:
            raise ValueError("Development training membership changed")
        print(json.dumps({"stage": "development_models_frozen", "year": year}), flush=True)
    train = candidate[(candidate.date >= pd.Timestamp(CONFIG["trainingStart"])) & (candidate.date < pd.Timestamp(CONFIG["fitEndExclusive"]))]
    calibration = candidate[(candidate.date >= pd.Timestamp(CONFIG["fitEndExclusive"])) & (candidate.date < pd.Timestamp(CONFIG["calibrationEndExclusive"]))]
    if len(train) != 5557 or len(calibration) != 1016 or set(train.event_id) & set(calibration.event_id):
        raise ValueError("Final training/calibration membership changed or overlaps")
    model = challenger.fit_model("gradient_boosting", train, columns)
    raw = base.symmetric_probability(model, calibration[columns])
    optimized = minimize_scalar(lambda log_t: log_loss(calibration.target, base.apply_temperature(raw, math.exp(log_t)), labels=[0, 1]), bounds=(-3, 3), method="bounded")
    if not optimized.success:
        raise ValueError("Temperature fit failed")
    temperature = math.exp(optimized.x)
    joblib.dump({"model": model, "temperature": temperature, "feature_columns": columns, "config": CONFIG,
                 "protocolSHA256": base.digest(output / "protocol.json"), "validated_for_betting": False}, output / "candidate.joblib", compress=3)
    audit.write(output / "fit_receipt.json", {"developmentFolds": folds, "temperature": temperature,
                "fitRows": len(train), "calibrationRows": len(calibration),
                "fitIds": audit.id_hash(train.fight_id), "calibrationIds": audit.id_hash(calibration.fight_id)})
    seal = {path.name: base.digest(path) for path in output.iterdir() if path.is_file()}
    audit.write(output / "pre-score-manifest.json", {"frozenAt": audit.now(), "files": seal})
    print(json.dumps({"stage": "all_models_frozen_before_scoring"}), flush=True)
    development = []
    for year in CONFIG["developmentYears"]:
        mask = original.date.dt.year == year
        rows = original.loc[mask, ["fight_id", "event_id", "date", "target"]].copy().assign(test_year=year)
        for name, data in (("corrected48", original), ("normalized48", candidate)):
            fitted = joblib.load(output / folds[year][name]["file"])
            rows[name] = audit.probability([fitted], 1.0, data.loc[mask, columns])
        development.append(rows)
    development = pd.concat(development, ignore_index=True)
    development.to_csv(output / "development_predictions.csv", index=False)
    full_original = features(fights, rounds, fighters, normalize=False)
    full = features(fights, rounds, fighters)
    assert_compatible(full_original, full)
    pd.testing.assert_frame_equal(full[full.date < pd.Timestamp(CONFIG["calibrationEndExclusive"])].reset_index(drop=True), candidate, check_exact=True)
    archive = pd.read_csv(source_tools.BASELINE / "predictions.csv", parse_dates=["date"])
    if len(archive) != 985 or audit.id_hash(archive.fight_id) != audit.COHORT_HASHES["same985"]:
        raise ValueError("Replay cohort changed")
    replay = full.set_index("fight_id").loc[archive.fight_id].reset_index()
    reference = full_original.set_index("fight_id").loc[archive.fight_id].reset_index()
    for side in ("red", "blue"):
        if not np.array_equal(replay[f"{side}_fighter_id"], archive[f"corrected_{side}_fighter_id"]):
            raise ValueError("Replay identity mismatch")
    for column in ("target", "date", "event_id"):
        if not np.array_equal(replay[column], archive[column]):
            raise ValueError(f"Replay {column} mismatch")
    reproduced = audit.probability([model48["model"]], model48["temperature"], reference[columns])
    if not np.allclose(reproduced, archive.original48_after, atol=1e-12, rtol=0):
        raise ValueError("Corrected48 prediction parity failed")
    rows = replay[["fight_id", "event_id", "date", "red_fighter_id", "blue_fighter_id", "target"]].copy()
    rows["corrected48"] = reproduced
    rows["normalized48"] = audit.probability([model], temperature, replay[columns])
    rows.to_csv(output / "predictions.csv", index=False)
    replay.to_parquet(output / "replay_features.parquet", index=False)
    cohorts = {**{str(year): development[development.test_year == year] for year in CONFIG["developmentYears"]},
               "development": development, "same985": rows, "primary285": rows[rows.date > pd.Timestamp("2026-05-09")],
               "recent49": rows[rows.date.isin(pd.to_datetime(audit.RECENT_DATES))]}
    expected_cohorts = {"primary285": audit.id_hash(archive.loc[archive.date > pd.Timestamp("2026-05-09"), "fight_id"]),
                        "recent49": audit.COHORT_HASHES["recent49"]}
    for name, count in (("primary285", 285), ("recent49", 49)):
        if len(cohorts[name]) != count or audit.id_hash(cohorts[name].fight_id) != expected_cohorts[name]:
            raise ValueError(f"Secondary cohort changed: {name}")
    results = {name: {"metrics": {column: audit.metrics(part, column) for column in ("corrected48", "normalized48")},
                       "paired": audit.paired(part, "corrected48", "normalized48")} for name, part in cohorts.items()}
    for path, checksum in hashes.items():
        if base.digest(Path(path)) != checksum:
            raise ValueError("Input changed during experiment")
    for filename, checksum in seal.items():
        if base.digest(output / filename) != checksum:
            raise ValueError("Frozen model changed during experiment")
    report = {"config": CONFIG, "results": results, "temperature": temperature,
              "sourceHashesUnchanged": True, "baselineParityPassed": True, "runtimeChanged": False,
              "limitations": ["Previously consumed outcomes and repeated model design; not independent confirmation.",
                              "Two connected representation corrections are bundled; this comparison cannot isolate their individual effects.",
                              "Original title flag remains broad and includes tournament titles.",
                              "Event bootstrap omits repeated-fighter, model-design and source-revision uncertainty."]}
    audit.write(output / "report.json", report)
    audit.write(output / "artifact_hashes.json", {path.name: base.digest(path) for path in output.iterdir() if path.is_file()})
    print(json.dumps({"stage": "complete", "results": {name: results[name]["metrics"] for name in ("development", "same985", "recent49")}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed48 context-normalization ablation; no source or runtime changes.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.source.resolve(), arguments.output.resolve())
