import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
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

WINDOWS = (3, 5)
REPLACED = [f"diff_td_{kind}_last{window}" for window in WINDOWS for kind in ("accuracy", "defense")]
ADDED = [f"{prefix}_td_{kind}_exposure_last{window}" for window in WINDOWS
         for kind in ("attack", "defense") for prefix in ("diff", "ctx_min")]
CONFIG = {
    "candidate": "pooled_grappling56", "developmentYears": list(range(2018, 2023)),
    "trainingStart": "2010-01-01", "fitEndExclusive": "2023-01-01", "calibrationEndExclusive": "2025-01-01",
    "learner": "Unchanged GB100/depth3/learning_rate0.05/seed42, existing median+indicator preprocessing",
    "ratios": "For last3/5 bouts, sum valid attempts and successes; Beta(1,1) fixed smoothing. Defense successes are opponent attempts minus landed.",
    "zeroAttempts": "Valid history with zero attempts receives prior0.5 and exposure0; no valid statistics remains NaN and exposure0.",
    "exposure": "Add log1p attempted-count difference and minimum for attack/defense in each window; total56 features.",
    "history": "Strict event date; same-day updates occur only after all feature rows for that day. Invalid/absent stats occupy history windows but contribute no counts.",
    "development": "For yearY, fit2010..Y-1 and scoreY, raw probabilities for both models; fixed ablation, no hyperparameter search.",
    "calibration": "Final model fits2010..2022; one positive temperature fits2023..2024 logloss, log(T) bounds[-3,3].",
    "gate": "Review only if pooled development and985 replay both improve Brier+logloss, with recent49 degradation<=0.01 Brier and<=0.03 logloss. No automatic promotion.",
    "status": "Exploratory historical ablation motivated by already-inspected market errors; no untouched validation or market-edge claim.",
    "bootstrap": audit.BOOTSTRAP, "tieRule": "Red whenp>=0.5", "promotionAllowed": False,
}


def state(history: list[dict | None], window: int) -> dict:
    valid = [row for row in history[-window:] if row is not None]
    result = {}
    for kind in ("attack", "defense"):
        attempted = sum(row[f"{kind}_attempted"] for row in valid)
        successes = sum(row[f"{kind}_successes"] for row in valid)
        result[kind] = (successes + 1) / (attempted + 2) if valid else np.nan
        result[f"{kind}_exposure"] = np.log1p(attempted)
    return result


def add_features(frame: pd.DataFrame, fights: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
    if set(ADDED) & set(frame):
        raise ValueError("Grappling features already present")
    totals, _ = base.prepare_round_stats(rounds)
    history, records = defaultdict(list), []
    requested = set(frame.fight_id)
    usable = fights.dropna(subset=["date", "red_fighter_id", "blue_fighter_id"])
    for _, day in usable.sort_values(["date", "fight_id"]).groupby("date", sort=True):
        for row in day.itertuples(index=False):
            if row.fight_id not in requested:
                continue
            features = {"fight_id": row.fight_id}
            for window in WINDOWS:
                red, blue = (state(history[fid], window) for fid in (row.red_fighter_id, row.blue_fighter_id))
                for kind, ratio in (("attack", "accuracy"), ("defense", "defense")):
                    features[f"diff_td_{ratio}_last{window}"] = red[kind] - blue[kind]
                    r, b = red[f"{kind}_exposure"], blue[f"{kind}_exposure"]
                    features[f"diff_td_{kind}_exposure_last{window}"] = r - b
                    features[f"ctx_min_td_{kind}_exposure_last{window}"] = min(r, b)
            records.append(features)
        for row in day.itertuples(index=False):
            for fid, opponent in ((row.red_fighter_id, row.blue_fighter_id), (row.blue_fighter_id, row.red_fighter_id)):
                valid = np.isfinite(base.bout_rates(row, fid, opponent, totals)["sig_landed_pm"])
                entry = None
                if valid:
                    own, opp = totals[(row.fight_id, fid)], totals[(row.fight_id, opponent)]
                    entry = {"attack_attempted": own["td_attempted"], "attack_successes": own["td_landed"],
                             "defense_attempted": opp["td_attempted"], "defense_successes": opp["td_attempted"] - opp["td_landed"]}
                history[fid].append(entry)
    additions = pd.DataFrame(records)
    if additions.fight_id.duplicated().any() or set(additions.fight_id) != requested:
        raise ValueError("Grappling feature identity coverage changed")
    result = frame.drop(columns=REPLACED).merge(additions, on="fight_id", how="left", validate="one_to_one", sort=False)
    return result[list(frame.columns) + ADDED]


def run(source: Path, output: Path) -> None:
    baseline_path = source_tools.BASELINE / "provisional_challenger.joblib"
    if base.digest(baseline_path) != source_tools.MODEL_SHA:
        raise ValueError("Corrected48 model changed")
    inputs = [Path(__file__), Path(base.__file__), Path(challenger.__file__), Path(audit.__file__),
              Path(source_tools.__file__), ROOT / "tests/test_grappling_features.py", baseline_path,
              source_tools.BASELINE / "development_and_calibration_features.parquet", source_tools.BASELINE / "predictions.csv",
              audit.RECENT, source / "manifest.json"]
    inputs += [source / name for name in json.loads((source / "manifest.json").read_text())["files"]]
    hashes = {str(path): base.digest(path) for path in inputs}
    output.mkdir(parents=True, exist_ok=False)
    audit.write(output / "protocol.json", {"createdAt": datetime.now(timezone.utc).isoformat(), "config": CONFIG, "hashes": hashes})
    for path in inputs[:6]:
        (output / path.name).write_bytes(path.read_bytes())
    print(json.dumps({"stage": "protocol_frozen", "output": str(output)}), flush=True)
    original = pd.read_parquet(source_tools.BASELINE / "development_and_calibration_features.parquet")
    fights, rounds, fighters = source_tools.source_frames(source)
    features = add_features(original, fights, rounds)
    model48 = joblib.load(baseline_path)
    columns48 = model48["feature_columns"]
    columns56 = [name for name in columns48 if name != "weightclass"] + ADDED + ["weightclass"]
    if len(columns56) != 56:
        raise ValueError("Unexpected feature count")
    unchanged = [name for name in original if name not in REPLACED]
    pd.testing.assert_frame_equal(original[unchanged], features[unchanged], check_exact=True)
    prefix_checks = []
    for cutoff in ("2018-01-01", "2021-01-01", "2023-01-01", "2025-01-01"):
        prior = fights[fights.date < pd.Timestamp(cutoff)]
        subset = original[original.date < pd.Timestamp(cutoff)].reset_index(drop=True)
        rebuilt = add_features(subset, prior, rounds[rounds.fight_id.isin(prior.fight_id)])
        pd.testing.assert_frame_equal(rebuilt, features[features.date < pd.Timestamp(cutoff)].reset_index(drop=True), check_exact=True)
        prefix_checks.append({"cutoffExclusive": cutoff, "rows": len(subset), "passed": True})
    audit.write(output / "prefix_checks.json", prefix_checks)
    folds, development = {}, []
    for year in CONFIG["developmentYears"]:
        folds[year] = {}
        for name, data, columns in (("corrected48", original, columns48), ("pooled56", features, columns56)):
            train = data[data.date < pd.Timestamp(f"{year}-01-01")]
            model = challenger.fit_model("gradient_boosting", train, columns)
            filename = f"{year}_{name}.joblib"
            joblib.dump(model, output / filename, compress=3)
            folds[year][name] = {"file": filename, "sha256": base.digest(output / filename), "trainRows": len(train), "trainIds": audit.id_hash(train.fight_id)}
        print(json.dumps({"stage": "development_models_frozen", "year": year}), flush=True)
    train = features[features.date < pd.Timestamp(CONFIG["fitEndExclusive"])]
    calibration = features[(features.date >= pd.Timestamp(CONFIG["fitEndExclusive"])) & (features.date < pd.Timestamp(CONFIG["calibrationEndExclusive"]))]
    if len(train) != 5557 or len(calibration) != 1016:
        raise ValueError("Final fit/calibration membership changed")
    model = challenger.fit_model("gradient_boosting", train, columns56)
    raw = base.symmetric_probability(model, calibration[columns56])
    optimized = minimize_scalar(lambda log_t: log_loss(calibration.target, base.apply_temperature(raw, math.exp(log_t)), labels=[0, 1]), bounds=(-3, 3), method="bounded")
    if not optimized.success:
        raise ValueError("Temperature fit failed")
    temperature = math.exp(optimized.x)
    joblib.dump({"model": model, "temperature": temperature, "feature_columns": columns56, "config": CONFIG,
                 "protocolSHA256": base.digest(output / "protocol.json"), "validated_for_betting": False}, output / "candidate.joblib", compress=3)
    audit.write(output / "fit_receipt.json", {"developmentFolds": folds, "temperature": temperature, "fitRows": len(train), "calibrationRows": len(calibration)})
    seal = {path.name: base.digest(path) for path in output.iterdir() if path.is_file()}
    audit.write(output / "pre-score-manifest.json", {"frozenAt": datetime.now(timezone.utc).isoformat(), "files": seal})
    print(json.dumps({"stage": "all_models_frozen_before_scoring"}), flush=True)
    for year in CONFIG["developmentYears"]:
        mask = original.date.dt.year == year
        rows = original.loc[mask, ["fight_id", "event_id", "date", "target"]].copy().assign(test_year=year)
        for name, data, columns in (("corrected48", original, columns48), ("pooled56", features, columns56)):
            fitted = joblib.load(output / folds[year][name]["file"])
            rows[name] = audit.probability([fitted], 1.0, data.loc[mask, columns])
        development.append(rows)
    development = pd.concat(development, ignore_index=True)
    development.to_csv(output / "development_predictions.csv", index=False)
    totals, _ = base.prepare_round_stats(rounds)
    full_original = challenger.add_context(base.reconstruct(fights, totals), fights, fighters)
    pd.testing.assert_frame_equal(full_original[full_original.date < pd.Timestamp("2025-01-01")].reset_index(drop=True), original, check_exact=True)
    full = add_features(full_original, fights, rounds)
    archive = pd.read_csv(source_tools.BASELINE / "predictions.csv", parse_dates=["date"])
    if len(archive) != 985 or audit.id_hash(archive.fight_id) != audit.COHORT_HASHES["same985"]:
        raise ValueError("Replay cohort changed")
    replay = full.set_index("fight_id").loc[archive.fight_id].reset_index()
    reference = full_original.set_index("fight_id").loc[archive.fight_id].reset_index()
    for side in ("red", "blue"):
        if not np.array_equal(replay[f"{side}_fighter_id"], archive[f"corrected_{side}_fighter_id"]):
            raise ValueError("Replay identity mismatch")
    if not np.array_equal(replay.target, archive.target):
        raise ValueError("Replay target mismatch")
    reproduced = audit.probability([model48["model"]], model48["temperature"], reference[columns48])
    if not np.allclose(reproduced, archive.original48_after, atol=1e-12, rtol=0):
        raise ValueError("Corrected48 prediction parity failed")
    rows = replay[["fight_id", "event_id", "date", "red_fighter_id", "blue_fighter_id", "target"]].copy()
    rows["corrected48"] = reproduced
    rows["pooled56"] = audit.probability([model], temperature, replay[columns56])
    rows.to_csv(output / "predictions.csv", index=False)
    results = {}
    cohorts = {**{str(year): development[development.test_year == year] for year in CONFIG["developmentYears"]},
               "development": development, "same985": rows, "primary285": rows[rows.date > pd.Timestamp("2026-05-09")],
               "recent49": rows[rows.date.isin(pd.to_datetime(audit.RECENT_DATES))]}
    if len(cohorts["primary285"]) != 285 or len(cohorts["recent49"]) != 49:
        raise ValueError("Secondary cohort mismatch")
    for name, part in cohorts.items():
        results[name] = {"metrics": {column: audit.metrics(part, column) for column in ("corrected48", "pooled56")},
                         "paired": audit.paired(part, "corrected48", "pooled56")}
    conditions = {f"{cohort}_{metric}": results[cohort]["metrics"]["pooled56"][metric] < results[cohort]["metrics"]["corrected48"][metric]
                  for cohort in ("development", "same985") for metric in ("brier", "logLoss")}
    for metric, limit in (("brier", .01), ("logLoss", .03)):
        recent = results["recent49"]["metrics"]
        conditions[f"recent49_{metric}"] = recent["pooled56"][metric] - recent["corrected48"][metric] <= limit
    for path, checksum in hashes.items():
        if base.digest(Path(path)) != checksum:
            raise ValueError("Input changed during experiment")
    for filename, checksum in seal.items():
        if base.digest(output / filename) != checksum:
            raise ValueError("Frozen model changed during experiment")
    report = {"config": CONFIG, "results": results, "gate": {"conditions": conditions, "eligibleForReview": all(conditions.values())},
              "temperature": temperature, "sourceHashesUnchanged": True, "baselineParityPassed": True, "runtimeChanged": False,
              "limitations": ["Historical outcomes have been reused; the missingness hypothesis was prompted by later market errors.",
                              "Event bootstrap uncertainty omits model design selection, repeated fighters and source revision uncertainty.",
                              "The fixed Beta(1,1) prior is a representation choice, not an externally validated wrestling-success prior."]}
    audit.write(output / "report.json", report)
    audit.write(output / "artifact_hashes.json", {path.name: base.digest(path) for path in output.iterdir() if path.is_file()})
    print(json.dumps({"stage": "complete", "gate": report["gate"], "results": {name: results[name]["metrics"] for name in ("development", "same985", "recent49")}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed pooled-takedown ablation with chronological development and reused replay.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve())
