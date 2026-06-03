import argparse
import json
import math
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

if __package__:
    from . import evaluate_challenger as challenger
    from . import evaluate_predictors as previous
    from . import evaluate_research as base
    from . import predictor_features
else:
    import evaluate_challenger as challenger
    import evaluate_predictors as previous
    import evaluate_research as base
    import predictor_features


CONFIG = {
    "developmentYears": [2018, 2019, 2020, 2021, 2022, 2023],
    "candidates": ["gb48", "gb92", "blend92", "cat92_depth4", "cat92_depth6",
                   "cat92_weighted", "cat92_window5", "cat48_depth6"],
    "reference": "blend92",
    "catboost": {"iterations": 500, "learning_rate": 0.035, "l2_leaf_reg": 10,
                 "random_seed": 42, "thread_count": 2, "boosting_type": "Ordered",
                 "loss_function": "Logloss", "bootstrap_type": "Bayesian", "bagging_temperature": 1,
                 "random_strength": 1, "border_count": 128, "grow_policy": "SymmetricTree",
                 "leaf_estimation_method": "Newton", "verbose": False, "allow_writing_files": False},
    "trainingStart": "2010-01-01", "weightHalfLifeDays": 3 * 365.25, "windowCalendarYears": 5,
    "refitCutoffExclusive": "2024-01-01", "calibrationEndExclusive": "2025-01-01",
    "replayEndInclusive": "2026-10-31",
    "selection": "Highest pooled accuracy across six reused annual development folds; ties lower Brier then declared order. No automatic runtime promotion.",
    "foldPolicy": "Fit 2010 through previous December31; predict all decisive known-ID fights in each development year. Keep full prior history when restricting learner rows to five years.",
    "weightPolicy": "Only cat92_weighted: exp2(-days_before_training_cutoff/(3*365.25)), normalized to mean1 before identical corner augmentation.",
    "preprocessing": "Existing training-only median imputation, missing indicators, standardization and one-hot division encoding. No CatBoost target encoding; no eval_set or early stopping.",
    "diagnostics": "Report all yearly deltas, worst year, and leave-one-year-out selection stability without new fitting. Diagnostics cannot change the declared pooled winner.",
    "calibration": "Selected model and fixed blend92 reference refitted through2023; separate positive temperature on2024 only. It cannot change winner picks.",
    "replay": "2025-October2026 outcomes were already inspected. One selected/reference replay; never call it untouched evidence or tune after it.",
    "source": "https://catboost.ai/docs/en/references/training-parameters/common",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def code_hashes() -> dict:
    return previous.code_hashes() | {Path(__file__).name: base.digest(Path(__file__))}


def versions() -> dict:
    packages = ("catboost", "numpy", "pandas", "scipy", "scikit-learn", "fastparquet", "joblib")
    return {"python": platform.python_version(), **{name: version(name) for name in packages}}


def columns_for(name: str, columns: dict) -> list[str]:
    return columns["original" if name in ("gb48", "cat48_depth6") else "expanded"]


def fit_candidate(name: str, data: pd.DataFrame, columns: list[str], cutoff: str) -> tuple[list, dict]:
    end = pd.Timestamp(cutoff)
    training = data[(data.date >= pd.Timestamp(CONFIG["trainingStart"])) & (data.date < end)]
    if name == "cat92_window5":
        training = training[training.date >= end - pd.DateOffset(years=CONFIG["windowCalendarYears"])]
    if training.empty:
        raise ValueError("No training rows")
    receipt = {"count": len(training), "from": str(training.date.min().date()),
               "through": str(training.date.max().date()), "cutoffExclusive": cutoff, "features": len(columns)}
    if name in ("gb48", "gb92", "blend92"):
        kind = "expanded_blend" if name == "blend92" else "expanded_gb"
        return previous.fit(kind, training, columns), receipt
    params = CONFIG["catboost"] | {"depth": 4 if name == "cat92_depth4" else 6}
    pipeline = challenger.make_pipeline("gradient_boosting", columns)
    pipeline.set_params(model=CatBoostClassifier(**params))
    X = pd.concat([training[columns], base.swapped(training[columns])], ignore_index=True)
    y = np.concatenate([training.target.to_numpy(), 1 - training.target.to_numpy()])
    options = {}
    if name == "cat92_weighted":
        days = (end - training.date).dt.days.to_numpy()
        weights = np.exp2(-days / CONFIG["weightHalfLifeDays"])
        weights /= weights.mean()
        options["model__sample_weight"] = np.concatenate([weights, weights])
        receipt["weights"] = {"min": float(weights.min()), "max": float(weights.max()), "mean": float(weights.mean())}
    pipeline.fit(X, y, **options)
    return [pipeline], receipt


def rank(scores: dict) -> str:
    return min(CONFIG["candidates"], key=lambda name: (-scores[name]["accuracy"], scores[name]["brier"]))


def develop(source: Path, cached: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    cache_inventory = json.loads((cached / "artifact_hashes.json").read_text())
    for filename, expected in cache_inventory.items():
        if base.digest(cached / filename) != expected:
            raise ValueError(f"Original feature experiment changed: {filename}")
    manifest = {"createdAt": now(), "config": CONFIG, "sourceDirectory": str(source),
                "sourceHashes": previous.source_hashes(source), "codeHashes": code_hashes(),
                "featureCache": {"directory": str(cached), "inventorySha256": base.digest(cached / "artifact_hashes.json")},
                "versions": versions(),
                "status": "Further development authorized after earlier historical comparisons; all periods previously exposed."}
    cached_protocol = json.loads((cached / "protocol.json").read_text())
    if cached_protocol["sourceHashes"] != manifest["sourceHashes"]:
        raise ValueError("Cached features and current source data differ")
    for filename, expected in cached_protocol["codeHashes"].items():
        if manifest["codeHashes"].get(filename) != expected:
            raise ValueError(f"Cached feature construction code changed: {filename}")
    if manifest["versions"]["catboost"] != "1.2.10":
        raise ValueError("This experiment pins CatBoost1.2.10")
    base.write_json(out / "protocol.json", manifest)
    for filename in manifest["codeHashes"]:
        (out / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    data = pd.read_parquet(cached / "development_features.parquet")
    columns = json.loads((cached / "feature_columns.json").read_text())
    if len(columns["original"]) != 48 or len(columns["expanded"]) != 92 or data.date.max() >= pd.Timestamp(CONFIG["calibrationEndExclusive"]):
        raise ValueError("Unexpected frozen feature schema or dates")
    prefix_checks = []
    for year in (2018, 2019, 2020):
        cutoff = f"{year}-01-01"
        prefix, _, _ = previous.load_features(source, cutoff)
        pd.testing.assert_frame_equal(data[data.date < pd.Timestamp(cutoff)].reset_index(drop=True), prefix)
        prefix_checks.append({"cutoffExclusive": cutoff, "count": len(prefix), "passed": True})
        print(json.dumps({"prefixCheck": cutoff, "passed": True}), flush=True)
    base.write_json(out / "additional_prefix_checks.json", prefix_checks)
    annual, rows, fit_receipts = {}, [], {}
    for year in CONFIG["developmentYears"]:
        cutoff = f"{year}-01-01"
        validation = data[data.date.dt.year == year]
        predictions = validation[["fight_id", "event_id", "date", "target"]].copy()
        annual[str(year)], fit_receipts[str(year)] = {}, {}
        fitted = {}
        for name in CONFIG["candidates"]:
            features = columns_for(name, columns)
            if name == "blend92":
                training = data[(data.date >= pd.Timestamp(CONFIG["trainingStart"])) & (data.date < pd.Timestamp(cutoff))]
                models = fitted["gb92"] + previous.fit("expanded_logistic", training, features)
                receipt = fit_receipts[str(year)]["gb92"] | {"members": ["gb92", "logistic92"]}
            else:
                models, receipt = fit_candidate(name, data, features, cutoff)
            fitted[name] = models
            predictions[name] = previous.predict(models, validation[features])
            score = previous.summary(predictions, name)
            annual[str(year)][name] = score
            fit_receipts[str(year)][name] = receipt
            print(json.dumps({"year": year, "candidate": name, "accuracy": score["accuracy"], "brier": score["brier"]}), flush=True)
        rows.append(predictions)
    pooled = pd.concat(rows, ignore_index=True)
    scores = {name: previous.summary(pooled, name) for name in CONFIG["candidates"]}
    selected = rank(scores)
    deltas = {name: {year: annual[year][name]["accuracy"] - annual[year][CONFIG["reference"]]["accuracy"] for year in annual}
              for name in CONFIG["candidates"]}
    loyo = {}
    for year in CONFIG["developmentYears"]:
        subset = pooled[pooled.date.dt.year != year]
        omitted_scores = {name: previous.summary(subset, name) for name in CONFIG["candidates"]}
        loyo[str(year)] = {"winner": rank(omitted_scores), "metrics": {name: {key: score[key] for key in ("accuracy", "brier")}
                                                                              for name, score in omitted_scores.items()}}
    selection = {"createdAt": now(), "selected": selected, "pooled": scores, "annual": annual,
                 "yearlyAccuracyDeltasVsReference": deltas, "worstYearAccuracyDelta": {name: min(values.values()) for name, values in deltas.items()},
                 "leaveOneYearOut": loyo, "pairedWithReference": previous.paired(pooled, CONFIG["reference"], selected),
                 "policy": CONFIG["selection"]}
    base.write_json(out / "selection.json", selection)
    base.write_json(out / "fit_receipts.json", fit_receipts)
    pooled.to_csv(out / "development_predictions.csv", index=False)
    calibration = data[(data.date >= pd.Timestamp(CONFIG["refitCutoffExclusive"])) & (data.date < pd.Timestamp(CONFIG["calibrationEndExclusive"]))]
    calibration_rows = calibration[["fight_id", "event_id", "date", "target"]].copy()
    artifacts = {}
    for name in dict.fromkeys([CONFIG["reference"], selected]):
        features = columns_for(name, columns)
        models, receipt = fit_candidate(name, data, features, CONFIG["refitCutoffExclusive"])
        raw = previous.predict(models, calibration[features])
        fitted_temperature = minimize_scalar(lambda log_t: log_loss(calibration.target, base.apply_temperature(raw, math.exp(log_t))),
                                             bounds=(-3, 3), method="bounded")
        if not fitted_temperature.success:
            raise ValueError("Temperature optimization failed")
        temperature = math.exp(fitted_temperature.x)
        calibration_rows[name] = base.apply_temperature(raw, temperature)
        artifact = {"models": models, "columns": features, "temperature": temperature, "name": name,
                    "fit": receipt, "calibratedThrough": str(calibration.date.max().date()),
                    "codeHashes": manifest["codeHashes"], "sourceHashes": manifest["sourceHashes"], "validatedForBetting": False}
        joblib.dump(artifact, out / f"{name}.joblib", compress=3)
        artifacts[name] = {"sha256": base.digest(out / f"{name}.joblib"), "fit": receipt,
                           "temperature": temperature, "calibrationMetricsInSample": previous.summary(calibration_rows, name)}
    calibration_rows.to_csv(out / "calibration_predictions.csv", index=False)
    base.write_json(out / "report.json", {"createdAt": now(), "selected": selected, "development": selection,
                                          "artifacts": artifacts, "runtimeChanged": False, "replayScored": False,
                                          "supports70PercentClaim": False, "status": "Reused historical development; no independent confirmation."})
    if code_hashes() != manifest["codeHashes"] or previous.source_hashes(source) != manifest["sourceHashes"] or versions() != manifest["versions"]:
        raise ValueError("Code or source changed during the experiment")
    previous.inventory(out)
    print(json.dumps({"selected": selected, "accuracy": scores[selected]["accuracy"], "brier": scores[selected]["brier"],
                      "referenceAccuracy": scores[CONFIG["reference"]]["accuracy"]}), flush=True)


def replay(source: Path, frozen: Path, previous_replay: Path, out: Path) -> None:
    for folder in (frozen, previous_replay):
        for filename, expected in json.loads((folder / "artifact_hashes.json").read_text()).items():
            if base.digest(folder / filename) != expected:
                raise ValueError(f"Frozen artifact changed: {folder / filename}")
    protocol = json.loads((frozen / "protocol.json").read_text())
    if protocol["codeHashes"] != code_hashes() or protocol["sourceHashes"] != previous.source_hashes(source) or protocol["versions"] != versions():
        raise ValueError("Code or source changed before replay")
    selected = json.loads((frozen / "selection.json").read_text())["selected"]
    out.mkdir(parents=True, exist_ok=False)
    base.write_json(out / "protocol.json", {"createdAt": now(), "selected": selected, "status": CONFIG["replay"],
                                           "developmentInventorySHA256": base.digest(frozen / "artifact_hashes.json"),
                                           "previousReplayInventorySHA256": base.digest(previous_replay / "artifact_hashes.json")})
    data, _, _ = previous.load_features(source, "2026-11-01")
    cached = Path(protocol["featureCache"]["directory"])
    pd.testing.assert_frame_equal(data[data.date < pd.Timestamp(CONFIG["calibrationEndExclusive"])].reset_index(drop=True),
                                  pd.read_parquet(cached / "development_features.parquet"))
    frame = data[data.date >= pd.Timestamp(CONFIG["calibrationEndExclusive"])]
    rows = frame[["fight_id", "event_id", "date", "red_name", "blue_name", "target"]].reset_index(drop=True)
    for name in dict.fromkeys([CONFIG["reference"], selected]):
        artifact = joblib.load(frozen / f"{name}.joblib")
        rows[name] = base.apply_temperature(previous.predict(artifact["models"], frame[artifact["columns"]]), artifact["temperature"])
    old = pd.read_csv(previous_replay / "predictions.csv", parse_dates=["date"])
    pairs = rows.merge(old, on="fight_id", suffixes=("", "_old"), validate="one_to_one")
    if len(pairs) != len(rows) or len(pairs) != len(old):
        raise ValueError("Replay membership changed")
    for column in ("event_id", "date", "red_name", "blue_name", "target"):
        if not (pairs[column] == pairs[f"{column}_old"]).all():
            raise ValueError(f"Replay identities changed: {column}")
    if not np.allclose(pairs.blend92, pairs.expanded_blend, atol=1e-12, rtol=0):
        raise ValueError("Previous frozen blend reproduction failed")
    old_by_id = old.set_index("fight_id")
    rows["original48"] = rows.fight_id.map(old_by_id.original_frozen)
    periods = {"all_2025_2026": rows, "2025": rows[rows.date.dt.year == 2025], "2026": rows[rows.date.dt.year == 2026],
               "same_285": rows[rows.date > pd.Timestamp("2026-05-09")]}
    results = {period: {"metrics": {name: previous.summary(part, name) for name in dict.fromkeys(["original48", CONFIG["reference"], selected])},
                        "selectedVsReference": previous.paired(part, CONFIG["reference"], selected),
                        "selectedVsOriginal": previous.paired(part, "original48", selected)} for period, part in periods.items()}
    rows.to_csv(out / "predictions.csv", index=False)
    base.write_json(out / "report.json", {"createdAt": now(), "selected": selected, "results": results,
                                          "prefixInvariant": True, "previousBlendReproduced": True,
                                          "runtimeChanged": False, "supports70PercentClaim": False,
                                          "limitations": [CONFIG["replay"], "Descriptive event bootstrap does not remove selection bias or repeated-fighter dependence.",
                                                          "All known-ID decisive bouts scored; no confidence exclusions or market predictors."]})
    if code_hashes() != protocol["codeHashes"] or previous.source_hashes(source) != protocol["sourceHashes"] or versions() != protocol["versions"]:
        raise ValueError("Code or source changed during replay")
    previous.inventory(out)
    print(json.dumps({period: {name: {key: score[key] for key in ("count", "accuracy", "brier")}
                              for name, score in result["metrics"].items()} for period, result in results.items()}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the fixed advanced-learner and recency comparison, then a separately invoked historical replay.")
    parser.add_argument("stage", choices=("develop", "replay"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cached-features", type=Path)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--previous-replay", type=Path)
    args = parser.parse_args()
    if args.stage == "develop":
        if args.cached_features is None:
            parser.error("--cached-features is required for development")
        develop(args.source.resolve(), args.cached_features.resolve(), args.output.resolve())
    else:
        if args.development is None or args.previous_replay is None:
            parser.error("--development and --previous-replay are required for replay")
        replay(args.source.resolve(), args.development.resolve(), args.previous_replay.resolve(), args.output.resolve())
