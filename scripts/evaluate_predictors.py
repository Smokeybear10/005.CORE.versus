import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

if __package__:
    from . import evaluate_challenger as challenger
    from . import evaluate_research as base
    from . import predictor_features
else:
    import evaluate_challenger as challenger
    import evaluate_research as base
    import predictor_features


CONFIG = {
    "schemaVersion": 1,
    "developmentYears": [2021, 2022, 2023],
    "trainingStart": "2010-01-01",
    "refitEndExclusive": "2024-01-01",
    "calibrationEndExclusive": "2025-01-01",
    "replayEndInclusive": "2026-10-31",
    "candidates": ["control48", "expanded_gb", "expanded_logistic", "expanded_hist", "expanded_forest", "expanded_blend"],
    "learners": {
        "gradient_boosting": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "random_state": 42},
        "logistic_regression": {"C": 0.5, "max_iter": 2000, "random_state": 42},
        "hist_gradient_boosting": {"max_iter": 250, "max_leaf_nodes": 15, "min_samples_leaf": 30,
                                     "l2_regularization": 5.0, "learning_rate": 0.05,
                                     "early_stopping": False, "random_state": 42},
        "random_forest": {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 10, "n_jobs": 2, "random_state": 42},
    },
    "selection": "Highest pooled winner accuracy on rolling 2021, 2022, 2023 development; ties use lower Brier then declared order. No subset selection.",
    "blend": "Equal probability average of expanded gradient boosting and logistic regression; fixed before scoring.",
    "calibration": "One positive temperature per selected/control model, fitted on 2024 log loss after final training through 2023. Temperature does not change winner picks.",
    "replay": "2025 through October 2026 was previously inspected. Historical replay only, not new independent validation. One selected candidate and control; no subsequent tuning.",
    "promotion": "No automatic runtime replacement. Report paired event-bootstrap uncertainty and all-bout coverage. A 70% claim requires independently evaluated evidence, not this previously exposed replay.",
    "exclusions": "Only missing fighter identities and nondecisive outcomes excluded from scoring; no low-confidence/debutant filtering. Existing source quarantine remains intact.",
    "bootstrap": {"replicates": 2000, "seed": 944},
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_hashes(source: Path) -> dict:
    return {name: base.digest(source / name) for name in ("manifest.json", "fights.parquet", "rounds.parquet", "fighters.parquet")}


def code_hashes() -> dict:
    return {Path(module.__file__).name: base.digest(Path(module.__file__))
            for module in (base, challenger, predictor_features)} | {Path(__file__).name: base.digest(Path(__file__))}


def load_features(source: Path, end_exclusive: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    manifest = json.loads((source / "manifest.json").read_text())
    for filename, expected in manifest["files"].items():
        if base.digest(source / filename) != expected:
            raise ValueError(f"Input checksum mismatch: {filename}")
    fights = pd.read_parquet(source / "fights.parquet")
    fights = fights[fights.date < pd.Timestamp(end_exclusive)].copy()
    rounds = pd.read_parquet(source / "rounds.parquet")
    rounds = rounds[rounds.fight_id.isin(fights.fight_id)]
    fighters = pd.read_parquet(source / "fighters.parquet")
    if fights.fight_id.duplicated().any() or fighters.fighter_id.duplicated().any():
        raise ValueError("Duplicate source identities")
    totals, _ = base.prepare_round_stats(rounds)
    original = challenger.add_context(base.reconstruct(fights, totals), fights, fighters)
    columns = [name for name in original if name.startswith(("diff_", "ctx_"))] + ["weightclass"]
    if len(columns) != 48:
        raise ValueError("The control feature set no longer matches the original 48 predictors")
    expanded = predictor_features.add_predictors(original, fights, rounds, fighters)
    pd.testing.assert_frame_equal(expanded[original.columns], original)
    expanded_columns = [name for name in expanded if name.startswith(("diff_", "ctx_"))] + ["weightclass"]
    if np.isinf(expanded[[name for name in expanded_columns if name != "weightclass"]].to_numpy(dtype=float)).any():
        raise ValueError("Nonfinite predictor values")
    for side in ("red", "blue"):
        through = expanded[f"{side}_history_through"]
        if not (through.isna() | (through < expanded.date)).all():
            raise ValueError("Non-prior fight history")
    return expanded, columns, expanded_columns


def fit(name: str, frame: pd.DataFrame, columns: list[str]) -> list:
    if name == "expanded_blend":
        return fit("expanded_gb", frame, columns) + fit("expanded_logistic", frame, columns)
    learner = {"control48": "gradient_boosting", "expanded_gb": "gradient_boosting",
               "expanded_logistic": "logistic_regression", "expanded_hist": "hist_gradient_boosting",
               "expanded_forest": "random_forest"}[name]
    pipeline = challenger.make_pipeline("gradient_boosting" if learner == "random_forest" else learner, columns)
    if learner == "random_forest":
        pipeline.set_params(model=RandomForestClassifier(**CONFIG["learners"][learner]))
    else:
        pipeline.named_steps["model"].set_params(**CONFIG["learners"][learner])
    X = pd.concat([frame[columns], base.swapped(frame[columns])], ignore_index=True)
    y = np.concatenate([frame.target.to_numpy(), 1 - frame.target.to_numpy()])
    return [pipeline.fit(X, y)]


def predict(models: list, frame: pd.DataFrame) -> np.ndarray:
    probabilities = np.mean([base.symmetric_probability(model, frame) for model in models], axis=0)
    reverse = np.mean([base.symmetric_probability(model, base.swapped(frame)) for model in models], axis=0)
    if not np.allclose(probabilities + reverse, 1, atol=1e-12, rtol=0):
        raise ValueError("Corner symmetry failed")
    return probabilities


def summary(frame: pd.DataFrame, column: str) -> dict:
    labels = frame.target.to_numpy()
    probabilities = frame[column].to_numpy()
    return {"events": int(frame.event_id.nunique()), "wins": int(((probabilities >= 0.5) == labels).sum()),
            **base.metrics(labels, probabilities)}


def paired(frame: pd.DataFrame, first: str, second: str) -> dict:
    frame = frame.reset_index(drop=True)
    y = frame.target.to_numpy()
    a, b = frame[first].to_numpy(), frame[second].to_numpy()
    da = ((b >= 0.5) == y).astype(float) - ((a >= 0.5) == y).astype(float)
    db = (a - y) ** 2 - (b - y) ** 2
    groups = [group.index.to_numpy() for _, group in frame.groupby("event_id")]
    rng = np.random.default_rng(CONFIG["bootstrap"]["seed"])
    results = []
    for _ in range(CONFIG["bootstrap"]["replicates"]):
        indices = np.concatenate([groups[index] for index in rng.integers(0, len(groups), len(groups))])
        results.append([float(da[indices].mean()), float(db[indices].mean())])
    intervals = np.quantile(results, [0.025, 0.975], axis=0)
    return {"accuracyImprovement": float(da.mean()), "accuracyImprovementEventBootstrap95": intervals[:, 0].tolist(),
            "brierImprovement": float(db.mean()), "brierImprovementEventBootstrap95": intervals[:, 1].tolist(),
            "interpretation": "Descriptive uncertainty; does not remove model selection or previously exposed outcome bias, or fully capture repeated-fighter and temporal dependence."}


def inventory(directory: Path) -> None:
    base.write_json(directory / "artifact_hashes.json", {p.name: base.digest(p) for p in sorted(directory.iterdir())
                                                        if p.is_file() and p.name != "artifact_hashes.json"})


def develop(source: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"createdAt": now(), "config": CONFIG, "sourceDirectory": str(source), "sourceHashes": source_hashes(source), "codeHashes": code_hashes(),
                "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
                "featurePolicy": predictor_features.CONFIG,
                "status": "Development on previously used historical periods; not independent validation"}
    base.write_json(out / "protocol.json", manifest)
    for filename in manifest["codeHashes"]:
        (out / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    data, original_columns, expanded_columns = load_features(source, CONFIG["calibrationEndExclusive"])
    prefix_checks = []
    for year in [*CONFIG["developmentYears"], max(CONFIG["developmentYears"]) + 1]:
        cutoff = f"{year}-01-01"
        prefix, prefix_original, prefix_expanded = load_features(source, cutoff)
        pd.testing.assert_frame_equal(data[data.date < pd.Timestamp(cutoff)].reset_index(drop=True), prefix)
        if prefix_original != original_columns or prefix_expanded != expanded_columns:
            raise ValueError("Feature columns changed with a future history suffix")
        prefix_checks.append({"cutoffExclusive": cutoff, "rows": len(prefix), "passed": True})
        print(json.dumps({"prefixCheck": cutoff, "passed": True}), flush=True)
    base.write_json(out / "prefix_checks.json", prefix_checks)
    data.to_parquet(out / "development_features.parquet", index=False)
    base.write_json(out / "feature_columns.json", {"original": original_columns, "expanded": expanded_columns})
    predictions, annual = [], {}
    for year in CONFIG["developmentYears"]:
        training = data[data.date < pd.Timestamp(f"{year}-01-01")]
        validation = data[data.date.dt.year == year]
        rows = validation[["fight_id", "event_id", "date", "target"]].copy()
        fitted = {}
        annual[str(year)] = {}
        for name in CONFIG["candidates"]:
            columns = original_columns if name == "control48" else expanded_columns
            models = fitted["expanded_gb"] + fitted["expanded_logistic"] if name == "expanded_blend" else fit(name, training, columns)
            fitted[name] = models
            rows[name] = predict(models, validation[columns])
            annual[str(year)][name] = summary(rows, name)
            print(json.dumps({"year": year, "candidate": name, "accuracy": annual[str(year)][name]["accuracy"],
                              "brier": annual[str(year)][name]["brier"]}), flush=True)
        predictions.append(rows)
    pooled = pd.concat(predictions, ignore_index=True)
    scores = {name: summary(pooled, name) for name in CONFIG["candidates"]}
    selected = min(CONFIG["candidates"], key=lambda name: (-scores[name]["accuracy"], scores[name]["brier"]))
    selection = {"createdAt": now(), "selected": selected, "pooled": scores, "annual": annual,
                 "pairedWithControl": paired(pooled, "control48", selected), "selectionPolicy": CONFIG["selection"]}
    base.write_json(out / "selection.json", selection)
    pooled.to_csv(out / "development_predictions.csv", index=False)
    training = data[data.date < pd.Timestamp(CONFIG["refitEndExclusive"])]
    calibration = data[(data.date >= pd.Timestamp(CONFIG["refitEndExclusive"])) & (data.date < pd.Timestamp(CONFIG["calibrationEndExclusive"]))]
    calibrated_rows = calibration[["fight_id", "event_id", "date", "target"]].copy()
    artifacts = {}
    for name in dict.fromkeys(["control48", selected]):
        columns = original_columns if name == "control48" else expanded_columns
        models = fit(name, training, columns)
        raw = predict(models, calibration[columns])
        result = minimize_scalar(lambda log_t: log_loss(calibration.target, base.apply_temperature(raw, math.exp(log_t))),
                                 bounds=(-3, 3), method="bounded")
        if not result.success:
            raise ValueError("Temperature fit did not converge")
        temperature = math.exp(result.x)
        calibrated_rows[name] = base.apply_temperature(raw, temperature)
        artifact = {"models": models, "columns": columns, "temperature": temperature, "name": name,
                    "trainedThrough": str(training.date.max().date()), "calibratedThrough": str(calibration.date.max().date()),
                    "sourceHashes": manifest["sourceHashes"], "codeHashes": manifest["codeHashes"], "validatedForBetting": False}
        joblib.dump(artifact, out / f"{name}.joblib", compress=3)
        artifacts[name] = {"sha256": base.digest(out / f"{name}.joblib"), "temperature": temperature,
                           "calibrationMetricsInSample": summary(calibrated_rows, name)}
    calibrated_rows.to_csv(out / "calibration_predictions.csv", index=False)
    report = {"createdAt": now(), "selected": selected, "featureCounts": {"original": len(original_columns), "expanded": len(expanded_columns)},
              "development": selection, "fitCount": len(training), "calibrationCount": len(calibration), "artifacts": artifacts,
              "replayScored": False, "prefixChecks": prefix_checks, "runtimeChanged": False, "supports70PercentClaim": False,
              "limitations": ["Development and calibration years were used in earlier research; results are exploratory.",
                              "Retrospective source snapshots lack original observation timestamps.",
                              "Calibration scores are in sample for temperature only.",
                              "Source tables are loaded in full then filtered before constructing features; replay-period rows never enter development fitting or scoring.",
                              "Height and reach are current identity snapshots, not verified historical measurements."]}
    base.write_json(out / "report.json", report)
    if code_hashes() != manifest["codeHashes"] or source_hashes(source) != manifest["sourceHashes"]:
        raise ValueError("Sources changed during development")
    inventory(out)
    print(json.dumps({"selected": selected, "developmentAccuracy": scores[selected]["accuracy"], "features": report["featureCounts"]}), flush=True)


def replay(source: Path, frozen: Path, out: Path) -> None:
    expected = json.loads((frozen / "artifact_hashes.json").read_text())
    for filename, checksum in expected.items():
        if base.digest(frozen / filename) != checksum:
            raise ValueError(f"Frozen artifact changed: {filename}")
    protocol = json.loads((frozen / "protocol.json").read_text())
    if protocol["codeHashes"] != code_hashes() or protocol["sourceHashes"] != source_hashes(source):
        raise ValueError("Development code or inputs changed before replay")
    selection = json.loads((frozen / "selection.json").read_text())
    selected = selection["selected"]
    out.mkdir(parents=True, exist_ok=False)
    base.write_json(out / "protocol.json", {"createdAt": now(), "developmentManifestSha256": base.digest(frozen / "artifact_hashes.json"),
                                           "selected": selected, "status": CONFIG["replay"], "config": CONFIG})
    end = str((pd.Timestamp(CONFIG["replayEndInclusive"]) + pd.Timedelta(days=1)).date())
    data, _, _ = load_features(source, end)
    prefix = data[data.date < pd.Timestamp(CONFIG["calibrationEndExclusive"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix, pd.read_parquet(frozen / "development_features.parquet"))
    replay_frame = data[data.date >= pd.Timestamp(CONFIG["calibrationEndExclusive"])].copy()
    rows = replay_frame[["fight_id", "event_id", "date", "red_name", "blue_name", "target"]].copy()
    for name in dict.fromkeys(["control48", selected]):
        artifact = joblib.load(frozen / f"{name}.joblib")
        raw = predict(artifact["models"], replay_frame[artifact["columns"]])
        rows[name] = base.apply_temperature(raw, artifact["temperature"])
    original_path = source.parent / "challenger-v1" / "provisional_challenger.joblib"
    original_contract = json.loads((source.parent / "candidate_contract.json").read_text())
    if "sha256:" + base.digest(original_path) != original_contract["modelVersion"]:
        raise ValueError("Original model checksum mismatch")
    original = joblib.load(original_path)
    rows["original_frozen"] = base.apply_temperature(predict([original["model"]], replay_frame[original["feature_columns"]]), original["temperature"])
    periods = {"all_2025_2026": rows, "2025": rows[rows.date.dt.year == 2025], "2026": rows[rows.date.dt.year == 2026],
               "original_285_bout_audit": rows[rows.date > pd.Timestamp("2026-05-09")]}
    results = {}
    for period, frame in periods.items():
        results[period] = {"metrics": {name: summary(frame, name) for name in dict.fromkeys(["original_frozen", "control48", selected])},
                           "selectedVsOriginal": paired(frame, "original_frozen", selected),
                           "selectedVsUpdatedControl": paired(frame, "control48", selected)}
    old_audit = pd.read_csv(source.parent / "fresh-audit-v1" / "predictions.csv")
    check = rows.merge(old_audit[["fight_id", "target", "p_red_challenger_temperature"]], on="fight_id", suffixes=("", "_old"), validate="one_to_one")
    if len(check) != len(old_audit) or not (check.target == check.target_old).all() or not np.allclose(check.original_frozen, check.p_red_challenger_temperature, atol=1e-12, rtol=0):
        raise ValueError("Frozen original audit reproduction failed")
    rows.to_csv(out / "predictions.csv", index=False)
    report = {"createdAt": now(), "selected": selected, "results": results, "prefixInvariant": True,
              "originalAuditReproduced": True, "runtimeChanged": False, "supports70PercentClaim": False,
              "status": "Previously exposed historical replay; development-selected candidate, no test-driven retuning",
              "limitations": ["Temporal row separation is not new independent validation after these historical outcomes were inspected.",
                              "All decisive bouts with known identities scored, including cases the public forecast would abstain on.",
                              "Accuracy is not profit. No market prices are predictors in this experiment.",
                              "Event-cluster intervals are descriptive and do not account for model selection."]}
    base.write_json(out / "report.json", report)
    if protocol["codeHashes"] != code_hashes() or protocol["sourceHashes"] != source_hashes(source):
        raise ValueError("Sources changed during replay")
    inventory(out)
    print(json.dumps({period: {name: {key: metrics[key] for key in ("count", "accuracy", "brier")}
                              for name, metrics in item["metrics"].items()} for period, item in results.items()}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare fixed prior-fight predictors with rolling development and a separately invoked historical replay.")
    parser.add_argument("stage", choices=("develop", "replay"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", type=Path)
    args = parser.parse_args()
    if args.stage == "replay" and args.development is None:
        parser.error("--development is required for replay")
    if args.stage == "develop":
        develop(args.source.resolve(), args.output.resolve())
    else:
        replay(args.source.resolve(), args.development.resolve(), args.output.resolve())
