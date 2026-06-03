import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.base import clone
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_challenger as challenger
from scripts import evaluate_predictors as evaluation
from scripts import evaluate_research as base
from scripts import predictor_features


REFERENCE = ROOT / "data/evaluation/inference"
ARCHIVE = ROOT / "data/evaluation/expanded-predictors-replay-v1/predictions.csv"
RECENT = ROOT / "data/evaluation/recent-cards-20261104/fight-by-fight.csv"
MODELS = {
    "original48": (ROOT / "data/evaluation/challenger-v1/provisional_challenger.joblib",
                   "915d2a35f6a3aacc320dc11da0f302e6260c43f070ed12c43f82cd4476ac0321"),
    "expanded92": (ROOT / "data/evaluation/expanded-predictors-v1/expanded_blend.joblib",
                   "8652fb31f9a8c1b33ba0e6f7cfc90d2c423a220d0d4f2ea8c72480b2a57cbbf4"),
}
CODE = {
    "evaluate_research.py": "b657ad7ac141166d629b1af55415f94d9bb31ac4e5dbc2776d634c2d73743859",
    "evaluate_challenger.py": "012f3d9b51c5f92c1840a8ef940859fea569e1487ca0a7560e32d5339a277825",
    "predictor_features.py": "4d241a11ad78ffa49c155d9850bec573f25ee357b2e16d69193153bc53429ca0",
    "evaluate_predictors.py": "e4570bf8610933c7f55f558b7e6171a8b1d1fd5181ea2d529358dda3405a4935",
}
COHORT_HASHES = {
    "same985": "529f6be09618f048cb1fa67eb77aa9dace05637a1eace75e919cc6876523b5cd",
    "recent49": "439d81806d82c7b81ece17e04be6a14797dbe58da8b46070c1414c727daf2f55",
}
RECENT_DATES = ["2026-10-10", "2026-10-17", "2026-10-24", "2026-10-31"]
BOOTSTRAP = {"seed": 944, "replicates": 2000}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def id_hash(values: pd.Series) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def fingerprint(source: Path) -> dict:
    paths = [Path(__file__), ARCHIVE, RECENT, *[path for path, _ in MODELS.values()],
             *[ROOT / "scripts" / name for name in CODE]]
    for directory in (source, REFERENCE):
        manifest = directory / "manifest.json"
        paths.extend([manifest, *[directory / name for name in json.loads(manifest.read_text())["files"]]])
    return {str(path): base.digest(path) for path in paths}


def probability(models: list, temperature: float, frame: pd.DataFrame) -> np.ndarray:
    values = base.apply_temperature(evaluation.predict(models, frame), temperature)
    reverse = base.apply_temperature(evaluation.predict(models, base.swapped(frame)), temperature)
    if not np.isfinite(values).all() or not np.allclose(values + reverse, 1, atol=1e-12, rtol=0):
        raise ValueError("Calibrated probability or corner symmetry check failed")
    return values


def metrics(frame: pd.DataFrame, column: str) -> dict:
    y, p = frame.target.to_numpy(), frame[column].to_numpy()
    correct = (p >= 0.5) == y
    return {"count": len(frame), "events": int(frame.event_id.nunique()), "correct": int(correct.sum()),
            "accuracy": float(correct.mean()), "accuracyWilson95": base.wilson(int(correct.sum()), len(frame)),
            "brier": float(np.mean((p - y) ** 2)), "logLoss": float(log_loss(y, p, labels=[0, 1]))}


def paired(frame: pd.DataFrame, before: str, after: str) -> dict:
    frame = frame.reset_index(drop=True)
    y = frame.target.to_numpy()
    first, second = frame[before].to_numpy(), frame[after].to_numpy()
    losses = []
    for p in (first, second):
        p = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
        losses.append(-y * np.log(p) - (1 - y) * np.log1p(-p))
    differences = np.column_stack([((second >= 0.5) == y).astype(float) - ((first >= 0.5) == y),
                                   (first - y) ** 2 - (second - y) ** 2, losses[0] - losses[1]])
    groups = [group.index.to_numpy() for _, group in frame.groupby("event_id")]
    intervals = None
    if len(groups) >= 2:
        rng = np.random.default_rng(BOOTSTRAP["seed"])
        samples = [differences[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])].mean(axis=0)
                   for _ in range(BOOTSTRAP["replicates"])]
        intervals = np.quantile(samples, [0.025, 0.975], axis=0)
    names = ["accuracyImprovement", "brierImprovement", "logLossImprovement"]
    return {"before": before, "after": after, "events": len(groups),
            "direction": "after minus before for accuracy; before minus after for losses; positive favors after",
            **{name: float(value) for name, value in zip(names, differences.mean(axis=0))},
            "eventBootstrap95": {name: intervals[:, i].tolist() if intervals is not None else None
                                 for i, name in enumerate(names)},
            "limitation": "Descriptive exposed-outcome comparison; few-event intervals are unstable. No interval for one event."}


def strict_recent_check(source: Path, full: pd.DataFrame, columns48: list[str], columns92: list[str]) -> list[dict]:
    fights = pd.read_parquet(source / "fights.parquet")
    rounds = pd.read_parquet(source / "rounds.parquet")
    fighters = pd.read_parquet(source / "fighters.parquet")
    checks = []
    for date in RECENT_DATES:
        card = full[full.date == pd.Timestamp(date)].copy()
        history = fights[fights.date < pd.Timestamp(date)].copy()
        prior_rounds = rounds[rounds.fight_id.isin(history.fight_id)].copy()
        totals, _ = base.prepare_round_stats(prior_rounds)
        requests = []
        contexts = fights.set_index("fight_id").loc[card.fight_id]
        for row in contexts.itertuples():
            frame, quality = challenger.build_inference_features(
                history, prior_rounds, fighters, row.red_fighter_id, row.blue_fighter_id, date,
                row.weightclass, int(row.num_rounds), bool(row.is_title_bout), prepared_totals=totals)
            if any(item["historyThrough"] is not None and item["historyThrough"] >= date for item in quality.values()):
                raise ValueError("Same-date history entered strict-prefix inference")
            requests.append(frame.assign(fight_id=row.Index, date=pd.Timestamp(date),
                                         red_fighter_id=row.red_fighter_id, blue_fighter_id=row.blue_fighter_id))
        strict = predictor_features.add_predictors(pd.concat(requests, ignore_index=True), history, prior_rounds, fighters)
        expected = card.set_index("fight_id").loc[strict.fight_id]
        for columns in (columns48, columns92):
            pd.testing.assert_frame_equal(strict[columns].reset_index(drop=True), expected[columns].reset_index(drop=True),
                                          check_exact=True, check_dtype=False)
        checks.append({"eventDate": date, "scoredRows": len(card), "historyRows": len(history),
                       "strictHistoryThrough": str(history.date.max().date()), "currentCardInHistory": False,
                       "original48AndExpanded92MatchFullReconstruction": True})
        print(json.dumps({"stage": "strict_recent_prefix", **checks[-1]}), flush=True)
    return checks


def run(source: Path, output: Path) -> None:
    if source == REFERENCE.resolve():
        raise ValueError("Supply a new corrected bundle; the immutable reference is not a repair input")
    for name, expected in CODE.items():
        if base.digest(ROOT / "scripts" / name) != expected:
            raise ValueError(f"Frozen feature/training implementation changed: {name}")
    frozen = {}
    for name, (path, expected) in MODELS.items():
        if base.digest(path) != expected:
            raise ValueError(f"Frozen reference model changed: {name}")
        frozen[name] = joblib.load(path)
    if frozen["original48"]["selected_model"] != "gradient_boosting" or frozen["expanded92"]["name"] != "expanded_blend":
        raise ValueError("Fixed learner selection no longer matches the archived models")
    columns48, columns92 = frozen["original48"]["feature_columns"], frozen["expanded92"]["columns"]
    if len(columns48) != 48 or len(columns92) != 92 or len(frozen["expanded92"]["models"]) != 2:
        raise ValueError("Unexpected frozen feature or model count")
    originals = {"original48": [frozen["original48"]["model"]], "expanded92": frozen["expanded92"]["models"]}
    before_hashes = fingerprint(source)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"schemaVersion": 1, "createdAt": now(), "source": str(source), "referenceSource": str(REFERENCE),
                "status": "fixed identity-repair refit and historical replay; all outcomes previously exposed",
                "selection": "None. Clone fixed original48 GB and expanded92 equal-probability GB/LR pipelines; no tuning.",
                "splits": {"original48": {"fitFromInclusive": "2010-01-01", "fitEndExclusive": "2023-01-01",
                                          "calibrationFromInclusive": "2023-01-01", "calibrationEndExclusive": "2025-01-01"},
                           "expanded92": {"fitFromInclusive": "2010-01-01", "fitEndExclusive": "2024-01-01",
                                          "calibrationFromInclusive": "2024-01-01", "calibrationEndExclusive": "2025-01-01"}},
                "calibration": "Refit one positive temperature per fixed model on its original calibration years; bounded log(T) [-3,3], log loss only.",
                "history": "Retain all pre-2010 history; create decisive known-identity feature rows from2010; strictly prior-date, same-date batches.",
                "cohorts": {"same985": 985, "primary285": 285, "recent49": 49, "lastTwo26": 26, "latest12": 12},
                "cohortIdHashes": COHORT_HASHES, "recentDates": RECENT_DATES,
                "newCoverage": "Newly eligible corrected-source rows are listed separately and never added to fixed-cohort scores.",
                "comparison": "Old frozen model on old source versus corrected fixed refit on corrected source; event/date/target must agree by unique fight_id.",
                "bootstrap": BOOTSTRAP, "tieRule": "Red for probability >=0.5, for every model.",
                "markets": "No market quotes, picks, blends or old market archives are recomputed or changed.",
                "hyperparameters": {name: [model.named_steps["model"].get_params() for model in models]
                                    for name, models in originals.items()},
                "hashes": before_hashes,
                "versions": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                             for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "fastparquet")}},
                "runtimeChanged": False, "supports70PercentClaim": False, "highConfidenceValidated": False}
    write(output / "protocol.json", protocol)
    for path in [Path(__file__), *[ROOT / "scripts" / name for name in CODE]]:
        (output / path.name).write_bytes(path.read_bytes())
    print(json.dumps({"stage": "protocol_frozen", "output": str(output)}), flush=True)
    development, original_columns, expanded_columns = evaluation.load_features(source, "2025-01-01")
    if original_columns != columns48 or expanded_columns != columns92:
        raise ValueError("Corrected feature columns differ from frozen model definitions")
    prefix_checks = []
    for cutoff in ("2023-01-01", "2024-01-01"):
        prefix, first, second = evaluation.load_features(source, cutoff)
        if first != columns48 or second != columns92:
            raise ValueError("Prefix feature definitions changed")
        pd.testing.assert_frame_equal(development[development.date < pd.Timestamp(cutoff)].reset_index(drop=True), prefix,
                                      check_exact=True)
        prefix_checks.append({"endExclusive": cutoff, "rows": len(prefix), "passed": True})
        print(json.dumps({"stage": "training_prefix", **prefix_checks[-1]}), flush=True)
    original_frame = development.drop(columns=predictor_features.ADDED_FEATURES)
    original_frame.to_parquet(output / "development_and_calibration_features.parquet", index=False)
    development.to_parquet(output / "development_features.parquet", index=False)
    write(output / "feature_columns.json", {"original": columns48, "expanded": columns92})
    fitted, fit_receipts = {}, {}
    for name, columns in (("original48", columns48), ("expanded92", columns92)):
        split = protocol["splits"][name]
        training = development[(development.date >= pd.Timestamp(split["fitFromInclusive"])) &
                               (development.date < pd.Timestamp(split["fitEndExclusive"]))]
        calibration = development[(development.date >= pd.Timestamp(split["calibrationFromInclusive"])) &
                                  (development.date < pd.Timestamp(split["calibrationEndExclusive"]))]
        if training.empty or calibration.empty or training.date.max() >= calibration.date.min() or set(training.event_id) & set(calibration.event_id):
            raise ValueError("Training and calibration periods are invalid or overlap")
        X = pd.concat([training[columns], base.swapped(training[columns])], ignore_index=True)
        y = np.concatenate([training.target.to_numpy(), 1 - training.target.to_numpy()])
        models = [clone(model).fit(X, y) for model in originals[name]]
        raw = evaluation.predict(models, calibration[columns])
        optimized = minimize_scalar(lambda log_t: log_loss(calibration.target, base.apply_temperature(raw, math.exp(log_t)), labels=[0, 1]),
                                    bounds=(-3, 3), method="bounded")
        if not optimized.success:
            raise ValueError("Temperature calibration did not converge")
        temperature = math.exp(optimized.x)
        calibrated = probability(models, temperature, calibration[columns])
        if not np.array_equal(raw >= 0.5, calibrated >= 0.5):
            raise ValueError("Positive temperature unexpectedly changed picks")
        frame = calibration[["fight_id", "event_id", "date", "target"]].assign(raw=raw, calibrated=calibrated)
        frame.to_csv(output / f"{name}_calibration_predictions.csv", index=False)
        fit_receipts[name] = {"fitCount": len(training), "calibrationCount": len(calibration),
                              "fitFrom": str(training.date.min().date()), "fitThrough": str(training.date.max().date()),
                              "calibrationFrom": str(calibration.date.min().date()), "calibrationThrough": str(calibration.date.max().date()),
                              "fitFightIdsSHA256": id_hash(training.fight_id), "calibrationFightIdsSHA256": id_hash(calibration.fight_id),
                              "temperature": temperature, "calibrationMetricsInSample": metrics(frame, "calibrated")}
        if name == "original48":
            artifact = {"model": models[0], "selected_model": "gradient_boosting", "temperature": temperature,
                        "feature_columns": columns, "config": frozen[name]["config"], "manifest": protocol,
                        "validated_for_betting": False}
            filename = "provisional_challenger.joblib"
        else:
            artifact = {"models": models, "columns": columns, "temperature": temperature, "name": "expanded_blend",
                        "trainedThrough": fit_receipts[name]["fitThrough"], "calibratedThrough": fit_receipts[name]["calibrationThrough"],
                        "sourceHashes": evaluation.source_hashes(source), "codeHashes": evaluation.code_hashes(), "validatedForBetting": False}
            filename = "expanded_blend.joblib"
        joblib.dump(artifact, output / filename, compress=3)
        fit_receipts[name]["artifact"] = filename
        fit_receipts[name]["artifactSHA256"] = base.digest(output / filename)
        fitted[name] = (models, temperature, columns)
        print(json.dumps({"stage": "model_frozen", "name": name, "fitCount": len(training), "calibrationCount": len(calibration)}), flush=True)
    write(output / "fit_receipts.json", fit_receipts)
    write(output / "pre-replay-manifest.json", {"frozenAt": now(), "files": {p.name: base.digest(p) for p in output.iterdir() if p.is_file()}})
    archived = pd.read_csv(ARCHIVE, parse_dates=["date"])
    recent = pd.read_csv(RECENT)
    recent_ids = recent.loc[recent.scored == True, "fight_id"]
    if archived.fight_id.duplicated().any() or len(archived) != 985 or id_hash(archived.fight_id) != COHORT_HASHES["same985"]:
        raise ValueError("Original replay cohort changed")
    if len(recent_ids) != 49 or id_hash(recent_ids) != COHORT_HASHES["recent49"]:
        raise ValueError("Original recent49 cohort changed")
    full, first, second = evaluation.load_features(source, "2026-11-01")
    old_full, old_first, old_second = evaluation.load_features(REFERENCE, "2026-11-01")
    if first != old_first or second != old_second or first != columns48 or second != columns92:
        raise ValueError("Old and corrected feature definitions differ")
    pd.testing.assert_frame_equal(full[full.date < pd.Timestamp("2025-01-01")].reset_index(drop=True), development, check_exact=True)
    records = archived[["fight_id", "event_id", "date", "target"]].copy()
    replay_ids = set(records.fight_id)
    for label, frame in (("old", old_full), ("corrected", full)):
        if frame.fight_id.duplicated().any() or not replay_ids <= set(frame.fight_id):
            raise ValueError(f"The {label} feature set cannot cover the unchanged replay cohort")
        aligned = frame.set_index("fight_id").loc[records.fight_id].reset_index()
        pd.testing.assert_frame_equal(records[["fight_id", "event_id", "date", "target"]],
                                      aligned[["fight_id", "event_id", "date", "target"]], check_dtype=False)
        for field in ("red_fighter_id", "blue_fighter_id", "red_name", "blue_name", "red_history_through", "blue_history_through"):
            records[f"{label}_{field}"] = aligned[field].to_numpy()
        for name, columns in (("original48", columns48), ("expanded92", columns92)):
            models, temperature = (originals[name], frozen[name]["temperature"]) if label == "old" else fitted[name][:2]
            values = probability(models, temperature, aligned[columns])
            if label == "old":
                expected = archived["original_frozen" if name == "original48" else "expanded_blend"].to_numpy()
                if not np.allclose(values, expected, atol=1e-12, rtol=0):
                    raise ValueError(f"Archived before-prediction reproduction failed: {name}")
                values = expected
            records[f"{name}_{'before' if label == 'old' else 'after'}"] = values
    extra = full[(full.date >= pd.Timestamp("2025-01-01")) & ~full.fight_id.isin(replay_ids)]
    extra[["fight_id", "event_id", "date", "red_fighter_id", "blue_fighter_id", "red_name", "blue_name"]].to_csv(output / "newly_eligible_unscored.csv", index=False)
    same_features = full.set_index("fight_id").loc[records.fight_id].reset_index()
    strict_checks = strict_recent_check(source, same_features[same_features.fight_id.isin(recent_ids)], columns48, columns92)
    records.to_csv(output / "predictions.csv", index=False)
    changed = records[(records.old_red_fighter_id != records.corrected_red_fighter_id) |
                      (records.old_blue_fighter_id != records.corrected_blue_fighter_id)]
    changed.to_csv(output / "replay_identity_changes.csv", index=False)
    cohorts = {"same985": records, "primary285": records[records.date > pd.Timestamp("2026-05-09")],
               "recent49": records[records.fight_id.isin(recent_ids)],
               "lastTwo26": records[records.date.isin(pd.to_datetime(RECENT_DATES[-2:]))],
               "latest12": records[records.date == pd.Timestamp(RECENT_DATES[-1])]}
    columns = [f"{name}_{stage}" for name in MODELS for stage in ("before", "after")]
    comparisons = [("original48_before", "original48_after"), ("expanded92_before", "expanded92_after"),
                   ("original48_before", "expanded92_before"), ("original48_after", "expanded92_after")]
    results = {}
    for label, frame in cohorts.items():
        if len(frame) != protocol["cohorts"][label]:
            raise ValueError(f"Fixed cohort size changed: {label}")
        results[label] = {"fightIdsSHA256": id_hash(frame.fight_id), "metrics": {column: metrics(frame, column) for column in columns},
                          "paired": [paired(frame, first, second) for first, second in comparisons]}
    if fingerprint(source) != before_hashes:
        raise ValueError("Source, model, code or archived comparator changed during the run")
    report = {"schemaVersion": 1, "createdAt": now(), "status": protocol["status"], "protocolSHA256": base.digest(output / "protocol.json"),
              "fits": fit_receipts, "results": results, "changedReplayIdentities": len(changed),
              "newlyEligibleReplayRowsNotScored": len(extra), "trainingPrefixChecks": prefix_checks,
              "strictRecentPrefixChecks": strict_checks, "archiveBeforeParityPassed": True, "sourceHashesUnchanged": True,
              "runtimeChanged": False, "marketComparisonsChanged": False, "supports70PercentClaim": False, "highConfidenceValidated": False,
              "runtimeCompatibility": "Artifact schemas retained. Root must separately version the contract, model/training-feature hashes and inference manifest; existing hard-pinned runtime cannot load these as a silent replacement.",
              "limitations": ["All replay outcomes were previously inspected; identity repair does not create a new holdout.",
                              "Before/after combines corrected source and a fixed refit; no model selection or hyperparameter tuning.",
                              "Calibration metrics are in sample for temperature only and kept separate from replay.",
                              "Historical DOB/height/reach snapshots lack original publication and revision timestamps.",
                              "Every original decisive known-identity replay bout is scored without runtime confidence/domain gates.",
                              "Small recent-card samples and event bootstrap intervals do not establish generalization or profit."]}
    write(output / "report.json", report)
    write(output / "artifact_hashes.json", {p.name: base.digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    print(json.dumps({"stage": "complete", "output": str(output), "results": {
        label: {name: {key: item[key] for key in ("count", "accuracy", "brier", "logLoss")}
                for name, item in result["metrics"].items()} for label, result in results.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refit the two fixed models after source identity repair, then replay unchanged exposed cohorts.")
    parser.add_argument("--source", type=Path, required=True, help="New corrected inference bundle; immutable old source stays untouched.")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are rejected.")
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve())
