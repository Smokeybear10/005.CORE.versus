import argparse
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
from scripts import career_features as career
from scripts import evaluate_career_supplement as previous
from scripts import evaluate_identity_repair as audit
from scripts import evaluate_research as base


PRIOR = ROOT / "data/evaluation/career-supplement-evaluation-v1"
FEATURES = PRIOR / "development_and_calibration_features.parquet"
PINS = {
    FEATURES: "155f3feb13db72e5e43552f5b6093606e60141b688467229151697cce344fafb",
    Path(career.__file__): "d601a14dc17572945ad4dafa7e05828442dda50b0feb0d2d7293135246aed562",
    previous.BASELINE / "provisional_challenger.joblib": previous.MODEL_SHA,
    PRIOR / "career_candidate.joblib": "ec8ff99f23638319c203135b0f4bb2d23f0d6a074828a7706714ef1a0db3e528",
}
FOLDS = {2021: (4113, 442, 496), 2022: (4555, 496, 506), 2023: (5051, 506, 504), 2024: (5557, 504, 512)}
CONFIG = {
    "schemaVersion": 1, "testYears": list(FOLDS), "trainingStart": "2010-01-01",
    "chronology": "For testyearY, fit2010 throughY-2, calibrate onlyY-1, scoreY; calendar boundaries exclusive",
    "models": ["corrected48", "career60"], "learning": "Clone each frozen pipeline from scratch per fold; same GB100/depth3/.05/seed42",
    "calibration": "One positive temperature per fold/model fitted only onY-1; log(T) bounds[-3,3]",
    "featureDefinitions": "Exactly the prior frozen48/60 definitions; no new predictors, row filters or hyperparameters",
    "preprocessing": "Imputer/scaler/division encoder refit only on corner-augmented training rows for each model/fold",
    "history": "Each test bout can use earlier completed bouts, including earlier bouts within the testyear; never same-day history",
    "selection": "None. All4years and both models reported; no model promotion or tuning from these results",
    "fitOrder": "All8pipelines and temperatures frozen before any testyear prediction or metric computation",
    "bootstrap": audit.BOOTSTRAP, "tieRule": "Red whenp>=0.5",
    "status": "Historical rolling-origin stresschecks, not untouched future validation; these years and later outcomes informed earlier model development",
    "pooling": "Pool only the4disjoint yearly test cohorts here; do not merge with985replay into an independent-validation claim",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def split(frame: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit_end, test_start, test_end = (pd.Timestamp(f"{value}-01-01") for value in (year - 1, year, year + 1))
    train = frame[(frame.date >= pd.Timestamp(CONFIG["trainingStart"])) & (frame.date < fit_end)]
    calibration = frame[(frame.date >= fit_end) & (frame.date < test_start)]
    test = frame[(frame.date >= test_start) & (frame.date < test_end)]
    if tuple(map(len, (train, calibration, test))) != FOLDS[year]:
        raise ValueError(f"Frozen fold membership counts changed: {year}")
    if not train.date.max() < calibration.date.min() <= calibration.date.max() < test.date.min():
        raise ValueError("Fit, calibration and test dates overlap")
    event_sets = [set(part.event_id) for part in (train, calibration, test)]
    if event_sets[0] & event_sets[1] or event_sets[0] & event_sets[2] or event_sets[1] & event_sets[2]:
        raise ValueError("An event crosses a fold boundary")
    if any(set(part.target) != {0, 1} for part in (train, calibration, test)):
        raise ValueError("A fold lacks one outcome class")
    return train, calibration, test


def fingerprint(source: Path, supplement: Path) -> dict:
    paths = [Path(__file__), Path(previous.__file__), Path(audit.__file__), Path(base.__file__),
             ROOT / "scripts/evaluate_challenger.py", *PINS,
             PRIOR / "protocol.json", PRIOR / "report.json", PRIOR / "feature_columns.json",
             previous.BASELINE / "predictions.csv"]
    for directory in (source, supplement):
        manifest = directory / "manifest.json"
        paths.extend([manifest, *[directory / name for name in json.loads(manifest.read_text())["files"]]])
    return {str(path): base.digest(path) for path in paths}


def run(source: Path, supplement_path: Path, output: Path) -> None:
    for path, checksum in PINS.items():
        if base.digest(path) != checksum:
            raise ValueError(f"Frozen model or feature definition changed: {path}")
    baseline = joblib.load(previous.BASELINE / "provisional_challenger.joblib")
    candidate = joblib.load(PRIOR / "career_candidate.joblib")
    if baseline["model"].named_steps["model"].get_params() != candidate["model"].named_steps["model"].get_params():
        raise ValueError("The two fixed learners differ")
    old_source = Path(baseline["manifest"]["source"])
    for filename in ("fights.parquet", "rounds.parquet", "fighters.parquet"):
        if base.digest(source / filename) != baseline["manifest"]["hashes"][str(old_source / filename)]:
            raise ValueError("Corrected48 source changed")
    if base.digest(supplement_path / "manifest.json") != candidate["supplement_manifest_sha256"]:
        raise ValueError("Frozen career supplement changed")
    before_hashes = fingerprint(source, supplement_path)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"createdAt": now(), "config": CONFIG, "careerFeaturePolicy": career.CONFIG,
                "source": str(source), "supplement": str(supplement_path), "frozenFeatureArchive": str(FEATURES),
                "hyperparameters": baseline["model"].named_steps["model"].get_params(), "hashes": before_hashes,
                "priorResultKnown": "Career60 failed the985cohort promotion gate before this additional robustness check was requested",
                "expectedFoldCounts": {str(year): dict(zip(("train", "calibration", "test"), counts)) for year, counts in FOLDS.items()},
                "versions": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                             for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "fastparquet")}},
                "runtimeChanged": False, "promotionAllowed": False}
    audit.write(output / "protocol.json", protocol)
    for path in (Path(__file__), Path(career.__file__), Path(previous.__file__), Path(audit.__file__),
                 Path(base.__file__), ROOT / "scripts/evaluate_challenger.py"):
        (output / path.name).write_bytes(path.read_bytes())
    print(json.dumps({"stage": "protocol_frozen", "output": str(output)}), flush=True)
    features = pd.read_parquet(FEATURES)
    if features.fight_id.duplicated().any() or len(features) != 6573:
        raise ValueError("Frozen feature archive membership changed")
    columns = {"corrected48": baseline["feature_columns"], "career60": candidate["feature_columns"]}
    expected = [name for name in columns["corrected48"] if name != "weightclass"] + career.ADDED_FEATURES + ["weightclass"]
    if len(columns["corrected48"]) != 48 or columns["career60"] != expected or len(expected) != 60:
        raise ValueError("Frozen feature columns differ")
    folds = {year: split(features, year) for year in FOLDS}
    cohort_manifest = {}
    for year, parts in folds.items():
        cohort_manifest[str(year)] = {name: {"count": len(part), "events": int(part.event_id.nunique()),
                                             "from": str(part.date.min().date()), "through": str(part.date.max().date()),
                                             "fightIdsSHA256": audit.id_hash(part.fight_id)}
                                    for name, part in zip(("train", "calibration", "test"), parts)}
    audit.write(output / "cohorts.json", cohort_manifest)
    fights, rounds, fighters = previous.source_frames(source)
    supplement, _ = career.load_supplement(supplement_path)
    prefix_checks = []
    for year in range(2020, 2026):
        cutoff = f"{year}-01-01"
        prior = fights[fights.date < pd.Timestamp(cutoff)].copy()
        prior_rounds = rounds[rounds.fight_id.isin(prior.fight_id)].copy()
        prior_supplement = supplement[supplement.date < pd.Timestamp(cutoff)].copy()
        reconstructed, first, second = previous.features(prior, prior_rounds, fighters, prior_supplement, cutoff)
        cached = features[features.date < pd.Timestamp(cutoff)].reset_index(drop=True)
        pd.testing.assert_frame_equal(reconstructed, cached, check_exact=True)
        if first != columns["corrected48"] or second != columns["career60"]:
            raise ValueError("Physical prefix feature columns changed")
        prefix_checks.append({"cutoffExclusive": cutoff, "rows": len(cached), "sourceFightRows": len(prior),
                              "supplementRows": len(prior_supplement), "passed": True})
        print(json.dumps({"stage": "physical_prefix_passed", **prefix_checks[-1]}), flush=True)
    audit.write(output / "prefix_checks.json", prefix_checks)
    pipelines = {"corrected48": baseline["model"], "career60": candidate["model"]}
    receipts, calibration_rows = {}, []
    for year, (training, calibration, _) in folds.items():
        receipts[str(year)] = {}
        cal = calibration[["fight_id", "event_id", "date", "target"]].copy().assign(test_year=year)
        for name in CONFIG["models"]:
            selected_columns = columns[name]
            X = pd.concat([training[selected_columns], base.swapped(training[selected_columns])], ignore_index=True)
            y = np.concatenate([training.target.to_numpy(), 1 - training.target.to_numpy()])
            model = clone(pipelines[name]).fit(X, y)
            raw = base.symmetric_probability(model, calibration[selected_columns])
            fitted = minimize_scalar(lambda log_t: log_loss(calibration.target, base.apply_temperature(raw, math.exp(log_t)), labels=[0, 1]),
                                    bounds=(-3, 3), method="bounded")
            if not fitted.success:
                raise ValueError("Walk-forward calibration did not converge")
            temperature = math.exp(fitted.x)
            cal[name] = audit.probability([model], temperature, calibration[selected_columns])
            if not np.array_equal(raw >= 0.5, cal[name].to_numpy() >= 0.5):
                raise ValueError("Positive temperature changed picks")
            artifact = {"model": model, "feature_columns": selected_columns, "temperature": temperature,
                        "test_year": year, "name": name, "splits": cohort_manifest[str(year)], "config": CONFIG,
                        "protocolSHA256": base.digest(output / "protocol.json"), "validated_for_betting": False}
            filename = f"{year}_{name}.joblib"
            joblib.dump(artifact, output / filename, compress=3)
            receipts[str(year)][name] = {"artifact": filename, "sha256": base.digest(output / filename),
                                        "temperature": temperature, "calibrationMetricsInSample": audit.metrics(cal, name)}
            print(json.dumps({"stage": "fold_model_frozen", "year": year, "model": name, "fitRows": len(training),
                              "calibrationRows": len(calibration)}), flush=True)
        calibration_rows.append(cal)
    pd.concat(calibration_rows, ignore_index=True).to_csv(output / "calibration_predictions.csv", index=False)
    audit.write(output / "fit_receipts.json", receipts)
    seal = {p.name: base.digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    audit.write(output / "pre-test-manifest.json", {"frozenAt": now(), "modelsFrozen": 8, "testPredictionsComputed": False, "files": seal})
    results, predictions = {}, []
    for year, (_, _, test) in folds.items():
        rows = test[["fight_id", "event_id", "date", "red_fighter_id", "blue_fighter_id", "red_name", "blue_name", "target",
                     "red_history_through", "blue_history_through"]].copy().assign(test_year=year)
        for name in CONFIG["models"]:
            receipt = receipts[str(year)][name]
            path = output / receipt["artifact"]
            if base.digest(path) != receipt["sha256"]:
                raise ValueError("Sealed fold model changed before scoring")
            model = joblib.load(path)
            rows[name] = audit.probability([model["model"]], model["temperature"], test[model["feature_columns"]])
        results[str(year)] = {"metrics": {name: audit.metrics(rows, name) for name in CONFIG["models"]},
                              "paired": audit.paired(rows, "corrected48", "career60")}
        predictions.append(rows)
    pooled = pd.concat(predictions, ignore_index=True)
    if len(pooled) != 2018 or pooled.fight_id.duplicated().any() or pooled.event_id.nunique() != 170:
        raise ValueError("Pooled disjoint yearly test cohort changed")
    results["pooled2021_2024"] = {"metrics": {name: audit.metrics(pooled, name) for name in CONFIG["models"]},
                                  "paired": audit.paired(pooled, "corrected48", "career60")}
    pooled.to_csv(output / "predictions.csv", index=False)
    for filename, checksum in seal.items():
        if base.digest(output / filename) != checksum:
            raise ValueError("An artifact changed after the pre-test seal")
    if fingerprint(source, supplement_path) != before_hashes:
        raise ValueError("Frozen sources, definitions or prior artifacts changed during the experiment")
    report = {"schemaVersion": 1, "createdAt": now(), "status": CONFIG["status"],
              "protocolSHA256": base.digest(output / "protocol.json"), "cohorts": cohort_manifest,
              "results": results, "fitReceipts": receipts, "physicalPrefixChecks": prefix_checks,
              "modelsFrozenBeforeTestPredictions": 8, "symmetryPassed": True, "sourceAndPriorArtifactHashesUnchanged": True,
              "runtimeChanged": False, "modelPromotion": False, "supports70PercentClaim": False, "highConfidenceValidated": False,
              "interpretation": "Eight fixed pipeline refits compare feature definitions across earlier calendar years; they do not replace the current corrected48 or failed career60 replay artifacts.",
              "limitations": ["These calendar years were used previously in model development and calibration; they are not untouched outcome sets.",
                              "The feature/model design and source snapshots were selected after later history existed. Chronological fitting alone cannot remove that design/revision bias.",
                              "Later folds deliberately learn from earlier years; fold results and fighter histories are not independent.",
                              "Additional professional records are incomplete and have no certified original publication/revision timestamps.",
                              "Event bootstrap intervals are descriptive and do not fully account for repeated fighters, model design or source uncertainty.",
                              "All4years and pooled results are reported without selecting a favorable year or making a promotion decision."]}
    audit.write(output / "report.json", report)
    audit.write(output / "artifact_hashes.json", {p.name: base.digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    print(json.dumps({"stage": "complete", "output": str(output), "results": {
        year: value["metrics"] for year, value in results.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed48-versus60 historical rolling-origin stresschecks, testyears2021-2024.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.supplement.resolve(), args.output.resolve())
