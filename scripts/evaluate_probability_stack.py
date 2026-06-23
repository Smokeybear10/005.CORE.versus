import argparse
import importlib.metadata
import json
import math
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_challenger as challenger
from scripts import evaluate_identity_repair as audit
from scripts import evaluate_predictors as previous
from scripts import evaluate_research as base
from scripts import predictor_features


SOURCE = ROOT / "data/evaluation/identity-repaired-source-v1"
PRIOR = ROOT / "data/evaluation/identity-repair-evaluation-v1"
CACHE = PRIOR / "development_features.parquet"
COMPONENTS = ("gb48", "lr92_strong", "gb92", "lr92_standard")
MODELS = ("candidate_stack", "gb48", "fixed_blend92")
META = ["fight_id", "event_id", "date", "target", "red_fighter_id", "blue_fighter_id"]
PINS = {
    CACHE: "68835656bf48acf69af6d3d8488f8bee59ebff0a08a57dfce98449c669c5b6fe",
    SOURCE / "manifest.json": "3fe41b67d9e987cbded3693a631bfea9a4c03b20f60c944fcdf20f657a86e8a8",
    PRIOR / "feature_columns.json": "646b31e814f755212da7f33fa544b90a5677f786f7869d8752b85b6d482d158f",
    PRIOR / "predictions.csv": "bc7ef73bdebe1775dda9ee7219a86eb4a50e48299a2d94ab82937aae7d0adb00",
    **{ROOT / "scripts" / name: checksum for name, checksum in audit.CODE.items()},
}
CONFIG = {
    "schemaVersion": 1, "candidate": "convex_gb48_strong_lr92", "developmentYears": [2018, 2019, 2020, 2021, 2022],
    "trainingStart": "2010-01-01", "finalFitEndExclusive": "2023-01-01", "calibrationEndExclusive": "2025-01-01",
    "replayEndExclusive": "2026-11-01", "features": "Exactly the corrected48 and corrected92 definitions; no new predictors or row filters",
    "learners": {"gradient_boosting": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "random_state": 42},
                 "logistic_regression": {"C": 0.05, "max_iter": 2000, "random_state": 42},
                 "reference_logistic_C": 0.5},
    "blend": "p=w*LR92(C=.05)+(1-w)*GB48; one nonnegative mixing weight, no intercept",
    "mixingFit": "Closed-form Brier minimizer over chronological2018-2022 OOF rows; clip to[0,1]; if denominator exactly zero use0.5",
    "forwardMeta": "For each2019-2022 evaluation year, estimate mixing weight using strictly earlier OOF years only;2018 is warmup",
    "developmentPolicy": "Base components fit2010 through prior December31. Report raw forward-meta probability quality; no meta training score called validation",
    "finalPolicy": "Fit all four components2010-2022, freeze final weight from2018-2022 OOF, calibrate each of3fixed outputs on2023-2024",
    "calibration": "One positive temperature per fixed output, log(T) bounded[-3,3], calibration log loss only; no change to picks",
    "preprocessing": "Training-only median imputation, missing indicators, scaling and division OHE; identical corner augmentation and symmetrized predictions",
    "comparators": "GB48 and fixed50:50 GB92+LR92(C=.5), refitted and calibrated on exactly the same final periods",
    "selection": "ONE declared candidate; no learner, regularization, feature, threshold or weight-grid search; all outputs reported",
    "primaryMetrics": ["Brier loss", "log loss"], "secondaryMetrics": ["winner accuracy", "paired event-bootstrap intervals"],
    "descriptiveGate": "Candidate must have strictly lower Brier AND log loss than BOTH fair-fit controls on forward2019-2022 and fixed985 replay; passing is only descriptive",
    "promotionGate": "No automatic promotion. Market superiority additionally requires lower paired Brier and log loss versus timestamp-qualified market quotes on the same bouts, supporting event-bootstrap intervals, and new prospective evidence",
    "bootstrap": audit.BOOTSTRAP, "tieRule": "Red ifp>=.5 for every output",
    "markets": "No odds, market probabilities or market labels used in fitting, mixing, calibration or selection",
    "status": "All historical periods were previously examined;2025-2026 is reused replay, never a new holdout",
    "runtimeChanged": False, "promotionAllowed": False,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def versions() -> dict:
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "fastparquet")}}


def fingerprints() -> dict:
    paths = [Path(__file__), Path(audit.__file__), *PINS]
    for directory in (SOURCE, audit.REFERENCE):
        manifest = directory / "manifest.json"
        paths.extend([manifest, *[directory / name for name in json.loads(manifest.read_text())["files"]]])
    return {str(path): base.digest(path) for path in paths}


def assert_integrity(protocol: dict) -> None:
    if fingerprints() != protocol["hashes"] or versions() != protocol["versions"] or CONFIG != protocol["config"]:
        raise ValueError("Frozen inputs, code, configuration or dependencies changed")


def mixing_weight(frame: pd.DataFrame) -> float:
    if frame.empty:
        raise ValueError("No strictly earlier OOF predictions for mixing")
    difference = frame.lr92_strong.to_numpy() - frame.gb48.to_numpy()
    denominator = float(np.dot(difference, difference))
    if denominator == 0:
        return 0.5
    return float(np.clip(np.dot(difference, frame.target.to_numpy() - frame.gb48.to_numpy()) / denominator, 0, 1))


def combine(frame: pd.DataFrame, weight: float) -> pd.DataFrame:
    return frame[META].assign(candidate_stack=weight * frame.lr92_strong + (1 - weight) * frame.gb48,
                              gb48=frame.gb48, fixed_blend92=(frame.gb92 + frame.lr92_standard) / 2)


def fit_components(frame: pd.DataFrame, columns: dict) -> dict:
    fitted = {}
    for name in COMPONENTS:
        learner = "logistic_regression" if name.startswith("lr") else "gradient_boosting"
        selected = columns["original" if name == "gb48" else "expanded"]
        model = challenger.make_pipeline(learner, selected)
        params = CONFIG["learners"][learner].copy()
        if name == "lr92_standard":
            params["C"] = CONFIG["learners"]["reference_logistic_C"]
        model.named_steps["model"].set_params(**params)
        x = pd.concat([frame[selected], base.swapped(frame[selected])], ignore_index=True)
        y = np.concatenate([frame.target.to_numpy(), 1 - frame.target.to_numpy()])
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            fitted[name] = model.fit(x, y)
    return fitted


def predict_components(models: dict, frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[META].copy()
    for name, model in models.items():
        result[name] = previous.predict([model], frame)
    return result


def report_comparison(frame: pd.DataFrame) -> dict:
    return {"metrics": {name: audit.metrics(frame, name) for name in MODELS},
            "paired": {name: audit.paired(frame, name, "candidate_stack") for name in MODELS[1:]}}


def self_checks() -> None:
    # Identical predictors must use the declared neutral fallback, not divide by zero.
    same = pd.DataFrame({"lr92_strong": [.2, .8], "gb48": [.2, .8], "target": [0, 1]})
    if mixing_weight(same) != .5:
        raise ValueError("Identical-component fallback failed")
    good_linear = same.assign(lr92_strong=[.01, .99], gb48=[.6, .4])
    if mixing_weight(good_linear) != 1 or mixing_weight(good_linear.rename(columns={"lr92_strong": "gb48", "gb48": "lr92_strong"})) != 0:
        raise ValueError("Convex mixing endpoints failed")
    rng = np.random.default_rng(649)
    sample = pd.DataFrame({"lr92_strong": rng.uniform(.05, .95, 100), "gb48": rng.uniform(.05, .95, 100),
                           "target": rng.integers(0, 2, 100)})
    weight = mixing_weight(sample)
    if abs(weight - mixing_weight(1 - sample)) > 1e-12:
        raise ValueError("Mixing fit depends on corner orientation")
    residual = weight * sample.lr92_strong + (1 - weight) * sample.gb48 - sample.target
    optimal = float(np.mean(residual ** 2))
    for alternative in (0, .25, .5, .75, 1):
        value = alternative * sample.lr92_strong + (1 - alternative) * sample.gb48
        if optimal > np.mean((value - sample.target) ** 2) + 1e-14:
            raise ValueError("Closed-form weight is not a Brier minimizer")


def develop(output: Path) -> None:
    for path, expected in PINS.items():
        if base.digest(path) != expected:
            raise ValueError(f"Pinned source changed: {path}")
    for directory in (SOURCE, audit.REFERENCE):
        for name, expected in json.loads((directory / "manifest.json").read_text())["files"].items():
            if base.digest(directory / name) != expected:
                raise ValueError(f"Source bundle mismatch: {name}")
    self_checks()
    data = pd.read_parquet(CACHE)
    columns = json.loads((PRIOR / "feature_columns.json").read_text())
    if len(data) != 6573 or data.fight_id.duplicated().any() or len(columns["original"]) != 48 or len(columns["expanded"]) != 92:
        raise ValueError("Frozen feature membership or columns differ")
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"createdAt": now(), "config": CONFIG, "hashes": fingerprints(), "versions": versions(),
                "source": str(SOURCE), "featureCache": str(CACHE), "columns": columns, "cohorts": {}}
    for year in CONFIG["developmentYears"]:
        training = data[(data.date >= CONFIG["trainingStart"]) & (data.date < f"{year}-01-01")]
        test = data[(data.date >= f"{year}-01-01") & (data.date < f"{year + 1}-01-01")]
        if training.empty or test.empty or training.date.max() >= test.date.min() or set(training.event_id) & set(test.event_id):
            raise ValueError("Development chronology or event separation failed")
        protocol["cohorts"][str(year)] = {"train": len(training), "test": len(test), "trainIdsSHA256": audit.id_hash(training.fight_id),
                                         "testIdsSHA256": audit.id_hash(test.fight_id)}
    audit.write(output / "protocol.json", protocol)
    for path in [Path(__file__), Path(audit.__file__), *[ROOT / "scripts" / name for name in audit.CODE]]:
        (output / path.name).write_bytes(path.read_bytes())
    print(json.dumps({"stage": "protocol_frozen", "sha256": base.digest(output / "protocol.json")}), flush=True)
    prefix_checks = []
    for cutoff in ("2018-01-01", "2023-01-01"):
        prefix, first, second = previous.load_features(SOURCE, cutoff)
        pd.testing.assert_frame_equal(prefix, data[data.date < cutoff].reset_index(drop=True), check_exact=True)
        if first != columns["original"] or second != columns["expanded"]:
            raise ValueError("Prefix feature definitions changed")
        prefix_checks.append({"cutoffExclusive": cutoff, "rows": len(prefix), "passed": True})
    audit.write(output / "prefix_checks.json", prefix_checks)
    oof = []
    for year in CONFIG["developmentYears"]:
        training = data[(data.date >= CONFIG["trainingStart"]) & (data.date < f"{year}-01-01")]
        test = data[(data.date >= f"{year}-01-01") & (data.date < f"{year + 1}-01-01")]
        models = fit_components(training, columns)
        predictions = predict_components(models, test).assign(development_year=year)
        oof.append(predictions)
        assert_integrity(protocol)
        print(json.dumps({"stage": "oof_components_complete", "year": year, "train": len(training), "test": len(test)}), flush=True)
    oof = pd.concat(oof, ignore_index=True)
    oof.to_csv(output / "development_oof_components.csv", index=False)
    forward, weights = [], {}
    for year in CONFIG["developmentYears"][1:]:
        earlier = oof[oof.development_year < year]
        weight = mixing_weight(earlier)
        forward.append(combine(oof[oof.development_year == year], weight))
        weights[str(year)] = {"linearWeight": weight, "metaFitCount": len(earlier), "metaFitThrough": str(earlier.date.max().date()),
                             "metaFitIdsSHA256": audit.id_hash(earlier.fight_id)}
    forward = pd.concat(forward, ignore_index=True)
    forward.to_csv(output / "forward_meta_predictions.csv", index=False)
    weight = mixing_weight(oof)
    development = {"forward2019_2022": report_comparison(forward), "yearly": {str(year): report_comparison(forward[forward.date.dt.year == year])
                    for year in CONFIG["developmentYears"][1:]}, "forwardWeights": weights, "finalLinearWeight": weight,
                   "metaFitTrainingMetricsNotValidation": {name: audit.metrics(combine(oof, weight), name) for name in MODELS}}
    audit.write(output / "development-report.json", development)
    training = data[(data.date >= CONFIG["trainingStart"]) & (data.date < CONFIG["finalFitEndExclusive"])]
    calibration = data[(data.date >= CONFIG["finalFitEndExclusive"]) & (data.date < CONFIG["calibrationEndExclusive"])]
    if len(training) != 5557 or len(calibration) != 1016 or set(training.event_id) & set(calibration.event_id):
        raise ValueError("Final fit/calibration membership changed")
    models = fit_components(training, columns)
    raw = combine(predict_components(models, calibration), weight)
    temperatures, calibrated = {}, raw[META].copy()
    for name in MODELS:
        optimized = minimize_scalar(lambda log_t: log_loss(raw.target, base.apply_temperature(raw[name].to_numpy(), math.exp(log_t)), labels=[0, 1]),
                                    bounds=(-3, 3), method="bounded")
        if not optimized.success:
            raise ValueError("Temperature optimization failed")
        temperatures[name] = math.exp(optimized.x)
        calibrated[name] = base.apply_temperature(raw[name].to_numpy(), temperatures[name])
        calibrated[f"{name}_raw"] = raw[name]
        if not np.array_equal(raw[name] >= .5, calibrated[name] >= .5):
            raise ValueError("Positive temperature changed winner picks")
    calibrated.to_csv(output / "calibration_predictions.csv", index=False)
    artifact = {"models": models, "columns": columns, "linearWeight": weight, "temperatures": temperatures,
                "protocolSHA256": base.digest(output / "protocol.json"), "validatedForBetting": False, "runtimePromotion": False}
    joblib.dump(artifact, output / "probability_stack.joblib", compress=3)
    assert_integrity(protocol)
    audit.write(output / "pre-replay-manifest.json", {"createdAt": now(), "protocolSHA256": base.digest(output / "protocol.json"),
        "files": {path.name: base.digest(path) for path in output.iterdir() if path.is_file()},
        "fitCount": len(training), "fitThrough": str(training.date.max().date()), "fitIdsSHA256": audit.id_hash(training.fight_id),
        "calibrationCount": len(calibration), "calibrationIdsSHA256": audit.id_hash(calibration.fight_id),
        "finalLinearWeight": weight, "temperatures": temperatures, "modelsFrozenBeforeReplay": True,
        "runtimeChanged": False, "modelsFitted": 24})
    print(json.dumps({"stage": "models_frozen_before_replay", "linearWeight": weight,
                      "modelSHA256": base.digest(output / "probability_stack.joblib")}), flush=True)


def replay(output: Path) -> None:
    if any((output / name).exists() for name in ("predictions.csv", "replay_raw_components.csv", "replay_features.parquet", "report.json")):
        raise ValueError("Replay outputs already exist; frozen experiment outputs cannot be overwritten")
    protocol = json.loads((output / "protocol.json").read_text())
    assert_integrity(protocol)
    manifest = json.loads((output / "pre-replay-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if base.digest(output / name) != expected:
            raise ValueError(f"Frozen fit artifact changed: {name}")
    artifact = joblib.load(output / "probability_stack.joblib")
    full, first, second = previous.load_features(SOURCE, CONFIG["replayEndExclusive"])
    if first != artifact["columns"]["original"] or second != artifact["columns"]["expanded"]:
        raise ValueError("Replay feature columns changed")
    cached = pd.read_parquet(CACHE)
    pd.testing.assert_frame_equal(full[full.date < CONFIG["calibrationEndExclusive"]].reset_index(drop=True), cached, check_exact=True)
    archived = pd.read_csv(PRIOR / "predictions.csv", parse_dates=["date"])
    if len(archived) != 985 or audit.id_hash(archived.fight_id) != audit.COHORT_HASHES["same985"]:
        raise ValueError("Fixed replay cohort changed")
    requests = full.set_index("fight_id").loc[archived.fight_id].reset_index()
    pd.testing.assert_frame_equal(requests[["fight_id", "event_id", "date", "target"]], archived[["fight_id", "event_id", "date", "target"]],
                                  check_exact=True, check_dtype=False)
    for side in ("red", "blue"):
        if not np.array_equal(requests[f"{side}_fighter_id"], archived[f"corrected_{side}_fighter_id"]):
            raise ValueError("Replay fighter identity changed")
    raw_components = predict_components(artifact["models"], requests)
    raw = combine(raw_components, artifact["linearWeight"])
    result = raw[META].copy()
    reverse_components = predict_components(artifact["models"], base.swapped(requests))
    reverse = combine(reverse_components, artifact["linearWeight"])
    for name in MODELS:
        result[name] = base.apply_temperature(raw[name].to_numpy(), artifact["temperatures"][name])
        opposite = base.apply_temperature(reverse[name].to_numpy(), artifact["temperatures"][name])
        if not np.isfinite(result[name]).all() or not np.allclose(result[name] + opposite, 1, atol=1e-12, rtol=0):
            raise ValueError("Calibrated replay symmetry failed")
    result.to_csv(output / "predictions.csv", index=False)
    raw_components.to_csv(output / "replay_raw_components.csv", index=False)
    requests.to_parquet(output / "replay_features.parquet", index=False)
    cohorts = {"same985": result, "primary285": result[result.date > "2026-05-09"],
               "recent49": result[result.date.isin(pd.to_datetime(audit.RECENT_DATES))]}
    if [len(cohorts[name]) for name in cohorts] != [985, 285, 49]:
        raise ValueError("Fixed scored cohort counts changed")
    results = {name: report_comparison(frame) for name, frame in cohorts.items()}
    development = json.loads((output / "development-report.json").read_text())
    def improves_both(metrics):
        return all(metrics["candidate_stack"][metric] < metrics[control][metric]
                   for control in MODELS[1:] for metric in ("brier", "logLoss"))
    gate = {"forwardDevelopment": improves_both(development["forward2019_2022"]["metrics"]),
            "reused985": improves_both(results["same985"]["metrics"])}
    assert_integrity(protocol)
    audit.write(output / "report.json", {"createdAt": now(), "status": CONFIG["status"],
        "protocolSHA256": base.digest(output / "protocol.json"), "modelSHA256": base.digest(output / "probability_stack.joblib"),
        "development": development, "results": results, "descriptiveGate": gate | {"passed": all(gate.values())},
        "same985GB48MatchesFixedCorrectedReference": bool(np.allclose(result.gb48, archived.original48_after, atol=1e-12, rtol=0)),
        "calibration": {"count": 1016, "fromInclusive": "2023-01-01", "through": "2024-12-31", "temperatures": artifact["temperatures"]},
        "finalLinearWeight": artifact["linearWeight"], "sourceCodeAndOldBundleHashesUnchanged": True,
        "runtimeChanged": False, "promotionAllowed": False, "marketSuperiorityEstablished": False,
        "limitations": ["All periods and outcomes were previously examined; these are descriptive historical experiments.",
            "Development forward-meta diagnostics are uncalibrated and final calibration is2023-2024 only.",
            "No market benchmark is computed here; compare timestamp-qualified quotes on identical bouts separately.",
            "Retrospective source snapshots cannot certify original historical publication or revision availability.",
            "Event bootstrap does not remove selection bias or all repeated-fighter/temporal dependence."]})
    previous.inventory(output)
    print(json.dumps({"stage": "replay_complete", "gate": gate, "same985": results["same985"]["metrics"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="One fixed probability-focused standalone stack; no runtime promotion")
    parser.add_argument("stage", choices=("develop", "replay"))
    parser.add_argument("--output", type=Path, default=ROOT / "data/evaluation/probability-stack-v1")
    args = parser.parse_args()
    (develop if args.stage == "develop" else replay)(args.output.resolve())


if __name__ == "__main__":
    main()
