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
from scripts import career_features as career
from scripts import evaluate_challenger as challenger
from scripts import evaluate_identity_repair as audit
from scripts import evaluate_research as base


BASELINE = ROOT / "data/evaluation/identity-repair-evaluation-v1"
DECLARATION = ROOT / "data/evaluation/career-candidate-declaration-v1.json"
MODEL_SHA = "6fde3bc72e4788c944efa65ad8a911c2e18d840404a337e2cd09559a8530a924"
CONFIG = {
    "schemaVersion": 1, "candidate": "corrected48_plus12_observed_career_gb", "featureCount": 60,
    "fitFromInclusive": "2010-01-01", "fitEndExclusive": "2023-01-01",
    "calibrationFromInclusive": "2023-01-01", "calibrationEndExclusive": "2025-01-01",
    "replayEndExclusive": "2026-11-01", "selection": "One fixed candidate; no search, selection, thresholds or retuning",
    "learner": "Clone corrected48 gradient-boosting estimator settings; rebuild identical preprocessing for60 columns",
    "calibration": "Positive temperature; minimize2023-24 logloss with log(T) bounded[-3,3]",
    "regionalUse": "Supplemental records supply only12 prior-date features; never training targets or UFC round statistics",
    "cohorts": {"same985": 985, "primary285": 285, "recent49": 49, "lastTwo26": 26, "latest12": 12},
    "promotionGate": {"same985AccuracyAtLeastBaseline": True, "same985BrierStrictlyBetter": True,
                      "same985LogLossStrictlyBetter": True, "recent49MaximumBrierDegradation": 0.01,
                      "recent49MaximumLogLossDegradation": 0.03,
                      "action": "Eligible for separate review only when every condition passes; never automatic deployment"},
    "bootstrap": audit.BOOTSTRAP, "tieRule": "Red when p>=0.5",
    "status": "Previously inspected historical replay; not independent validation or proof of70% accuracy",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(source: Path, supplement: Path) -> dict:
    paths = [Path(__file__), Path(career.__file__), Path(challenger.__file__), Path(base.__file__), Path(audit.__file__),
             ROOT / "tests/test_career_features.py", BASELINE / "provisional_challenger.joblib",
             BASELINE / "predictions.csv", BASELINE / "development_and_calibration_features.parquet", audit.RECENT, DECLARATION]
    for directory in (source, supplement):
        manifest = directory / "manifest.json"
        paths.extend([manifest, *[directory / filename for filename in json.loads(manifest.read_text())["files"]]])
    return {str(path): base.digest(path) for path in paths}


def source_frames(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((source / "manifest.json").read_text())
    for filename, expected in manifest["files"].items():
        if base.digest(source / filename) != expected:
            raise ValueError(f"Corrected source checksum mismatch: {filename}")
    fights, rounds, fighters = (pd.read_parquet(source / f"{name}.parquet") for name in ("fights", "rounds", "fighters"))
    if fights.fight_id.duplicated().any() or fighters.fighter_id.duplicated().any():
        raise ValueError("Corrected source identities are duplicated")
    return fights, rounds, fighters


def features(fights: pd.DataFrame, rounds: pd.DataFrame, fighters: pd.DataFrame, supplement: pd.DataFrame,
             cutoff: str, prepared_history: dict | None = None) -> tuple[pd.DataFrame, list[str], list[str]]:
    prior = fights[fights.date < pd.Timestamp(cutoff)].copy()
    prior_rounds = rounds[rounds.fight_id.isin(prior.fight_id)]
    totals, _ = base.prepare_round_stats(prior_rounds)
    original = challenger.add_context(base.reconstruct(prior, totals), prior, fighters)
    columns48 = [name for name in original if name.startswith(("diff_", "ctx_"))] + ["weightclass"]
    augmented = career.add_career_features(original, prior, supplement, prepared_history)
    pd.testing.assert_frame_equal(augmented[original.columns], original, check_exact=True)
    columns60 = [name for name in columns48 if name != "weightclass"] + career.ADDED_FEATURES + ["weightclass"]
    if len(columns48) != 48 or len(columns60) != 60 or len(set(columns60)) != 60:
        raise ValueError("Career candidate feature definitions changed")
    if np.isinf(augmented[[name for name in columns60 if name != "weightclass"]].to_numpy(dtype=float)).any():
        raise ValueError("Infinite career feature")
    for side in ("red", "blue"):
        through = augmented[f"{side}_history_through"]
        if not (through.isna() | (through < augmented.date)).all():
            raise ValueError("Non-prior base history")
    return augmented, columns48, columns60


def strict_recent(fights: pd.DataFrame, rounds: pd.DataFrame, fighters: pd.DataFrame, supplement: pd.DataFrame,
                  replay: pd.DataFrame, recent_ids: set[str], columns: list[str]) -> list[dict]:
    checks = []
    for date in audit.RECENT_DATES:
        card = replay[(replay.date == pd.Timestamp(date)) & replay.fight_id.isin(recent_ids)]
        prior = fights[fights.date < pd.Timestamp(date)].copy()
        prior_rounds = rounds[rounds.fight_id.isin(prior.fight_id)].copy()
        prior_career = supplement[supplement.date < pd.Timestamp(date)].copy()
        prepared = career.prepare_history(prior, prior_career)
        totals, _ = base.prepare_round_stats(prior_rounds)
        contexts = fights.set_index("fight_id").loc[card.fight_id]
        rows = []
        for row in contexts.itertuples():
            frame, quality = career.build_inference_features(prior, prior_rounds, fighters, prior_career,
                row.red_fighter_id, row.blue_fighter_id, date, row.weightclass, int(row.num_rounds), bool(row.is_title_bout),
                prepared_totals=totals, prepared_history=prepared)
            if any(x["historyThrough"] is not None and x["historyThrough"] >= date for x in quality.values()):
                raise ValueError("Strict inference exposed current-card history")
            rows.append(frame.assign(fight_id=row.Index))
        physical = pd.concat(rows, ignore_index=True)
        expected = card.set_index("fight_id").loc[physical.fight_id]
        pd.testing.assert_frame_equal(physical[columns].reset_index(drop=True), expected[columns].reset_index(drop=True),
                                      check_exact=True, check_dtype=False)
        checks.append({"eventDate": date, "rows": len(card), "ufcHistoryRows": len(prior),
                       "supplementHistoryRows": len(prior_career), "all60FeaturesMatchStrictPriorHistory": True})
        print(json.dumps({"stage": "strict_recent", **checks[-1]}), flush=True)
    return checks


def run(source: Path, supplement_path: Path, output: Path) -> None:
    declaration = json.loads(DECLARATION.read_text())
    if declaration["config"] != CONFIG or declaration["careerFeaturePolicy"] != career.CONFIG:
        raise ValueError("Candidate or promotion gate differs from its prior declaration")
    model_path = BASELINE / "provisional_challenger.joblib"
    if base.digest(model_path) != MODEL_SHA:
        raise ValueError("Corrected48 reference model changed")
    baseline = joblib.load(model_path)
    if baseline["selected_model"] != "gradient_boosting" or len(baseline["feature_columns"]) != 48:
        raise ValueError("Unexpected corrected48 reference")
    original_source = Path(baseline["manifest"]["source"])
    for filename in ("fights.parquet", "rounds.parquet", "fighters.parquet"):
        if base.digest(source / filename) != baseline["manifest"]["hashes"][str(original_source / filename)]:
            raise ValueError(f"The fixed corrected48 source changed: {filename}")
    before_hashes = fingerprint(source, supplement_path)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"createdAt": now(), "declarationSHA256": base.digest(DECLARATION), "config": CONFIG, "careerFeaturePolicy": career.CONFIG,
                "source": str(source), "supplement": str(supplement_path), "baselineModelSHA256": MODEL_SHA,
                "hyperparameters": baseline["model"].named_steps["model"].get_params(), "hashes": before_hashes,
                "cohortIdHashes": audit.COHORT_HASHES, "runtimeChanged": False,
                "versions": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                             for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "fastparquet")}}}
    audit.write(output / "protocol.json", protocol)
    for path in (Path(__file__), Path(career.__file__), Path(challenger.__file__), Path(base.__file__), Path(audit.__file__),
                 ROOT / "tests/test_career_features.py"):
        (output / path.name).write_bytes(path.read_bytes())
    print(json.dumps({"stage": "protocol_frozen", "output": str(output)}), flush=True)
    supplement, supplement_manifest = career.load_supplement(supplement_path)
    fights, rounds, fighters = source_frames(source)
    history = career.prepare_history(fights, supplement)
    development, columns48, columns60 = features(fights, rounds, fighters, supplement, "2025-01-01", history)
    if columns48 != baseline["feature_columns"]:
        raise ValueError("Original48 feature ordering changed")
    original = development.drop(columns=career.ADDED_FEATURES)
    pd.testing.assert_frame_equal(original, pd.read_parquet(BASELINE / "development_and_calibration_features.parquet"), check_exact=True)
    prefix_fights = fights[fights.date < pd.Timestamp(CONFIG["fitEndExclusive"])]
    prefix_supplement = supplement[supplement.date < pd.Timestamp(CONFIG["fitEndExclusive"])]
    prefix, first, second = features(prefix_fights, rounds[rounds.fight_id.isin(prefix_fights.fight_id)], fighters,
                                     prefix_supplement, CONFIG["fitEndExclusive"])
    pd.testing.assert_frame_equal(development[development.date < pd.Timestamp(CONFIG["fitEndExclusive"])].reset_index(drop=True), prefix,
                                  check_exact=True)
    if first != columns48 or second != columns60:
        raise ValueError("Physical training-prefix columns changed")
    print(json.dumps({"stage": "training_prefix_passed", "rows": len(prefix)}), flush=True)
    development.to_parquet(output / "development_and_calibration_features.parquet", index=False)
    audit.write(output / "feature_columns.json", {"original48": columns48, "candidate60": columns60})
    training = development[(development.date >= pd.Timestamp(CONFIG["fitFromInclusive"])) &
                           (development.date < pd.Timestamp(CONFIG["fitEndExclusive"]))]
    calibration = development[(development.date >= pd.Timestamp(CONFIG["calibrationFromInclusive"])) &
                              (development.date < pd.Timestamp(CONFIG["calibrationEndExclusive"]))]
    if len(training) != 5557 or len(calibration) != 1016 or set(training.event_id) & set(calibration.event_id):
        raise ValueError("Fixed corrected48 training/calibration cohorts changed")
    pipeline = challenger.make_pipeline("gradient_boosting", columns60)
    pipeline.set_params(model=clone(baseline["model"].named_steps["model"]))
    X = pd.concat([training[columns60], base.swapped(training[columns60])], ignore_index=True)
    y = np.concatenate([training.target.to_numpy(), 1 - training.target.to_numpy()])
    model = pipeline.fit(X, y)
    raw = base.symmetric_probability(model, calibration[columns60])
    fitted = minimize_scalar(lambda log_t: log_loss(calibration.target, base.apply_temperature(raw, math.exp(log_t)), labels=[0, 1]),
                            bounds=(-3, 3), method="bounded")
    if not fitted.success:
        raise ValueError("Career candidate temperature calibration failed")
    temperature = math.exp(fitted.x)
    calibrated = audit.probability([model], temperature, calibration[columns60])
    if not np.array_equal(raw >= 0.5, calibrated >= 0.5):
        raise ValueError("Temperature calibration changed picks")
    calibration_rows = calibration[["fight_id", "event_id", "date", "target"]].assign(raw=raw, candidate60=calibrated)
    calibration_rows.to_csv(output / "calibration_predictions.csv", index=False)
    fit_receipt = {"fitCount": len(training), "calibrationCount": len(calibration),
                   "fitFrom": str(training.date.min().date()), "fitThrough": str(training.date.max().date()),
                   "calibrationFrom": str(calibration.date.min().date()), "calibrationThrough": str(calibration.date.max().date()),
                   "fitFightIdsSHA256": audit.id_hash(training.fight_id), "calibrationFightIdsSHA256": audit.id_hash(calibration.fight_id),
                   "temperature": temperature, "calibrationMetricsInSample": audit.metrics(calibration_rows, "candidate60")}
    artifact = {"model": model, "selected_model": "gradient_boosting", "temperature": temperature, "feature_columns": columns60,
                "config": CONFIG, "career_config": career.CONFIG, "manifest": protocol,
                "supplement_manifest_sha256": base.digest(supplement_path / "manifest.json"),
                "validated_for_betting": False, "high_confidence_validated": False}
    joblib.dump(artifact, output / "career_candidate.joblib", compress=3)
    audit.write(output / "fit_receipt.json", fit_receipt)
    audit.write(output / "pre-replay-manifest.json", {"frozenAt": now(), "files": {p.name: base.digest(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({"stage": "candidate_frozen_before_replay", "modelSHA256": base.digest(output / "career_candidate.joblib")}), flush=True)
    archive = pd.read_csv(BASELINE / "predictions.csv", parse_dates=["date"])
    recent = pd.read_csv(audit.RECENT)
    recent_ids = set(recent.loc[recent.scored == True, "fight_id"])
    if len(archive) != 985 or archive.fight_id.duplicated().any() or audit.id_hash(archive.fight_id) != audit.COHORT_HASHES["same985"]:
        raise ValueError("Fixed same985 cohort changed")
    if len(recent_ids) != 49 or audit.id_hash(pd.Series(sorted(recent_ids))) != audit.COHORT_HASHES["recent49"]:
        raise ValueError("Fixed recent49 cohort changed")
    full, first, second = features(fights, rounds, fighters, supplement, CONFIG["replayEndExclusive"], history)
    if first != columns48 or second != columns60:
        raise ValueError("Replay feature definitions changed")
    pd.testing.assert_frame_equal(full[full.date < pd.Timestamp("2025-01-01")].reset_index(drop=True), development, check_exact=True)
    if not set(archive.fight_id) <= set(full.fight_id):
        raise ValueError("Candidate cannot cover the unchanged985 cohort")
    replay = full.set_index("fight_id").loc[archive.fight_id].reset_index()
    pd.testing.assert_frame_equal(replay[["fight_id", "event_id", "date", "target"]],
                                  archive[["fight_id", "event_id", "date", "target"]], check_dtype=False)
    for side in ("red", "blue"):
        if not np.array_equal(replay[f"{side}_fighter_id"], archive[f"corrected_{side}_fighter_id"]):
            raise ValueError("Replay fighter identities differ from corrected48 reference")
    before = audit.probability([baseline["model"]], baseline["temperature"], replay[columns48])
    if not np.allclose(before, archive.original48_after, atol=1e-12, rtol=0):
        raise ValueError("Corrected48 archived baseline reproduction failed")
    rows = replay[["fight_id", "event_id", "date", "red_fighter_id", "blue_fighter_id", "red_name", "blue_name", "target",
                   "red_history_through", "blue_history_through"]].copy()
    rows["corrected48"] = archive.original48_after.to_numpy()
    rows["career60"] = audit.probability([model], temperature, replay[columns60])
    checks = strict_recent(fights, rounds, fighters, supplement, replay, recent_ids, columns60)
    rows.to_csv(output / "predictions.csv", index=False)
    replay.to_parquet(output / "replay_features.parquet", index=False)
    cohorts = {"same985": rows, "primary285": rows[rows.date > pd.Timestamp("2026-05-09")],
               "recent49": rows[rows.fight_id.isin(recent_ids)], "lastTwo26": rows[rows.date.isin(pd.to_datetime(audit.RECENT_DATES[-2:]))],
               "latest12": rows[rows.date == pd.Timestamp(audit.RECENT_DATES[-1])]}
    results = {}
    for name, frame in cohorts.items():
        if len(frame) != CONFIG["cohorts"][name]:
            raise ValueError(f"Fixed cohort changed: {name}")
        results[name] = {"fightIdsSHA256": audit.id_hash(frame.fight_id),
                         "metrics": {column: audit.metrics(frame, column) for column in ("corrected48", "career60")},
                         "paired": audit.paired(frame, "corrected48", "career60")}
    main, recent_metrics = results["same985"]["metrics"], results["recent49"]["metrics"]
    conditions = {"same985AccuracyAtLeastBaseline": main["career60"]["correct"] >= main["corrected48"]["correct"],
                  "same985BrierStrictlyBetter": main["career60"]["brier"] < main["corrected48"]["brier"],
                  "same985LogLossStrictlyBetter": main["career60"]["logLoss"] < main["corrected48"]["logLoss"],
                  "recent49BrierWithinLimit": recent_metrics["career60"]["brier"] - recent_metrics["corrected48"]["brier"] <= 0.01,
                  "recent49LogLossWithinLimit": recent_metrics["career60"]["logLoss"] - recent_metrics["corrected48"]["logLoss"] <= 0.03}
    if fingerprint(source, supplement_path) != before_hashes:
        raise ValueError("A frozen source, code, model or archive changed during the run")
    report = {"schemaVersion": 1, "createdAt": now(), "status": CONFIG["status"], "protocolSHA256": base.digest(output / "protocol.json"),
              "fit": fit_receipt, "results": results, "promotionGate": {"conditions": conditions, "eligibleForReview": all(conditions.values())},
              "sourceSupplement": {"manifestSHA256": base.digest(supplement_path / "manifest.json"), "source": supplement_manifest["source"],
                                   "counts": supplement_manifest["counts"], "featureUse": career.CONFIG},
              "strictRecentChecks": checks, "original48FeaturesUnchanged": True, "original48ArchiveParityPassed": True,
              "trainingPrefixPassed": True, "fullHistoryPrefixPassed": True, "cornerSymmetryPassed": True,
              "coverage": {"training": development.attrs.get("career_features"), "replay": full.attrs.get("career_features")},
              "runtimeChanged": False, "marketComparisonsChanged": False, "supports70PercentClaim": False, "highConfidenceValidated": False,
              "limitations": ["Outcomes were already inspected; this is one prespecified exploratory comparison, not new validation.",
                              "Event dates enable computational prior-date features but do not prove past publication or revision availability.",
                              "Observed histories are incomplete and source/promotion coverage may differ systematically between fighters.",
                              "Additional results never replace UFC round statistics or serve as extra training targets.",
                              "Known individual UFC sides remain in new observed-career history when an opponent ID is missing; no missing identity is guessed.",
                              "A positive gate permits separate review; it does not establish generalization,90%confidence,70%accuracy or profit."]}
    audit.write(output / "report.json", report)
    audit.write(output / "artifact_hashes.json", {p.name: base.digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    print(json.dumps({"stage": "complete", "promotionGate": report["promotionGate"],
                      "results": {name: value["metrics"] for name, value in results.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One fixed corrected48 plus observed-career candidate; unchanged historical replay cohorts.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.supplement.resolve(), args.output.resolve())
