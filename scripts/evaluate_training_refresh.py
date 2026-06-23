import argparse
import importlib.metadata
import math
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.base import clone
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_challenger as challenger
from scripts import evaluate_identity_repair as audit
from scripts import evaluate_market_edge as edge
from scripts import evaluate_predictors as features
from scripts import evaluate_research as base


SOURCE = edge.DATA / "identity-repaired-source-v1"
PRIOR = edge.DATA / "identity-repair-evaluation-v1"
MODEL = PRIOR / "provisional_challenger.joblib"
MARKET = edge.DATA / "market-edge-v1/verified-quotes.csv"
PINS = {
    SOURCE / "manifest.json": "3fe41b67d9e987cbded3693a631bfea9a4c03b20f60c944fcdf20f657a86e8a8",
    MODEL: "6fde3bc72e4788c944efa65ad8a911c2e18d840404a337e2cd09559a8530a924",
    PRIOR / "development_features.parquet": "68835656bf48acf69af6d3d8488f8bee59ebff0a08a57dfce98449c669c5b6fe",
    PRIOR / "predictions.csv": "bc7ef73bdebe1775dda9ee7219a86eb4a50e48299a2d94ab82937aae7d0adb00",
    MARKET: "88520cca5cbfa51277d130d8ad1fceada6f976d97f8723462ed44b2e2a107a86",
    **{edge.ROOT / "scripts" / name: checksum for name, checksum in audit.CODE.items()},
}
CONFIG = {
    "candidate": "refreshed_corrected_GB48",
    "hypothesis": "Refit the deployed learner on newer strictly prior fights, separating this from recent temperature recalibration",
    "fitFromInclusive": "2010-01-01", "fitEndExclusive": "2026-01-01",
    "calibrationEndExclusive": "2026-05-10", "testEndExclusive": "2026-11-01",
    "expectedCounts": {"fit": 7084, "calibration": 189, "test": 285, "matchedMarket": 217},
    "learner": "Exact clone of frozen corrected48 pipeline: GB100/depth3/learning_rate0.05/seed42; same corner augmentation and train-only preprocessing",
    "features": "Same48 columns and ordering; no new predictors, source repairs, row filters or parameter search",
    "controls": ["Frozen corrected48 with original temperature", "Same frozen classifier, temperature recalibrated on the identical189 rows"],
    "temperature": "Positive temperature minimizing calibration logloss, log(T) bounds[-3,3]; no classifier fit on calibration rows",
    "evaluation": "All285 fixed decisive bouts and exact217 timestamp-qualified market pairs; no confidence filtering",
    "featureTiming": "Strictly earlier-date history; rebuild all217 market feature rows from history before earliest actual quote UTC-date per card",
    "primaryMetrics": ["Brier", "logLoss"], "secondaryMetrics": ["accuracy"],
    "primaryComparison": "Refresh versus recalibrated old classifier; also report frozen current classifier and matched market",
    "bootstrap": {"eventReplicates": 2000, "seed": 945},
    "selection": "One fixed refit. Same date windows as earlier uncorrected blend refresh; no threshold/window/learner search",
    "status": "Exploratory previously examined historical replay; not untouched validation",
    "runtimeChanged": False, "promotionAllowed": False,
}
MODEL_NAMES = ["frozen_current", "recalibrated_current", "refreshed"]


def versions() -> dict:
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "fastparquet"]}}


def split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dates = pd.to_datetime(frame.date)
    parts = {
        "fit": frame[(dates >= CONFIG["fitFromInclusive"]) & (dates < CONFIG["fitEndExclusive"])],
        "calibration": frame[(dates >= CONFIG["fitEndExclusive"]) & (dates < CONFIG["calibrationEndExclusive"])],
        "test": frame[(dates >= CONFIG["calibrationEndExclusive"]) & (dates < CONFIG["testEndExclusive"])],
    }
    if frame.fight_id.duplicated().any():
        raise ValueError("Duplicate fight identity")
    for first, second in [("fit", "calibration"), ("fit", "test"), ("calibration", "test")]:
        if set(parts[first].event_id) & set(parts[second].event_id):
            raise ValueError("An event crosses chronological splits")
    return parts


def calibrate(raw: np.ndarray, target: np.ndarray) -> float:
    if len(raw) != len(target) or not np.isfinite(raw).all() or not ((raw > 0) & (raw < 1)).all():
        raise ValueError("Invalid calibration probabilities")
    fitted = minimize_scalar(lambda log_t: log_loss(target, base.apply_temperature(raw, math.exp(log_t)), labels=[0, 1]),
                             bounds=(-3, 3), method="bounded")
    if not fitted.success or not np.isfinite(fitted.fun):
        raise ValueError("Fixed temperature calibration failed")
    temperature = math.exp(fitted.x)
    if not np.array_equal(raw >= .5, base.apply_temperature(raw, temperature) >= .5):
        raise ValueError("Positive temperature changed winner picks")
    return temperature


def quote_history(fights: pd.DataFrame, observations: pd.Series) -> tuple[pd.DataFrame, pd.Timestamp]:
    parsed = [pd.Timestamp(value) for value in observations]
    if not parsed or any(pd.isna(value) or value.tzinfo is None for value in parsed):
        raise ValueError("Market observation requires an explicit timezone")
    cutoff = min(value.tz_convert("UTC") for value in parsed).floor("D").tz_localize(None)
    return fights[fights.date < cutoff].copy(), cutoff


def verify_market_features(full: pd.DataFrame, market: pd.DataFrame, columns: list[str]) -> list[dict]:
    fights = pd.read_parquet(SOURCE / "fights.parquet")
    rounds = pd.read_parquet(SOURCE / "rounds.parquet")
    fighters = pd.read_parquet(SOURCE / "fighters.parquet")
    dobs = pd.to_datetime(fighters.set_index("fighter_id").dob, errors="coerce").to_dict()
    source_rows, expected_rows = fights.set_index("fight_id"), full.set_index("fight_id")
    checks = []
    for date, group in market.groupby("date", sort=True):
        date = pd.Timestamp(date)
        history_rows, cutoff = quote_history(fights, group.quote_observed_at)
        prior_rounds = rounds[rounds.fight_id.isin(history_rows.fight_id)]
        totals, _ = base.prepare_round_stats(prior_rounds)
        history, elo = base.history_before(history_rows, totals, date)
        rebuilt = []
        for row in source_rows.loc[group.fight_id].itertuples():
            red, blue = (base.fighter_state(history[fid], elo[fid], date) for fid in (row.red_fighter_id, row.blue_fighter_id))
            red_context, blue_context = (challenger.context_state(history[fid], date, dobs.get(fid, pd.NaT))
                                         for fid in (row.red_fighter_id, row.blue_fighter_id))
            rebuilt.append({**base.matchup_features(red, blue, row.weightclass, int(row.num_rounds), bool(row.is_title_bout)),
                            **challenger.matchup_context(red_context, blue_context, elo[row.red_fighter_id], elo[row.blue_fighter_id])})
        pd.testing.assert_frame_equal(pd.DataFrame(rebuilt)[columns].reset_index(drop=True),
                                      expected_rows.loc[group.fight_id, columns].reset_index(drop=True),
                                      check_exact=True, check_dtype=False)
        checks.append({"eventDate": str(date.date()), "rows": len(group), "strictHistoryCutoffExclusive": str(cutoff.date()),
                       "historyRows": len(history_rows), "original48ExactlyMatchesReplay": True})
    return checks


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {output}")
    for path, checksum in PINS.items():
        if base.digest(path) != checksum:
            raise ValueError(f"Pinned input changed: {path}")
    inputs = set(PINS) | {Path(__file__), Path(audit.__file__), Path(edge.__file__), edge.COVERAGE / "score_coverage.py",
                         edge.ROOT / "tests/test_training_refresh.py"}
    for directory in [SOURCE, edge.DATA / "inference"]:
        manifest = directory / "manifest.json"
        inputs.add(manifest)
        for filename, checksum in edge.read(manifest)["files"].items():
            path = directory / filename
            if base.digest(path) != checksum:
                raise ValueError(f"Frozen source changed: {path}")
            inputs.add(path)
    hashes = {str(path): base.digest(path) for path in sorted(inputs)}
    dependencies = versions()
    artifact = joblib.load(MODEL)
    original = artifact["model"]
    columns = artifact["feature_columns"]
    if len(columns) != 48 or artifact["selected_model"] != "gradient_boosting":
        raise ValueError("Current model is not the fixed48 gradient booster")
    output.mkdir(parents=True, exist_ok=False)
    audit.write(output / "protocol.json", {"createdAt": audit.now(), "config": CONFIG, "inputHashes": hashes,
                "versions": dependencies, "learnerParameters": original.named_steps["model"].get_params(),
                "oldTemperature": artifact["temperature"]})
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print("Protocol frozen; reconstructing fit/calibration features", flush=True)
    early, columns48, _ = features.load_features(SOURCE, CONFIG["calibrationEndExclusive"])
    if columns48 != columns:
        raise ValueError("Feature columns or ordering changed")
    cached = pd.read_parquet(PRIOR / "development_features.parquet")
    pd.testing.assert_frame_equal(early[early.date < "2025-01-01"].reset_index(drop=True), cached, check_exact=True)
    parts = split(early)
    train, calibration = parts["fit"], parts["calibration"]
    if len(train) != 7084 or len(calibration) != 189 or len(parts["test"]):
        raise ValueError("Fixed7084/189 fitting periods changed")
    early.to_parquet(output / "fit_and_calibration_features.parquet", index=False)
    audit.write(output / "fit_inputs.json", {name: {"count": len(part), "events": int(part.event_id.nunique()),
                "from": str(part.date.min().date()), "through": str(part.date.max().date()),
                "fightIdsSHA256": audit.id_hash(part.fight_id)} for name, part in [("fit", train), ("calibration", calibration)]})
    raw_current = base.symmetric_probability(original, calibration[columns])
    recalibrated_temperature = calibrate(raw_current, calibration.target.to_numpy())
    refreshed = clone(original)
    X = pd.concat([train[columns], base.swapped(train[columns])], ignore_index=True)
    y = np.concatenate([train.target.to_numpy(), 1 - train.target.to_numpy()])
    refreshed.fit(X, y)
    if refreshed.named_steps["model"].get_params() != original.named_steps["model"].get_params():
        raise ValueError("Refresh changed learner parameters")
    raw_refreshed = base.symmetric_probability(refreshed, calibration[columns])
    refreshed_temperature = calibrate(raw_refreshed, calibration.target.to_numpy())
    models = {"frozen_current": (original, artifact["temperature"]),
              "recalibrated_current": (original, recalibrated_temperature), "refreshed": (refreshed, refreshed_temperature)}
    calibration_rows = calibration[["fight_id", "event_id", "date", "target"]].copy()
    receipts = {}
    for name, (model, temperature) in models.items():
        calibration_rows[name] = audit.probability([model], temperature, calibration[columns])
        filename = name + ".joblib"
        joblib.dump({"model": model, "feature_columns": columns, "temperature": temperature,
                     "protocolSHA256": base.digest(output / "protocol.json"), "researchOnly": True}, output / filename, compress=3)
        receipts[name] = {"artifact": filename, "sha256": base.digest(output / filename), "temperature": temperature,
                          "calibrationMetricsInSample": audit.metrics(calibration_rows, name)}
    calibration_rows.to_csv(output / "calibration_predictions.csv", index=False)
    audit.write(output / "fit_receipts.json", receipts)
    seal = {path.name: base.digest(path) for path in output.iterdir() if path.is_file()}
    audit.write(output / "pre-test-manifest.json", {"createdAt": audit.now(), "files": seal,
                "testPredictionsComputed": False, "historicalOutcomesPreviouslyExamined": True})
    print("Models and temperatures frozen; reconstructing replay", flush=True)
    full, full_columns, _ = features.load_features(SOURCE, CONFIG["testEndExclusive"])
    if full_columns != columns:
        raise ValueError("Replay feature columns changed")
    pd.testing.assert_frame_equal(full[full.date < CONFIG["calibrationEndExclusive"]].reset_index(drop=True), early, check_exact=True)
    test = split(full)["test"].copy()
    if len(test) != 285 or audit.id_hash(test.fight_id) != "93c0f37bd82803fab5ef62061e8dc35bf17ea0c11d1d63158e4fdbd53e84e65b":
        raise ValueError("Fixed285 replay changed")
    old = pd.read_csv(PRIOR / "predictions.csv", float_precision="round_trip").set_index("fight_id").loc[test.fight_id]
    for side in ["red", "blue"]:
        if not np.array_equal(test[f"{side}_fighter_id"].to_numpy(), old[f"corrected_{side}_fighter_id"].to_numpy()):
            raise ValueError("Replay fighter identity changed")
    if not np.array_equal(test.target, old.target) or not np.array_equal(test.event_id, old.event_id):
        raise ValueError("Replay outcome/event changed")
    test.to_parquet(output / "replay_features.parquet", index=False)
    market = pd.read_csv(MARKET, float_precision="round_trip")
    market = market[market.date >= "2026-05-16"].copy()
    if len(market) != 217 or not set(market.fight_id) <= set(test.fight_id):
        raise ValueError("Fixed217 market cohort changed")
    if pd.to_datetime(market.quote_observed_at, utc=True).min().tz_localize(None) <= calibration.date.max():
        raise ValueError("Calibration overlaps earliest market observation")
    aligned = test.set_index("fight_id").loc[market.fight_id]
    for column in ["red_fighter_id", "blue_fighter_id", "event_id", "target"]:
        if not np.array_equal(aligned[column].to_numpy(), market[column].to_numpy()):
            raise ValueError("Market/source identity or target mismatch")
    if not np.array_equal(aligned.date.to_numpy(), pd.to_datetime(market.date).to_numpy()):
        raise ValueError("Market event date mismatch")
    timing = verify_market_features(full, market, columns)
    audit.write(output / "market-timing-checks.json", timing)
    rows = test[["fight_id", "event_id", "date", "target", "red_fighter_id", "blue_fighter_id"]].copy()
    for name, (model, temperature) in models.items():
        rows[name] = audit.probability([model], temperature, test[columns])
        rows[f"pick_{name}"] = rows[name] >= .5
    if not np.allclose(rows.frozen_current, old.original48_after.to_numpy(), atol=1e-12, rtol=0):
        raise ValueError("Frozen current model does not reproduce corrected archive")
    if not np.array_equal(rows.pick_frozen_current, rows.pick_recalibrated_current):
        raise ValueError("Temperature-only control changed picks")
    matched = market.merge(rows[["fight_id", *MODEL_NAMES, *["pick_" + name for name in MODEL_NAMES]]], on="fight_id", validate="one_to_one")
    predictors = {name: (name, "pick_" + name) for name in MODEL_NAMES}
    pairs = [("refreshed", "frozen_current"), ("refreshed", "recalibrated_current"), ("recalibrated_current", "frozen_current")]
    helper = edge.frozen_helpers()
    results = {"all285": {"metrics": {name: edge.metrics(rows, *fields) for name, fields in predictors.items()},
                          "paired": helper.bootstrap(rows, predictors, pairs)}}
    market_predictors = {**predictors, "market": ("p_market", "pick_market")}
    results["matched217"] = {"metrics": {name: edge.metrics(matched, *fields) for name, fields in market_predictors.items()},
                             "paired": helper.bootstrap(matched, market_predictors, pairs + [("refreshed", "market")])}
    rows.to_csv(output / "predictions.csv", index=False)
    matched.to_csv(output / "market-predictions.csv", index=False)
    if versions() != dependencies or any(base.digest(Path(path)) != checksum for path, checksum in hashes.items()):
        raise ValueError("Frozen code/source/dependencies changed during run")
    if any(base.digest(output / name) != checksum for name, checksum in seal.items()):
        raise ValueError("Pre-test artifact changed")
    audit.write(output / "report.json", {"createdAt": audit.now(), "status": CONFIG["status"], "results": results,
                "fits": receipts, "sourceHashesUnchanged": True, "cachedPrefixExactlyReproduced": True,
                "frozenCurrentArchiveParity": True, "all217MarketFeatureRowsRebuiltExactly": True,
                "runtimeChanged": False, "promoted": False,
                "limitations": ["Previously consumed historical outcomes; no independent future validation.",
                    "One fixed refresh; cannot attribute all probability changes to training without comparing recalibrated-old control.",
                    "Retrospective DOB/static profiles and result revisions retain their original source limitations.",
                    "No automatic promotion or executable-profit conclusion; bootstrap does not correct repeated experimentation."]})
    audit.write(output / "artifact-hashes.json", {path.name: base.digest(path) for path in output.iterdir() if path.is_file()})
    print({cohort: {name: {key: score[key] for key in ["count", "correct", "accuracy", "brier", "logLoss"]}
                   for name, score in result["metrics"].items()} for cohort, result in results.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One fixed corrected48 training refresh with a temperature-only control")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.output.resolve())
