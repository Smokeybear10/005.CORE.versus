import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_market_edge as edge


FEATURES = ["market_logit", "price_drift", "volume_imbalance"]
CONFIG = {
    "status": "Exploratory: historical outcomes already inspected; no untouched validation claim",
    "hypothesis": "Earlier price movement and relative contract activity may add information beyond the last market price",
    "cohort": "Frozen 499 exact-identity market pairs; preserve all rows including missing history",
    "chronology": {name: edge.CONFIG[name] for name in ("fit", "calibration", "test")},
    "features": FEATURES,
    "window": "Original acquisition start < candle end <= actual selected paired quote; no forward fill",
    "price_drift": "Final paired normalized-midpoint logit minus earliest valid common-minute logit; zero if fewer than two common observations",
    "valid_midpoint": "Both sides have finite 0 < bid <= ask < 1 and spread <= 0.05+1e-9, matching frozen benchmark; paired midpoint normalization",
    "volume_imbalance": "(red cumulative volume - blue cumulative volume) / (red + blue + 1) in the available window; not buyer/seller order flow",
    "missingVolume": "Imbalance zero if either side lacks candles or any included volume is missing, negative or nonfinite",
    "fit": "Market-logit offset plus linear adjustment, no intercept, mean logloss + 0.5 * squared coefficient norm",
    "scaling": "Uncentered RMS learned only on 127 fit rows; zero RMS becomes one",
    "control": "Same fit and calibration using only market_logit",
    "optimizer": "BFGS analytic gradient from zero, gtol=1e-9,maxiter=1000; precision-loss accepted only if finite and gradient L2 <=1e-7",
    "calibration": "Positive temperature fit on the fixed 155 calibration rows, log(T) bounds [-3,3]",
    "comparison": "Same 217 later historical bouts, Brier/logloss primary; accuracy secondary",
    "ties": "Preserve frozen market picks; fitted probabilities pick red at >=0.5",
    "selection": "One fixed feature block, no threshold, regularization or feature search",
    "runtimeChanged": False,
}


def number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def midpoint(candle: dict) -> float | None:
    values = [number(candle.get(side, {}).get("close_dollars", candle.get(side, {}).get("close")))
              for side in ("yes_bid", "yes_ask")]
    bid, ask = values
    if bid is None or ask is None or not 0 < bid <= ask < 1 or ask - bid > .05 + 1e-9:
        return None
    return (bid + ask) / 2


def window(candles: list[dict], start: int, end: int) -> dict[int, dict]:
    selected = {}
    for candle in candles:
        timestamp = candle.get("end_period_ts")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise ValueError("Non-integer candle timestamp")
        if not start < timestamp <= end:
            continue
        if timestamp in selected:
            raise ValueError("Duplicate candle timestamp")
        selected[timestamp] = candle
    return selected


def history_features(red: list[dict], blue: list[dict], start: int, observed: int,
                     market_probability: float) -> dict:
    if not 0 < market_probability < 1 or start > observed:
        raise ValueError("Invalid market probability or window")
    left, right = window(red, start, observed), window(blue, start, observed)
    points = []
    for timestamp in sorted(left.keys() & right.keys()):
        a, b = midpoint(left[timestamp]), midpoint(right[timestamp])
        if a is not None and b is not None:
            points.append((timestamp, a / (a + b)))
    if points and points[-1][0] != observed:
        raise ValueError("Selected quote absent from valid common history")
    if points and abs(points[-1][1] - market_probability) > 1e-12:
        raise ValueError("Selected quote probability differs from history")
    enough_history = len(points) >= 2
    volumes = [[number(c.get("volume_fp", c.get("volume"))) for c in side.values()] for side in (left, right)]
    volume_available = all(values and all(v is not None and v >= 0 for v in values) for values in volumes)
    totals = [sum(values) for values in volumes] if volume_available else [0.0, 0.0]
    return {
        "market_logit": float(logit(market_probability)),
        "price_drift": float(logit(market_probability) - logit(points[0][1])) if enough_history else 0.0,
        "volume_imbalance": (totals[0] - totals[1]) / (sum(totals) + 1),
        "common_observations": len(points),
        "history_span_seconds": observed - points[0][0] if points else 0,
        "drift_available": enough_history,
        "volume_available": bool(volume_available),
        "red_observed_volume": totals[0] if volume_available else None,
        "blue_observed_volume": totals[1] if volume_available else None,
        "red_observations": len(left), "blue_observations": len(right),
        "observed_start_ts": start, "observed_end_ts": observed,
    }


def design(frame: pd.DataFrame, columns: list[str], scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    offset = frame.market_logit.to_numpy(float)
    x = frame[columns].to_numpy(float) / scales
    if not np.isfinite(x).all() or not np.isfinite(offset).all():
        raise ValueError("Nonfinite model input")
    return offset, x


def predict(frame: pd.DataFrame, fitted: dict) -> np.ndarray:
    offset, x = design(frame, fitted["columns"], np.asarray(fitted["scales"]))
    return expit((offset + x @ np.asarray(fitted["coefficients"])) / fitted["temperature"])


def fit(train: pd.DataFrame, calibration: pd.DataFrame, columns: list[str]) -> dict:
    scales = np.sqrt(np.mean(train[columns].to_numpy(float) ** 2, axis=0))
    scales[scales == 0] = 1.0
    offset, x = design(train, columns, scales)
    y = train.target.to_numpy(float)

    def objective(beta):
        z = offset + x @ beta
        return (np.mean(np.logaddexp(0, z) - y * z) + .5 * beta @ beta,
                x.T @ (expit(z) - y) / len(y) + beta)

    result = minimize(objective, np.zeros(len(columns)), jac=True, method="BFGS",
                      options={"gtol": 1e-9, "maxiter": 1000})
    norm = float(np.linalg.norm(result.jac))
    finite = np.isfinite(result.fun) and np.isfinite(result.x).all() and np.isfinite(result.jac).all()
    precision_accepted = result.status == 2 and "precision loss" in str(result.message).lower() and norm <= 1e-7
    if not finite or not (result.success or precision_accepted):
        raise ValueError(f"Fixed optimizer failed: {result.message}")
    fitted = {"columns": columns, "scales": scales.tolist(), "coefficients": result.x.tolist(),
              "temperature": 1.0, "gradientL2Norm": norm, "optimizerStatus": int(result.status),
              "optimizerMessage": str(result.message), "fitCount": len(train),
              "precisionLossAccepted": bool(not result.success and precision_accepted)}
    offset, x = design(calibration, columns, scales)
    z = offset + x @ result.x
    y = calibration.target.to_numpy(float)
    temperature = minimize_scalar(
        lambda t: np.mean(np.logaddexp(0, z / math.exp(t)) - y * z / math.exp(t)),
        bounds=(-3, 3), method="bounded")
    if not temperature.success or not np.isfinite(temperature.fun):
        raise ValueError("Temperature fit failed")
    fitted.update(temperature=math.exp(temperature.x), calibrationCount=len(calibration))
    return fitted


def extract(quoted: pd.DataFrame) -> pd.DataFrame:
    manifests = {version: {r["ticker"]: r for r in edge.read(directory / "quote-manifest.json")["files"]}
                 for version, directory in [("v1", edge.OLD / "source"), ("v2", edge.COVERAGE)]}
    rows = []
    for row in quoted.itertuples(index=False):
        directory = edge.OLD / "source" if row.quote_source_version == "v1" else edge.COVERAGE
        files = manifests[row.quote_source_version]
        raw = []
        for ticker in (row.red_ticker, row.blue_ticker):
            receipt = files[ticker]
            path = directory / receipt["file"]
            if edge.digest(path) != receipt["sha256"]:
                raise ValueError("Candle source changed")
            response = edge.read(path)
            if response.get("ticker") != ticker:
                raise ValueError("Candle ticker mismatch")
            raw.append(response["candlesticks"])
        start = int(datetime.fromisoformat(row.cutoff).timestamp()) - 3600
        observed = int(datetime.fromisoformat(row.quote_observed_at).timestamp())
        values = history_features(*raw, start, observed, row.p_market)
        reverse = history_features(raw[1], raw[0], start, observed, 1 - row.p_market)
        if not np.allclose([values[c] for c in FEATURES], [-reverse[c] for c in FEATURES], atol=1e-12, rtol=0):
            raise ValueError("Features are not corner symmetric")
        rows.append(values)
    return pd.concat([quoted.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def run(source: Path, output: Path) -> None:
    inputs = edge.initial_inputs(source) | {Path(__file__), Path(edge.__file__)}
    hashes = {str(p): edge.digest(p) for p in sorted(inputs)}
    output.mkdir(parents=True, exist_ok=False)
    edge.write(output / "protocol.json", {"createdAt": edge.now(), "config": CONFIG, "inputHashes": hashes,
               "documentation": ["https://docs.kalshi.com/api-reference/market/get-market-candlesticks",
                                 "https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks"]})
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print('Protocol frozen', flush=True)
    fights, _, _ = edge.verify_source(source)
    quoted, exclusions = edge.quotes(fights, output, inputs)
    if set(map(str, inputs)) - set(hashes):
        raise ValueError("Unpinned input discovered")
    frame = extract(quoted)
    frame.to_csv(output / "features.csv", index=False)
    cohorts = edge.split(frame)
    fitted = {name: fit(cohorts["fit"], cohorts["calibration"], columns)
              for name, columns in [("market_control", FEATURES[:1]), ("market_history", FEATURES)]}
    edge.write(output / "fitted-models.json", fitted)
    seal = {p.name: edge.digest(p) for p in output.iterdir() if p.is_file()}
    edge.write(output / "pre-test-manifest.json", {"createdAt": edge.now(), "files": seal,
               "candidateTestPredictionsComputed": False, "historicalTargetsPreviouslyExposed": True})
    print('Models frozen; scoring fixed cohorts', flush=True)
    for name, model in fitted.items():
        frame[name] = predict(frame, model)
        frame[f"pick_{name}"] = frame[name] >= .5
        reverse = frame.copy()
        reverse[FEATURES] *= -1
        if not np.allclose(frame[name] + predict(reverse, model), 1, atol=1e-12, rtol=0):
            raise ValueError("Predictions are not corner symmetric")
    predictors = {"market": ("p_market", "pick_market"),
                  **{name: (name, f"pick_{name}") for name in fitted}}
    helper = edge.frozen_helpers()
    results = {}
    for name, part in edge.split(frame).items():
        results[name] = {
            "role": "Later historical exploratory evaluation" if name == "test" else "Fitting/calibration data; not generalization evidence",
            "metrics": {model: edge.metrics(part, *fields) for model, fields in predictors.items()},
            "paired": helper.bootstrap(part, predictors, [("market_history", "market"),
                                                          ("market_history", "market_control")]),
            "coverage": {"rows": len(part), "driftAvailable": int(part.drift_available.sum()),
                         "volumeAvailable": int(part.volume_available.sum()),
                         "nonzeroDrift": int((part.price_drift != 0).sum()),
                         "nonzeroVolumeImbalance": int((part.volume_imbalance != 0).sum()),
                         "observedSpanSeconds": part.history_span_seconds.describe().to_dict()},
        }
    frame.to_csv(output / "predictions.csv", index=False)
    for filename, checksum in seal.items():
        if edge.digest(output / filename) != checksum:
            raise ValueError("Pretest artifact changed")
    if any(edge.digest(Path(p)) != checksum for p, checksum in hashes.items()):
        raise ValueError("Input changed")
    report = {"createdAt": edge.now(), "status": CONFIG["status"], "results": results,
              "fitted": fitted, "exclusions": exclusions, "allInputHashesVerified": True,
              "symmetryVerified": True, "runtimeChanged": False, "promoted": False,
              "limitations": ["Only127 fitting bouts; all historical results previously inspected",
                  "Quote windows are variably short; missing history stays in the denominator",
                  "Candle end times are observation bounds, not last quote-update times",
                  "Relative volume is contract activity, not signed order flow or independent consensus",
                  "Binary decisive-result subset; no draw/NC or executable-profit evidence",
                  "Bootstrap intervals do not adjust for repeated experimentation"]}
    edge.write(output / "report.json", report)
    edge.write(output / "artifact-hashes.json", {p.name: edge.digest(p) for p in output.iterdir() if p.is_file()})
    print({name: {k: value[k] for k in ("count", "correct", "accuracy", "brier", "logLoss")}
           for name, value in results["test"]["metrics"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed market-history residual experiment")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve())
