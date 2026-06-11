import argparse
import hashlib
import importlib.util
import json
import math
import platform
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_challenger as challenger
from scripts import evaluate_identity_repair as audit
from scripts import evaluate_research as base


DATA = ROOT / "data/evaluation"
OLD = DATA / "market-benchmark-v1"
COVERAGE = DATA / "market-coverage-v2"
BASELINE = DATA / "identity-repair-evaluation-v1"
MODEL = BASELINE / "provisional_challenger.joblib"
MODEL_SHA = "6fde3bc72e4788c944efa65ad8a911c2e18d840404a337e2cd09559a8530a924"
AMENDMENT = DATA / "market-edge-v1/numerical-recovery-amendment.json"
STALE_IDS = {"20d74ed23d3e9b3a", "fc9168fd102e2e66"}
CONFIG = {
    "schemaVersion": 1,
    "status": "Exploratory previously exposed historical outcomes; not untouched validation",
    "coverage": "Frozen coverage_v2, minus two market rows with stale Bruno Silva fighter IDs; no remapping",
    "expectedMatched": 499,
    "fit": {"fromInclusive": "2025-01-01", "endExclusive": "2026-01-01", "rows": 127, "events": 20},
    "calibration": {"fromInclusive": "2026-01-01", "endExclusive": "2026-05-16", "rows": 155, "events": 15},
    "test": {"fromInclusive": "2026-05-16", "endExclusive": "2026-11-01", "rows": 217, "events": 23},
    "residual": {
        "formula": "logit(q) = logit(market) + a*logit(market) + b*logit(corrected48)",
        "fitObjective": "mean binary logloss + 0.5*lambda*(a*a+b*b)",
        "lambda": 1.0,
        "intercept": False,
        "initialCoefficients": [0.0, 0.0],
        "probabilityClip": 1e-6,
        "optimizer": "BFGS with analytic gradient; gtol=1e-9, maxiter=1000; no restart/search",
        "numericalAcceptance": "Also accept BFGS precision-loss status2 only when objective, coefficients and gradient are finite and analytic L2 gradient norm<=1e-7. Withlambda1, parameter L2 error<=gradient norm and objectivegap<=norm²/2.",
    },
    "marketOnlyControl": "Same objective with b fixed to zero; estimate a on the same 127 rows",
    "temperature": "One positive T per fitted adjustment, minimize calibration logloss with log(T) bounds [-3,3]",
    "fitOrder": "Freeze both fitted adjustments and calibration temperatures before any test predictions or metrics",
    "standalone": "Corrected48 frozen through 2022 with calibration2023-24; no refit",
    "primaryComparison": "Residual versus unaltered market and identically trained market-only adjustment on 217 test bouts",
    "primaryMetrics": ["brier", "logLoss"],
    "secondaryMetrics": ["accuracy", "calibrationBins", "meanProbability", "observedRedWinRate"],
    "bootstrap": {"seed": 945, "replicates": 2000, "cluster": "UFC event_id", "interval": 0.95},
    "tieRules": "Preserve frozen market pick_market for exact midpoint ties; fitted adjustments and corrected48 pick red at p>=0.5",
    "cutoffSafety": "Exact ticker/rules/registry calendar date D; cutoff D-1 at00:00UTC. Even D at00:00 inUTC+14 is10h later. Actual per-card bell timestamps are not archived; the bound relies on correct calendar identity.",
    "fallback": "Full985 and full285 use corrected48 where market is absent; these are not market-alone comparisons",
    "selection": "No alternative penalty, formula, threshold, quote eligibility or model selection",
    "runtimeChanged": False,
    "modelPromotion": False,
    "profitEstablished": False,
}
DOCS = [
    {"url": "https://docs.kalshi.com/getting_started/historical_data",
     "fact": "Historical markets and candles move to historical endpoints according to a published settlement-time partition cutoff."},
    {"url": "https://docs.kalshi.com/api-reference/market/get-market-candlesticks",
     "fact": "Minute candle end timestamps bound the observation; OHLC bid/ask fields and interval volume do not certify last quote-update age or executable depth."},
    {"url": "https://kalshi.com/docs/kalshi-fee-schedule.pdf",
     "fact": "Reviewed current July7,2026 schedule: general taker fees depend on price, contract count and multiplier. It is not a verified historical fee schedule for every archived quote."},
    {"url": "https://docs.polymarket.com/api-reference/markets/get-prices-history",
     "fact": "Documented token price time series has timestamp and price; it does not itself supply historical bid/ask depth. No Polymarket records are added here."},
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def frozen_helpers():
    path = COVERAGE / "score_coverage.py"
    spec = importlib.util.spec_from_file_location("frozen_coverage_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def initial_inputs(source: Path) -> set[Path]:
    paths = {Path(__file__), Path(base.__file__), Path(challenger.__file__), Path(audit.__file__),
             ROOT / "scripts/evaluate_predictors.py", ROOT / "scripts/predictor_features.py", MODEL,
             BASELINE / "predictions.csv", COVERAGE / "score_coverage.py",
             COVERAGE / "results/predictions.csv", COVERAGE / "results/manifest.json",
             COVERAGE / "qualified-quotes.json", COVERAGE / "eligible-events.json",
             COVERAGE / "accepted-aliases.json", COVERAGE / "quote-manifest.json",
             COVERAGE / "quote-evaluation-protocol.json", COVERAGE / "completion-inventory.json",
             OLD / "qualified-quotes.json", OLD / "predictions.csv", OLD / "manifest.json",
             OLD / "source/quote-manifest.json", OLD / "source/protocol.json",
             OLD / "source/supplementary-protocol.json", source / "manifest.json"}
    paths.update(source / filename for filename in read(source / "manifest.json")["files"])
    if AMENDMENT.exists():
        paths.update([AMENDMENT, AMENDMENT.parent / "protocol.json", AMENDMENT.parent / "evaluate_market_edge.py",
                      AMENDMENT.parent / "failed-attempt-receipt.json"])
    paths.update(OLD / "source" / name for name in ["historical-markets-1.json", "historical-markets-2.json", "current-settled-markets.json"])
    for directory in [OLD / "source", COVERAGE]:
        paths.update(directory / row["file"] for row in read(directory / "quote-manifest.json")["files"] if row.get("file"))
    return paths


def verify_source(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if digest(MODEL) != MODEL_SHA:
        raise ValueError("Corrected48 model changed")
    for filename, checksum in read(source / "manifest.json")["files"].items():
        if digest(source / filename) != checksum:
            raise ValueError(f"Corrected source changed: {filename}")
    frames = tuple(pd.read_parquet(source / f"{name}.parquet") for name in ("fights", "rounds", "fighters"))
    if frames[0].fight_id.duplicated().any() or frames[2].fighter_id.duplicated().any():
        raise ValueError("Source identity duplication")
    model = joblib.load(MODEL)
    old_source = Path(model["manifest"]["source"])
    for filename in ("fights.parquet", "rounds.parquet", "fighters.parquet"):
        if digest(source / filename) != model["manifest"]["hashes"][str(old_source / filename)]:
            raise ValueError("Source does not match fitted corrected48")
    return frames


def quotes(fights: pd.DataFrame, output: Path, inputs: set[Path]) -> tuple[pd.DataFrame, list]:
    helper = frozen_helpers()
    for filename, checksum in read(OLD / "manifest.json")["files"].items():
        if digest(OLD / filename) != checksum:
            raise ValueError(f"Original market archive changed: {filename}")
    for filename, receipt in read(COVERAGE / "completion-inventory.json")["files"].items():
        if digest(COVERAGE / filename) != receipt["sha256"]:
            raise ValueError(f"Coverage archive changed: {filename}")
    full = pd.read_csv(COVERAGE / "results/predictions.csv")
    full = full[full.coverage_version == "coverage_v2"].set_index("fight_id")
    if len(full) != 985 or full.index.duplicated().any() or int(full.has_market.sum()) != 501:
        raise ValueError("Frozen coverage membership changed")
    registry = fights.set_index("fight_id")
    metadata = helper.metadata_index(OLD / "source")
    sources = [(OLD / "qualified-quotes.json", OLD / "source"), (COVERAGE / "qualified-quotes.json", COVERAGE)]
    qualified = {}
    receipts = {}
    for path, directory in sources:
        for item in read(path):
            if item["fightId"] in qualified:
                raise ValueError("Quote appears in both source versions")
            qualified[item["fightId"]] = (item, directory)
        manifest = read(directory / "quote-manifest.json")
        if directory == COVERAGE and (not manifest.get("completed") or manifest.get("expectedContracts") != 116):
            raise ValueError("Coverage quote acquisition is incomplete")
        receipts[directory] = {row["ticker"]: row for row in manifest["files"]}
    aliases = {helper.normalized_name(row["alias"]): row["fighterId"] for row in read(COVERAGE / "accepted-aliases.json")}
    rows, exclusions, verified = [], [], []
    for fight_id, saved in full[full.has_market].sort_index().iterrows():
        quote, directory = qualified[fight_id]
        fight = registry.loc[fight_id]
        sides = quote["markets"]
        actual_ids = {fight.red_fighter_id, fight.blue_fighter_id}
        quoted_ids = {side["fighterId"] for side in sides}
        if actual_ids != quoted_ids:
            exclusions.append({"fightId": fight_id, "reason": "quote_fighter_ids_do_not_match_corrected_registry",
                               "marketIds": sorted(quoted_ids), "correctedIds": sorted(actual_ids)})
            continue
        if len(sides) != 2 or len(actual_ids) != 2 or not all(re.fullmatch(r"[0-9a-f]{16}", value) for value in actual_ids):
            raise ValueError("Invalid quote identities")
        date = str(pd.Timestamp(fight.date).date())
        cutoff = datetime.fromisoformat(date).replace(tzinfo=timezone.utc) - timedelta(days=1)
        if quote["fightDate"] != date or helper.utc(quote["cutoff"]) != cutoff:
            raise ValueError("Quote cutoff or date changed")
        token = re.match(r"KXUFCFIGHT-(\d{2}[A-Z]{3}\d{2})", quote["eventTicker"])
        if not token or datetime.strptime(token[1], "%y%b%d").strftime("%Y-%m-%d") != date:
            raise ValueError("Contract ticker date differs from fight")
        for side in sides:
            receipt = receipts[directory][side["ticker"]]
            raw_path = directory / receipt["file"]
            query = parse_qs(urlparse(receipt["url"]).query)
            expected_query = {"start_ts": [str(int(cutoff.timestamp()) - 3600)],
                              "end_ts": [str(int(cutoff.timestamp()))], "period_interval": ["1"]}
            if receipt.get("status") != 200 or receipt.get("error") or digest(raw_path) != receipt["sha256"] or query != expected_query:
                raise ValueError("Raw candle receipt, hash or query mismatch")
            raw = metadata[side["ticker"]]
            corner = "red" if side["fighterId"] == fight.red_fighter_id else "blue"
            name = helper.normalized_name(raw["yes_sub_title"])
            if name != helper.normalized_name(fight[f"{corner}_name"]) and aliases.get(name) != side["fighterId"]:
                raise ValueError("Market name not supported by exact name or frozen alias")
            if raw["event_ticker"] != quote["eventTicker"] or helper.rules_date(raw["rules_primary"]) != date:
                raise ValueError("Market rule identity or date mismatch")
            if helper.utc(raw["open_time"]) > cutoff:
                raise ValueError("Market opened after fixed cutoff")
        selected = helper.latest_quote(sides, cutoff, directory, inputs)
        if selected is None:
            raise ValueError("Archived qualified quote has no valid simultaneous raw candles")
        timestamp, prices, volumes = selected
        if helper.utc(quote["quoteEndTimestamp"]).timestamp() != timestamp:
            raise ValueError("Stored quote is not latest eligible common minute")
        normalized = []
        for side, (bid, ask), volume in zip(sides, prices, volumes):
            if abs(side["bid"] - bid) > 1e-12 or abs(side["ask"] - ask) > 1e-12 or abs(side["windowVolume"] - volume) > 1e-8:
                raise ValueError("Selected quote differs from raw prices or volume")
            if helper.utc(metadata[side["ticker"]]["open_time"]).timestamp() > timestamp:
                raise ValueError("Selected quote precedes contract opening")
            normalized.append((bid + ask) / 2)
        red_index = [side["fighterId"] for side in sides].index(fight.red_fighter_id)
        p_market = normalized[red_index] / sum(normalized)
        if abs(p_market - saved.p_market) > 1e-12:
            raise ValueError("Market probability changed")
        settlement = [metadata[side["ticker"]] for side in sides]
        if sorted(row.get("result", "") for row in settlement) != ["no", "yes"]:
            raise ValueError("Nondecisive or scalar settlement entered matched cohort")
        if any(float(row.get("settlement_value_dollars", -1)) != int(row["result"] == "yes") for row in settlement):
            raise ValueError("Settlement value disagrees with binary outcome")
        winning_id = sides[[row["result"] for row in settlement].index("yes")]["fighterId"]
        target = int(winning_id == fight.red_fighter_id)
        if target != int(saved.target) or str(fight.winner).lower() != ("red" if target else "blue"):
            raise ValueError("Market settlement, archived target and corrected source disagree")
        row = {"fight_id": fight_id, "event_id": fight.event_id, "date": date,
               "red_fighter_id": fight.red_fighter_id, "blue_fighter_id": fight.blue_fighter_id,
               "red_name": fight.red_name, "blue_name": fight.blue_name, "target": target,
               "event_ticker": quote["eventTicker"], "cutoff": cutoff.isoformat(),
               "quote_observed_at": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
               "observation_age_seconds": cutoff.timestamp() - timestamp, "quote_update_age_known": False,
               "p_market": p_market, "pick_market": bool(saved.pick_market), "midpoint_sum": sum(normalized),
               "quote_source_version": "v1" if directory == OLD / "source" else "v2"}
        for label, index in [("red", red_index), ("blue", 1 - red_index)]:
            side = sides[index]
            row.update({f"{label}_ticker": side["ticker"], f"{label}_bid": prices[index][0],
                        f"{label}_ask": prices[index][1], f"{label}_window_volume": volumes[index],
                        f"{label}_retrieved_at": receipts[directory][side["ticker"]]["retrievedAt"]})
        rows.append(row)
        verified.append({"fightId": fight_id, "rawCandleFiles": [str(directory / side["rawCandleFile"]) for side in sides],
                         "latestCommonMinuteVerified": True, "exactCurrentIdentity": True, "binarySettlementReconciled": True})
    if {row["fightId"] for row in exclusions} != STALE_IDS or len(rows) != 499:
        raise ValueError("Declared exact-ID cohort changed")
    matched = pd.DataFrame(rows).sort_values(["date", "event_id", "fight_id"]).reset_index(drop=True)
    matched.to_csv(output / "verified-quotes.csv", index=False)
    write(output / "quote-verification.json", verified)
    write(output / "exclusions.json", exclusions)
    return matched, exclusions


def reconstruct(fights: pd.DataFrame, rounds: pd.DataFrame, fighters: pd.DataFrame) -> pd.DataFrame:
    totals, _ = base.prepare_round_stats(rounds)
    return challenger.add_context(base.reconstruct(fights, totals), fights, fighters)


def logits(frame: pd.DataFrame, model: bool) -> tuple[np.ndarray, np.ndarray]:
    epsilon = CONFIG["residual"]["probabilityClip"]
    market = logit(np.clip(frame.p_market.to_numpy(float), epsilon, 1 - epsilon))
    standalone = logit(np.clip(frame.corrected48.to_numpy(float), epsilon, 1 - epsilon))
    return market, np.column_stack([market, standalone]) if model else market[:, None]


def fitted_probability(frame: pd.DataFrame, fitted: dict) -> np.ndarray:
    offset, X = logits(frame, fitted["usesCorrected48"])
    return expit((offset + X @ np.asarray(fitted["coefficients"])) / fitted["temperature"])


def fit_adjustment(train: pd.DataFrame, calibration: pd.DataFrame, model: bool) -> dict:
    offset, X = logits(train, model)
    y = train.target.to_numpy(float)
    strength = CONFIG["residual"]["lambda"]

    def objective(beta):
        z = offset + X @ beta
        value = np.mean(np.logaddexp(0, z) - y * z) + .5 * strength * np.dot(beta, beta)
        gradient = X.T @ (expit(z) - y) / len(y) + strength * beta
        return value, gradient

    result = minimize(objective, np.zeros(X.shape[1]), jac=True, method="BFGS", options={"gtol": 1e-9, "maxiter": 1000})
    gradient_norm = float(np.linalg.norm(result.jac))
    finite = np.isfinite(result.fun) and np.isfinite(result.x).all() and np.isfinite(result.jac).all()
    precision_accepted = bool(result.status == 2 and "precision loss" in str(result.message).lower() and finite and gradient_norm <= 1e-7)
    if not finite or not (result.success or precision_accepted):
        raise ValueError(f"Fixed residual optimizer failed: {result.message}")
    fitted = {"usesCorrected48": model, "coefficients": result.x.tolist(), "temperature": 1.0,
              "fitRows": len(train), "fitEvents": int(train.event_id.nunique()), "objective": float(result.fun),
              "gradientMaximum": float(np.max(np.abs(result.jac))), "gradientL2Norm": gradient_norm,
              "optimizerIterations": int(result.nit), "optimizerReportedSuccess": bool(result.success),
              "optimizerStatus": int(result.status), "optimizerMessage": str(result.message),
              "precisionLossAcceptedByFrozenCriterion": precision_accepted,
              "parameterL2ErrorUpperBound": gradient_norm / strength,
              "objectiveGapUpperBound": gradient_norm ** 2 / (2 * strength)}
    offset, X = logits(calibration, model)
    z = offset + X @ result.x
    y = calibration.target.to_numpy(float)
    temperature = minimize_scalar(lambda log_t: np.mean(np.logaddexp(0, z / math.exp(log_t)) - y * z / math.exp(log_t)),
                                  bounds=(-3, 3), method="bounded")
    if not temperature.success or not np.isfinite(temperature.fun) or not np.isfinite(temperature.x) or not math.exp(temperature.x) > 0:
        raise ValueError("Calibration did not converge")
    fitted.update({"temperature": math.exp(temperature.x), "calibrationRows": len(calibration),
                   "calibrationEvents": int(calibration.event_id.nunique()), "calibrationLogLossInSample": float(temperature.fun)})
    return fitted


def split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dates = pd.to_datetime(frame.date)
    cohorts = {name: frame[(dates >= config["fromInclusive"]) & (dates < config["endExclusive"])].copy()
               for name, config in ((name, CONFIG[name]) for name in ["fit", "calibration", "test"])}
    event_sets = []
    for name, part in cohorts.items():
        if len(part) != CONFIG[name]["rows"] or part.event_id.nunique() != CONFIG[name]["events"]:
            raise ValueError(f"Frozen market chronology changed: {name}")
        event_sets.append(set(part.event_id))
    if any(event_sets[i] & event_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Events cross market train/calibration/test boundaries")
    return cohorts


def metrics(frame: pd.DataFrame, probability: str, pick: str) -> dict:
    helper = frozen_helpers()
    result = helper.metrics(frame, probability, pick)
    result.update({"meanProbability": float(frame[probability].mean()), "observedRedWinRate": float(frame.target.mean())})
    bins = []
    edges = np.linspace(0, 1, 11)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        part = frame[(frame[probability] >= low) & ((frame[probability] < high) if index < 9 else (frame[probability] <= high))]
        bins.append({"lower": float(low), "upper": float(high), "count": len(part),
                     "meanProbability": float(part[probability].mean()) if len(part) else None,
                     "observedRedWinRate": float(part.target.mean()) if len(part) else None})
    if sum(row["count"] for row in bins) != len(frame):
        raise ValueError("Calibration bins do not partition the cohort")
    result["calibrationBins"] = bins
    return result


def run(source: Path, output: Path) -> None:
    inputs = initial_inputs(source)
    hashes = {str(path): digest(path) for path in sorted(inputs)}
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"createdAt": now(), "config": CONFIG, "inputHashes": hashes,
                "source": str(source), "corrected48ModelSHA256": MODEL_SHA,
                "numericalRecoveryAmendment": {"path": str(AMENDMENT), "sha256": digest(AMENDMENT)} if AMENDMENT.exists() else None,
                "priorExposure": "All historical outcomes and prior market/model results already inspected. Protocol fixed before these residual fits; no untouched test claim.",
                "officialDocumentationCheckedAt": now(), "officialDocumentation": DOCS,
                "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}
    write(output / "protocol.json", protocol)
    for path in [Path(__file__), Path(base.__file__), Path(challenger.__file__), Path(audit.__file__), COVERAGE / "score_coverage.py"]:
        (output / path.name).write_bytes(path.read_bytes())
    print(json.dumps({"stage": "protocol_frozen", "sha256": digest(output / "protocol.json")}), flush=True)
    fights, rounds, fighters = verify_source(source)
    quoted, exclusions = quotes(fights, output, inputs)
    if set(map(str, inputs)) - set(hashes):
        raise ValueError("Unpinned input discovered during quote validation")
    features = reconstruct(fights, rounds, fighters)
    features = features[(features.date >= "2025-01-01") & (features.date < "2026-11-01")].reset_index(drop=True)
    if len(features) != 985 or features.fight_id.duplicated().any():
        raise ValueError("Full replay cohort changed")
    frozen = joblib.load(MODEL)
    features["corrected48"] = audit.probability([frozen["model"]], frozen["temperature"], features[frozen["feature_columns"]])
    old_predictions = pd.read_csv(BASELINE / "predictions.csv").set_index("fight_id").loc[features.fight_id]
    if not np.allclose(features.corrected48, old_predictions.original48_after, atol=1e-14, rtol=0):
        raise ValueError("Recomputed corrected48 differs from pinned replay")
    for corner in ["red", "blue"]:
        if not features[f"{corner}_fighter_id"].tolist() == old_predictions[f"corrected_{corner}_fighter_id"].tolist():
            raise ValueError("Current corner identity differs from corrected replay")
        through = features[f"{corner}_history_through"]
        if not (through.isna() | (through < features.date)).all():
            raise ValueError("Non-prior feature history")
    prefix = []
    for cutoff in ["2026-01-01", "2026-05-16", "2026-11-01"]:
        history = fights[fights.date < pd.Timestamp(cutoff)]
        earlier = reconstruct(history, rounds[rounds.fight_id.isin(history.fight_id)], fighters)
        earlier = earlier[earlier.date >= "2025-01-01"].reset_index(drop=True)
        expected = features[features.date < pd.Timestamp(cutoff)].reset_index(drop=True)
        pd.testing.assert_frame_equal(earlier[frozen["feature_columns"]], expected[frozen["feature_columns"]], check_exact=True)
        if earlier.fight_id.tolist() != expected.fight_id.tolist():
            raise ValueError("Physical prefix identities changed")
        prefix.append({"cutoffExclusive": cutoff, "rows": len(earlier), "all48FeaturesExact": True})
    write(output / "prefix-checks.json", prefix)
    feature_map = features.set_index("fight_id")
    selected = feature_map.loc[quoted.fight_id]
    for field in ["event_id", "red_fighter_id", "blue_fighter_id", "target"]:
        if quoted[field].tolist() != selected[field].tolist():
            raise ValueError("Quote and current model identity or target mismatch")
    quoted["corrected48"] = selected.corrected48.to_numpy()
    quoted["pick_corrected48"] = quoted.corrected48 >= .5
    cohorts = split(quoted)
    write(output / "cohorts.json", {name: {"count": len(frame), "events": int(frame.event_id.nunique()),
          "from": frame.date.min(), "through": frame.date.max(), "fightIdsSHA256": audit.id_hash(frame.fight_id)} for name, frame in cohorts.items()})
    fitted = {name: fit_adjustment(cohorts["fit"], cohorts["calibration"], model)
              for name, model in [("market_calibrated", False), ("market_residual", True)]}
    write(output / "fitted-adjustments.json", fitted)
    seal = {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()}
    write(output / "pre-test-manifest.json", {"frozenAt": now(), "adjustedTestPredictionsComputed": False,
          "frozenBaselineRecomputedForParityBeforeSeal": True, "files": seal})
    print(json.dumps({"stage": "adjustments_frozen", "models": fitted}), flush=True)
    for name, model in fitted.items():
        quoted[name] = fitted_probability(quoted, model)
        quoted[f"pick_{name}"] = quoted[name] >= .5
        reverse = quoted.copy()
        reverse.p_market = 1 - reverse.p_market
        reverse.corrected48 = 1 - reverse.corrected48
        if not np.allclose(quoted[name].to_numpy() + fitted_probability(reverse, model), 1, atol=1e-12, rtol=0):
            raise ValueError("Market adjustment lost corner symmetry")
    predictors = {"market": ("p_market", "pick_market"), "corrected48": ("corrected48", "pick_corrected48"),
                  **{name: (name, f"pick_{name}") for name in fitted}}
    pairs = [("market_residual", "market"), ("market_residual", "market_calibrated"),
             ("market_calibrated", "market"), ("corrected48", "market")]
    helper = frozen_helpers()
    scored = split(quoted)
    scored["all_matched"] = quoted
    report = {name: {"role": "primary historical test" if name == "test" else "Includes adjustment training/calibration labels; not residual generalization evidence",
                    "metrics": {model: metrics(frame, *fields) for model, fields in predictors.items()},
                    "paired": helper.bootstrap(frame, predictors, pairs)} for name, frame in scored.items()}
    if any(result["paired"]["seed"] != CONFIG["bootstrap"]["seed"] or result["paired"]["replicates"] != CONFIG["bootstrap"]["replicates"] for result in report.values()):
        raise ValueError("Bootstrap implementation differs from declared settings")
    quoted.to_csv(output / "predictions.csv", index=False)
    full = features[["fight_id", "event_id", "date", "red_fighter_id", "blue_fighter_id", "target", "corrected48"]].copy()
    additions = quoted[["fight_id", "p_market", "pick_market", "market_calibrated", "market_residual"]]
    full = full.merge(additions, on="fight_id", how="left", validate="one_to_one")
    full["has_market"] = full.p_market.notna()
    full["pick_corrected48"] = full.corrected48 >= .5
    fallback_predictors = {"corrected48": ("corrected48", "pick_corrected48")}
    for name, probability in [("market", "p_market"), ("market_calibrated", "market_calibrated"), ("market_residual", "market_residual")]:
        column = f"{name}_with48fallback"
        full[column] = full[probability].fillna(full.corrected48)
        full[f"pick_{column}"] = full[column] >= .5
        if name == "market":
            full.loc[full.has_market, f"pick_{column}"] = full.loc[full.has_market, "pick_market"].astype(bool)
        fallback_predictors[column] = (column, f"pick_{column}")
    full.to_csv(output / "full-predictions.csv", index=False)
    fallback = {}
    for name, frame in [("full985", full), ("primary285", full[full.date >= "2026-05-16"])]:
        fallback[name] = {"count": len(frame), "quoted": int(frame.has_market.sum()), "marketAlone": False,
                          "metrics": {model: metrics(frame, *fields) for model, fields in fallback_predictors.items()}}
    for filename, checksum in seal.items():
        if digest(output / filename) != checksum:
            raise ValueError("A pre-test artifact changed")
    if any(digest(Path(path)) != checksum for path, checksum in hashes.items()):
        raise ValueError("A frozen input changed during evaluation")
    result = {"schemaVersion": 1, "createdAt": now(), "protocolSHA256": digest(output / "protocol.json"),
              "status": CONFIG["status"], "coverage": {"originalV1": 462, "duplicateQuarantineOnly": 460, "coverageV2": 501,
              "currentExactIdentity": len(quoted), "full985Fraction": len(quoted) / 985, "primary285Matched": 217,
              "excludedIdentityRows": exclusions, "events": int(quoted.event_id.nunique()),
              "earliestQuoteFightDate": quoted.date.min(), "latestQuoteFightDate": quoted.date.max()},
              "results": report, "fullUniverseFallbackDiagnostics": fallback,
              "quoteQuality": {"zeroVolumeEitherSide": int(((quoted.red_window_volume == 0) | (quoted.blue_window_volume == 0)).sum()),
               "zeroVolumeBothSides": int(((quoted.red_window_volume == 0) & (quoted.blue_window_volume == 0)).sum()),
               "observationAgeSeconds": quoted.observation_age_seconds.describe().to_dict(),
               "actualQuoteUpdateAgeKnown": False, "midpointSumRange": [quoted.midpoint_sum.min(), quoted.midpoint_sum.max()],
               "maximumSpread": float(max((quoted.red_ask - quoted.red_bid).max(), (quoted.blue_ask - quoted.blue_bid).max()))},
              "fittedAdjustments": fitted, "physicalPrefixChecks": prefix, "symmetryPassed": True,
              "current48ReplayParityPassed": True, "inputHashesUnchanged": True,
              "runtimeChanged": False, "modelPromotion": False, "profitEstablished": False,
              "limitations": [
                  "All historical outcomes were previously consumed. Chronological splitting does not undo design or source revision bias.",
                  "Only127 quoted bouts on20 events fit the adjustments; coverage and selection are nonrandom.",
                  "Normalized paired YES midpoints are expected-payout proxies; draw/NC/cancellation terms differ from a conditional decisive-win target.",
                  "Decisive-only scoring excludes draws, no contests and scalar settlements and is not an all-contract P&L backtest.",
                  "Minute observations do not certify quote freshness; many fetched windows have no trades on one or both sides.",
                  "Bid/ask observations lack historical orderbook depth, fillability, queue, slippage and contemporaneous fees. No realized ROI or tradable edge is established.",
                  "Full-universe fallback diagnostics mix market and model probabilities and are not market-alone performance.",
                  "Event bootstrap intervals are descriptive; repeated fighters and previously selected models create additional dependence.",
                  "The early-November prospective archive has no completed outcomes and uses different quote timing and older model versions; it is not pooled here.",
              ]}
    write(output / "report.json", result)
    write(output / "artifact-hashes.json", {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()})
    print(json.dumps({"stage": "complete", "output": str(output), "test": result["results"]["test"]["metrics"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed retrospective market residual and corrected48 matched comparison")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve())
