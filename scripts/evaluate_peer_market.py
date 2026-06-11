import argparse
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.special import logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_market_edge as edge
from scripts import evaluate_market_history as history


PROPOSAL = edge.DATA / "peer-residual-proposal-v1/protocol.json"
PROPOSAL_SHA = "0ce72e37b502388f9d661173cd9f6638bde55dd44b0038c73cf193341dabb3c2"
MARKET = edge.DATA / "market-edge-v1/verified-quotes.csv"
MARKET_SHA = "88520cca5cbfa51277d130d8ad1fceada6f976d97f8723462ed44b2e2a107a86"
HISTORY_SHA = "87f7e3ea29f9d238deb89f270b1a3baecd03751b57619fa060873b2182311ad6"
FEATURES = ["market_logit", "peer_difference"]
IDENTITY = ["event_id", "date", "red_fighter_id", "blue_fighter_id", "red_name", "blue_name"]
FORBIDDEN = {"target", "winner", "result", "winning_outcome_label", "resolution_status", "resolved_at", "p_event", "D"}
MISSING = {"identity_unresolved", "no_execution_in_window", "ambiguous_latest_second", "source_unavailable"}


def versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__}


def checked(path: Path, checksum: str, inputs: set[Path]) -> Path:
    if edge.digest(path) != checksum:
        raise ValueError(f"Source checksum mismatch: {path}")
    inputs.add(path.resolve())
    return path


def scoped(directory: Path, name: str) -> Path:
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Source path escapes its bundle")
    return path


def timestamp(value: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("Observation needs a valid timezone")
    return result.tz_convert("UTC")


def validate_candidate(row: dict, identity: dict, decision: dict) -> None:
    primary = decision["primary"]
    probability, age = row["peer_1h"], row["peer_1h_age_seconds"]
    if (not isinstance(probability, (int, float)) or isinstance(probability, bool)
            or not math.isfinite(probability) or not 0 < probability < 1
            or not isinstance(age, (int, float)) or isinstance(age, bool)
            or not math.isfinite(age) or int(age) != age or not 0 <= age <= 3600):
        raise ValueError("Invalid one-hour execution probability or age")
    if (primary["status"] != "candidate_execution" or primary["maxAgeSeconds"] != 3600
            or primary.get("activityCountIsLowerBound") is not False
            or abs(primary["redProbabilityProxy"] - probability) > 1e-12
            or primary["ageSeconds"] != age):
        raise ValueError("Primary execution decision differs from CSV")
    observed = timestamp(row["quote_observed_at"])
    latest = int(observed.timestamp()) - int(age)
    if primary["latestTimestamp"] != latest or timestamp(identity["createdAt"]).timestamp() > latest:
        raise ValueError("Execution timestamp precedes creation or differs from cutoff")
    tokens = {item["tokenId"]: item for item in identity["tokens"]}
    records = primary["latestRecords"]
    if (not records or len(identity["tokens"]) != 2 or len(tokens) != 2
            or {item["side"] for item in tokens.values()} != {"red", "blue"}
            or len({item["fighterId"] for item in tokens.values()}) != 2
            or any(item["fighterId"] != row[f"{item['side']}_fighter_id"] for item in tokens.values())):
        raise ValueError("Latest execution lacks exact token identities")
    for record in records:
        token = tokens.get(record["tokenId"])
        price, amount = record["tokenPrice"], record["usdcAmount"]
        if (token is None or record["timestamp"] != latest or record["ageSeconds"] != age
                or isinstance(price, bool) or not isinstance(price, (int, float))
                or not math.isfinite(price) or not 0 < price < 1
                or isinstance(amount, bool) or not isinstance(amount, (int, float))
                or not math.isfinite(amount) or amount <= 0):
            raise ValueError("Invalid latest execution evidence")
        expected_id = row[f"{token['side']}_fighter_id"]
        proxy = price if token["side"] == "red" else 1 - price
        if token["fighterId"] != expected_id or abs(round(proxy, 12) - probability) > 1e-12:
            raise ValueError("Execution token orientation or same-second price mismatch")


def verify_partitions(directory: Path, manifest: dict, inputs: set[Path]) -> None:
    if (manifest.get("completed") is not True or manifest.get("status") not in {"complete", "complete_with_declared_missing"}
            or manifest.get("failedFiles")):
        raise ValueError("Peer source acquisition is incomplete")
    plan = edge.read(checked(directory / "plan.json", manifest["planSHA256"], inputs))
    if "catalogSHA256" in manifest:
        checked(directory / "catalog.json", manifest["catalogSHA256"], inputs)
    else:
        checked(directory.parent / "protocol.json", manifest["parentProtocolSHA256"], inputs)
        checked(directory.parent.parent / "2026-cost/estimate.json", plan["costEstimateSHA256"], inputs)
        if (manifest["requiredSources"] != plan["requiredSources"]
                or manifest["requiredSourceKeysByFightId"] != plan["requiredSourceKeysByFightId"]
                or manifest["missingSources"] != plan["missingSources"]):
            raise ValueError("Later source membership differs from its acquisition plan")
        required = {row["sourceKey"] for row in manifest["requiredSources"]}
        available = {row["sourceKey"] for row in manifest["files"]}
        missing = {row["sourceKey"] for row in manifest["missingSources"]}
        if available & missing or available | missing != required or len(available) != len(manifest["files"]):
            raise ValueError("Later source coverage has unaccounted gaps or duplicates")
    record_rows = manifest.get("records", manifest["files"])
    records = {(row["sourceRevision"], row["sourcePath"]): row for row in record_rows}
    if len(records) != len(manifest["files"]) or len(records) != len(record_rows):
        raise ValueError("Duplicate or missing source partitions")
    for item in manifest["files"]:
        record = records[(item["sourceRevision"], item["sourcePath"])]
        source_sha = item.get("sourceSHA256") or record["publishedLFSsha256"]
        references = record.get("fullHTTPrawReferences", record.get("rawReferences", []))
        if not item.get("sourceComplete") or not item.get("fullSourceHashVerified"):
            raise ValueError("Peer partition lacks complete source verification")
        checked(scoped(directory, item["projectedFile"]), item["projectedSHA256"], inputs)
        source = Path(record["localSourcePath"])
        if not source.is_absolute():
            source = scoped(directory, str(source))
        deleted = record.get("rawSourceDeletedAfterVerification") is True
        if source.exists():
            checked(source, source_sha, inputs)
        elif not deleted or not references:
            raise ValueError("Full source is missing without retained verification receipts")
        if record["publishedLFSsha256"] != source_sha or not record.get("fullSourceHashVerified"):
            raise ValueError("Full-source verification differs from published hash")
        partition_date = item.get("partitionDate") or record.get("date") or item["sourceKey"].split("/")[-1]
        start = int(pd.Timestamp(partition_date, tz="UTC").timestamp())
        empty = item.get("rows") == 0 and item["minTimestamp"] is None and item["maxTimestamp"] is None
        valid_bounds = (item["minTimestamp"] is not None and item["maxTimestamp"] is not None
                        and start <= item["minTimestamp"] <= item["maxTimestamp"] < start + 86400)
        if not empty and not valid_bounds:
            raise ValueError("Peer partition timestamps do not match its UTC date")
        if deleted:
            verification_path = scoped(directory, record["sourceVerificationReceipt"])
            deletion_path = scoped(directory, record["deletionReceipt"])
            deletion = edge.read(deletion_path)
            inputs.add(deletion_path)
            verification = edge.read(checked(verification_path, deletion["verificationReceiptSHA256"], inputs))
            if (deletion["sourceSHA256VerifiedBeforeDeletion"] != source_sha
                    or deletion["projectedSHA256"] != item["projectedSHA256"]
                    or deletion["sourceBytes"] != record["sourceBytes"]
                    or verification["sourceSHA256"] != source_sha
                    or verification["projectedSHA256"] != item["projectedSHA256"]
                    or verification.get("fullSourceHashVerified") is not True
                    or verification.get("beforeAnySourceDeletion") is not True):
                raise ValueError("Deleted source lacks a consistent pre-deletion verification chain")
        for reference in references:
            if "receiptFile" in reference:
                receipt_path = Path(reference["receiptFile"])
                if not receipt_path.is_absolute():
                    receipt_path = scoped(directory, str(receipt_path))
                checked(receipt_path, reference["receiptSHA256"], inputs)
            else:
                path = scoped(directory, reference["file"])
                if path.exists():
                    checked(path, reference["sha256"], inputs)
                elif not deleted:
                    raise ValueError("Full-source receipt body is unexpectedly missing")
                receipt_path = path.with_name(path.name + ".receipt.json")
            receipt = edge.read(receipt_path)
            inputs.add(receipt_path)
            if (receipt.get("error") or receipt.get("status") != 200
                    or receipt["sha256"] != reference["sha256"]
                    or reference["sha256"] != source_sha
                    or receipt["bytes"] != record["sourceBytes"]
                    or item["sourceRevision"] not in receipt["url"]):
                raise ValueError("Invalid complete-file acquisition receipt")


def read_peer(path: Path, inputs: set[Path]) -> pd.DataFrame:
    if not path.name.startswith("peer-prices-") or path.suffix != ".csv":
        raise ValueError("Expected peer-prices-<cohort>.csv")
    directory = path.parent
    cohort = path.stem.removeprefix("peer-prices-")
    report_path = directory / f"coverage-report-{cohort}.json"
    report = edge.read(report_path)
    inputs.add(report_path)
    if report["cohort"] != cohort or report["pricesCSV"] != path.name:
        raise ValueError("Peer report names a different cohort")
    if report.get("noOutcomeScoring") is not True or report.get("noModelFitting") is not True:
        raise ValueError("Peer acquisition contains outcome scoring or fitting")
    checked(path, report["pricesSHA256"], inputs)
    freeze_path = scoped(directory, report.get("inputFreezeFile", "coverage-input-freeze.json"))
    freeze = edge.read(freeze_path)
    inputs.add(freeze_path)
    identity_path = checked(directory / f"identity-decisions-{cohort}-final.json", report["identityManifestSHA256"], inputs)
    manifest_path = checked(directory / "partitions/manifest.json", report["sourceManifestSHA256"], inputs)
    if (freeze["identitySHA256"] != report["identityManifestSHA256"]
            or freeze["partitionManifestSHA256"] != report["sourceManifestSHA256"]
            or freeze["collectorSHA256"] != report["codeSHA256"]
            or freeze["primaryWindowSeconds"] != 3600):
        raise ValueError("Peer coverage freeze differs from report")
    snapshots = ([scoped(directory, report["collectorSnapshotFile"])] if report.get("collectorSnapshotFile") else
                 [p for p in (directory / "source-snapshots").glob("*.py") if edge.digest(p) == report["codeSHA256"]])
    if not snapshots:
        raise ValueError("Frozen collector code snapshot is missing")
    checked(sorted(snapshots)[0], report["codeSHA256"], inputs)
    manifest = edge.read(manifest_path)
    verify_partitions(manifest_path.parent, manifest, inputs)
    identity_rows = edge.read(identity_path)
    identities = {row["fightId"]: row for row in identity_rows}
    decision_path = directory / f"coverage-decisions-{cohort}.json"
    decision_rows = edge.read(decision_path)
    inputs.add(decision_path)
    decisions = {row["fight_id"]: row for row in decision_rows}
    frame = pd.read_csv(path)
    if FORBIDDEN & set(frame.columns):
        raise ValueError("Outcome-derived field in peer input")
    if (frame.fight_id.duplicated().any() or len(frame) != report["candidateBouts"]
            or len(identities) != len(identity_rows) or len(decisions) != len(decision_rows)
            or set(frame.fight_id) != set(identities) or set(frame.fight_id) != set(decisions)):
        raise ValueError("Peer identity/decision membership differs from CSV")
    for row in frame.to_dict("records"):
        identity, decision = identities[row["fight_id"]], decisions[row["fight_id"]]
        if row["status"] == "identity_unresolved":
            if identity["status"] == "exact_pair" or decision["status"] != "identity_unresolved":
                raise ValueError("Unresolved identity has conflicting evidence")
            continue
        if identity["status"] != "exact_pair" or len(identity["identities"]) != 1:
            raise ValueError("Peer execution lacks an exact identity")
        identity = identity["identities"][0]
        if (identity["redFighterId"] != row["red_fighter_id"] or identity["blueFighterId"] != row["blue_fighter_id"]
                or identity["fightDate"] != row["date"] or timestamp(identity["quoteObservedAt"]) != timestamp(row["quote_observed_at"])
                or identity["conditionId"] != row["condition_id"] or identity["conditionId"] != decision["conditionId"]):
            raise ValueError("Peer condition identity, date or observation mismatch")
        checked(scoped(directory, identity["metadataFile"]), identity["metadataSHA256"], inputs)
        if "requiredSourceKeysByFightId" in manifest:
            keys = manifest["requiredSourceKeysByFightId"][row["fight_id"]]
            available = {item["sourceKey"] for item in manifest["files"]}
            missing = set(keys) - available
            reported_missing = {item["sourceKey"] if isinstance(item, dict) else item
                                for item in decision.get("missingRequiredSources", [])}
            if missing != reported_missing or bool(missing) != (row["status"] == "source_unavailable"):
                raise ValueError("Peer execution window differs from declared source availability")
        if row["status"] == "source_unavailable":
            if decision["status"] != "source_unavailable" or not decision.get("missingRequiredSources"):
                raise ValueError("Unavailable source lacks explicit missing-partition evidence")
            continue
        if decision.get("missingRequiredSources"):
            raise ValueError("Candidate window has missing source partitions")
        if row["status"] != decision["primary"]["status"]:
            raise ValueError("Peer CSV status differs from execution decision")
        if row["status"] == "candidate_execution":
            validate_candidate(row, identity, decision)
    if int((frame.status == "candidate_execution").sum()) != report["strict1hCandidateBouts"]:
        raise ValueError("Peer coverage count differs from frozen report")
    return frame


def combine(market: pd.DataFrame, peer: pd.DataFrame) -> pd.DataFrame:
    if (market.fight_id.duplicated().any() or peer.fight_id.duplicated().any()
            or set(market.fight_id) != set(peer.fight_id)):
        raise ValueError("Market/peer membership differs or duplicates exist")
    if FORBIDDEN & set(peer.columns):
        raise ValueError("Outcome-derived field in peer input")
    peer = peer.set_index("fight_id").loc[market.fight_id].reset_index()
    market = market.reset_index(drop=True).copy()
    for column in IDENTITY:
        if not market[column].equals(peer[column]):
            raise ValueError(f"Exact peer identity mismatch: {column}")
    for column in ["cutoff", "quote_observed_at"]:
        if any(timestamp(a) != timestamp(b) for a, b in zip(market[column], peer[column])):
            raise ValueError(f"Peer observation mismatch: {column}")
    eligible = peer.status == "candidate_execution"
    if not peer.status.isin(MISSING | {"candidate_execution"}).all():
        raise ValueError("Unknown peer status")
    probabilities = pd.to_numeric(peer.peer_1h, errors="raise")
    ages = pd.to_numeric(peer.peer_1h_age_seconds, errors="raise")
    if (not probabilities[eligible].between(0, 1, inclusive="neither").all()
            or not ages[eligible].between(0, 3600).all() or not (ages[eligible] == np.floor(ages[eligible])).all()
            or probabilities[~eligible].notna().any() or ages[~eligible].notna().any()):
        raise ValueError("Invalid candidate or nonmissing unqualified one-hour price")
    market["peer_status"] = peer.status
    market["peer_available"] = eligible
    market["peer_1h"] = probabilities
    market["peer_age_seconds"] = ages
    market["market_logit"] = logit(market.p_market.clip(1e-6, 1 - 1e-6))
    market["peer_difference"] = np.where(eligible, logit(probabilities.clip(1e-6, 1 - 1e-6)) - market.market_logit, 0.0)
    return market


def fit_gate(frame: pd.DataFrame) -> None:
    if frame.loc[frame.peer_available, "fight_id"].nunique() < 40:
        raise ValueError("At least40 unique strict one-hour training bouts are required; no fit")


def run(training: Path, later: list[Path], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an experiment: {output}")
    inputs = {Path(__file__).resolve(), Path(edge.__file__).resolve(), edge.COVERAGE / "score_coverage.py",
              edge.ROOT / "tests/test_peer_market.py"}
    checked(PROPOSAL, PROPOSAL_SHA, inputs)
    checked(MARKET, MARKET_SHA, inputs)
    checked(Path(history.__file__), HISTORY_SHA, inputs)
    peer_train = read_peer(training, inputs)
    peer_later = pd.concat([read_peer(path, inputs) for path in later], ignore_index=True)
    if (len(peer_train) != 127 or len(peer_later) != 372 or not (peer_train.date < "2026-01-01").all()
            or not (peer_later.date >= "2026-01-01").all()):
        raise ValueError("Peer training/later cohort dates or counts changed")
    frame = combine(pd.read_csv(MARKET), pd.concat([peer_train, peer_later], ignore_index=True))
    cohorts = edge.split(frame)
    fit_gate(cohorts["fit"])
    hashes = {str(path): edge.digest(path) for path in sorted(inputs)}
    dependencies = versions()
    output.mkdir(parents=True, exist_ok=False)
    edge.write(output / "protocol.json", {"createdAt": edge.now(), "proposalSHA256": PROPOSAL_SHA,
               "proposal": edge.read(PROPOSAL), "features": FEATURES, "inputHashes": hashes, "versions": dependencies})
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    frame.to_csv(output / "features.csv", index=False)
    fitted = {name: history.fit(cohorts["fit"], cohorts["calibration"], columns)
              for name, columns in [("market_control", FEATURES[:1]), ("market_peer", FEATURES)]}
    edge.write(output / "fitted-models.json", fitted)
    seal = {path.name: edge.digest(path) for path in output.iterdir() if path.is_file()}
    edge.write(output / "pre-test-manifest.json", {"createdAt": edge.now(), "files": seal,
               "candidateTestPredictionsComputed": False, "historicalTargetsPreviouslyExposed": True})
    for name, model in fitted.items():
        frame[name] = history.predict(frame, model)
        frame[f"pick_{name}"] = frame[name] >= .5
        reverse = frame.copy()
        reverse[FEATURES] *= -1
        if not np.allclose(frame[name] + history.predict(reverse, model), 1, atol=1e-12, rtol=0):
            raise ValueError("Peer model lost corner symmetry")
    predictors = {"market": ("p_market", "pick_market"), **{name: (name, f"pick_{name}") for name in fitted}}
    results = {}
    for name, part in edge.split(frame).items():
        results[name] = {
            "role": "Exploratory previously examined historical outcomes" if name == "test" else "Fit/calibration data; not generalization evidence",
            "metrics": {model: edge.metrics(part, *fields) for model, fields in predictors.items()},
            "paired": edge.frozen_helpers().bootstrap(part, predictors, [("market_peer", "market"), ("market_peer", "market_control")]),
            "coverage": {"rows": len(part), "peerAvailable": int(part.peer_available.sum()),
                         "statusCounts": part.peer_status.value_counts().to_dict(),
                         "ageSeconds": part.peer_age_seconds.dropna().describe().replace({np.nan: None}).to_dict()},
        }
    frame.to_csv(output / "predictions.csv", index=False)
    if versions() != dependencies or any(edge.digest(Path(path)) != sha for path, sha in hashes.items()):
        raise ValueError("Frozen inputs or dependency versions changed")
    if any(edge.digest(output / path) != sha for path, sha in seal.items()):
        raise ValueError("Pre-test artifact changed")
    edge.write(output / "report.json", {"createdAt": edge.now(), "results": results, "fitted": fitted,
               "inputHashesVerified": True, "symmetryVerified": True, "promoted": False, "runtimeChanged": False,
               "limitations": ["All historical outcomes previously inspected; no untouched validation claim.",
                   "Peer executions and metadata timestamps remain provisional retrospective source data.",
                   "Deleted temporary full files are checked through retained download/LFS attestations; their bytes cannot be independently rehashed after deletion.",
                   "No bid/ask or fillability evidence; this is not an executable trading-profit estimate.",
                   "Missing peer observations remain in the denominator with zero peer adjustment input.",
                   "One-hour rule unchanged;24-hour diagnostic fields are never model inputs.",
                   "Bootstrap intervals do not account for repeated experimentation."]})
    edge.write(output / "artifact-hashes.json", {path.name: edge.digest(path) for path in output.iterdir() if path.is_file()})
    print({name: {key: value[key] for key in ["count", "correct", "accuracy", "brier", "logLoss"]}
           for name, value in results["test"]["metrics"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One fixed chronological peer-execution residual experiment")
    parser.add_argument("--peer-training", type=Path, required=True)
    parser.add_argument("--peer-later", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.peer_training.resolve(), [path.resolve() for path in arguments.peer_later], arguments.output.resolve())
