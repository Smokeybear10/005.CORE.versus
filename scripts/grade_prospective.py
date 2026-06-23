import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

from record_prospective import CANDIDATE_CLASS, CANDIDATE_COVERAGE, MODEL_VERSIONS, STACK_COVERAGE, digest, now, write_bytes, write_json


BINS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0)
STATUSES = {"decisive", "draw", "no_contest", "cancelled", "changed"}
CONTEXT_FIELDS = ("weightClass", "rounds", "titleBout")
PIPELINE_CODE = ("backend/forecast.py", "backend/research.py", "backend/model_engine.py",
                 "scripts/evaluate_research.py", "scripts/evaluate_challenger.py", "scripts/record_prospective.py")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def timestamp(value: str) -> datetime:
    require(isinstance(value, str), "Timestamps must be timezone-aware ISO strings")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "Timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def event_date(value: str) -> date:
    require(isinstance(value, str), "Event dates must be YYYY-MM-DD strings")
    parsed = date.fromisoformat(value)
    require(parsed.isoformat() == value, "Event dates must use YYYY-MM-DD")
    return parsed


def pair_key(payload: object) -> tuple | None:
    if not isinstance(payload, dict):
        return None
    ids = (payload.get("firstID"), payload.get("secondID"))
    if not all(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{16}", value) for value in ids) or ids[0] == ids[1]:
        return None
    if not isinstance(payload.get("eventDate"), str):
        return None
    return (payload["eventDate"], *sorted(ids))


def load_archive(path: Path) -> dict:
    manifest_bytes = (path / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    require(manifest.get("schemaVersion") == 1 and manifest.get("inputIntegrityVerified") is True,
            f"{path}: not an integrity-verified recorder archive; older prospective-v1 format is unsupported")
    hashes = manifest.get("files")
    require(isinstance(hashes, dict), "Missing archive file manifest")
    actual = {str(item.relative_to(path)) for item in path.rglob("*") if item.is_file() and item != path / "manifest.json"}
    require(actual == set(hashes), f"{path}: archive files do not match its manifest")
    contents = {}
    for name, sha in hashes.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts and not (path / relative).is_symlink(), "Unsafe archive filename")
        content = (path / relative).read_bytes()
        require(digest(content) == sha, f"{path}: checksum mismatch for {name}")
        contents[name] = content
    required = {"input.json", "protocol.json", "model-status-before.json", "input-fingerprints.json",
                "input-fingerprints-after.json", "pre-prediction-manifest.json", "summary.json"}
    require(required <= contents.keys(), "Unsupported or incomplete recorder archive")
    documents = {name: json.loads(contents[name]) for name in required}
    frozen = documents["pre-prediction-manifest.json"]
    pre_files = {"input.json", "protocol.json", "model-status-before.json", "input-fingerprints.json"}
    if "linked-manifest.json" in contents:
        pre_files.add("linked-manifest.json")
    require(set(frozen.get("files", {})) == pre_files, "Incomplete pre-prediction freeze")
    require(all(digest(contents[name]) == sha for name, sha in frozen["files"].items()), "Pre-prediction checksum mismatch")
    summary = documents["summary.json"]
    require(summary.get("inputIntegrityVerified") is True and summary.get("changedInputs") == [], "Archive inputs changed")
    require(documents["input-fingerprints.json"]["files"] == documents["input-fingerprints-after.json"]["files"], "Input fingerprints changed")
    status = documents["model-status-before.json"]
    observation_class = status.get("observationClass", "publicForecasts")
    require(observation_class in ("publicForecasts", CANDIDATE_CLASS), "Unsupported frozen observation class")
    model_version = status.get("modelVersion") or status.get("requestedModelVersion")
    require(model_version is None or isinstance(model_version, str) and re.fullmatch(r"sha256:[a-f0-9]{64}", model_version), "Invalid frozen model identity")
    if status.get("modelVersion") and status.get("requestedModelVersion"):
        require(status["modelVersion"] == status["requestedModelVersion"], "Requested and loaded model versions differ")
    pipeline_code = PIPELINE_CODE + (("scripts/predictor_features.py", "scripts/evaluate_predictors.py") if observation_class == CANDIDATE_CLASS else ())
    if model_version == MODEL_VERSIONS["probability_stack"]:
        require(observation_class == CANDIDATE_CLASS, "Probability stack must remain a research candidate")
        if status.get("available"):
            require(status.get("domainGateModelVersion") == MODEL_VERSIONS["corrected48"], "Probability stack domain gate is not corrected48")
        pipeline_code += ("scripts/evaluate_probability_stack.py",)
    code = {name: documents["input-fingerprints.json"]["files"].get(name) for name in pipeline_code}
    require(all(isinstance(sha, str) and re.fullmatch(r"[a-f0-9]{64}", sha) for sha in code.values()), "Missing forecasting pipeline code hashes")
    protocol = documents["protocol.json"]
    require(isinstance(protocol.get("packages"), dict) and isinstance(protocol.get("python"), str), "Missing pipeline runtime versions")
    pipeline = {"codeSHA256": code, "packages": protocol["packages"], "python": protocol["python"]}
    pipeline_fingerprint = "sha256:" + digest(json.dumps(pipeline, sort_keys=True, separators=(",", ":")).encode())
    card = documents["input.json"]
    requests = card.get("requests", [])
    require(card.get("schemaVersion") == 1 and card.get("fullCard") is True and isinstance(requests, list) and requests,
            "Archive does not contain a declared full-card request list")
    require(card.get("expectedBoutCount") == len(requests) == summary.get("requests") == len(summary.get("records", [])), "Incomplete full-card archive")
    ids = [item.get("boutId") for item in requests]
    require(all(isinstance(value, str) and value for value in ids) and len(set(ids)) == len(ids), "Invalid or duplicate bout IDs")
    scheduled = event_date(card["event"]["date"])
    frozen_at = timestamp(frozen["frozenAt"])
    require(timestamp(documents["protocol.json"]["frozenAt"]) <= frozen_at and frozen_at.date() < scheduled, "Capture was not frozen before the event date")
    completed = timestamp(manifest["completedAt"])
    require(completed <= datetime.now(timezone.utc), "Archive completion timestamp is in the future")
    rows = []
    for index, (request, entry) in enumerate(zip(requests, summary["records"]), 1):
        name = f"responses/{index:03}.json"
        require(entry.get("responseFile") == name and name in contents and entry.get("sha256") == digest(contents[name]), "Response manifest mismatch")
        record = json.loads(contents[name])
        require(record.get("request") == request and record.get("boutId") == entry.get("boutId") == request["boutId"], "Response does not match frozen request")
        started, finished = timestamp(record["startedAt"]), timestamp(record["finishedAt"])
        require(frozen_at <= started <= finished <= completed and finished.date() < scheduled, "Response was not recorded prospectively")
        rows.append({"archiveSHA256": digest(manifest_bytes), "archivePath": str(path), "recordFile": name,
                     "boutId": request["boutId"], "frozenAt": frozen["frozenAt"], "eventDate": card["event"]["date"],
                     "inputSHA256": digest(contents["input.json"]), "modelVersion": model_version,
                     "observationClass": observation_class,
                     "pipelineFingerprint": pipeline_fingerprint,
                     "conservativeTimingEligible": finished < datetime.combine(scheduled - timedelta(days=1), time.min, timezone.utc),
                     "record": record})
    return {"path": path, "manifestSHA256": digest(manifest_bytes), "prePredictionSHA256": digest(contents["pre-prediction-manifest.json"]),
            "pipelineFingerprint": pipeline_fingerprint, "pipeline": pipeline,
            "hashes": {"manifest.json": digest(manifest_bytes), **hashes}, "rows": rows}


def select_records(archives: list[dict]) -> list[dict]:
    rows = sorted((row for archive in archives for row in archive["rows"]),
                  key=lambda row: (timestamp(row["frozenAt"]), row["archiveSHA256"], row["recordFile"]))
    earliest = {}
    for row in rows:
        payload = row["record"]["request"].get("payload")
        payload = payload if isinstance(payload, dict) else {}
        model_key = row["modelVersion"] or ("unknown_model", row["archiveSHA256"])
        key = (model_key, pair_key(payload) or ("unresolved", row["inputSHA256"], row["boutId"]))
        row["selection"] = "selected" if key not in earliest else "duplicate"
        row["selectedCapture"] = None
        row["conflicts"] = []
        if key in earliest:
            first = earliest[key]
            row["selectedCapture"] = {"archiveSHA256": first["archiveSHA256"], "recordFile": first["recordFile"]}
            original = first["record"]["request"].get("payload")
            original = original if isinstance(original, dict) else {}
            if any(payload.get(field) != original.get(field) for field in CONTEXT_FIELDS):
                row["conflicts"].append("different_context")
            if row["pipelineFingerprint"] != first["pipelineFingerprint"]:
                row["conflicts"].append("different_pipeline")
            if row["observationClass"] != first["observationClass"]:
                row["conflicts"].append("different_observation_class")
        else:
            earliest[key] = row
    return rows


def load_outcomes(content: bytes, graded_at: datetime) -> dict:
    source = json.loads(content)
    require(isinstance(source, dict) and source.get("schemaVersion") == 1 and isinstance(source.get("outcomes"), list), "Expected schemaVersion: 1 and outcomes: [...]")
    results = {}
    for item in source["outcomes"]:
        require(isinstance(item, dict), "Each outcome must be an object")
        key = pair_key(item)
        require(key is not None, "Outcomes need two distinct exact UFCStats fighter IDs and an event date")
        scheduled = event_date(item["eventDate"])
        require(key not in results, "Duplicate outcome pair/date, including reversed or conflicting entries")
        require(item.get("status") in STATUSES, "Unsupported outcome status")
        require(isinstance(item.get("sourceUrl"), str), "Outcome sourceUrl must be a string")
        source_url = urlparse(item["sourceUrl"])
        require(source_url.scheme in ("http", "https") and bool(source_url.netloc) and not source_url.username, "Outcome sourceUrl must be an HTTP(S) source URL")
        observed = timestamp(item.get("observedAt"))
        require(observed <= graded_at, "Outcome observedAt cannot be in the future")
        incomplete = item["status"] in {"cancelled", "changed"}
        if not incomplete:
            require(scheduled < graded_at.date(), "Final results cannot be supplied for today/future event dates")
            require(observed.date() >= scheduled, "Final-result observedAt must be on or after the event date")
        require(all(field in item for field in (*CONTEXT_FIELDS, "winnerID")), "Outcomes need actual context and winnerID (null for nondecisive results)")
        require((isinstance(item["weightClass"], str) and bool(item["weightClass"])) or (incomplete and item["weightClass"] is None), "Missing actual weightClass")
        require((type(item["rounds"]) is int and item["rounds"] in (3, 5)) or (incomplete and item["rounds"] is None), "rounds means scheduled rounds (3 or 5), not the finish round")
        require(type(item["titleBout"]) is bool or (incomplete and item["titleBout"] is None), "titleBout must be boolean, or null for cancelled/changed bouts")
        require(item["winnerID"] in key[1:] if item["status"] == "decisive" else item["winnerID"] is None, "Invalid winnerID for the supplied status/pair")
        results[key] = item
    return results


def prediction(row: dict, payload: dict) -> tuple[str | None, float | None, str | None]:
    record = row["record"]
    if record.get("error") is not None:
        return None, None, "request_error"
    response = record.get("response") or {}
    shadow = record.get("unvalidatedShadowObservation")
    candidate = record.get("researchCandidateObservation")
    channel = "publicForecasts"
    values = response
    if row.get("observationClass", "publicForecasts") == CANDIDATE_CLASS:
        require(shadow is None, "Research candidate was relabeled as an original-model shadow observation")
        if candidate is None:
            require(response.get("status") == "abstained", "Missing research candidate observation")
            return None, None, "no_forecast"
        channel, values = CANDIDATE_CLASS, candidate
        require(response.get("observationClass") == candidate.get("observationClass") == CANDIDATE_CLASS, "Research candidate classification mismatch")
        require(response.get("eligiblePublicPrediction") is False and candidate.get("eligiblePublicPrediction") is False and candidate.get("highConfidenceSupported") is False and candidate.get("validatedForBetting") is False, "Research candidate is incorrectly labeled")
        require(candidate.get("modelVersion") == response.get("modelVersion") and candidate.get("provenance") == response.get("provenance"), "Research candidate model/provenance mismatch")
        if row["modelVersion"] == MODEL_VERSIONS["probability_stack"]:
            require(response.get("provenance", {}).get("domainGateModelVersion") == MODEL_VERSIONS["corrected48"], "Probability stack has the wrong domain-gate model")
            require(response.get("coveragePolicy") == candidate.get("coveragePolicy") == STACK_COVERAGE, "Probability stack has the wrong coverage policy")
        require(all(response.get(side, {}).get("probability") is None for side in ("fighter1", "fighter2")), "Research candidate contains a public probability")
        require((response.get("status") == "research_observation" and response.get("reasonCodes") == [] and candidate.get("confidenceBandUnvalidated") is False) or
                (response.get("status") == "abstained" and response.get("reasonCodes") == ["unvalidated_confidence_band"] and candidate.get("confidenceBandUnvalidated") is True), "Research candidate bypasses a domain gate or mislabels its confidence band")
    elif response.get("status") == "abstained":
        require(candidate is None, "Research candidate was relabeled as a public-model observation")
        if shadow is None:
            return None, None, "no_forecast"
        channel, values = "unvalidatedShadowObservations", shadow
        require(response.get("reasonCodes") == ["unvalidated_confidence_band"] and shadow.get("reason") == "unvalidated_confidence_band", "Shadow observation bypasses a domain gate")
        require(shadow.get("eligiblePublicPrediction") is False and shadow.get("highConfidenceSupported") is False and shadow.get("validatedForBetting") is False, "Shadow observation is incorrectly labeled")
        require(shadow.get("modelVersion") == response.get("modelVersion") and shadow.get("provenance") == response.get("provenance"), "Shadow model/provenance mismatch")
        require(all(response.get(side, {}).get("probability") is None for side in ("fighter1", "fighter2")), "Public abstention contains a probability")
    else:
        require(response.get("status") == "forecast" and response.get("reasonCodes") == [] and shadow is None and candidate is None, "Invalid public forecast response")
    require(isinstance(row["modelVersion"], str) and re.fullmatch(r"sha256:[a-f0-9]{64}", row["modelVersion"]) and response.get("modelVersion") == row["modelVersion"], "Forecast model version mismatch")
    expected = {"eventDate": payload["eventDate"], "weightClass": payload["weightClass"], "scheduledRounds": payload["rounds"], "isTitleBout": payload["titleBout"]}
    require(all(response.get("context", {}).get(key) == value for key, value in expected.items()), "Response context differs from request")
    probabilities = []
    for side, id_field in (("fighter1", "firstID"), ("fighter2", "secondID")):
        item = values.get(side, {})
        p = item.get("probability")
        require(item.get("id") == payload[id_field] and type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1, "Invalid fighter probability")
        probabilities.append(p)
    require(math.isclose(sum(probabilities), 1, abs_tol=1e-12, rel_tol=0), "Fighter probabilities are not complementary")
    require(channel != "unvalidatedShadowObservations" or max(probabilities) >= 0.9, "Shadow observation is outside its declared confidence band")
    require(channel != "publicForecasts" or max(probabilities) < 0.9, "Public probability contradicts the recorder's high-confidence abstention policy")
    require(channel != CANDIDATE_CLASS or candidate["confidenceBandUnvalidated"] == (max(probabilities) >= 0.9), "Research candidate confidence-band flag contradicts its probability")
    return channel, probabilities[0], None


def wilson(wins: int, count: int) -> list[float] | None:
    if not count:
        return None
    z = 1.959963984540054
    p = wins / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    margin = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return [max(0, center - margin), min(1, center + margin)]


def rates(rows: list[dict]) -> dict:
    n, wins = len(rows), sum(row["correct"] for row in rows)
    return {"count": n, "wins": wins, "eventDates": len({row["eventDate"] for row in rows}),
            "meanConfidence": sum(row["confidence"] for row in rows) / n if n else None,
            "observedWinRate": wins / n if n else None, "wilson95": wilson(wins, n)}


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    return {**rates(rows), "eventDates": len({row["eventDate"] for row in rows}),
            "accuracy": sum(row["correct"] for row in rows) / n if n else None,
            "brier": sum((row["probability"] - row["outcomeFirstFighterWon"]) ** 2 for row in rows) / n if n else None,
            "logLoss": -sum(math.log(max(1e-15, min(1 - 1e-15, row["probability"] if row["outcomeFirstFighterWon"] else 1 - row["probability"]))) for row in rows) / n if n else None,
            "confidenceBins": [{"lower": lower, "upper": upper, **rates([row for row in rows if lower <= row["confidence"] and (row["confidence"] < upper or upper == 1)])} for lower, upper in zip(BINS, BINS[1:])],
            "atLeast90Percent": rates([row for row in rows if row["confidence"] >= 0.9])}


def grade_records(rows: list[dict], outcomes: dict, graded_at: datetime) -> list[dict]:
    graded = []
    for row in rows:
        payload = row["record"]["request"].get("payload")
        payload = payload if isinstance(payload, dict) else {}
        key = pair_key(payload)
        outcome = outcomes.get(key)
        item = {name: value for name, value in row.items() if name != "record"}
        response = row["record"].get("response") or {}
        item.update({"request": row["record"]["request"], "outcome": outcome, "exclusions": [], "channel": None, "probability": None,
                     "captureError": row["record"].get("error"), "publicStatus": response.get("status"),
                     "abstentionReasonCodes": response.get("reasonCodes", [])})
        reasons = item["exclusions"]
        if row["selection"] == "duplicate":
            reasons.append("duplicate_capture")
        if key is None:
            reasons.append("unresolved_identity")
        if item["captureError"] is not None:
            reasons.append("request_error")
        if event_date(row["eventDate"]) >= graded_at.date():
            reasons.append("event_not_completed")
        if not row["conservativeTimingEligible"]:
            reasons.append("capture_too_close_to_event_date")
        if payload.get("eventDate") != row["eventDate"] or any(field not in payload for field in CONTEXT_FIELDS):
            reasons.append("invalid_request_context")
        if key is not None and "invalid_request_context" not in reasons and "request_error" not in reasons:
            try:
                item["channel"], item["probability"], error = prediction(row, payload)
                if error:
                    reasons.append(error)
            except (ValueError, TypeError, KeyError) as error:
                reasons.append("invalid_prediction_record")
                item["predictionError"] = str(error)
        if outcome is None:
            reasons.append("missing_result")
        elif outcome["status"] != "decisive":
            reasons.append(outcome["status"])
        elif any(payload.get(field) != outcome[field] for field in CONTEXT_FIELDS):
            reasons.append("actual_context_mismatch")
        if not reasons:
            p = item["probability"]
            winner = payload["firstID"] if p > 0.5 else payload["secondID"] if p < 0.5 else min(payload["firstID"], payload["secondID"])
            item.update({"outcomeFirstFighterWon": int(outcome["winnerID"] == payload["firstID"]),
                         "correct": int(winner == outcome["winnerID"]), "confidence": max(p, 1 - p)})
        item["scored"] = not reasons
        graded.append(item)
    return graded


def grade(archive_paths: list[Path], outcomes_path: Path, output: Path) -> dict:
    require(all(not output.is_relative_to(path) for path in archive_paths), "Output must be outside every source archive")
    archives = [load_archive(path) for path in archive_paths]
    rows = select_records(archives)
    outcomes_bytes = outcomes_path.read_bytes()
    code_paths = [Path(__file__).resolve(), Path(__file__).with_name("record_prospective.py").resolve()]
    code_hashes = {str(path): digest(path.read_bytes()) for path in code_paths}
    output.mkdir(parents=True, exist_ok=False)
    write_bytes(output / "outcomes.json", outcomes_bytes)
    references = [{"path": str(archive["path"]), "manifestSHA256": archive["manifestSHA256"], "prePredictionSHA256": archive["prePredictionSHA256"],
                   "pipelineFingerprint": archive["pipelineFingerprint"], "pipeline": archive["pipeline"]} for archive in archives]
    write_json(output / "archive-references.json", references)
    write_json(output / "selection.json", [{key: value for key, value in row.items() if key != "record"} for row in rows])
    protocol = {"schemaVersion": 1, "frozenAt": now(), "codeSHA256": code_hashes,
                "purpose": "Local deterministic grading of caller-supplied outcomes; no model execution",
                "outcomeProvenance": "Caller-supplied source URLs and observedAt timestamps are not independently verified",
                "selectionRule": "For each model artifact hash, select earliest integrity-valid frozen capture by event date and unordered exact IDs, regardless of forecast/abstention/error; ties use archive hash then record filename. Different models remain paired observations, never independent bouts.",
                "unresolvedRule": "Unscoreable; within each model deduplicate only identical full-card input hashes plus boutId. Unknown model identities remain archive-specific.",
                "contextRule": "Exact IDs, advertised event date, actual division, scheduled rounds and title status must match",
                "timeRule": "Event date must be strictly before today's UTC date; captures must precede event date",
                "unknownStartTimeRule": "Without verified UTC start times, scoring requires capture strictly before 00:00 UTC on the day before the advertised date; closer captures are excluded conservatively",
                "confidenceBins": list(BINS), "probabilityTieRule": "Pick lexicographically smaller fighter ID at exactly 0.5",
                "logLossClip": 1e-15, "interval": "Wilson 95%, binomial approximation; does not account for event clustering",
                "stratification": "Public forecasts, original-model unvalidated shadow observations, and research candidate observations are separate, each grouped by model version and forecasting code/runtime fingerprint; rolling data hashes are excluded from that grouping",
                "candidateCoveragePolicy": CANDIDATE_COVERAGE,
                "claims": "Descriptive local results only; no automatic high-confidence validation or betting claim, no external preregistration"}
    write_json(output / "protocol.json", protocol)
    write_json(output / "pre-grading-manifest.json", {"frozenAt": now(), "files": {path.name: digest(path.read_bytes()) for path in output.iterdir() if path.is_file()}})
    graded_at = datetime.now(timezone.utc)
    outcomes = load_outcomes(outcomes_bytes, graded_at)
    graded = grade_records(rows, outcomes, graded_at)
    selected = [row for row in graded if row["selection"] == "selected"]
    channel_metrics = {}
    for channel in ("publicForecasts", "unvalidatedShadowObservations", CANDIDATE_CLASS):
        observation_class = CANDIDATE_CLASS if channel == CANDIDATE_CLASS else "publicForecasts"
        eligible = [row for row in selected if row["observationClass"] == observation_class]
        groups = defaultdict(list)
        for row in eligible:
            groups[(row["modelVersion"], row["pipelineFingerprint"])].append(row)
        strata = []
        for (version, pipeline), group in sorted(groups.items(), key=lambda item: (item[0][0] or "", item[0][1])):
            observations = [row for row in group if row["channel"] == channel]
            strata.append({"modelVersion": version, "pipelineFingerprint": pipeline,
                           "selectedRecords": len(group), "predictionCount": len(observations),
                           "coverageOfSelectedRecords": len(observations) / len(group),
                           **metrics([row for row in observations if row["scored"]])})
        prediction_count = sum(row["channel"] == channel for row in eligible)
        channel_metrics[channel] = {"selectedRecords": len(eligible), "predictionCount": prediction_count,
                                    "scoredCount": sum(row["count"] for row in strata),
                                    "coverageOfSelectedRecords": prediction_count / len(eligible) if eligible else None,
                                    "coverageDenominator": "Selected requests of this observation class, including errors and abstentions; each stratum uses only its own model/pipeline requests",
                                    "mixedModels": len({key[0] for key in groups}) > 1,
                                    "mixedPipelines": len({key[1] for key in groups}) > 1,
                                    "strata": strata}
    changed = []
    for archive in archives:
        for name, sha in archive["hashes"].items():
            path = archive["path"] / name
            if not path.is_file() or digest(path.read_bytes()) != sha:
                changed.append(str(path))
    changed.extend(str(path) for path in code_paths if digest(path.read_bytes()) != code_hashes[str(path)])
    keys = {pair_key(row["record"]["request"].get("payload") or {}) for row in rows}
    coverage_policies = [CANDIDATE_COVERAGE]
    if any(row["modelVersion"] == MODEL_VERSIONS["probability_stack"] for row in rows):
        coverage_policies = [STACK_COVERAGE]
        if any(row["modelVersion"] == MODEL_VERSIONS["expanded92"] for row in rows):
            coverage_policies.insert(0, CANDIDATE_COVERAGE)
    summary = {"gradedAt": graded_at.isoformat(), "inputIntegrityVerified": not changed, "changedInputs": changed,
               "counts": {"archiveRecords": len(rows), "selectedRecords": len(selected), "duplicateRecords": len(rows) - len(selected),
                          "uniqueSelectedResolvedBouts": len({pair_key(row["request"].get("payload")) for row in selected if pair_key(row["request"].get("payload")) is not None}),
                          "selectedResolvedPairs": sum("unresolved_identity" not in row["exclusions"] for row in selected),
                          "selectedUnresolvedIdentities": sum("unresolved_identity" in row["exclusions"] for row in selected),
                          "selectedScored": sum(row["scored"] for row in selected), "selectedExcluded": sum(not row["scored"] for row in selected)},
               "selectedExclusionCounts": dict(Counter(reason for row in selected for reason in row["exclusions"])),
               "exclusionCountsMayOverlap": True, "metrics": channel_metrics,
               "unmatchedOutcomes": [item for key, item in outcomes.items() if key not in keys],
               "highConfidenceSupported": False, "validatedForBetting": False,
               "limitations": ["Caller-supplied outcomes are not independently verified.", "Local hashes are not an external timestamp authority.", "Repeated grading is not a predeclared stopping rule; descriptive intervals do not establish validation.", "Wilson intervals treat bouts as independent and do not adjust for event clustering.", "Different models observing the same bout are paired observations, not additional independent fights.", *coverage_policies]}
    write_json(output / "graded-records.json", graded)
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", {"schemaVersion": 1, "completedAt": now(), "inputIntegrityVerified": not changed,
                                         "files": {path.name: digest(path.read_bytes()) for path in output.iterdir() if path.is_file()}})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Grade new recorder archives offline into a new directory; never rerun the model. Final outcomes require an event date before today's UTC date and observation on/after that date.",
                                     epilog="Outcomes JSON: schemaVersion=1, outcomes=[{firstID,secondID,eventDate,weightClass,rounds,titleBout,status,winnerID,sourceUrl,observedAt}]. rounds means scheduled rounds. status: decisive, draw, no_contest, cancelled, changed; winnerID is null unless decisive.")
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = grade([path.resolve() for path in args.archive], args.outcomes.resolve(), args.output.resolve())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Grading failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output.resolve()), "counts": result["counts"], "inputIntegrityVerified": result["inputIntegrityVerified"]}, indent=2))
    return 0 if result["inputIntegrityVerified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
