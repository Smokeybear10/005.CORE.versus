import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from io import BytesIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUEST_FIELDS = ("firstID", "secondID", "eventDate", "weightClass", "rounds", "titleBout")
MODEL_VERSIONS = {
    "original48": "sha256:915d2a35f6a3aacc320dc11da0f302e6260c43f070ed12c43f82cd4476ac0321",
    "expanded92": "sha256:8652fb31f9a8c1b33ba0e6f7cfc90d2c423a220d0d4f2ea8c72480b2a57cbbf4",
    "corrected48": "sha256:6fde3bc72e4788c944efa65ad8a911c2e18d840404a337e2cd09559a8530a924",
    "probability_stack": "sha256:54230545da5225df536b096facee3ec90f80e7f18e893fa463247a06b7f2174b",
}
PREDICTOR_SHA256 = "4d241a11ad78ffa49c155d9850bec573f25ee357b2e16d69193153bc53429ca0"
CANDIDATE_CLASS = "researchCandidateObservations"
CANDIDATE_COVERAGE = "Conservative original-model domain gates apply. This selected prospective coverage is not directly comparable to historical replay accuracy across all decisive known-identity bouts."
STACK_COVERAGE = "Conservative corrected48 domain gates apply. This research-only prospective coverage is not directly comparable to historical replay accuracy across all decisive known-identity bouts."
STACK_PROTOCOL_SHA256 = "4d2e01cb79c64427b1cbc4e60c84e93117fc7f00f38be8d476301c2770520e51"
CURRENT_CODE_SHA256 = {
    "backend/forecast.py": "dd2d3b4712a655eaf997822a98af925d3ee31ebc5201ff917773f0f36b776a4e",
    "backend/research.py": "7766d5417c8b1ff67d3eb1165415afeabbe361170c0a9e2bbc4785cae1c4bb73",
    "backend/model_engine.py": "294326a027fddaf4e2846183beab1ec4fc699e946df573e0a18251e58d062590",
    "scripts/evaluate_research.py": "b657ad7ac141166d629b1af55415f94d9bb31ac4e5dbc2776d634c2d73743859",
    "scripts/evaluate_challenger.py": "012f3d9b51c5f92c1840a8ef940859fea569e1487ca0a7560e32d5339a277825",
}


def research_candidate(model: str) -> bool:
    return model in ("expanded92", "probability_stack")


def candidate_coverage(model: str) -> str:
    return STACK_COVERAGE if model == "probability_stack" else CANDIDATE_COVERAGE


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def capture_deadline(event_date: str) -> datetime:
    return datetime.combine(date.fromisoformat(event_date) - timedelta(days=1), time.min, timezone.utc)


def write_bytes(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def write_json(path: Path, value: object) -> None:
    write_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def validate_card(card: object) -> None:
    if not isinstance(card, dict) or card.get("schemaVersion") != 1 or card.get("fullCard") is not True:
        raise ValueError("The input must declare schemaVersion: 1 and fullCard: true")
    event = card.get("event")
    if not isinstance(event, dict) or not isinstance(event.get("date"), str):
        raise ValueError("Include event.date in YYYY-MM-DD format")
    event_date = date.fromisoformat(event["date"])
    if event_date.isoformat() != event["date"] or event_date <= datetime.now(timezone.utc).date():
        raise ValueError("A prospective card must have a date strictly after today's UTC date")
    if datetime.now(timezone.utc) >= capture_deadline(event["date"]):
        raise ValueError("Record before 00:00 UTC on the day before the advertised event date; no verified start time is available")
    for field in ("sources", "contextAssumptions"):
        if not isinstance(card.get(field), (dict, list, str)) or not card[field]:
            raise ValueError(f"Include explicit {field} in the input")
    requests = card.get("requests")
    if not isinstance(requests, list) or not requests or type(card.get("expectedBoutCount")) is not int or card["expectedBoutCount"] != len(requests):
        raise ValueError("expectedBoutCount must equal the nonempty full requests list")
    ids = [item.get("boutId") if isinstance(item, dict) else None for item in requests]
    if any(not isinstance(bout_id, str) or not bout_id.strip() for bout_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Every request needs a unique nonempty boutId")
    pairs = set()
    for item in requests:
        payload = item.get("payload")
        if not isinstance(payload, dict) or not isinstance(payload.get("eventDate"), str):
            continue
        pair = (payload.get("firstID"), payload.get("secondID"))
        if all(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{16}", value) for value in pair):
            key = (payload.get("eventDate"), *sorted(pair))
            if key in pairs:
                raise ValueError("The card contains a duplicate fighter pair on the same event date")
            pairs.add(key)


def input_paths(data: Path, model: str = "original48") -> list[Path]:
    code = ["scripts/record_prospective.py", "scripts/evaluate_research.py", "scripts/evaluate_challenger.py",
            "backend/forecast.py", "backend/research.py", "backend/model_engine.py", "requirements.txt", ".python-version"]
    if model in ("corrected48", "probability_stack"):
        runtime = data / "evaluation/identity-repair-v1"
        paths = [*(ROOT / name for name in code), runtime / "runtime-manifest.json", runtime / "candidate_contract.json",
                 runtime / "models/provisional_challenger.joblib", runtime / "models/development_and_calibration_features.parquet",
                 *[runtime / folder / name for folder in ("research-data", "inference")
                   for name in ("manifest.json", "fights.parquet", "rounds.parquet", "fighters.parquet")]]
        if model == "probability_stack":
            paths.extend([ROOT / name for name in ("scripts/predictor_features.py", "scripts/evaluate_predictors.py", "scripts/evaluate_probability_stack.py")])
            paths.extend(data / "evaluation/probability-stack-v1" / name for name in ("probability_stack.joblib", "protocol.json", "pre-replay-manifest.json"))
        return paths
    legacy = ["model_v2.pkl", "model.pkl", "ufc-master.csv", "fighter_snapshots.parquet", "fights.parquet", "events.parquet", "rounds.parquet"]
    evaluation = ["candidate_contract.json", "challenger-v1/provisional_challenger.joblib",
                  "challenger-v1/development_and_calibration_features.parquet"]
    if model == "expanded92":
        code.extend(["scripts/predictor_features.py", "scripts/evaluate_predictors.py"])
        evaluation.append("expanded-predictors-v1/expanded_blend.joblib")
    bundles = [data / directory / name for directory in ("research-data", "evaluation/inference")
               for name in ("manifest.json", "fights.parquet", "rounds.parquet", "fighters.parquet")]
    return [*(ROOT / name for name in code), *(data / name for name in legacy),
            *(data / "evaluation" / name for name in evaluation), *bundles]


def fingerprints(paths: list[Path]) -> dict:
    result = {}
    for path in paths:
        name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        result[name] = digest(path.read_bytes()) if path.is_file() else None
    return result


class ResearchCandidateEngine:
    model_name = "expanded92"
    gate_model_name = "original48"
    coverage_policy = CANDIDATE_COVERAGE
    artifact_label = "expanded_blend"
    learner_label = "equal_weight_gradient_boosting_logistic_regression"
    calibration_method = "positive temperature on equal-weight symmetric blend"
    extra_provenance = {}

    def __init__(self, gate, data: Path):
        import joblib
        if not gate.available:
            raise RuntimeError("The original domain-gate engine is unavailable")
        content = (data / "evaluation/expanded-predictors-v1/expanded_blend.joblib").read_bytes()
        if "sha256:" + digest(content) != MODEL_VERSIONS["expanded92"]:
            raise ValueError("Research candidate artifact checksum mismatch")
        if digest((ROOT / "scripts/predictor_features.py").read_bytes()) != PREDICTOR_SHA256:
            raise ValueError("Research candidate feature code checksum mismatch")
        from scripts import predictor_features
        artifact = joblib.load(BytesIO(content))
        for name, checksum in artifact["codeHashes"].items():
            if digest((ROOT / "scripts" / name).read_bytes()) != checksum:
                raise ValueError(f"Research candidate source code changed: {name}")
        for name, checksum in artifact["sourceHashes"].items():
            if digest((data / "evaluation/inference" / name).read_bytes()) != checksum:
                raise ValueError(f"Research candidate inference source changed: {name}")
        expected = [name for name in gate.feature_columns if name != "weightclass"] + predictor_features.ADDED_FEATURES + ["weightclass"]
        if artifact["name"] != "expanded_blend" or artifact["columns"] != expected or len(expected) != 92 or len(artifact["models"]) != 2:
            raise ValueError("Research candidate does not match the frozen 92-feature blend")
        if not math.isfinite(artifact["temperature"]) or artifact["temperature"] <= 0 or any(list(model.classes_) != [0, 1] for model in artifact["models"]):
            raise ValueError("Research candidate calibration or outcome classes are invalid")
        self.gate, self.artifact, self.features = gate, artifact, predictor_features

    def status(self) -> dict:
        status = deepcopy(self.gate.status())
        status.update({"modelVersion": MODEL_VERSIONS[self.model_name], "artifact": self.artifact_label,
                       "learner": self.learner_label,
                       "training": {"through": self.artifact["trainedThrough"]},
                       "calibration": {"through": self.artifact["calibratedThrough"], "temperature": self.artifact["temperature"]},
                       "audit": None, "observationClass": CANDIDATE_CLASS, "eligiblePublicPrediction": False,
                       "domainGateModelVersion": MODEL_VERSIONS[self.gate_model_name], "coveragePolicy": self.coverage_policy})
        return status

    @lru_cache(maxsize=64)
    def _features(self, *args):
        import pandas as pd
        frame, _ = self.gate._features(*args)
        frame = frame.assign(date=pd.Timestamp(args[2]), red_fighter_id=args[0], blue_fighter_id=args[1])
        return self.features.add_predictors(frame, self.gate.fights, self.gate.rounds, self.gate.fighters)[self.artifact["columns"]]

    def predict(self, *args) -> dict:
        import numpy as np
        response = deepcopy(self.gate.predict(*args))
        gate_codes = response["reasonCodes"]
        domain_supported = not gate_codes or gate_codes == ["unvalidated_confidence_band"]
        response.update({"modelVersion": MODEL_VERSIONS[self.model_name], "observationClass": CANDIDATE_CLASS,
                         "eligiblePublicPrediction": False, "coveragePolicy": self.coverage_policy,
                         "researchCandidateObservation": None})
        for side in ("fighter1", "fighter2"):
            response[side]["probability"] = None
        response["provenance"].update({"trainingThrough": self.artifact["trainedThrough"],
                                       "calibrationThrough": self.artifact["calibratedThrough"],
                                       "domainGateModelVersion": MODEL_VERSIONS[self.gate_model_name],
                                       "candidateFeatureCode": "sha256:" + PREDICTOR_SHA256, **self.extra_provenance})
        response["calibration"] = {"through": self.artifact["calibratedThrough"], "temperature": self.artifact["temperature"],
                                   "method": self.calibration_method}
        response["warnings"] = ["Unvalidated research candidate observation; never an eligible public prediction.", self.coverage_policy,
                                "Supplied matchup context is hypothetical unless separately verified; probabilities are conditional on a decisive result.",
                                "Height/reach are retrospective snapshots with no original historical observation timestamps."]
        if not domain_supported:
            return response
        frame = self._features(*args)
        numeric = frame.drop(columns="weightclass").to_numpy(dtype=float)
        if np.isinf(numeric).any():
            raise ValueError("Research candidate features contain unsupported numeric inputs")
        response["missingInputs"] = [name for name in frame if name != "weightclass" and frame[name].isna().iloc[0]]
        if response["missingInputs"]:
            response["warnings"].append(f"{len(response['missingInputs'])} missing inputs use the candidate's frozen training preprocessing.")
        raw = self._raw_probability(frame)
        probability = float(self.gate.apply_temperature(raw, self.artifact["temperature"])[0])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Research candidate returned an invalid probability")
        high_band = max(probability, 1 - probability) >= 0.9
        response["status"] = "abstained" if high_band else "research_observation"
        response["reasonCodes"] = ["unvalidated_confidence_band"] if high_band else []
        response["reasons"] = ["The candidate's 90% band is unvalidated; retained only as a research observation."] if high_band else []
        response["researchCandidateObservation"] = {
            "label": "Unvalidated research candidate observation; not an eligible public prediction",
            "observationClass": CANDIDATE_CLASS, "eligiblePublicPrediction": False,
            "highConfidenceSupported": False, "validatedForBetting": False, "confidenceBandUnvalidated": high_band,
            "fighter1": {"id": args[0], "probability": probability},
            "fighter2": {"id": args[1], "probability": 1 - probability},
            "symmetricProbabilityBeforeTemperature": float(raw[0]), "temperature": self.artifact["temperature"],
            "modelVersion": MODEL_VERSIONS[self.model_name], "provenance": deepcopy(response["provenance"]),
            "coveragePolicy": self.coverage_policy,
        }
        return response

    def _raw_probability(self, frame):
        import numpy as np
        return np.mean([self.gate.symmetric_probability(model, frame) for model in self.artifact["models"]], axis=0)


class ProbabilityStackEngine(ResearchCandidateEngine):
    model_name = "probability_stack"
    gate_model_name = "corrected48"
    coverage_policy = STACK_COVERAGE
    artifact_label = "probability_stack"
    learner_label = "convex_gradient_boosting48_logistic92"
    calibration_method = "positive temperature on fixed symmetric convex blend"

    def __init__(self, gate, data: Path):
        import joblib
        if not gate.available or gate.model_version != MODEL_VERSIONS["corrected48"]:
            raise ValueError("The stack requires the pinned corrected48 domain-gate model")
        directory = data / "evaluation/probability-stack-v1"
        content = (directory / "probability_stack.joblib").read_bytes()
        if "sha256:" + digest(content) != MODEL_VERSIONS[self.model_name]:
            raise ValueError("Probability stack artifact checksum mismatch")
        protocol_bytes = (directory / "protocol.json").read_bytes()
        if digest(protocol_bytes) != STACK_PROTOCOL_SHA256:
            raise ValueError("Probability stack protocol checksum mismatch")
        protocol = json.loads(protocol_bytes)
        recorded_root = next(Path(name).parents[1] for name in protocol["hashes"]
                             if name.endswith("/scripts/evaluate_probability_stack.py"))
        for name in ("scripts/predictor_features.py", "scripts/evaluate_predictors.py", "scripts/evaluate_probability_stack.py",
                     "scripts/evaluate_research.py", "scripts/evaluate_challenger.py"):
            if digest((ROOT / name).read_bytes()) != protocol["hashes"][str(recorded_root / name)]:
                raise ValueError(f"Probability stack feature/training code changed: {name}")
        for name in ("fights.parquet", "rounds.parquet", "fighters.parquet"):
            expected = protocol["hashes"][str(recorded_root / "data/evaluation/identity-repaired-source-v1" / name)]
            if digest((data / "evaluation/identity-repair-v1/inference" / name).read_bytes()) != expected:
                raise ValueError(f"Probability stack corrected source changed: {name}")
        from scripts import predictor_features
        artifact = joblib.load(BytesIO(content))
        expected = [name for name in gate.feature_columns if name != "weightclass"] + predictor_features.ADDED_FEATURES + ["weightclass"]
        if (artifact["protocolSHA256"] != STACK_PROTOCOL_SHA256 or artifact["columns"]["original"] != gate.feature_columns
                or artifact["columns"]["expanded"] != expected or len(expected) != 92
                or artifact["linearWeight"] != 0.6225102979980173):
            raise ValueError("Probability stack feature or mixing contract changed")
        models = [artifact["models"][name] for name in ("gb48", "lr92_strong")]
        temperature = artifact["temperatures"]["candidate_stack"]
        if not math.isfinite(temperature) or temperature <= 0 or any(list(model.classes_) != [0, 1] for model in models):
            raise ValueError("Probability stack calibration or classes are invalid")
        self.gate, self.features, self.linear_weight = gate, predictor_features, artifact["linearWeight"]
        self.artifact = {"models": models, "columns": expected, "temperature": temperature,
                         "trainedThrough": "2022-12-17", "calibratedThrough": "2024-12-14"}
        self.extra_provenance = {"candidateProtocol": "sha256:" + STACK_PROTOCOL_SHA256, "linearWeight": self.linear_weight}

    def _raw_probability(self, frame):
        gb, linear = self.artifact["models"]
        return (self.linear_weight * self.gate.symmetric_probability(linear, frame)
                + (1 - self.linear_weight) * self.gate.symmetric_probability(gb, frame))


def initialize_engine(data: Path, model: str = "original48"):
    if model not in MODEL_VERSIONS:
        raise ValueError("Unsupported prospective model")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "backend"))
    from forecast import ForecastEngine
    from model_engine import FightPredictor
    from research import ResearchData

    predictor = FightPredictor()
    if model in ("corrected48", "probability_stack"):
        for name, checksum in CURRENT_CODE_SHA256.items():
            if digest((ROOT / name).read_bytes()) != checksum:
                raise ValueError(f"Corrected forecasting code changed: {name}")
        runtime = data / "evaluation/identity-repair-v1"
        if digest((runtime / "runtime-manifest.json").read_bytes()) != "93e7ff9758df33fe536039027f90c882e50808b88e7c09183eb368f2de177e59":
            raise ValueError("Corrected runtime manifest checksum mismatch")
        artifact = runtime / "models/provisional_challenger.joblib"
        research = ResearchData(predictor, str(artifact), bundle_directory=runtime / "research-data")
        gate = ForecastEngine(artifact, runtime / "candidate_contract.json", research, runtime / "inference/fighters.parquet")
        if not gate.available or gate.model_version != MODEL_VERSIONS["corrected48"]:
            raise ValueError("The pinned corrected48 model and data are unavailable or mismatched")
        return ProbabilityStackEngine(gate, data) if model == "probability_stack" else gate
    artifact = data / ("model_v2.pkl" if (data / "model_v2.pkl").exists() else "model.pkl")
    if artifact.is_file():
        predictor.load_artifact(str(artifact), str(data / "ufc-master.csv"))
    research = ResearchData(predictor, str(artifact))
    gate = ForecastEngine(data / "evaluation/challenger-v1/provisional_challenger.joblib",
                          data / "evaluation/candidate_contract.json", research,
                          data / "evaluation/inference/fighters.parquet")
    return ResearchCandidateEngine(gate, data) if model == "expanded92" else gate


def record_request(engine, item: dict, event_date: str, initialization_error: str | None = None) -> dict:
    record = {"boutId": item["boutId"], "request": item, "startedAt": now(),
              "response": None, "unvalidatedShadowObservation": None, "researchCandidateObservation": None, "error": None}
    try:
        payload = item.get("payload")
        if not isinstance(payload, dict) or set(payload) != set(REQUEST_FIELDS):
            raise ValueError(f"payload must contain exactly: {', '.join(REQUEST_FIELDS)}")
        if payload["eventDate"] != event_date:
            raise ValueError("The request date must match the frozen card date")
        if datetime.now(timezone.utc) >= capture_deadline(event_date):
            raise ValueError("The conservative capture deadline has passed (00:00 UTC on the day before the advertised event date)")
        if initialization_error:
            raise RuntimeError(initialization_error)
        args = tuple(payload[field] for field in REQUEST_FIELDS)
        response = engine.predict(*args)
        record["researchCandidateObservation"] = response.pop("researchCandidateObservation", None)
        record["response"] = response
        if response.get("observationClass") != CANDIDATE_CLASS and response.get("status") == "abstained" and response.get("reasonCodes") == ["unvalidated_confidence_band"]:
            frame, _ = engine._features(*args)
            raw = engine.symmetric_probability(engine.model, frame[engine.feature_columns])
            probability = float(engine.apply_temperature(raw, engine.temperature)[0])
            if not math.isfinite(probability) or not 0 <= probability <= 1 or max(probability, 1 - probability) < 0.9:
                raise ValueError("The internal observation does not match the confidence-band abstention")
            record["unvalidatedShadowObservation"] = {
                "label": "Unvalidated research observation; not an eligible public prediction",
                "eligiblePublicPrediction": False, "highConfidenceSupported": False,
                "validatedForBetting": False, "reason": "unvalidated_confidence_band",
                "fighter1": {"id": payload["firstID"], "probability": probability},
                "fighter2": {"id": payload["secondID"], "probability": 1 - probability},
                "symmetricProbabilityBeforeTemperature": float(raw[0]),
                "temperature": engine.temperature, "modelVersion": response["modelVersion"],
                "provenance": response["provenance"],
            }
        if datetime.now(timezone.utc) >= capture_deadline(event_date):
            raise ValueError("Prediction finished after the conservative capture deadline; this record cannot be scored")
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    record["finishedAt"] = now()
    return record


def archive_hashes(output: Path) -> dict:
    return {str(path.relative_to(output)): digest(path.read_bytes())
            for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"}


def record_card(card_path: Path, output: Path, data: Path, linked_manifest: Path | None = None, model: str = "original48") -> dict:
    if model not in MODEL_VERSIONS:
        raise ValueError(f"Choose one of: {', '.join(MODEL_VERSIONS)}")
    content = card_path.read_bytes()
    card = json.loads(content)
    validate_card(card)
    linked_bytes = linked_manifest.read_bytes() if linked_manifest else None
    if linked_bytes is not None:
        json.loads(linked_bytes)
    paths = input_paths(data, model)
    before = fingerprints(paths)
    output.mkdir(parents=True, exist_ok=False)
    write_bytes(output / "input.json", content)
    if linked_bytes is not None:
        write_bytes(output / "linked-manifest.json", linked_bytes)
    write_json(output / "input-fingerprints.json", {"capturedAt": now(), "files": before})
    engine = None
    initialization_error = None
    try:
        engine = initialize_engine(data, model)
        if fingerprints(paths) != before:
            raise RuntimeError("Model, data or code changed during initialization; no predictions will be made")
        status = engine.status()
    except Exception as error:
        initialization_error = f"{type(error).__name__}: {error}"
        status = {"available": False, "error": initialization_error}
    status.update({"requestedModel": model, "requestedModelVersion": MODEL_VERSIONS[model],
                   "observationClass": CANDIDATE_CLASS if research_candidate(model) else "publicForecasts"})
    write_json(output / "model-status-before.json", status)
    versions = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "fastparquet", "pyarrow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    protocol = {
        "schemaVersion": 1, "frozenAt": now(), "eventDate": card["event"]["date"],
        "rosterSize": len(card["requests"]), "purpose": "Local full-card research observation",
        "requestedModel": model, "requestedModelVersion": MODEL_VERSIONS[model],
        "observationClass": status["observationClass"],
        "candidateCoveragePolicy": candidate_coverage(model) if research_candidate(model) else None,
        "registrationScope": "Local hashes and clock only; no external preregistration or timestamp authority",
        "captureTiming": "Actual recorded time; not a fixed historical day-before cutoff",
        "captureBeforeUTC": capture_deadline(card["event"]["date"]).isoformat().replace("+00:00", "Z"),
        "unknownStartTimeRule": "Without verified UTC start times, capture must finish before 00:00 UTC on the day before the advertised date",
        "linkedArchiveManifestSHA256": digest(linked_bytes) if linked_bytes is not None else None,
        "linkedArchivePolicy": "A new observation, never a replacement or backdated registration",
        "selectionRule": "Every supplied request in source order, including invalid identities, abstentions and errors",
        "fullCardVerification": "Declared by caller; this offline recorder does not independently verify the source roster",
        "contextRule": "Explicit supplied assumptions retained in input.json; no claim of verified scheduled conditions",
        "shadowRule": "Record internal probability only when the sole public abstention reason is unvalidated_confidence_band",
        "domainRule": "Never bypass identity, date, division, history, age, layoff or other domain gates",
        "requestRule": "One engine call per request; no retries, context changes, fitting or outcome scoring",
        "gradingRule": "Any later outcomes belong in separate linked artifacts; this archive contains no performance claim",
        "python": platform.python_version(), "packages": versions,
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "pre-prediction-manifest.json", {"frozenAt": now(), "files": archive_hashes(output)})
    (output / "responses").mkdir()
    records = []
    for index, item in enumerate(card["requests"], 1):
        record = record_request(engine, item, card["event"]["date"], initialization_error)
        response_file = output / "responses" / f"{index:03}.json"
        write_json(response_file, record)
        records.append({"boutId": item["boutId"], "responseFile": str(response_file.relative_to(output)),
                        "sha256": digest(response_file.read_bytes()),
                        "status": (record["response"] or {}).get("status", "error"),
                        "hasError": record["error"] is not None,
                        "hasResearchCandidateObservation": record["researchCandidateObservation"] is not None,
                        "hasUnvalidatedShadowObservation": record["unvalidatedShadowObservation"] is not None})
    after = fingerprints(paths)
    write_json(output / "input-fingerprints-after.json", {"capturedAt": now(), "files": after})
    changed = [name for name in before if before[name] != after[name]]
    summary = {
        "finishedAt": now(), "requests": len(records), "records": records,
        "publicForecasts": sum(item["status"] == "forecast" for item in records),
        "abstentions": sum(item["status"] == "abstained" for item in records),
        "errors": sum(item["hasError"] for item in records),
        "unvalidatedShadowObservations": sum(item["hasUnvalidatedShadowObservation"] for item in records),
        "researchCandidateObservations": sum(item["hasResearchCandidateObservation"] for item in records),
        "inputIntegrityVerified": not changed, "changedInputs": changed,
        "eligibleForPerformanceClaims": False,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", {"schemaVersion": 1, "completedAt": now(),
                                           "inputIntegrityVerified": not changed, "files": archive_hashes(output)})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Record an offline full-card research observation before 00:00 UTC on the day before the advertised event date; actual capture timestamps are retained.",
                                     epilog="Input JSON: schemaVersion=1, fullCard=true, expectedBoutCount, event={date,...}, sources, contextAssumptions, requests=[{boutId,payload:{firstID,secondID,eventDate,weightClass,rounds,titleBout}}].")
    parser.add_argument("card", type=Path, help="Full-card request JSON, including sources and context assumptions")
    parser.add_argument("output", type=Path, help="New archive directory; existing paths are never overwritten")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--linked-manifest", type=Path, help="Original archive manifest to preserve and link by hash")
    parser.add_argument("--model", choices=tuple(MODEL_VERSIONS), default="original48", help="expanded92 and probability_stack are frozen research candidates, never public forecasts")
    args = parser.parse_args()
    try:
        result = record_card(args.card.resolve(), args.output.resolve(), args.data_dir.resolve(),
                             args.linked_manifest.resolve() if args.linked_manifest else None, args.model)
    except (OSError, ValueError) as error:
        print(f"Recording failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"archive": str(args.output.resolve()), **{key: result[key] for key in
                     ("requests", "publicForecasts", "abstentions", "errors", "unvalidatedShadowObservations", "researchCandidateObservations", "inputIntegrityVerified")}}, indent=2))
    return 0 if result["inputIntegrityVerified"] and not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
