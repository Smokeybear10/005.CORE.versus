import hashlib
import json
import math
import re
from datetime import date, timedelta
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from research import ResearchData


MODEL_VERSION = "sha256:915d2a35f6a3aacc320dc11da0f302e6260c43f070ed12c43f82cd4476ac0321"
TRAINING_FEATURES_SHA = "8bdb92c642b3c73200ec0aaeb0f70cebb2817eda979a175cdaf5ef8880995864"
AUDITED_MODELS = {
    MODEL_VERSION: {"trainingFeaturesSha256": TRAINING_FEATURES_SHA, "artifact": "challenger-v1"},
    "sha256:6fde3bc72e4788c944efa65ad8a911c2e18d840404a337e2cd09559a8530a924": {
        "trainingFeaturesSha256": "39d191e73b1b0ff903114dac125e245afbe372db1670097f16dcc392e1a51c98",
        "artifact": "identity-repair-v1",
        "contractSha256": "eb92479bf7986df6451919df2f4698e3e5643b3a74f3ffe178eefb030cc0c445",
        "inferenceManifestSha256": "ea21687f2dad98f247168cfd0bbc99a059f6ab5c80c6a0143afe4744797fd3ad",
        "researchManifestSha256": "027e79362e4631dcc47b2f52c797893bc7431d9c31ec1803422c9d5e745b0839",
        "evaluationSha256": "2a3609d2f809eea922d41af5f24ec9c0bbc2b02ea5a05f6c1093beb20d1ee1ac",
    },
}
STANDARD_DIVISIONS = (
    "Flyweight", "Bantamweight", "Featherweight", "Lightweight", "Welterweight",
    "Middleweight", "Light Heavyweight", "Heavyweight", "Women's Strawweight",
    "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight",
)
MINIMUM_STAT_BOUTS = 3
FORECAST_HORIZON_DAYS = 365
DOMAIN_FEATURE_LABELS = {
    "ctx_min_age": "younger fighter's age",
    "ctx_max_age": "older fighter's age",
    "diff_age": "difference in ages",
    "diff_age_squared": "age comparison",
    "diff_layoff_days": "difference in time since each fighter's last recorded bout",
}


def normalized_division(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    division = value.removeprefix("Interim ")
    return division if division in STANDARD_DIVISIONS else None


class ForecastEngine:
    def __init__(self, artifact_path: str | Path, contract_path: str | Path,
                 research_data: ResearchData, fighter_identity_path: str | Path):
        self.available = False
        self.error = "The audited forecast model is unavailable. Matchup research is still available."
        self.research_data = research_data
        self.contract = {}
        self.model_version = None
        self.supported_divisions = []
        self.forecast_date_range = None
        self.domain_bounds = None
        self.names = {fighter_id: name for name, fighter_id in research_data.fighters.values()}
        try:
            from scripts.evaluate_challenger import build_inference_features
            from scripts.evaluate_research import apply_temperature, prepare_round_stats, symmetric_probability

            artifact_path = Path(artifact_path)
            identity_path = Path(fighter_identity_path)
            contract_bytes = Path(contract_path).read_bytes()
            self.contract = json.loads(contract_bytes)
            self.model_version = self.contract.get("modelVersion")
            self.model_spec = AUDITED_MODELS.get(self.model_version)
            if self.model_spec is None or self.contract.get("supports90PercentClaims") is not False:
                raise ValueError("Forecast contract does not match the audited model")
            if self.model_spec.get("contractSha256") and hashlib.sha256(contract_bytes).hexdigest() != self.model_spec["contractSha256"]:
                raise ValueError("Forecast contract checksum does not match its model")
            artifact_bytes = artifact_path.read_bytes()
            if "sha256:" + hashlib.sha256(artifact_bytes).hexdigest() != self.model_version:
                raise ValueError("Forecast model checksum does not match its audit")
            bundle = identity_path.parent
            manifest_bytes = (bundle / "manifest.json").read_bytes()
            manifest = json.loads(manifest_bytes)
            if self.model_spec.get("inferenceManifestSha256") and hashlib.sha256(manifest_bytes).hexdigest() != self.model_spec["inferenceManifestSha256"]:
                raise ValueError("Forecast inputs do not match the model's corrected data")
            files = manifest.get("files")
            names = ("fights.parquet", "rounds.parquet", "fighters.parquet")
            if manifest.get("schemaVersion") != 1 or manifest.get("candidateModelVersion") != self.model_version or not isinstance(files, dict) or set(files) != set(names):
                raise ValueError("Forecast data manifest is incomplete")
            research_digest = research_data.research_manifest_sha256
            if self.model_spec.get("researchManifestSha256") and research_digest != self.model_spec["researchManifestSha256"]:
                raise ValueError("Research data does not match the model's corrected data")
            if manifest.get("researchManifestSha256") != research_digest or research_data.refresh_status != "loaded":
                raise ValueError("Forecast inputs do not match current research data")
            frames = {}
            for name in names:
                content = (bundle / name).read_bytes()
                if hashlib.sha256(content).hexdigest() != files[name]:
                    raise ValueError("Forecast data checksum mismatch")
                frames[name] = pd.read_parquet(BytesIO(content))
            self.fights = frames["fights.parquet"]
            self.rounds = frames["rounds.parquet"]
            self.fighters = frames["fighters.parquet"]
            if self.fighters.fighter_id.duplicated().any() or "dob" not in self.fighters:
                raise ValueError("Forecast fighter identities are invalid")
            self.birth_dates = pd.to_datetime(self.fighters.set_index("fighter_id").dob, errors="coerce").to_dict()
            self.data_through = str(self.fights.date.max().date())
            if self.data_through != manifest.get("dataThrough") or self.data_through != research_data.data_through:
                raise ValueError("Forecast and research dates do not match")
            artifact = joblib.load(BytesIO(artifact_bytes))
            self.model = artifact["model"]
            self.feature_columns = artifact["feature_columns"]
            self.temperature = artifact["temperature"]
            if self.feature_columns != self.contract["featureColumns"] or not math.isfinite(self.temperature) or self.temperature <= 0 or self.temperature != self.contract["calibration"]["temperature"]:
                raise ValueError("Forecast preprocessing does not match the audit")
            if list(self.model.classes_) != [0, 1]:
                raise ValueError("Forecast outcome classes are unsupported")
            learned = self.model.named_steps["preprocess"].named_transformers_["division"].categories_[0]
            self.supported_divisions = [division for division in STANDARD_DIVISIONS if division in learned]
            self.totals, self.round_quality = prepare_round_stats(self.rounds)
            self.build_inference_features = build_inference_features
            self.symmetric_probability = symmetric_probability
            self.apply_temperature = apply_temperature
            self.source_manifest = "sha256:" + research_digest
            self.inference_manifest = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
            self.domain_bounds = self._load_domain_bounds(artifact_path.parent)
            self.forecast_date_range = {
                "min": self.data_through,
                "max": str(date.fromisoformat(self.data_through) + timedelta(days=FORECAST_HORIZON_DAYS)),
                "basis": "researchDataThrough",
            }
            self.available = True
            self.error = None
        except (OSError, ValueError, TypeError, KeyError, AttributeError, ImportError, EOFError):
            self.available = False
            self.contract = {}
            self.supported_divisions = []
            self.forecast_date_range = None
            self.domain_bounds = None

    def _load_domain_bounds(self, artifact_directory: Path) -> dict:
        content = (artifact_directory / "development_and_calibration_features.parquet").read_bytes()
        if hashlib.sha256(content).hexdigest() != self.model_spec["trainingFeaturesSha256"]:
            raise ValueError("Forecast training features do not match the frozen source")
        features = pd.read_parquet(BytesIO(content))
        training = features[features.date.between(self.contract["trainedOn"]["from"], self.contract["trainedOn"]["through"])]
        if len(training) != self.contract["trainedOn"]["decisiveBouts"]:
            raise ValueError("Forecast domain bounds do not match the training split")
        bounds = {}
        for feature in DOMAIN_FEATURE_LABELS:
            minimum, maximum = float(training[feature].min()), float(training[feature].max())
            if not math.isfinite(minimum) or not math.isfinite(maximum):
                raise ValueError("Forecast training domain is incomplete")
            if feature.startswith("diff_"):
                maximum = max(abs(minimum), abs(maximum))
                minimum = -maximum
            bounds[feature] = {"minimum": minimum, "maximum": maximum}
        layoffs = pd.concat([
            (training.date - pd.to_datetime(training[column])).dt.days
            for column in ("red_history_through", "blue_history_through")
        ]).dropna()
        return {
            "source": "sha256:" + self.model_spec["trainingFeaturesSha256"],
            "trainingThrough": self.contract["trainedOn"]["through"],
            "bouts": len(training),
            "fighterAgeYears": {"minimum": bounds["ctx_min_age"]["minimum"], "maximum": bounds["ctx_max_age"]["maximum"]},
            "layoffDays": {"minimum": int(layoffs.min()), "maximum": int(layoffs.max())},
            "features": bounds,
        }

    def status(self) -> dict:
        audit = self.contract.get("historicalReplay", self.contract.get("independentAudit", {}))
        return {
            "available": self.available, "error": self.error,
            "modelVersion": self.model_version if self.available else None,
            "artifact": self.model_spec["artifact"] if self.available else None,
            "learner": self.contract.get("learner"),
            "training": self.contract.get("trainedOn"),
            "calibration": self.contract.get("calibration"),
            "audit": {key: audit[key] for key in ("from", "through", "bouts", "events", "metrics") if key in audit},
            "evaluationType": "historical_replay" if "historicalReplay" in self.contract else "original_audit",
            "supportedWeightClasses": self.supported_divisions,
            "minimumPriorStatBouts": MINIMUM_STAT_BOUTS,
            "minimumLast5StatBouts": MINIMUM_STAT_BOUTS,
            "researchDataThrough": self.research_data.data_through,
            "forecastDateRange": self.forecast_date_range,
            "domainBounds": self.domain_bounds,
            "highConfidenceSupported": False,
            "validatedForBetting": False,
        }

    @lru_cache(maxsize=64)
    def _features(self, first_id: str, second_id: str, event_date: str,
                  weight_class: str, rounds: int, title_bout: bool) -> tuple[pd.DataFrame, dict]:
        return self.build_inference_features(
            self.fights, self.rounds, self.fighters, first_id, second_id, event_date,
            weight_class, rounds, title_bout, prepared_totals=self.totals,
        )

    def predict(self, first_id: str, second_id: str, event_date: str,
                weight_class: str, rounds: int, title_bout: bool) -> dict:
        if not self.available:
            raise RuntimeError(self.error)
        for fighter_id in (first_id, second_id):
            if not isinstance(fighter_id, str) or not re.fullmatch(r"[a-f0-9]{16}", fighter_id) or fighter_id not in self.names:
                raise ValueError("Choose fighters with known UFCStats identities")
        if first_id == second_id:
            raise ValueError("Choose two different fighters")
        if not isinstance(event_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", event_date):
            raise ValueError("Enter an event date in YYYY-MM-DD format")
        try:
            parsed_date = date.fromisoformat(event_date)
        except ValueError as error:
            raise ValueError("Enter a valid event date") from error
        if not self.forecast_date_range["min"] <= event_date <= self.forecast_date_range["max"]:
            raise ValueError(f"Use an event date between {self.forecast_date_range['min']} and {self.forecast_date_range['max']}; the forecast horizon is limited to 365 days after the research cutoff")
        if not isinstance(weight_class, str) or weight_class not in self.supported_divisions:
            raise ValueError("Choose a supported standard weight class")
        if type(rounds) is not int or rounds not in (3, 5):
            raise ValueError("Scheduled rounds must be 3 or 5")
        if type(title_bout) is not bool:
            raise ValueError("Title bout must be true or false")
        if title_bout and rounds != 5:
            raise ValueError("This forecast supports five-round title bouts only")
        frame, quality = self._features(first_id, second_id, event_date, weight_class, rounds, title_bout)
        frame = frame[self.feature_columns]
        missing = [column for column in self.feature_columns if column != "weightclass" and pd.isna(frame.iloc[0][column])]
        if np.isinf(frame.drop(columns="weightclass").to_numpy(dtype=float)).any():
            raise ValueError("The supplied matchup produces unsupported numeric inputs")
        warnings = [
            "This is a hypothetical matchup with the context you supplied, not a verified scheduled fight.",
            "Probabilities are conditional on a decisive result; draws and no contests are not forecast.",
            "High-confidence accuracy and a betting edge have not been established.",
        ]
        reasons = []
        reason_codes = []
        coverage = {}
        for label, fighter_id in (("fighter1", first_id), ("fighter2", second_id)):
            item = quality[label]
            birth_date = self.birth_dates.get(fighter_id, pd.NaT)
            age = (parsed_date - birth_date.date()).days / 365.25 if pd.notna(birth_date) else None
            layoff = (parsed_date - date.fromisoformat(item["historyThrough"])).days if item["historyThrough"] else None
            coverage[label] = {"ageYears": age, "layoffDays": layoff}
            last_class = normalized_division(item["lastWeightClass"])
            if last_class != weight_class:
                reasons.append(f"{self.names[fighter_id]}'s latest recorded division is {last_class or 'unknown'}, not {weight_class}. A division change is outside this forecast's supported context.")
                reason_codes.append("unsupported_division")
            if item["validStatBouts"] < MINIMUM_STAT_BOUTS or item["last5StatBouts"] < MINIMUM_STAT_BOUTS:
                reasons.append(f"{self.names[fighter_id]} needs at least {MINIMUM_STAT_BOUTS} prior bouts with valid statistics, including {MINIMUM_STAT_BOUTS} of the most recent five.")
                reason_codes.append("insufficient_history")
            for key, value, text, code in (
                ("fighterAgeYears", age, "age in years", "age_outside_training_range"),
                ("layoffDays", layoff, "days since the last recorded bout", "layoff_outside_training_range"),
            ):
                bound = self.domain_bounds[key]
                if value is not None and not bound["minimum"] <= value <= bound["maximum"]:
                    display = f"{value:.1f}" if key == "fighterAgeYears" else str(value)
                    interval = f"{bound['minimum']:.1f}–{bound['maximum']:.1f}" if key == "fighterAgeYears" else f"{bound['minimum']}–{bound['maximum']}"
                    reasons.append(f"{self.names[fighter_id]}'s {text} ({display}) is outside the observed training range ({interval}).")
                    reason_codes.append(code)
            if item["historyThrough"] and item["historyThrough"] >= event_date:
                raise ValueError("Forecast history must precede the supplied event date")
        for feature, bound in self.domain_bounds["features"].items():
            value = frame.iloc[0][feature]
            if pd.notna(value) and not bound["minimum"] <= value <= bound["maximum"]:
                reasons.append(f"The matchup's {DOMAIN_FEATURE_LABELS[feature]} falls outside the observed training range.")
                reason_codes.append("feature_outside_training_range")
        if missing:
            warnings.append(f"{len(missing)} missing feature inputs use the model's original training medians; no current values were invented.")
        response = {
            "status": "abstained" if reasons else "forecast",
            "fighter1": {"id": first_id, "name": self.names[first_id], "probability": None, **quality["fighter1"], **coverage["fighter1"]},
            "fighter2": {"id": second_id, "name": self.names[second_id], "probability": None, **quality["fighter2"], **coverage["fighter2"]},
            "modelVersion": self.model_version,
            "context": {"eventDate": event_date, "weightClass": weight_class, "scheduledRounds": rounds, "isTitleBout": title_bout, "hypothetical": True},
            "provenance": {
                "trainingThrough": self.contract["trainedOn"]["through"],
                "calibrationThrough": self.contract["calibration"]["through"],
                "researchDataThrough": self.data_through,
                "sourceManifest": self.source_manifest, "inferenceManifest": self.inference_manifest,
                "domainBoundsSource": self.domain_bounds["source"],
                "sourceUrl": self.research_data.provenance["sourceUrl"],
            },
            "calibration": self.contract["calibration"],
            "highConfidenceSupported": False, "validatedForBetting": False,
            "missingInputs": missing, "warnings": warnings, "reasons": reasons,
            "reasonCodes": sorted(set(reason_codes)),
        }
        if reasons:
            return response
        raw = self.symmetric_probability(self.model, frame)
        probability = float(self.apply_temperature(raw, self.temperature)[0])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("The forecast model did not return a valid probability")
        if max(probability, 1 - probability) >= 0.9:
            response["status"] = "abstained"
            response["reasons"].append("The model returned a value in an unvalidated high-confidence range. No supported high-confidence forecast is available.")
            response["reasonCodes"].append("unvalidated_confidence_band")
            return response
        response["fighter1"]["probability"] = probability
        response["fighter2"]["probability"] = 1 - probability
        return response
