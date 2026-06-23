import importlib
import json
import shutil
import sys
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
recorder = importlib.import_module("record_prospective")
grader = importlib.import_module("grade_prospective")
from scripts import evaluate_research as base


@unittest.skipUnless((ROOT / "data/evaluation/probability-stack-v1/probability_stack.joblib").exists(), "Requires frozen local research models")
class ProspectiveCurrentModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = ROOT / "data"
        cls.paths = recorder.input_paths(cls.data, "probability_stack") + recorder.input_paths(cls.data, "original48")
        cls.before = recorder.fingerprints(cls.paths)
        cls.gate = recorder.initialize_engine(cls.data, "corrected48")
        cls.engine = recorder.ProbabilityStackEngine(cls.gate, cls.data)
        cls.features = pd.read_parquet(cls.data / "evaluation/probability-stack-v1/replay_features.parquet")
        cls.forecasts = pd.read_csv(cls.data / "evaluation/probability-stack-v1/predictions.csv", float_precision="round_trip")

    @classmethod
    def tearDownClass(cls):
        if cls.before != recorder.fingerprints(cls.paths):
            raise AssertionError("A frozen input or forecasting implementation changed during tests")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def request(self):
        row = self.features.iloc[0]
        return {"boutId": "synthetic-recorder-fixture", "payload": {"firstID": row.red_fighter_id, "secondID": row.blue_fighter_id,
                "eventDate": "2030-01-05", "weightClass": row.weightclass, "rounds": 3, "titleBout": False}}

    def response(self, reason_codes=None):
        payload = self.request()["payload"]
        codes = reason_codes or []
        return {"status": "abstained" if codes else "forecast", "reasonCodes": codes, "reasons": list(codes),
                "fighter1": {"id": payload["firstID"], "probability": None if codes else .6},
                "fighter2": {"id": payload["secondID"], "probability": None if codes else .4},
                "modelVersion": recorder.MODEL_VERSIONS["corrected48"],
                "context": {"eventDate": payload["eventDate"], "weightClass": payload["weightClass"], "scheduledRounds": 3, "isTitleBout": False},
                "provenance": {"sourceManifest": self.gate.source_manifest, "inferenceManifest": self.gate.inference_manifest}}

    def card(self):
        return {"schemaVersion": 1, "fullCard": True, "expectedBoutCount": 1, "event": {"date": "2030-01-05"},
                "sources": ["Synthetic unit-test fixture; not an actual scheduled card"],
                "contextAssumptions": ["Deliberately mocked domain response for recorder schema tests"], "requests": [self.request()]}

    def test_corrected_model_and_stack_provenance(self):
        status = self.gate.status()
        self.assertTrue(status["available"])
        self.assertEqual(status["modelVersion"], recorder.MODEL_VERSIONS["corrected48"])
        status = self.engine.status()
        self.assertEqual(status["modelVersion"], recorder.MODEL_VERSIONS["probability_stack"])
        self.assertEqual(status["domainGateModelVersion"], recorder.MODEL_VERSIONS["corrected48"])
        self.assertEqual(status["observationClass"], recorder.CANDIDATE_CLASS)
        self.assertFalse(status["eligiblePublicPrediction"])
        self.assertEqual(status["coveragePolicy"], recorder.STACK_COVERAGE)

    def test_all985_feature_forwards_match_frozen_predictions_and_reverse(self):
        frame = self.features[self.engine.artifact["columns"]]
        probability = self.gate.apply_temperature(self.engine._raw_probability(frame), self.engine.artifact["temperature"])
        reverse = self.gate.apply_temperature(self.engine._raw_probability(base.swapped(frame)), self.engine.artifact["temperature"])
        np.testing.assert_array_equal(probability, self.forecasts.candidate_stack.to_numpy())
        np.testing.assert_allclose(probability + reverse, 1, atol=1e-12, rtol=0)

    def test_dynamic_feature_builder_matches_two_frozen_historical_contexts(self):
        fights = self.gate.fights.set_index("fight_id")
        for index in (0, len(self.features) - 1):
            row = self.features.iloc[index]
            context = fights.loc[row.fight_id]
            frame = self.engine._features(row.red_fighter_id, row.blue_fighter_id, str(row.date.date()),
                                          context.weightclass, int(context.num_rounds), bool(context.is_title_bout))
            pd.testing.assert_frame_equal(frame.reset_index(drop=True), self.features.iloc[[index]][self.engine.artifact["columns"]].reset_index(drop=True),
                                          check_exact=True, check_dtype=False)

    def test_wrong_domain_model_and_tampered_stack_are_rejected(self):
        with patch.object(self.gate, "model_version", recorder.MODEL_VERSIONS["original48"]), self.assertRaisesRegex(ValueError, "corrected48"):
            recorder.ProbabilityStackEngine(self.gate, self.data)
        destination = self.directory / "evaluation/probability-stack-v1"
        destination.mkdir(parents=True)
        original = self.data / "evaluation/probability-stack-v1/probability_stack.joblib"
        (destination / original.name).write_bytes(original.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "artifact checksum"):
            recorder.ProbabilityStackEngine(self.gate, self.directory)

    def test_wrong_corrected_source_is_rejected(self):
        destination = self.directory / "evaluation"
        shutil.copytree(self.data / "evaluation/probability-stack-v1", destination / "probability-stack-v1")
        inference = destination / "identity-repair-v1/inference"
        inference.mkdir(parents=True)
        for name in ("fights.parquet", "rounds.parquet", "fighters.parquet"):
            shutil.copyfile(self.data / "evaluation/identity-repair-v1/inference" / name, inference / name)
        with (inference / "fights.parquet").open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "corrected source changed"):
            recorder.ProbabilityStackEngine(self.gate, self.directory)

    def test_relocated_checkout_preserves_frozen_protocol_and_code_checks(self):
        checkout = self.directory / "CORE/005-CORE.V3RSUS"
        for name in ("predictor_features.py", "evaluate_predictors.py", "evaluate_probability_stack.py",
                     "evaluate_research.py", "evaluate_challenger.py"):
            destination = checkout / "scripts" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "scripts" / name, destination)
        protocol = self.data / "evaluation/probability-stack-v1/protocol.json"
        before = protocol.read_bytes()
        with patch.object(recorder, "ROOT", checkout):
            engine = recorder.ProbabilityStackEngine(self.gate, self.data)
            self.assertEqual(engine.status()["modelVersion"], recorder.MODEL_VERSIONS["probability_stack"])
            with (checkout / "scripts/evaluate_research.py").open("a") as stream:
                stream.write("\n# altered code\n")
            with self.assertRaisesRegex(ValueError, "feature/training code changed"):
                recorder.ProbabilityStackEngine(self.gate, self.data)
        self.assertEqual(protocol.read_bytes(), before)

    def test_tampered_corrected_contract_is_rejected(self):
        destination = self.directory / "evaluation/identity-repair-v1"
        shutil.copytree(self.data / "evaluation/identity-repair-v1", destination)
        path = destination / "candidate_contract.json"
        contract = json.loads(path.read_text())
        contract["calibration"]["temperature"] += .1
        path.chmod(0o600)
        path.write_text(json.dumps(contract))
        with self.assertRaisesRegex(ValueError, "unavailable or mismatched"):
            recorder.initialize_engine(self.directory, "corrected48")

    def test_domain_abstentions_are_never_bypassed(self):
        args = tuple(self.request()["payload"][name] for name in recorder.REQUEST_FIELDS)
        for reasons in (["insufficient_history"], ["unvalidated_confidence_band", "layoff_outside_training_range"]):
            with self.subTest(reasons=reasons), patch.object(self.gate, "predict", return_value=self.response(reasons)), \
                    patch.object(self.engine, "_features", side_effect=AssertionError("A domain abstention reached the candidate")):
                response = self.engine.predict(*args)
                self.assertEqual(response["reasonCodes"], reasons)
                self.assertIsNone(response["researchCandidateObservation"])
                self.assertIsNone(response["fighter1"]["probability"])

    def test_sole_old_confidence_band_does_not_suppress_candidate(self):
        args = tuple(self.request()["payload"][name] for name in recorder.REQUEST_FIELDS)
        frame = self.features.iloc[[0]][self.engine.artifact["columns"]]
        with patch.object(self.gate, "predict", return_value=self.response(["unvalidated_confidence_band"])), patch.object(self.engine, "_features", return_value=frame):
            response = self.engine.predict(*args)
        self.assertEqual(response["status"], "research_observation")
        self.assertEqual(response["reasonCodes"], [])
        self.assertIsNone(response["fighter1"]["probability"])
        self.assertEqual(response["researchCandidateObservation"]["fighter1"]["probability"], self.forecasts.iloc[0].candidate_stack)
        self.assertEqual(response["provenance"]["domainGateModelVersion"], recorder.MODEL_VERSIONS["corrected48"])

    def test_new_candidate_confidence_band_stays_research_only(self):
        args = tuple(self.request()["payload"][name] for name in recorder.REQUEST_FIELDS)
        frame = self.features.iloc[[0]][self.engine.artifact["columns"]]
        with patch.object(self.gate, "predict", return_value=self.response()), patch.object(self.engine, "_features", return_value=frame), \
                patch.object(self.engine, "_raw_probability", return_value=np.array([.99])):
            response = self.engine.predict(*args)
        self.assertEqual(response["status"], "abstained")
        self.assertEqual(response["reasonCodes"], ["unvalidated_confidence_band"])
        self.assertTrue(response["researchCandidateObservation"]["confidenceBandUnvalidated"])
        self.assertIsNone(response["fighter1"]["probability"])

    def test_recording_and_grading_include_stack_code_without_overwriting(self):
        card = self.directory / "card.json"
        card.write_text(json.dumps(self.card()))
        output = self.directory / "archive"
        frame = self.features.iloc[[0]][self.engine.artifact["columns"]]
        with patch.object(recorder, "initialize_engine", return_value=self.engine), patch.object(self.gate, "predict", return_value=self.response()), \
                patch.object(self.engine, "_features", return_value=frame):
            summary = recorder.record_card(card, output, self.data, model="probability_stack")
            with self.assertRaises(FileExistsError):
                recorder.record_card(card, output, self.data, model="probability_stack")
        self.assertEqual(summary["researchCandidateObservations"], 1)
        self.assertEqual(summary["publicForecasts"], 0)
        archive = grader.load_archive(output)
        self.assertIn("scripts/evaluate_probability_stack.py", archive["pipeline"]["codeSHA256"])
        row = archive["rows"][0]
        channel, probability, excluded = grader.prediction(row, self.request()["payload"])
        self.assertEqual(channel, recorder.CANDIDATE_CLASS)
        self.assertIsNone(excluded)
        self.assertEqual(probability, self.forecasts.iloc[0].candidate_stack)
        wrong = deepcopy(row)
        for name in ("response", "researchCandidateObservation"):
            wrong["record"][name]["provenance"]["domainGateModelVersion"] = recorder.MODEL_VERSIONS["original48"]
        with self.assertRaisesRegex(ValueError, "wrong domain-gate"):
            grader.prediction(wrong, self.request()["payload"])

    def test_deadline_and_mismatched_request_dates_fail_before_prediction(self):
        class FixedDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 11, 4, 12, tzinfo=timezone.utc)
        card = self.card()
        for event in ("2026-11-04", "2026-11-05"):
            card["event"]["date"] = event
            with patch.object(recorder, "datetime", FixedDatetime), self.assertRaises(ValueError):
                recorder.validate_card(card)
        with patch.object(self.engine, "predict", side_effect=AssertionError("Invalid dates reached model")):
            record = recorder.record_request(self.engine, self.request(), "2030-01-06")
        self.assertEqual(record["error"]["type"], "ValueError")

    def test_old_choices_and_frozen_archive_grading_stay_supported(self):
        paths = [str(path) for path in recorder.input_paths(self.data, "original48")]
        self.assertTrue(any("challenger-v1/provisional_challenger.joblib" in path for path in paths))
        self.assertFalse(any("identity-repair-v1" in path for path in paths))
        for name in ("nov14-original48", "nov14-expanded92"):
            archive = grader.load_archive(self.data / "evaluation/new-fights-20261104" / name)
            self.assertEqual(len(archive["rows"]), 12)
            for row in archive["rows"]:
                if row["record"].get("error") is None:
                    grader.prediction(row, row["record"]["request"]["payload"])


if __name__ == "__main__":
    unittest.main()
