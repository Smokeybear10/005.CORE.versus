import hashlib
import importlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "data/evaluation/identity-repair-v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from forecast import ForecastEngine
from research import ResearchData


@unittest.skipUnless((RUNTIME / "candidate_contract.json").is_file(), "Requires the sealed identity-repair-v1 runtime bundle")
class ForecastIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sealed_files = [path for path in RUNTIME.rglob("*") if path.is_file()]
        cls.sealed_files.extend(path for path in (ROOT / "data/evaluation/inference").glob("*") if path.is_file())
        cls.before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in cls.sealed_files}
        cls.module = importlib.import_module("app")
        cls.research = cls.module.research_data
        cls.engine = cls.module.forecast_engine
        cls.model_version = json.loads((RUNTIME / "candidate_contract.json").read_bytes())["modelVersion"]

    @classmethod
    def tearDownClass(cls):
        after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in cls.sealed_files}
        if cls.before != after:
            raise AssertionError("A sealed runtime or original inference file changed during tests")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.client = self.module.app.test_client()

    def load_engine(self, *, contract=None, inference=None, research=None):
        return ForecastEngine(
            RUNTIME / "models/provisional_challenger.joblib",
            contract or RUNTIME / "candidate_contract.json",
            research or self.research,
            (inference or RUNTIME / "inference") / "fighters.parquet",
        )

    def assert_unavailable(self, engine):
        self.assertFalse(engine.available)
        self.assertFalse(engine.status()["available"])
        self.assertIsNone(engine.status()["modelVersion"])
        self.assertEqual(engine.supported_divisions, [])

    def test_current_runtime_and_research_api_are_ready(self):
        self.assertTrue(self.engine.available)
        response = self.client.get("/api/forecast-status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["modelVersion"], self.model_version)
        self.assertTrue(response.get_json()["available"])
        self.assertTrue(self.client.get("/api/research-status").get_json()["research"]["available"])
        data = self.client.get("/api/matchup-research", query_string={
            "fighter1": "Bruno Silva (Bulldog)", "fighter2": "Bruno Silva (Blindado)",
        })
        self.assertEqual(data.status_code, 200)
        self.assertEqual([fighter["totalRecordedBouts"] for fighter in data.get_json()["fighters"]], [12, 11])
        self.assertEqual(self.client.post("/api/predict", json={}).status_code, 410)

    def test_tampered_contract_is_rejected_even_with_valid_model_version(self):
        contract = json.loads((RUNTIME / "candidate_contract.json").read_bytes())
        contract["calibration"]["bouts"] += 1
        path = self.directory / "candidate_contract.json"
        path.write_text(json.dumps(contract))
        self.assert_unavailable(self.load_engine(contract=path))

    def test_relabeled_original_inference_cannot_pair_with_corrected_model(self):
        old = ROOT / "data/evaluation/inference"
        if not (old / "manifest.json").is_file():
            self.skipTest("Requires the original frozen inference bundle")
        inference = self.directory / "inference"
        inference.mkdir()
        for filename in ("fights.parquet", "rounds.parquet", "fighters.parquet"):
            shutil.copyfile(old / filename, inference / filename)
        manifest = json.loads((old / "manifest.json").read_bytes())
        manifest["candidateModelVersion"] = self.model_version
        manifest["researchManifestSha256"] = self.research.research_manifest_sha256
        for filename, expected in manifest["files"].items():
            self.assertEqual(hashlib.sha256((inference / filename).read_bytes()).hexdigest(), expected)
        (inference / "manifest.json").write_text(json.dumps(manifest))
        self.assert_unavailable(self.load_engine(inference=inference))

    def test_valid_but_stale_research_bundle_is_rejected(self):
        stub = SimpleNamespace(norm_fights=None, fights_df=None, snapshots=None, is_ready=lambda: False)
        stale = ResearchData(stub, str(ROOT / "data/model_v2.pkl"))
        self.assertEqual(stale.refresh_status, "loaded")
        self.assertNotEqual(stale.research_manifest_sha256, self.research.research_manifest_sha256)
        self.assert_unavailable(self.load_engine(research=stale))

    def test_current_evaluation_is_bound_to_model_and_not_cached(self):
        response = self.client.get("/evaluation.json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["candidate"]["modelVersion"], self.model_version)
        self.assertEqual(response.headers["Cache-Control"], "no-store")

    def test_tampered_evaluation_is_not_served_as_current_accuracy(self):
        report = json.loads((RUNTIME / "evaluation.json").read_bytes())
        report["candidate"]["audit"]["accuracy"] = 1.0
        (self.directory / "evaluation.json").write_text(json.dumps(report))
        with patch.object(self.module, "_runtime_dir", str(self.directory)):
            response = self.client.get("/evaluation.json")
        self.assertEqual(response.status_code, 503)
        self.assertIn("does not match", response.get_json()["error"])
        self.assertNotIn("candidate", response.get_json())

    def test_missing_evaluation_is_unavailable(self):
        with patch.object(self.module, "_runtime_dir", str(self.directory)):
            self.assertEqual(self.client.get("/evaluation.json").status_code, 503)


if __name__ == "__main__":
    unittest.main()
