import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from scripts import evaluate_peer_market as peer


class PeerMarketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.market = pd.DataFrame([
            {"fight_id": str(i), "event_id": "event", "date": "2025-05-31",
             "red_fighter_id": f"red{i}", "blue_fighter_id": f"blue{i}",
             "red_name": f"Red {i}", "blue_name": f"Blue {i}",
             "cutoff": "2025-05-30T00:00:00Z", "quote_observed_at": "2025-05-29T23:28:00Z",
             "p_market": probability, "pick_market": probability >= .5, "target": i % 2}
            for i, probability in enumerate([.6, .3, .5])])
        self.prices = self.market.drop(columns=["p_market", "pick_market", "target"]).copy()
        self.prices["status"] = ["candidate_execution", "no_execution_in_window", "identity_unresolved"]
        self.prices["peer_1h"] = [.7, np.nan, np.nan]
        self.prices["peer_1h_age_seconds"] = [30, np.nan, np.nan]
        self.prices["peer_24h_diagnostic"] = [.8, .9, .1]

    def test_missing_prices_remain_zero_without_24h_fallback(self) -> None:
        result = peer.combine(self.market, self.prices)
        self.assertEqual(len(result), 3)
        np.testing.assert_allclose(result.peer_difference, [logit(.7) - logit(.6), 0, 0])
        modified = self.prices.copy()
        modified.peer_24h_diagnostic = ["future result", float("inf"), -999]
        pd.testing.assert_frame_equal(result, peer.combine(self.market, modified))

    def test_exact_identity_and_actual_observation_are_required(self) -> None:
        for column in peer.IDENTITY:
            altered = self.prices.copy()
            altered.loc[0, column] = "wrong"
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "identity mismatch"):
                peer.combine(self.market, altered)
        for column in ["cutoff", "quote_observed_at"]:
            altered = self.prices.copy()
            altered.loc[0, column] = "2025-05-29T23:29:00Z"
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "observation mismatch"):
                peer.combine(self.market, altered)
        altered.loc[0, "quote_observed_at"] = "2025-05-29T23:28:00"
        with self.assertRaisesRegex(ValueError, "timezone"):
            peer.combine(self.market, altered)

    def test_duplicate_missing_and_extra_bouts_are_rejected(self) -> None:
        for altered in [self.prices.iloc[:2], pd.concat([self.prices, self.prices.iloc[:1]])]:
            with self.assertRaisesRegex(ValueError, "membership"):
                peer.combine(self.market, altered)
        shuffled = self.prices.iloc[::-1]
        pd.testing.assert_frame_equal(peer.combine(self.market, self.prices), peer.combine(self.market, shuffled))

    def test_future_stale_nonfinite_and_unqualified_prices_are_rejected(self) -> None:
        for field, value in [("peer_1h", 0), ("peer_1h", 1), ("peer_1h", np.nan),
                             ("peer_1h", float("inf")), ("peer_1h_age_seconds", -1),
                             ("peer_1h_age_seconds", 3601), ("peer_1h_age_seconds", .5),
                             ("peer_1h_age_seconds", np.nan)]:
            altered = self.prices.copy()
            altered.loc[0, field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                peer.combine(self.market, altered)
        altered = self.prices.copy()
        altered.loc[1, "peer_1h"] = .6
        with self.assertRaisesRegex(ValueError, "unqualified"):
            peer.combine(self.market, altered)

    def test_outcome_fields_and_unknown_status_are_rejected(self) -> None:
        for field in peer.FORBIDDEN:
            altered = self.prices.assign(**{field: "unused"})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Outcome-derived"):
                peer.combine(self.market, altered)
        altered = self.prices.copy()
        altered.loc[1, "status"] = "latest_regardless_of_age"
        with self.assertRaisesRegex(ValueError, "Unknown peer status"):
            peer.combine(self.market, altered)

    def test_probabilities_are_clipped_only_before_logit(self) -> None:
        market, prices = self.market.copy(), self.prices.copy()
        market.loc[0, "p_market"] = 1e-10
        prices.loc[0, "peer_1h"] = 1 - 1e-10
        result = peer.combine(market, prices)
        self.assertEqual(result.loc[0, "p_market"], 1e-10)
        self.assertEqual(result.loc[0, "peer_1h"], 1 - 1e-10)
        self.assertAlmostEqual(result.loc[0, "peer_difference"], logit(1 - 1e-6) - logit(1e-6))

    def test_corner_reversal_negates_features_and_complements_predictions(self) -> None:
        original = peer.combine(self.market, self.prices)
        market, prices = self.market.copy(), self.prices.copy()
        for frame in [market, prices]:
            for suffix in ["fighter_id", "name"]:
                frame[["red_" + suffix, "blue_" + suffix]] = frame[["blue_" + suffix, "red_" + suffix]].to_numpy()
        market.p_market = 1 - market.p_market
        prices.peer_1h = 1 - prices.peer_1h
        reversed_frame = peer.combine(market, prices)
        np.testing.assert_allclose(original[peer.FEATURES], -reversed_frame[peer.FEATURES], atol=1e-12)
        model = {"columns": peer.FEATURES, "scales": [.9, .4], "coefficients": [.02, -.03], "temperature": .8}
        np.testing.assert_allclose(peer.history.predict(original, model) + peer.history.predict(reversed_frame, model), 1, atol=1e-12)

    def test_fixed_training_availability_gate(self) -> None:
        frame = pd.DataFrame({"fight_id": [str(i) for i in range(127)], "peer_available": [True] * 39 + [False] * 88})
        with self.assertRaisesRegex(ValueError, "At least40"):
            peer.fit_gate(frame)
        frame.loc[39, "peer_available"] = True
        peer.fit_gate(frame)
        frame.loc[39, "fight_id"] = frame.loc[0, "fight_id"]
        with self.assertRaisesRegex(ValueError, "At least40"):
            peer.fit_gate(frame)

    def test_execution_evidence_is_bound_to_token_time_and_price(self) -> None:
        row = self.prices.iloc[0].to_dict()
        observed = int(peer.timestamp(row["quote_observed_at"]).timestamp())
        identity = {"createdAt": "2025-05-20T00:00:00Z", "tokens": [
            {"tokenId": "11", "side": "red", "fighterId": "red0"},
            {"tokenId": "22", "side": "blue", "fighterId": "blue0"}]}
        record = {"tokenId": "22", "timestamp": observed - 30, "ageSeconds": 30, "tokenPrice": .3, "usdcAmount": 2}
        decision = {"primary": {"status": "candidate_execution", "maxAgeSeconds": 3600,
                    "activityCountIsLowerBound": False, "redProbabilityProxy": .7, "ageSeconds": 30,
                    "latestTimestamp": observed - 30, "latestRecords": [record]}}
        peer.validate_candidate(row, identity, decision)
        for field, value in [("fighterId", "wrong_unused_fighter"), ("fighterId", "blue0"),
                             ("tokenId", "22"), ("side", "blue")]:
            altered = copy.deepcopy(identity)
            altered["tokens"][0][field] = value
            with self.subTest(unused_token_field=field, value=value), self.assertRaisesRegex(ValueError, "exact token identities"):
                peer.validate_candidate(row, altered, decision)
        for field, value in [("tokenId", "unknown"), ("timestamp", observed + 1), ("tokenPrice", .4),
                             ("usdcAmount", 0), ("usdcAmount", True), ("tokenPrice", True)]:
            altered = copy.deepcopy(decision)
            altered["primary"]["latestRecords"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                peer.validate_candidate(row, identity, altered)
        with self.assertRaisesRegex(ValueError, "precedes creation"):
            peer.validate_candidate(row, {**identity, "createdAt": row["quote_observed_at"]}, decision)

    def test_existing_experiment_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            sentinel = path / "original"
            sentinel.write_text("unchanged")
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                peer.run(path / "missing.csv", [path / "missing-later.csv"], path)
            self.assertEqual(sentinel.read_text(), "unchanged")

    def test_deleted_raw_sources_require_retained_full_download_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "http").mkdir()
            full = root / "http/source.parquet"
            full.write_bytes(b"verified full source")
            projected = root / "projected.parquet"
            projected.write_bytes(b"projected columns")
            sha = hashlib.sha256(full.read_bytes()).hexdigest()
            receipt = {"status": 200, "bytes": full.stat().st_size, "sha256": sha,
                       "url": "https://example.invalid/resolve/revision/daily_aligned/source.parquet"}
            receipt_path = full.with_name(full.name + ".receipt.json")
            receipt_path.write_text(json.dumps(receipt))
            for name in ["plan.json", "catalog.json"]:
                (root / name).write_text("{}")
            timestamp = int(pd.Timestamp("2025-05-29", tz="UTC").timestamp())
            record = {"date": "2025-05-29", "sourceRevision": "revision", "sourcePath": "daily_aligned/source.parquet",
                      "localSourcePath": str(full), "sourceBytes": full.stat().st_size,
                      "publishedLFSsha256": sha, "fullSourceHashVerified": True,
                      "rawReferences": [{"file": "http/source.parquet", "sha256": sha}]}
            item = {"partitionDate": "2025-05-29", "sourceRevision": "revision", "sourcePath": record["sourcePath"],
                    "sourceSHA256": sha, "sourceComplete": True, "fullSourceHashVerified": True,
                    "projectedFile": "projected.parquet", "projectedSHA256": peer.edge.digest(projected),
                    "minTimestamp": timestamp, "maxTimestamp": timestamp + 300}
            manifest = {"completed": True, "status": "complete", "failedFiles": [], "records": [record], "files": [item],
                        "planSHA256": peer.edge.digest(root / "plan.json"), "catalogSHA256": peer.edge.digest(root / "catalog.json")}
            peer.verify_partitions(root, manifest, set())
            item.update(rows=0, minTimestamp=None, maxTimestamp=None)
            peer.verify_partitions(root, manifest, set())
            item["rows"] = 1
            with self.assertRaisesRegex(ValueError, "timestamps"):
                peer.verify_partitions(root, manifest, set())
            item.update(rows=1, minTimestamp=timestamp, maxTimestamp=timestamp + 300)
            full.unlink()
            with self.assertRaisesRegex(ValueError, "missing without"):
                peer.verify_partitions(root, manifest, set())
            record["rawSourceDeletedAfterVerification"] = True
            record["sourceVerificationReceipt"] = "verification.json"
            record["deletionReceipt"] = "deletion.json"
            verification = {"sourceSHA256": sha, "projectedSHA256": item["projectedSHA256"],
                            "fullSourceHashVerified": True, "beforeAnySourceDeletion": True}
            (root / "verification.json").write_text(json.dumps(verification))
            deletion = {"verificationReceiptSHA256": peer.edge.digest(root / "verification.json"),
                        "sourceSHA256VerifiedBeforeDeletion": sha, "projectedSHA256": item["projectedSHA256"],
                        "sourceBytes": record["sourceBytes"]}
            (root / "deletion.json").write_text(json.dumps(deletion))
            peer.verify_partitions(root, manifest, set())
            receipt["sha256"] = "changed"
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "acquisition receipt"):
                peer.verify_partitions(root, manifest, set())


if __name__ == "__main__":
    unittest.main()
