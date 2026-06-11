import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import evaluate_prefight_market as evaluation


class PrefightMarketTests(unittest.TestCase):
    def setUp(self):
        self.row = {"fight_id": "fixture-fight", "event_id": "fixture-event", "date": "2025-06-07",
                    "red_fighter_id": "red-id", "blue_fighter_id": "blue-id", "red_name": "Red Fighter", "blue_name": "Blue Fighter",
                    "red_ticker": "FIXTURE-R", "blue_ticker": "FIXTURE-B"}
        self.line = "Welterweight Bout: Red Fighter (*172) vs Blue Fighter (170)"
        self.rule_span = "Welterweight standard limit is 170 pounds."
        self.allowance = "Non-title bouts have a one-pound allowance; championship bouts have no allowance."
        self.schedule = "Early prelims begin at 6pm ET."
        self.sources = {"fixture": {"text": " ".join([self.line, self.rule_span, self.allowance, self.schedule]),
                                     "knownBy": evaluation.timestamp("2025-06-06T17:00:00Z")}}
        self.record = {**self.row, "weight": {"sourceId": "fixture", "exactSourceLine": self.line},
                       "rule": {"sourceId": "fixture", "exactSpan": self.rule_span, "allowanceSpan": self.allowance,
                                "division": "Welterweight", "standardLimitPounds": 170, "nonTitleAllowancePounds": 1, "titleAllowancePounds": 0},
                       "schedule": {"sourceId": "fixture", "exactSpan": self.schedule, "firstPrelimUTC": "2025-06-07T22:00:00Z"}}
        self.quote = evaluation.timestamp("2025-06-06T23:30:00Z")

    def test_standard_excess_and_corner_reversal(self):
        result = evaluation.context_feature(self.record, self.row, self.quote, self.sources)
        self.assertTrue(result["context_available"])
        self.assertEqual(result["standard_excess_difference"], 1)
        reverse = dict(self.row, red_name=self.row["blue_name"], blue_name=self.row["red_name"],
                       red_fighter_id=self.row["blue_fighter_id"], blue_fighter_id=self.row["red_fighter_id"])
        self.assertEqual(evaluation.context_feature(self.record, reverse, self.quote, self.sources)["standard_excess_difference"], -1)

    def test_unknown_rule_is_not_certified_zero(self):
        self.record["rule"] = None
        result = evaluation.context_feature(self.record, self.row, self.quote, self.sources)
        self.assertEqual(result["standard_excess_difference"], 0)
        self.assertFalse(result["context_available"])
        self.assertEqual(result["context_reason"], "missing_rule_evidence")

    def test_known_zero_and_title_allowance_are_distinct(self):
        self.line = self.line.replace("172", "171")
        self.record["weight"]["exactSourceLine"] = self.line
        self.sources["fixture"]["text"] = " ".join([self.line, self.rule_span, self.allowance])
        self.assertEqual(evaluation.context_feature(self.record, self.row, self.quote, self.sources)["standard_excess_difference"], 0)
        self.line = self.line.replace("Welterweight Bout", "Welterweight Title Bout")
        self.record["weight"]["exactSourceLine"] = self.line
        self.sources["fixture"]["text"] = " ".join([self.line, self.rule_span, self.allowance])
        self.assertEqual(evaluation.context_feature(self.record, self.row, self.quote, self.sources)["standard_excess_difference"], 1)

    def test_bmf_and_catchweight_are_retained_unknown(self):
        for label in ["BMF Title Bout", "Catchweight Bout"]:
            self.record["weight"]["exactSourceLine"] = label + ": Red Fighter (172) vs Blue Fighter (170)"
            result = evaluation.context_feature(self.record, self.row, self.quote, self.sources)
            self.assertFalse(result["context_available"])
            self.assertEqual(result["context_reason"], "ambiguous_division_or_title")

    def test_recognized_division_without_bout_suffix(self):
        line = "Women’s Bantamweight: Red Fighter (136) vs Blue Fighter (135)"
        self.record["weight"]["exactSourceLine"] = line
        self.record["rule"].update(division="Women's Bantamweight", standardLimitPounds=135,
                                   exactSpan="Bantamweight standard limit is 135 pounds.")
        self.sources["fixture"]["text"] = " ".join([line, self.record["rule"]["exactSpan"], self.allowance])
        result = evaluation.context_feature(self.record, self.row, self.quote, self.sources)
        self.assertTrue(result["context_available"])
        self.assertEqual(result["standard_excess_difference"], 0)

    def test_source_after_actual_quote_is_not_admitted_before_planned_midnight(self):
        self.sources["fixture"]["knownBy"] = evaluation.timestamp("2025-06-06T23:45:00Z")
        self.assertFalse(evaluation.context_feature(self.record, self.row, self.quote, self.sources)["context_available"])
        self.assertEqual(evaluation.schedule_status(self.record, self.quote, self.sources), (False, "schedule_evidence_after_quote"))

    def test_schedule_requires_first_prelim_after_quote(self):
        self.assertEqual(evaluation.schedule_status(self.record, self.quote, self.sources), (True, "qualified_schedule"))
        self.record["schedule"]["firstPrelimUTC"] = self.quote.isoformat()
        self.assertFalse(evaluation.schedule_status(self.record, self.quote, self.sources)[0])

    def test_tampered_claim_identity_and_boolean_number_fail(self):
        bad = dict(self.row, blue_fighter_id="other")
        with self.assertRaisesRegex(ValueError, "identity"):
            evaluation.same_identity(self.record, bad)
        self.record["rule"]["standardLimitPounds"] = True
        with self.assertRaisesRegex(ValueError, "numeric"):
            evaluation.context_feature(self.record, self.row, self.quote, self.sources)
        self.record["rule"]["standardLimitPounds"] = 170
        self.record["weight"]["exactSourceLine"] = self.line.replace("172", "173")
        with self.assertRaisesRegex(ValueError, "span"):
            evaluation.context_feature(self.record, self.row, self.quote, self.sources)

    def test_nested_outcomes_and_naive_timestamps_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Outcome"):
            evaluation.no_outcomes({"records": [{"result": "win"}]})
        with self.assertRaisesRegex(ValueError, "timezone"):
            evaluation.timestamp("2025-06-06T23:30:00")

    def test_support_gate_requires_nonzero_events_not_only_many_weights(self):
        parts = {}
        for name, count in [("fit", 127), ("calibration", 155), ("test", 217)]:
            parts[name] = pd.DataFrame({"event_id": [str(i % 20) for i in range(count)], "market_available": True,
                                       "context_available": True, "standard_excess_difference": 0.0})
        with patch.object(evaluation.edge, "split", return_value=parts):
            result = evaluation.gates(pd.DataFrame())
            self.assertFalse(result["allFitGatesPassed"])
            self.assertEqual(result["trainingContextFailures"], ["nonzeroTrainingPairs", "nonzeroTrainingEvents"])
            parts["fit"].loc[:9, "standard_excess_difference"] = 1.0
            self.assertTrue(evaluation.gates(pd.DataFrame())["allFitGatesPassed"])
            parts["test"].loc[173:, "market_available"] = False
            self.assertFalse(evaluation.gates(pd.DataFrame())["allFitGatesPassed"])

    def test_archive_receipts_bind_raw_bytes_and_capture_time(self):
        with tempfile.TemporaryDirectory(dir=evaluation.edge.DATA) as temporary:
            directory = Path(temporary)
            body = b"<p>Synthetic exact evidence</p>"
            url = "https://www.ufc.com/news/synthetic-test-fixture"
            chosen = {"timestamp": "20250606170000", "original": url, "statuscode": "200", "mimetype": "text/html",
                      "digest": base64.b32encode(hashlib.sha1(body).digest()).decode().rstrip("=")}
            capture = "https://web.archive.org/web/20250606170000id_/" + url
            cbody = json.dumps([list(chosen), list(chosen.values())]).encode()
            values = {"body": body, "cdx": cbody,
                      "receipt": json.dumps({"status": 200, "requestedUrl": capture, "finalUrl": capture,
                                             "bodyBytes": len(body), "bodySha256": hashlib.sha256(body).hexdigest(),
                                             "headers": {"memento-datetime": "Fri, 06 Jun 2025 17:00:00 GMT"}}).encode(),
                      "selection": json.dumps({"selectedCDXRow": chosen, "requestUrl": capture}).encode(),
                      "cdx-receipt": json.dumps({"status": 200, "bodyBytes": len(cbody), "bodySha256": hashlib.sha256(cbody).hexdigest()}).encode()}
            files = {}
            for name, content in values.items():
                path = directory / name
                path.write_bytes(content)
                files[str(path.relative_to(evaluation.edge.ROOT))] = hashlib.sha256(content).hexdigest()
            source = {"originalURL": url, **{key: str((directory / name).relative_to(evaluation.edge.ROOT)) for key, name in
                        [("bodyPath", "body"), ("receiptPath", "receipt"), ("selectionPath", "selection"), ("cdxBodyPath", "cdx"), ("cdxReceiptPath", "cdx-receipt")]}}
            evidence = evaluation.archived_source(source, files, {})
            self.assertEqual(evidence["knownBy"], evaluation.timestamp("2025-06-06T17:00:00Z"))
            (directory / "body").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                evaluation.archived_source(source, files, {})

    def test_existing_output_fails_before_any_fit(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(evaluation.history, "fit") as fit:
            with self.assertRaisesRegex(ValueError, "overwrite"):
                evaluation.evaluate(Path("unused"), "unused", Path(temporary))
            fit.assert_not_called()

    @unittest.skipUnless(evaluation.REGISTRY.exists(), "Requires the frozen local market registry")
    def test_explicit_decision_file_retains_entire_missing_registry_without_fit(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(evaluation.history, "fit") as fit:
            directory = Path(temporary)
            decisions = directory / "decisions-fit2025.json"
            decisions.write_text("[]")
            manifest = directory / "manifest.json"
            manifest.write_text(json.dumps({decisions.name: evaluation.edge.digest(decisions)}))
            context = directory / "context.json"
            context.write_text(json.dumps({"schemaVersion": 1, "registrySHA256": evaluation.REGISTRY_SHA,
                                           "files": {}, "sources": [], "records": []}))
            result = evaluation.qualify(manifest, evaluation.edge.digest(manifest), context, evaluation.edge.digest(context),
                                        directory / "output", market_decisions=decisions.name)
            self.assertFalse(result["allFitGatesPassed"])
            frame = pd.read_csv(directory / "output/qualified-registry.csv")
            self.assertEqual(len(frame), 499)
            self.assertFalse(frame.market_available.any())
            self.assertFalse(frame.context_available.any())
            self.assertNotIn("target", frame)
            fit.assert_not_called()
            with self.assertRaisesRegex(ValueError, "hash-bound"):
                evaluation.qualify(manifest, evaluation.edge.digest(manifest), context, evaluation.edge.digest(context),
                                    directory / "rejected", market_decisions="../decisions-fit2025.json")

    def test_failed_gate_stops_before_targets_or_fit(self):
        receipt = {"proposalSHA256": evaluation.PROPOSAL_SHA, "dependencies": evaluation.versions(),
                   "inputHashes": {}, "gates": {"allFitGatesPassed": False}}
        with tempfile.TemporaryDirectory() as temporary, patch.object(evaluation, "read_manifest", return_value={"qualification.json": "fixture", "qualified-registry.csv": "fixture"}), \
                patch.object(evaluation.edge, "read", return_value=receipt), \
                patch.object(evaluation.pd, "read_csv", return_value=pd.DataFrame()) as read_csv, \
                patch.object(evaluation, "gates", return_value=receipt["gates"]), \
                patch.object(evaluation.history, "fit") as fit:
            with self.assertRaisesRegex(ValueError, "No fit"):
                evaluation.evaluate(Path(temporary) / "manifest.json", "unused", Path(temporary) / "output")
            self.assertEqual(read_csv.call_count, 1)
            fit.assert_not_called()

    def test_unlisted_qualification_csv_never_reaches_targets(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(evaluation, "read_manifest", return_value={"qualification.json": "fixture"}), \
                patch.object(evaluation.pd, "read_csv") as read_csv, patch.object(evaluation.history, "fit") as fit:
            with self.assertRaisesRegex(ValueError, "hash-bound"):
                evaluation.evaluate(Path(temporary) / "manifest.json", "fixture", Path(temporary) / "output")
            read_csv.assert_not_called()
            fit.assert_not_called()

    def test_raw_quotes_exclude_future_and_reject_relabeling(self):
        cutoff = int(evaluation.timestamp("2025-06-07T00:00:00Z").timestamp())
        observed = cutoff - 60
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "raw").mkdir()
            (directory / "receipts").mkdir()
            files, markets, provenance = {}, [], {}
            for side, mid in [("red", .6), ("blue", .4)]:
                ticker = self.row[side + "_ticker"]
                payload = {"ticker": ticker, "candlesticks": [
                    {"end_period_ts": observed, "yes_bid": {"close_dollars": mid - .01}, "yes_ask": {"close_dollars": mid + .01}},
                    {"end_period_ts": cutoff + 60, "yes_bid": {"close_dollars": .98}, "yes_ask": {"close_dollars": .99}}]}
                name = "raw/" + side + ".json"
                (directory / name).write_text(json.dumps(payload))
                files[name] = evaluation.edge.digest(directory / name)
                markets.append({"ticker": ticker, "fighterId": self.row[side + "_fighter_id"], "rawFile": name, "midpoint": mid})
                header_name = "raw/" + side + ".headers.json"
                (directory / header_name).write_text("{}")
                files[header_name] = evaluation.edge.digest(directory / header_name)
                receipt_name = "receipts/" + side + ".json"
                receipt = {"status": 200, "bodyComplete": True, "rawFile": name,
                           "bodySha256": files[name], "bodyBytes": (directory / name).stat().st_size,
                           "headersFile": header_name, "headersSha256": files[header_name], "headerBytes": 2,
                           "url": f"https://external-api.kalshi.com/trade-api/v2/historical/markets/{ticker}/candlesticks?start_ts={cutoff - 3600}&end_ts={cutoff}&period_interval=1"}
                (directory / receipt_name).write_text(json.dumps(receipt))
                files[receipt_name] = evaluation.edge.digest(directory / receipt_name)
                provenance.update({side + "_rawFile": name, side + "_receiptFile": receipt_name, side + "_apiPartition": "historical"})
            decision = {**self.row, "cutoff": "2025-06-07T00:00:00Z", "quoteObservedAt": "2025-06-06T23:59:00Z",
                        "quoteEndTimestamp": observed, "markets": markets, **provenance}
            p, when = evaluation.price_from_raw(decision, self.row, directory, files, {})
            self.assertAlmostEqual(p, .6)
            self.assertEqual(when, evaluation.timestamp(decision["quoteObservedAt"]))
            decision["quoteObservedAt"] = "2025-06-06T23:58:00Z"
            with self.assertRaisesRegex(ValueError, "latest"):
                evaluation.price_from_raw(decision, self.row, directory, files, {})
            decision["quoteObservedAt"] = "2025-06-06T23:59:00Z"
            decision["markets"][1]["fighterId"] = "unrelated-fighter"
            with self.assertRaisesRegex(ValueError, "identity"):
                evaluation.price_from_raw(decision, self.row, directory, files, {})
            decision["markets"][1]["fighterId"] = self.row["blue_fighter_id"]
            receipt_name = provenance["red_receiptFile"]
            original = json.loads((directory / receipt_name).read_text())
            for changes, message in [({"status": 403}, "HTTP"), ({"bodyComplete": False}, "HTTP"),
                                     ({"url": original["url"].replace("period_interval=1", "period_interval=60")}, "window")]:
                (directory / receipt_name).write_text(json.dumps({**original, **changes}))
                files[receipt_name] = evaluation.edge.digest(directory / receipt_name)
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                    evaluation.price_from_raw(decision, self.row, directory, files, {})


if __name__ == "__main__":
    unittest.main()
