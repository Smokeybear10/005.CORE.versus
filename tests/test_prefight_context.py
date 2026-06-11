import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts import collect_prefight_context as collector


class PrefightContextTests(unittest.TestCase):
    def make_bundle(self, root: Path) -> dict:
        cohort = pd.DataFrame([{"event_id": "event", "fight_id": "fight", "date": "2022-02-05", "red_name": "Alpha", "blue_name": "Beta", "red_fighter_id": "alpha", "blue_fighter_id": "beta"}])
        cohort.to_csv(root / "cohort.csv", index=False)
        (root / "protocol.json").write_text(json.dumps({"cohortSHA256": collector.digest(root / "cohort.csv"), "expectedCompletedBouts": 1, "expectedEvents": 1}))
        text = "## Test Official Weigh-In Results:\nFeatherweight Bout: Alpha (149)* vs Beta (145)\nAlpha replaced Gamma against Beta.\n"
        (root / "source.txt").write_text(text)
        (root / "raw.txt").write_text("Raw response:\n" + text)
        source = {"source_id": "weighin", "event_id": "event", "kind": "weighin", "url": "https://www.ufc.com/news/test", "text_path": "source.txt", "text_sha256": collector.digest(root / "source.txt"), "discovery_receipt": "raw.txt", "discovery_sha256": collector.digest(root / "raw.txt"), "transport": "public_search_index", "publisher_date": "2022-02-04", "captured_at": "2026-11-04T22:00:00Z"}
        (root / "sources.json").write_text(json.dumps([source]))
        (root / "aliases.json").write_text("[]")
        (root / "reviewed_assertions.json").write_text("[]")
        return source

    def test_weight_rows_with_stars_and_missing_vs(self) -> None:
        text = "## Test Official Weigh-In Results:\nFeatherweight Bout: Alpha Name (145.5) vs Beta Name (*149)\nFlyweight Bout: Gamma Name (126) Delta Name (125)\nHeavyweight Bout: Epsilon Name* (218) vs Zeta Name (206)\n"
        rows = collector.parse_weighins(text)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["source_name2"], "Beta Name")
        self.assertEqual(rows[0]["weight2_lbs"], 149)
        self.assertTrue(rows[0]["annotation2"])
        self.assertTrue(rows[2]["annotation1"])
        self.assertNotIn("weight_miss", rows[0])

    def test_catchweight_label_does_not_become_fighter_weight(self) -> None:
        rows = collector.parse_weighins("## Test Official Weigh-In Results:\nCo-Main Event - Catchweight (160 lbs) Bout: Alpha (160) vs Beta (158)\n")
        self.assertEqual(rows[0]["weight1_lbs"], 160)
        self.assertEqual(rows[0]["source_name1"], "Alpha")

    def test_multiple_footnote_stars_keep_names_and_annotations_on_the_right_side(self) -> None:
        text = "## Test Official Weigh-In Results:\nFlyweight Bout: Azat Maksum (129)** vs Mitch Raposo (125.5)\n"
        row = collector.parse_weighins(text)[0]
        self.assertEqual(row["source_name1"], "Azat Maksum")
        self.assertEqual(row["source_name2"], "Mitch Raposo")
        self.assertTrue(row["annotation1"])
        self.assertFalse(row["annotation2"])
        cohort = pd.DataFrame([{"event_id": "event", "fight_id": "fight", "red_name": "Azat Maksum", "blue_name": "Mitch Raposo", "red_fighter_id": "azat", "blue_fighter_id": "mitch"}])
        self.assertEqual(collector.resolve_pair(row, cohort, "event", {})["fight_id"], "fight")

    def test_division_without_bout_suffix_is_retained(self) -> None:
        text = "## Test Official Weigh-In Results:\nWomen’s Bantamweight: Irina Alekseeva (135.5) vs Klaudia Sygula (135)\nTickets: Alpha (125) vs Beta (126)\n"
        rows = collector.parse_weighins(text)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_name2"], "Klaudia Sygula")
        self.assertEqual(rows[0]["weight1_lbs"], 135.5)

    def test_spaced_footnote_stays_with_its_weight(self) -> None:
        for stars in ["*", "**"]:
            with self.subTest(stars=stars):
                text = f"## Official Weigh-In Results:\nFlyweight Bout: Alpha (129) {stars} vs Beta (125.5)\n"
                row = collector.parse_weighins(text)[0]
                self.assertTrue(row["annotation1"])
                self.assertFalse(row["annotation2"])
                self.assertEqual(row["source_name2"], "Beta")

    def test_vs_separator_does_not_truncate_a_name(self) -> None:
        text = "## Test Official Weigh-In Results:\nFlyweight Bout: Alpha (125) Vsevolod Example (126)\n"
        self.assertEqual(collector.parse_weighins(text)[0]["source_name2"], "Vsevolod Example")

    def test_duplicate_pair_rejected(self) -> None:
        text = "## Test Official Weigh-In Results:\nFlyweight Bout: Alpha (125) vs Beta (126)\nFlyweight Bout: Beta (125) vs Alpha (125)\n"
        with self.assertRaises(ValueError):
            collector.parse_weighins(text)

    def test_missing_heading_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            collector.parse_weighins("Flyweight Bout: Alpha (125) vs Beta (126)")

    def test_only_exact_event_pair_is_resolved(self) -> None:
        cohort = pd.DataFrame([{"event_id": "event", "fight_id": "fight", "red_name": "Bruno Silva", "blue_name": "Alex Pereira", "red_fighter_id": "blindado", "blue_fighter_id": "alex"},
                               {"event_id": "other", "fight_id": "otherfight", "red_name": "Bruno Silva", "blue_name": "Opponent", "red_fighter_id": "bulldog", "blue_fighter_id": "opponent"}])
        row = {"source_name1": "Alex Pereira", "source_name2": "Bruno Silva"}
        found = collector.resolve_pair(row, cohort, "event", {})
        self.assertEqual(found["fighter2_id"], "blindado")
        self.assertIsNone(collector.resolve_pair(row, cohort, "other", {})["fight_id"])
        row["source_name2"] = "Bruno Silv"
        self.assertIsNone(collector.resolve_pair(row, cohort, "event", {})["fight_id"])

    def test_explicit_alias_scoped_to_event_and_id(self) -> None:
        cohort = pd.DataFrame([{"event_id": "event", "fight_id": "fight", "red_name": "King Green", "blue_name": "Opponent", "red_fighter_id": "king", "blue_fighter_id": "opponent"}])
        row = {"source_name1": "Bobby Green", "source_name2": "Opponent"}
        self.assertIsNone(collector.resolve_pair(row, cohort, "event", {})["fight_id"])
        self.assertEqual(collector.resolve_pair(row, cohort, "event", {("event", "king"): {"bobbygreen"}})["fighter1_id"], "king")

    def test_dates_do_not_invent_historical_availability(self) -> None:
        source = {"publisher_date": "2022-02-10", "captured_at": "2026-11-04T22:00:00Z"}
        result = collector.availability(source, "2022-02-12")
        self.assertTrue(result["publisher_date_before_event"])
        self.assertEqual(result["publisher_date_gap_days"], 2)
        self.assertFalse(result["eligible_for_day_before_market_validation"])
        self.assertFalse(result["historical_availability_verified"])

    def test_receipt_hash_and_official_host(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "source.txt"
            path.write_text("evidence")
            raw = root / "raw.txt"
            raw.write_text("Raw search response: evidence")
            source = {"url": "https://www.ufc.com/news/test", "text_path": "source.txt", "text_sha256": hashlib.sha256(b"evidence").hexdigest(), "transport": "public_search_index", "discovery_receipt": "raw.txt", "discovery_sha256": collector.digest(raw)}
            self.assertEqual(collector.verified_text(root, source), "evidence")
            raw.write_text("changed")
            with self.assertRaisesRegex(ValueError, "Discovery receipt"):
                collector.verified_text(root, source)
            raw.write_text("Raw search response: evidence")
            path.write_text("changed")
            with self.assertRaises(ValueError):
                collector.verified_text(root, source)
            self.assertFalse(collector.official_url("https://www.ufc.com.evil.test/news/test"))
            self.assertFalse(collector.official_url("https://attacker@www.ufc.com/news/test"))

    def test_build_preserves_unknown_flags_and_timing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_bundle(root)
            collector.build(root, root / "output")
            row = pd.read_csv(root / "output/bout_context.csv").iloc[0]
            self.assertEqual(row.red_weight_lbs, 149)
            self.assertTrue(pd.isna(row.red_explicit_weight_miss))
            self.assertTrue(pd.isna(row.red_replacement_reported))
            self.assertFalse(row.strict_market_eligible)

    def test_reviewed_claim_span_must_remain_exact(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_bundle(root)
            text = (root / "source.txt").read_text()
            span = "Alpha replaced Gamma against Beta."
            claim = {"source_id": "weighin", "kind": "replacement", "target_event_id": "event", "fighter_name": "Alpha", "opponent_name": "Beta", "span_start": text.index(span), "span_end": text.index(span) + len(span), "span_sha256": hashlib.sha256(span.encode()).hexdigest()}
            (root / "reviewed_assertions.json").write_text(json.dumps([claim]))
            collector.build(root, root / "valid")
            row = pd.read_csv(root / "valid/bout_context.csv").iloc[0]
            self.assertTrue(row.red_replacement_reported)
            self.assertTrue(pd.isna(row.red_short_notice_reported))
            claim["span_start"] += 1
            (root / "reviewed_assertions.json").write_text(json.dumps([claim]))
            with self.assertRaisesRegex(ValueError, "assertion span"):
                collector.build(root, root / "invalid")

    def test_alias_evidence_hash_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_bundle(root)
            alias = {"event_id": "event", "fighter_id": "alpha", "source_name": "Old Alpha", "evidence": "Explicit name change", "source_urls": ["https://www.ufc.com/news/test"], "evidence_receipts": {"raw.txt": "0" * 64}}
            (root / "aliases.json").write_text(json.dumps([alias]))
            with self.assertRaisesRegex(ValueError, "Alias identity receipt"):
                collector.build(root, root / "invalid")


if __name__ == "__main__":
    unittest.main()
