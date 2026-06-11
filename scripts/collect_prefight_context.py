import argparse
import hashlib
import json
import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

WEIGHT = re.compile(r"([^()]+?)\s*\(\s*(\*?)(\d+(?:\.\d+)?)\s*\)\s*(\**)")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def name_key(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def official_url(value: str) -> bool:
    url = urlparse(value)
    return url.scheme == "https" and url.hostname == "www.ufc.com" and url.path.startswith(("/news/", "/index.php/news/")) and not url.username


def verified_text(bundle: Path, source: dict) -> str:
    if not official_url(source["url"]):
        raise ValueError("Only official UFC news sources are accepted")
    path = (bundle / source["text_path"]).resolve()
    if not path.is_relative_to(bundle.resolve()) or digest(path) != source["text_sha256"]:
        raise ValueError("Source receipt checksum/path mismatch")
    if source.get("transport") != "public_search_index":
        raise ValueError("Unsupported receipt transport")
    raw = (bundle / source["discovery_receipt"]).resolve()
    if not raw.is_relative_to(bundle.resolve()) or digest(raw) != source["discovery_sha256"]:
        raise ValueError("Discovery receipt checksum/path mismatch")
    text = path.read_text()
    if text.strip() not in raw.read_text():
        raise ValueError("Selected source is absent from discovery receipt")
    return text


def availability(source: dict, event_date: str) -> dict:
    published = date.fromisoformat(source["publisher_date"])
    event = date.fromisoformat(event_date)
    observed = datetime.fromisoformat(source["captured_at"].replace("Z", "+00:00"))
    if observed.tzinfo is None:
        raise ValueError("Receipt capture time needs a timezone")
    return {"publisher_date": published.isoformat(), "observed_at": observed.isoformat(),
            "source_quality_flags": ";".join(source.get("quality_flags", [])),
            "publisher_date_gap_days": (event - published).days,
            "publisher_date_before_event": published < event,
            "market_day_before_cutoff_utc": f"{(event - timedelta(days=1)).isoformat()}T00:00:00+00:00",
            "historical_availability_verified": False,
            "eligible_for_strict_prefight_validation": False,
            "eligible_for_day_before_market_validation": False,
            "availability_reason": "Current search-index capture; original publisher date does not timestamp later article edits or individual facts."}


def parse_weighins(text: str) -> list[dict]:
    start = re.search(r"^#+ .*Official Weigh-In Results:\s*$", text, re.MULTILINE)
    if not start:
        raise ValueError("Official weigh-in table heading missing")
    rows = []
    for line in text[start.end():].splitlines():
        line = line.replace("\u00a0", " ").strip()
        if ":" not in line:
            continue
        label, content = line.split(":", 1)
        if not re.search(r"\bBout\b|\b(?:Strawweight|Flyweight|Bantamweight|Featherweight|Lightweight|Welterweight|Middleweight|Heavyweight)(?:\s+Title)?\s*$", label, re.IGNORECASE):
            continue
        matches = list(WEIGHT.finditer(content))
        if len(matches) != 2:
            continue
        first, second = matches
        names = [re.sub(r"^vs(?:\.|\s)\s*", "", match[1].strip(" *"), flags=re.IGNORECASE).strip(" *") for match in matches]
        if any(not value or not re.search(r"[A-Za-z]", value) for value in names):
            continue
        row = {"source_line": line, "bout_label": label.strip(), "source_name1": names[0], "source_name2": names[1],
               "weight1_lbs": float(first[3]), "weight2_lbs": float(second[3]),
               "annotation1": bool(first[2] or first[4] or "*" in first[1]),
               "annotation2": bool(second[2] or second[4] or "*" in second[1])}
        if not all(80 <= row[f"weight{side}_lbs"] <= 400 for side in (1, 2)):
            raise ValueError("Unexpected weight units/range")
        rows.append(row)
    pairs = [tuple(sorted((name_key(row["source_name1"]), name_key(row["source_name2"])))) for row in rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Duplicate or conflicting weigh-in pair")
    return rows


def resolve_pair(row: dict, cohort: pd.DataFrame, event_id: str, aliases: dict) -> dict:
    first, second = name_key(row["source_name1"]), name_key(row["source_name2"])
    found = []
    for bout in cohort[cohort.event_id == event_id].itertuples(index=False):
        red_names = {name_key(bout.red_name), *aliases.get((event_id, bout.red_fighter_id), set())}
        blue_names = {name_key(bout.blue_name), *aliases.get((event_id, bout.blue_fighter_id), set())}
        if first in red_names and second in blue_names:
            found.append((bout, bout.red_fighter_id, bout.blue_fighter_id))
        if first in blue_names and second in red_names:
            found.append((bout, bout.blue_fighter_id, bout.red_fighter_id))
    if len(found) != 1:
        return {"fight_id": None, "fighter1_id": None, "fighter2_id": None,
                "identity_status": "ambiguous" if found else "unresolved_or_not_in_completed_roster"}
    bout, first_id, second_id = found[0]
    return {"fight_id": bout.fight_id, "fighter1_id": first_id, "fighter2_id": second_id, "identity_status": "exact_event_pair"}


def build(bundle: Path, output: Path) -> None:
    protocol = json.loads((bundle / "protocol.json").read_text())
    if digest(bundle / "cohort.csv") != protocol["cohortSHA256"]:
        raise ValueError("Frozen cohort changed")
    cohort = pd.read_csv(bundle / "cohort.csv", dtype=str)
    if cohort.fight_id.duplicated().any() or len(cohort) != protocol["expectedCompletedBouts"] or cohort.event_id.nunique() != protocol["expectedEvents"]:
        raise ValueError("Cohort membership mismatch")
    source_list = json.loads((bundle / "sources.json").read_text())
    sources = {source["source_id"]: source for source in source_list}
    if len(sources) != len(source_list):
        raise ValueError("Duplicate source IDs")
    texts = {key: verified_text(bundle, value) for key, value in sources.items()}
    aliases = {}
    alias_records = json.loads((bundle / "aliases.json").read_text())
    alias_identities = {}
    for item in alias_records:
        event_rows = cohort[cohort.event_id == item["event_id"]]
        if not item.get("evidence") or not item.get("source_urls") or not item.get("evidence_receipts") or item["fighter_id"] not in set(event_rows.red_fighter_id) | set(event_rows.blue_fighter_id):
            raise ValueError("Alias lacks explicit identity evidence")
        for name, checksum in item["evidence_receipts"].items():
            path = (bundle / name).resolve()
            if not path.is_relative_to(bundle.resolve()) or digest(path) != checksum:
                raise ValueError("Alias identity receipt changed")
        alias_key = (item["event_id"], name_key(item["source_name"]))
        if alias_key in alias_identities and alias_identities[alias_key] != item["fighter_id"]:
            raise ValueError("Conflicting fighter aliases")
        alias_identities[alias_key] = item["fighter_id"]
        aliases.setdefault((item["event_id"], item["fighter_id"]), set()).add(name_key(item["source_name"]))
    assertions = json.loads((bundle / "reviewed_assertions.json").read_text())
    claims = []
    for claim in assertions:
        source = sources[claim["source_id"]]
        text = texts[claim["source_id"]]
        span = text[claim["span_start"]:claim["span_end"]]
        if not span or hashlib.sha256(span.encode()).hexdigest() != claim["span_sha256"]:
            raise ValueError("Reviewed assertion span changed")
        if claim["kind"] not in ("replacement", "cancellation", "event_change", "division_change", "weight_miss"):
            raise ValueError("Unknown assertion type")
        event_rows = cohort[cohort.event_id == claim["target_event_id"]]
        if event_rows.empty:
            raise ValueError("Assertion outside frozen event calendar")
        pair = {"source_name1": claim["fighter_name"], "source_name2": claim["opponent_name"]}
        resolved = resolve_pair(pair, cohort, claim["target_event_id"], aliases)
        claims.append({**claim, **resolved, "event_date": event_rows.iloc[0].date,
                       "source_url": source["url"], "source_sha256": source["text_sha256"],
                       **availability(source, event_rows.iloc[0].date)})
    weights = []
    for source in source_list:
        if source["kind"] != "weighin":
            continue
        event_rows = cohort[cohort.event_id == source["event_id"]]
        if event_rows.empty:
            raise ValueError("Source outside frozen event calendar")
        for row in parse_weighins(texts[source["source_id"]]):
            resolved = resolve_pair(row, cohort, source["event_id"], aliases)
            weights.append({**row, **resolved, "source_id": source["source_id"], "event_id": source["event_id"],
                            "event_date": event_rows.iloc[0].date, "source_url": source["url"],
                            "source_sha256": source["text_sha256"], **availability(source, event_rows.iloc[0].date)})
    matched = [row for row in weights if row["fight_id"]]
    if len({row["fight_id"] for row in matched}) != len(matched):
        raise ValueError("Multiple weigh-in sources matched the same bout")
    by_fight = {row["fight_id"]: row for row in matched}
    summaries = []
    for bout in cohort.to_dict("records"):
        row = by_fight.get(bout["fight_id"])
        summary = {**bout, "weighin_found": row is not None, "source_id": row["source_id"] if row else None,
                   "red_weight_lbs": None, "blue_weight_lbs": None,
                   "red_explicit_weight_miss": None, "blue_explicit_weight_miss": None,
                   "red_replacement_reported": None, "blue_replacement_reported": None,
                   "red_short_notice_reported": None, "blue_short_notice_reported": None,
                   "historical_availability_verified": False, "strict_market_eligible": False}
        if row:
            for position in (1, 2):
                side = "red" if row[f"fighter{position}_id"] == bout["red_fighter_id"] else "blue"
                summary[f"{side}_weight_lbs"] = row[f"weight{position}_lbs"]
        for claim in claims:
            if claim["fight_id"] != bout["fight_id"]:
                continue
            side = "red" if claim["fighter1_id"] == bout["red_fighter_id"] else "blue"
            if claim["kind"] == "replacement":
                summary[f"{side}_replacement_reported"] = True
                if claim.get("short_notice_explicit") is True:
                    summary[f"{side}_short_notice_reported"] = True
            if claim["kind"] == "weight_miss":
                summary[f"{side}_explicit_weight_miss"] = True
        summaries.append(summary)
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(weights).to_csv(output / "weighins.csv", index=False)
    pd.DataFrame(claims).to_csv(output / "assertions.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "bout_context.csv", index=False)
    inputs = [bundle / name for name in ("protocol.json", "cohort.csv", "sources.json", "aliases.json", "reviewed_assertions.json")]
    inputs += [bundle / source["text_path"] for source in source_list]
    inputs += sorted({bundle / source["discovery_receipt"] for source in source_list})
    inputs += sorted({bundle / name for item in alias_records for name in item["evidence_receipts"]})
    inputs += [Path(__file__), Path(__file__).resolve().parents[1] / "tests/test_prefight_context.py"]
    per_event = [{"event_id": key, "date": group.iloc[0].date, "completed_bouts": len(group),
                  "matched_weighins": sum(row["fight_id"] in set(group.fight_id) for row in matched),
                  "weighin_sources": sum(source["kind"] == "weighin" and source["event_id"] == key for source in source_list),
                  "update_sources": sum(source["kind"] == "update" and source["event_id"] == key for source in source_list)}
                 for key, group in cohort.groupby("event_id")]
    report = {"createdAt": datetime.now(timezone.utc).isoformat(), "completedBoutDenominator": len(cohort),
              "events": len(per_event), "sources": len(source_list), "sourceWeighinRows": len(weights),
              "matchedWeighinBouts": len(matched), "unmatchedWeighinRows": len(weights) - len(matched),
              "reviewedAssertions": len(claims), "matchedAssertions": sum(bool(claim["fight_id"]) for claim in claims),
              "explicitWeightMissClaims": sum(claim["kind"] == "weight_miss" for claim in claims),
              "replacementClaims": sum(claim["kind"] == "replacement" for claim in claims),
              "sourceQuality": [{"source_id": source["source_id"], "flags": source.get("quality_flags", [])} for source in source_list if source.get("quality_flags")],
              "perEvent": per_event, "inputHashes": {str(path): digest(path) for path in inputs},
              "historicallyVerifiedRows": 0, "strictMarketEligibleRows": 0,
              "limitations": ["Current indexed official-source captures; publisher date does not prove fact-level historical availability.",
                              "Direct source access returned403 and was not retried or bypassed.",
                              "No mention of a miss/replacement remains unknown, never a negative label.",
                              "The completed-bout cohort is retrospective; unmatched source pairs and cancellation claims are retained separately.",
                              "Reported weights are transcribed observations, not a claim that every published number is correct.",
                              "Announcement-date gaps are not fighter notice periods. No model fitting or runtime changes."]}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "manifest.json").write_text(json.dumps({"files": {path.name: digest(path) for path in output.iterdir() if path.is_file()}}, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("events", "completedBoutDenominator", "sourceWeighinRows", "matchedWeighinBouts", "reviewedAssertions", "historicallyVerifiedRows")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce pre-fight context from hash-verified official search receipts. No network retries, inference or fitting.")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    build(arguments.bundle.resolve(), arguments.output.resolve())
