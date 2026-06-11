import argparse
import base64
import gzip
import html
import importlib.metadata
import json
import math
import platform
import re
import sys
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import collect_prefight_context as extraction
from scripts import evaluate_market_edge as edge
from scripts import evaluate_market_history as history


PROPOSAL = edge.DATA / "prefight-residual-proposal-v1/protocol.json"
PROPOSAL_SHA = "1fed07daf3dba8e24b786924e987b4da5e84caf188b4808058db69ce0a05cce9"
REGISTRY = edge.DATA / "market-edge-v1/verified-quotes.csv"
REGISTRY_SHA = "88520cca5cbfa51277d130d8ad1fceada6f976d97f8723462ed44b2e2a107a86"
HISTORY_SHA = "87f7e3ea29f9d238deb89f270b1a3baecd03751b57619fa060873b2182311ad6"
EXTRACTION_SHA = "80eb39324efdae378ceafa7676a35237a652955be37b7d3c2cb12ca1d0f713d7"
FEATURES = ["market_logit", "standard_excess_difference"]
IDENTITY = ["fight_id", "event_id", "date", "red_fighter_id", "blue_fighter_id", "red_name", "blue_name"]
REGISTRY_COLUMNS = IDENTITY + ["red_ticker", "blue_ticker"]
DIVISIONS = ["Strawweight", "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
             "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight"]
FORBIDDEN = {"target", "winner", "result", "winning_outcome_label", "resolution_status", "resolved_at"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def number(value: object) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value), "Invalid finite numeric evidence")
    return float(value)


def timestamp(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    require(not pd.isna(stamp) and stamp.tzinfo is not None, "Evidence timestamp needs an explicit timezone")
    return stamp.tz_convert("UTC")


def versions() -> dict:
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy"]}}


def checked(path: Path, expected: str, inputs: dict) -> Path:
    path = path.resolve()
    require(edge.digest(path) == expected, f"Input checksum changed: {path}")
    inputs[str(path)] = expected
    return path


def evidence_path(name: str) -> Path:
    path = (edge.ROOT / name).resolve()
    require(path.is_relative_to(edge.DATA.resolve()), "Evidence path escapes data/evaluation")
    return path


def no_outcomes(value: object) -> None:
    if isinstance(value, dict):
        require(not FORBIDDEN.intersection(value), "Outcome field in evidence input")
        for item in value.values():
            no_outcomes(item)
    elif isinstance(value, list):
        for item in value:
            no_outcomes(item)


def normalize_space(value: str) -> str:
    return " ".join(value.split())


def archived_source(source: dict, files: dict, inputs: dict) -> dict:
    import hashlib

    paths = {}
    for key in ["bodyPath", "receiptPath", "selectionPath", "cdxBodyPath", "cdxReceiptPath"]:
        name = source[key]
        require(name in files, "Archive evidence file is not hash-bound")
        paths[key] = checked(evidence_path(name), files[name], inputs)
    meta, selected, cmeta = (edge.read(paths[key]) for key in ["receiptPath", "selectionPath", "cdxReceiptPath"])
    cbody = paths["cdxBodyPath"].read_bytes()
    require(cmeta.get("status") == 200 and not cmeta.get("possiblyTruncated") and not cmeta.get("error"), "Invalid CDX receipt")
    require(edge.digest(paths["cdxBodyPath"]) == cmeta["bodySha256"] and len(cbody) == cmeta["bodyBytes"], "CDX body mismatch")
    table = json.loads(cbody)
    chosen = selected["selectedCDXRow"]
    require(chosen in [dict(zip(table[0], row)) for row in table[1:]], "Selected snapshot is absent from CDX response")
    require(chosen["original"] == source["originalURL"] and chosen["statuscode"] == "200" and chosen["mimetype"] == "text/html", "Wrong original archive source")
    require(source["originalURL"].startswith(("https://www.ufc.com/news/", "https://www.ufc.com/index.php/news/")), "Source is not official UFC news")
    url = f"https://web.archive.org/web/{chosen['timestamp']}id_/{chosen['original']}"
    require(meta.get("status") == 200 and not meta.get("possiblyTruncated") and not meta.get("error"), "Invalid archived body receipt")
    require(selected["requestUrl"] == meta["requestedUrl"] == meta["finalUrl"] == url, "Archive redirected or wrong snapshot")
    body = paths["bodyPath"].read_bytes()
    require(len(body) == meta["bodyBytes"] and edge.digest(paths["bodyPath"]) == meta["bodySha256"], "Archive body mismatch")
    require(base64.b32encode(hashlib.sha1(body).digest()).decode().rstrip("=") == chosen["digest"], "Archive CDX content digest mismatch")
    headers = {key.lower(): value for key, value in meta["headers"].items()}
    known = timestamp(parsedate_to_datetime(headers["memento-datetime"]).isoformat())
    require(known == pd.Timestamp(datetime.strptime(chosen["timestamp"], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)), "Archive timestamps disagree")
    encoding = headers.get("content-encoding", "identity")
    require(encoding in ["identity", "gzip"], "Unsupported archive encoding")
    body = gzip.decompress(body) if encoding == "gzip" else body
    require(len(body) < 5_000_000, "Decoded archive exceeds bound")
    text = re.sub(r"<(script|style)\b.*?</\1>", "", body.decode(), flags=re.S | re.I)
    text = normalize_space(html.unescape(re.sub("<[^>]+>", " ", text)))
    return {"knownBy": known, "text": text, "url": source["originalURL"], "bodySHA256": meta["bodySha256"]}


def claim_source(claim: dict, sources: dict, field: str = "exactSpan") -> dict:
    source = sources[claim["sourceId"]]
    span = normalize_space(claim[field])
    require(bool(span) and span in source["text"], "Claim span is absent from archived source")
    return source


def same_identity(record: dict, row: dict) -> None:
    require(all(record.get(key) == row[key] for key in IDENTITY[:5]), "Context exact identity/date mismatch")


def parse_numeric_pair(claim: dict, row: dict, sources: dict) -> tuple[dict, str, dict]:
    source = claim_source(claim, sources, "exactSourceLine")
    parsed = extraction.parse_weighins("## Official Weigh-In Results:\n" + claim["exactSourceLine"])
    require(len(parsed) == 1, "Expected one complete numeric weigh-in pair")
    pair = parsed[0]
    names = {extraction.name_key(row[f"{side}_name"]): row[f"{side}_fighter_id"] for side in ["red", "blue"]}
    require(len(names) == 2, "Ambiguous participant names")
    weights = {}
    for side in [1, 2]:
        key = extraction.name_key(pair[f"source_name{side}"])
        require(key in names, "Weigh-in participant does not match exact registry pair")
        weights[names[key]] = number(pair[f"weight{side}_lbs"])
    require(len(weights) == 2, "Duplicate weigh-in participant")
    return weights, pair["bout_label"], source


def parse_weight(claim: dict, row: dict, sources: dict) -> tuple[dict, str, bool, dict]:
    weights, label, source = parse_numeric_pair(claim, row, sources)
    require(not re.search(r"BMF|catch\s*weight|rebook|reschedul|changed|moved", label, re.I), "Ambiguous or changed division/title context")
    normalized = label.replace("’", "'")
    match = re.search(r"(?:(Women's)\s+)?(" + "|".join(sorted(DIVISIONS, key=len, reverse=True)) + r")(\s+Title)?(?:\s+Bout)?$", normalized, re.I)
    require(match is not None, "Ordinary division/title not explicit in source label")
    division = next(name for name in DIVISIONS if name.lower() == match[2].lower())
    if match[1]:
        division = "Women's " + division
    return weights, division, bool(match[3]), source


def schedule_status(record: dict | None, quote: pd.Timestamp, sources: dict) -> tuple[bool, str]:
    if record is None or record.get("schedule") is None:
        return False, "missing_schedule_evidence"
    claim = record["schedule"]
    source = claim_source(claim, sources)
    span = claim["exactSpan"].lower()
    require("prelim" in span and re.search(r"\d(?::\d\d)?\s*(?:am|pm|a\.m\.|p\.m\.)", span), "Schedule span lacks numeric prelim time")
    scheduled = timestamp(claim["firstPrelimUTC"])
    if source["knownBy"] > quote:
        return False, "schedule_evidence_after_quote"
    if not quote < scheduled <= quote + pd.Timedelta(days=2):
        return False, "quote_not_before_documented_first_prelim"
    return True, "qualified_schedule"


def context_feature(record: dict | None, row: dict, quote: pd.Timestamp, sources: dict) -> dict:
    empty = {"context_available": False, "standard_excess_difference": 0.0, "context_reason": "missing_weight_or_rule_evidence"}
    if record is None or record.get("weight") is None or record.get("rule") is None:
        return {**empty, "context_reason": "missing_rule_evidence" if record and record.get("weight") else "missing_weight_evidence"}
    if record.get("contextWarnings"):
        return {**empty, "context_reason": "reviewed_context_hold"}
    # Malformed identity/proof fails the input; documented ambiguity is a retained unknown.
    label = record["weight"]["exactSourceLine"].split(":", 1)[0]
    if re.search(r"BMF|catch\s*weight|rebook|reschedul|changed|moved", label, re.I):
        return {**empty, "context_reason": "ambiguous_division_or_title"}
    weights, division, title, weight_source = parse_weight(record["weight"], row, sources)
    rule = record["rule"]
    rule_source = claim_source(rule, sources)
    require(rule["division"] == division, "Rule division differs from archived bout label")
    limit, allowance = number(rule["standardLimitPounds"]), number(rule["nonTitleAllowancePounds"])
    require(80 <= limit <= 265 and allowance == 1 and number(rule["titleAllowancePounds"]) == 0, "Unsupported UFC standard allowance")
    require(extraction.name_key(division.replace("Women's ", "")) in extraction.name_key(rule["exactSpan"]), "Rule span lacks named division")
    require(re.search(r"(?<!\d)" + re.escape(f"{limit:g}") + r"(?!\d)", rule["exactSpan"]) is not None, "Rule span lacks standard limit")
    allowance_source = claim_source(rule, sources, "allowanceSpan")
    require("pound" in rule["allowanceSpan"].lower() and re.search(r"title|championship", rule["allowanceSpan"], re.I), "Allowance span lacks title distinction")
    if max(weight_source["knownBy"], rule_source["knownBy"], allowance_source["knownBy"]) > quote:
        return {**empty, "context_reason": "weight_or_rule_evidence_after_quote"}
    applicable = limit + (0 if title else allowance)
    red = max(0.0, weights[row["red_fighter_id"]] - applicable)
    blue = max(0.0, weights[row["blue_fighter_id"]] - applicable)
    return {"context_available": True, "standard_excess_difference": red - blue,
            "context_reason": "documented_standard_allowance", "red_weight_lbs": weights[row["red_fighter_id"]],
            "blue_weight_lbs": weights[row["blue_fighter_id"]], "standard_allowance_lbs": applicable,
            "source_division": division, "source_title": title}


def price_from_raw(decision: dict, row: dict, directory: Path, files: dict, inputs: dict) -> tuple[float, pd.Timestamp]:
    require(all(decision.get(key) == row[key] for key in REGISTRY_COLUMNS), "Market exact identity/ticker mismatch")
    cutoff = timestamp(row["date"] + "T00:00:00Z")
    require(timestamp(decision["cutoff"]) == cutoff, "Market cutoff is not event-day midnight")
    require(len(decision["markets"]) == 2, "Expected two market contracts")
    indexed = []
    for side in ["red", "blue"]:
        matches = [value for value in decision["markets"] if value["ticker"] == row[f"{side}_ticker"]]
        require(len(matches) == 1 and matches[0]["fighterId"] == row[f"{side}_fighter_id"], "Wrong contract-side identity")
        name = matches[0]["rawFile"]
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()) and name in files, "Unbound raw market file")
        require(decision.get(f"{side}_rawFile") == name, "Decision and selected raw source differ")
        receipt_name = decision[f"{side}_receiptFile"]
        receipt_path = (directory / receipt_name).resolve()
        require(receipt_path.is_relative_to(directory.resolve()) and receipt_name in files, "Unbound market HTTP receipt")
        receipt = edge.read(checked(receipt_path, files[receipt_name], inputs))
        require(receipt.get("status") == 200 and receipt.get("bodyComplete") is True and not receipt.get("error")
                and not receipt.get("stopped") and not receipt.get("stopReason"), "Market HTTP response is unsuccessful or incomplete")
        partition = decision[f"{side}_apiPartition"]
        require(partition in ["historical", "current"], "Unknown frozen market API route")
        ticker = row[f"{side}_ticker"]
        route = f"/historical/markets/{ticker}" if partition == "historical" else f"/series/KXUFCFIGHT/markets/{ticker}"
        expected_url = (f"https://external-api.kalshi.com/trade-api/v2{route}/candlesticks?"
                        f"start_ts={int(cutoff.timestamp()) - 3600}&end_ts={int(cutoff.timestamp())}&period_interval=1")
        require(receipt["url"] == expected_url, "Market request route/ticker/window differs from frozen contract")
        require(receipt.get("ticker", ticker) == ticker and receipt.get("apiPartition", partition) == partition,
                "HTTP receipt ticker/route mismatch")
        # Pilot receipts retain paths relative to their nested pilot bundle.
        receipt_base = receipt_path.parent.parent
        declared_body = (receipt_base / receipt.get("rawFile", receipt.get("bodyFile", ""))).resolve()
        require(declared_body == path, "HTTP receipt is for another response body")
        checked(path, files[name], inputs)
        require(edge.digest(path) == receipt["bodySha256"] and path.stat().st_size == receipt["bodyBytes"], "HTTP body hash/length mismatch")
        header_path = (receipt_base / receipt["headersFile"]).resolve()
        require(header_path.is_relative_to(directory.resolve()), "HTTP headers escape market bundle")
        header_name = str(header_path.relative_to(directory.resolve()))
        require(header_name in files, "Unbound HTTP response headers")
        checked(header_path, files[header_name], inputs)
        require(edge.digest(header_path) == receipt["headersSha256"] and header_path.stat().st_size == receipt["headerBytes"], "HTTP header hash/length mismatch")
        headers = {key.lower(): value for key, value in edge.read(header_path).items()}
        if "content-length" in headers:
            require(int(headers["content-length"]) == receipt["bodyBytes"], "HTTP Content-Length mismatch")
        payload = edge.read(checked(path, files[name], inputs))
        require(payload.get("ticker", row[f"{side}_ticker"]) == row[f"{side}_ticker"], "Raw market ticker mismatch")
        values = {}
        for candle in payload["candlesticks"]:
            when = candle["end_period_ts"]
            require(type(when) is int, "Invalid raw candle timestamp")
            if cutoff.timestamp() - 3600 <= when <= cutoff.timestamp():
                require(when not in values, "Duplicate raw candle timestamp")
                values[when] = candle
        indexed.append(values)
    for when in sorted(indexed[0].keys() & indexed[1].keys(), reverse=True):
        mids = [history.midpoint(values[when]) for values in indexed]
        if None in mids:
            continue
        observed = pd.Timestamp(when, unit="s", tz="UTC")
        require(observed == timestamp(decision["quoteObservedAt"]) and when == decision["quoteEndTimestamp"], "Selected quote is not latest valid common candle")
        for side, mid in zip(["red", "blue"], mids):
            declared = next(value for value in decision["markets"] if value["ticker"] == row[f"{side}_ticker"])
            require(abs(number(declared["midpoint"]) - mid) <= 1e-12, "Declared market midpoint differs from raw candle")
        return mids[0] / sum(mids), observed
    raise ValueError("Qualified quote has no valid simultaneous raw candles")


def gates(frame: pd.DataFrame) -> dict:
    parts = edge.split(frame)
    train = parts["fit"]
    known = train[train.market_available & train.context_available]
    nonzero = known[known.standard_excess_difference != 0]
    counts = {"knownTrainingPairs": len(known), "knownTrainingEvents": int(known.event_id.nunique()),
              "nonzeroTrainingPairs": len(nonzero), "nonzeroTrainingEvents": int(nonzero.event_id.nunique()),
              "knownZeroTrainingPairs": int((known.standard_excess_difference == 0).sum())}
    minimum = dict(zip(counts, [40, 10, 10, 5, 30]))
    failures = [key for key in counts if counts[key] < minimum[key]]
    market = {name: {"registryRows": len(part), "qualifiedQuotes": int(part.market_available.sum()),
                     "minimum": threshold} for (name, part), threshold in zip(parts.items(), [102, 124, 174])}
    return {"trainingContextCounts": counts, "trainingContextMinimums": minimum,
            "trainingContextFailures": failures, "trainingContextPassed": not failures,
            "marketCoverage": market, "allFitGatesPassed": not failures and all(v["qualifiedQuotes"] >= v["minimum"] for v in market.values()),
            "notAPowerGuarantee": True}


def read_manifest(path: Path, checksum: str, inputs: dict) -> dict:
    manifest = edge.read(checked(path, checksum, inputs))
    files = manifest.get("files", manifest)
    require(isinstance(files, dict) and all(isinstance(v, str) for v in files.values()), "Expected file-to-SHA256 manifest")
    for name, value in files.items():
        source = (path.parent / name).resolve()
        require(source.is_relative_to(path.parent.resolve()), "Manifest path escapes bundle")
        checked(source, value, inputs)
    return files


def qualify(market_manifest: Path, market_sha: str, context_path: Path, context_sha: str, output: Path,
            market_decisions: str = "decisions.json") -> dict:
    require(not output.exists(), "Refusing to overwrite qualification")
    inputs = {}
    for path, checksum in [(PROPOSAL, PROPOSAL_SHA), (REGISTRY, REGISTRY_SHA), (Path(history.__file__), HISTORY_SHA), (Path(extraction.__file__), EXTRACTION_SHA)]:
        checked(path, checksum, inputs)
    for path in [Path(__file__), Path(edge.__file__), edge.COVERAGE / "score_coverage.py", edge.ROOT / "tests/test_prefight_market.py"]:
        inputs[str(path.resolve())] = edge.digest(path)
    files = read_manifest(market_manifest, market_sha, inputs)
    decisions_path = (market_manifest.parent / market_decisions).resolve()
    require(market_decisions in files and decisions_path.is_relative_to(market_manifest.parent.resolve()), "Market decisions file is not hash-bound inside its bundle")
    decisions = edge.read(decisions_path)
    context = edge.read(checked(context_path, context_sha, inputs))
    no_outcomes(decisions)
    no_outcomes(context)
    require(context.get("schemaVersion") == 1 and context.get("registrySHA256") == REGISTRY_SHA, "Wrong context registry/schema")
    sources = {source["sourceId"]: archived_source(source, context["files"], inputs) for source in context["sources"]}
    require(len(sources) == len(context["sources"]), "Duplicate evidence source IDs")
    records = {row["fight_id"]: row for row in context["records"]}
    prices = {row["fight_id"]: row for row in decisions}
    require(len(records) == len(context["records"]) and len(prices) == len(decisions), "Duplicate context or market bout")
    registry = pd.read_csv(REGISTRY, usecols=REGISTRY_COLUMNS, dtype=str)
    require(set(records) <= set(registry.fight_id) and set(prices) <= set(registry.fight_id), "Unknown evidence bout ID")
    rows = []
    for row in registry.to_dict("records"):
        record, decision = records.get(row["fight_id"]), prices.get(row["fight_id"])
        result = {**row, "market_available": False, "market_reason": "not_collected", "p_market": None,
                  "quote_observed_at": None, "context_available": False, "context_reason": "no_qualified_quote",
                  "standard_excess_difference": 0.0, "archived_numeric_pair_available": False,
                  "numeric_pair_before_selected_quote": None, "weight_known_by": None}
        if record is not None:
            same_identity(record, row)
            if record.get("weight") is not None:
                _, _, weight_source = parse_numeric_pair(record["weight"], row, sources)
                result.update(archived_numeric_pair_available=True, weight_known_by=weight_source["knownBy"].isoformat())
        if decision is not None:
            require(all(decision.get(key) == row[key] for key in REGISTRY_COLUMNS), "Market decision identity mismatch")
            result["market_reason"] = decision["reason"]
            if decision["reason"] == "qualified":
                probability, observed = price_from_raw(decision, row, market_manifest.parent, files, inputs)
                valid, reason = schedule_status(record, observed, sources)
                result.update(market_reason=reason, quote_observed_at=observed.isoformat())
                if result["archived_numeric_pair_available"]:
                    result["numeric_pair_before_selected_quote"] = timestamp(result["weight_known_by"]) <= observed
                if valid:
                    result.update(market_available=True, p_market=probability, **context_feature(record, row, observed, sources))
        rows.append(result)
    frame = pd.DataFrame(rows)
    report = gates(frame)
    require(all(edge.digest(Path(path)) == checksum for path, checksum in inputs.items()), "Qualification inputs changed")
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "qualified-registry.csv", index=False)
    edge.write(output / "qualification.json", {"createdAt": edge.now(), "proposalSHA256": PROPOSAL_SHA, "inputHashes": inputs,
               "dependencies": versions(), "gates": report, "outcomesLoaded": False, "fits": 0,
               "semanticReview": "Rule applicability and first-prelim UTC conversion are reviewed exact-span interpretations pinned by context JSON checksum, not established by string containment alone.",
               "coverage": {name: {"registryRows": len(part), "archivedNumericPairs": int(part.archived_numeric_pair_available.sum()),
                                    "numericPairsBeforeSelectedQuote": int(part.numeric_pair_before_selected_quote.eq(True).sum()),
                                    "fullyQualifiedContextPairs": int(part.context_available.sum()), "marketReasons": part.market_reason.value_counts().to_dict(),
                                    "contextReasons": part.context_reason.value_counts().to_dict()} for name, part in edge.split(frame).items()}})
    edge.write(output / "manifest.json", {"files": {name: edge.digest(output / name) for name in ["qualified-registry.csv", "qualification.json"]}})
    return report


def evaluate(qualification_manifest: Path, qualification_sha: str, output: Path) -> dict:
    require(not output.exists(), "Refusing to overwrite experiment")
    inputs = {}
    qualified_files = read_manifest(qualification_manifest, qualification_sha, inputs)
    require({"qualification.json", "qualified-registry.csv"} <= set(qualified_files), "Qualification records are not hash-bound")
    directory = qualification_manifest.parent
    receipt = edge.read(directory / "qualification.json")
    require(receipt["proposalSHA256"] == PROPOSAL_SHA and receipt["dependencies"] == versions(), "Qualification protocol/dependency changed")
    for path, checksum in receipt["inputHashes"].items():
        checked(Path(path), checksum, inputs)
    frame = pd.read_csv(directory / "qualified-registry.csv", float_precision="round_trip")
    require(gates(frame) == receipt["gates"] and receipt["gates"]["allFitGatesPassed"], "No fit: predeclared support/coverage gates failed")
    require(not FORBIDDEN.intersection(frame.columns), "Outcome column in qualification records")
    checked(REGISTRY, REGISTRY_SHA, inputs)
    registry = pd.read_csv(REGISTRY, usecols=REGISTRY_COLUMNS, dtype=str)
    require(len(frame) == 499 and frame.fight_id.is_unique and set(frame.fight_id) == set(registry.fight_id), "Qualification fixed499 identities changed")
    require(np.array_equal(frame.sort_values("fight_id")[REGISTRY_COLUMNS].to_numpy(),
                           registry.sort_values("fight_id")[REGISTRY_COLUMNS].to_numpy()), "Qualification identity/date/cohort differs from registry")
    targets = pd.read_csv(REGISTRY, usecols=["fight_id", "target"])
    frame = frame.merge(targets, on="fight_id", validate="one_to_one")
    frame["market_logit"] = logit(frame.p_market.clip(1e-6, 1 - 1e-6))
    frame["pick_market"] = frame.p_market >= .5
    parts = edge.split(frame)
    eligible = {name: part[part.market_available].copy() for name, part in parts.items()}
    output.mkdir(parents=True, exist_ok=False)
    edge.write(output / "protocol.json", {"createdAt": edge.now(), "proposalSHA256": PROPOSAL_SHA,
               "qualificationSHA256": qualification_sha, "inputHashes": inputs, "dependencies": versions()})
    fitted = {name: history.fit(eligible["fit"], eligible["calibration"], columns)
              for name, columns in [("market_control", FEATURES[:1]), ("market_context", FEATURES)]}
    edge.write(output / "fitted-models.json", fitted)
    edge.write(output / "pre-test-manifest.json", {"createdAt": edge.now(), "files": {name: edge.digest(output / name) for name in ["protocol.json", "fitted-models.json"]}, "testPredictionsComputed": False})
    selected = frame.market_available
    for name, fitted_model in fitted.items():
        frame.loc[selected, name] = history.predict(frame[selected], fitted_model)
        reverse = frame[selected].copy()
        reverse[FEATURES] *= -1
        require(np.allclose(frame.loc[selected, name] + history.predict(reverse, fitted_model), 1, atol=1e-12, rtol=0), "Corner symmetry failed")
        frame[f"pick_{name}"] = frame[name].where(selected).ge(.5).where(selected)
    predictors = {"market": ("p_market", "pick_market"), **{name: (name, "pick_" + name) for name in fitted}}
    results = {}
    for name, part in edge.split(frame).items():
        scored = part[part.market_available].copy()
        results[name] = {"registryDenominator": len(part), "scoredIdenticalRows": len(scored), "unpredictedRows": len(part) - len(scored),
                         "metrics": {model: edge.metrics(scored, *fields) for model, fields in predictors.items()},
                         "paired": edge.frozen_helpers().bootstrap(scored, predictors, [("market_context", "market"), ("market_context", "market_control")])}
    require(all(edge.digest(Path(path)) == checksum for path, checksum in inputs.items()), "Evaluation input changed")
    require(versions() == receipt["dependencies"], "Evaluation dependencies changed")
    for name, checksum in edge.read(output / "pre-test-manifest.json")["files"].items():
        require(edge.digest(output / name) == checksum, "Sealed model changed")
    frame.to_csv(output / "predictions.csv", index=False)
    edge.write(output / "report.json", {"createdAt": edge.now(), "results": results, "promoted": False, "runtimeChanged": False,
               "limitations": ["Previously examined historical replay; no untouched validation or executable-profit claim.",
                   "Standard allowance excess is not a verified contractual miss.", "Unknown context contributes zero context input; fitted market recalibration can still change its probability.",
                   "Scheduled first prelim is not verified actual bell; true quote update age and fillability remain unknown.",
                   "Minimum support gates and event bootstrap do not establish power or correct repeated experimentation."]})
    edge.write(output / "manifest.json", {"files": {path.name: edge.digest(path) for path in output.iterdir() if path.is_file()}})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qualify a fixed standard-allowance market residual; evaluation is a separate gated action")
    sub = parser.add_subparsers(dest="command", required=True)
    admission = sub.add_parser("qualify")
    admission.add_argument("--market-manifest", type=Path, required=True)
    admission.add_argument("--market-sha", required=True)
    admission.add_argument("--market-decisions", default="decisions.json")
    admission.add_argument("--context", type=Path, required=True)
    admission.add_argument("--context-sha", required=True)
    admission.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("evaluate")
    run.add_argument("--qualification-manifest", type=Path, required=True)
    run.add_argument("--qualification-sha", required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (qualify(args.market_manifest.resolve(), args.market_sha, args.context.resolve(), args.context_sha, args.output.resolve(), args.market_decisions)
              if args.command == "qualify" else evaluate(args.qualification_manifest.resolve(), args.qualification_sha, args.output.resolve()))
    print(json.dumps(result, indent=2))
