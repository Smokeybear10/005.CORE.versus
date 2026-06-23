import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from http.client import HTTPException
from io import BytesIO
from pathlib import Path
from threading import Lock
from urllib.error import URLError
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen

import pandas as pd

from model_engine import FightPredictor


ODDS_PROVIDER = "The Odds API"
ODDS_UNAVAILABLE = "Prices unavailable; enter odds manually"


def odds_api_key() -> str:
    return os.environ.get("ODDS_API_KEY", "").strip() or os.environ.get("THE_ODDS_API_KEY", "").strip()


def date_string(value: object) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce")
    return parsed.date().isoformat() if pd.notna(parsed) else None


def utc_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None


def iso_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def text_value(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def recorded_record(fights: list[dict]) -> dict:
    return {
        key: sum(fight["result"] == result for fight in fights)
        for key, result in (("wins", "win"), ("losses", "loss"), ("draws", "draw"),
                            ("noContests", "no_contest"), ("unknown", "unknown"))
    }


def win_methods(fights: list[dict]) -> dict:
    counts = {"koTko": 0, "submission": 0, "decision": 0, "other": 0}
    for fight in fights:
        if fight["result"] != "win":
            continue
        method = (fight["method"] or "").casefold()
        key = "koTko" if "ko" in method else "submission" if "submission" in method else "decision" if "decision" in method else "other"
        counts[key] += 1
    return counts


def fight_duration(finish_round: int, finish_time: str, time_format: str) -> int | None:
    if not isinstance(finish_time, str) or not re.fullmatch(r"\d+:[0-5]\d", finish_time.strip()):
        return None
    minutes, seconds = (int(value) for value in finish_time.strip().split(":"))
    elapsed = minutes * 60 + seconds
    if finish_round == 1 and time_format == "No Time Limit":
        return elapsed or None
    if not isinstance(time_format, str):
        return None
    match = re.search(r"\((\d+(?:-\d+)*)\)\s*$", time_format)
    if match is None:
        return None
    lengths = [int(value) * 60 for value in match.group(1).split("-")]
    if finish_round < 1 or finish_round > len(lengths) or not all(lengths):
        return None
    if elapsed > lengths[finish_round - 1]:
        return None
    total = sum(lengths[:finish_round - 1]) + elapsed
    return total or None


class ResearchData:
    def __init__(self, predictor: FightPredictor, artifact_path: str,
                 bundle_directory: str | Path | None = None,
                 career_directory: str | Path | None = None,
                 career_manifest_sha256: str | None = None):
        self.predictor = predictor
        self.explicit_bundle = bundle_directory is not None
        self.research_bundle_directory = None
        self.research_manifest_sha256 = None
        self.model_version = None
        artifact = Path(artifact_path)
        if artifact.is_file():
            with artifact.open("rb") as stream:
                self.model_version = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()

        model_fights = predictor.norm_fights
        profiles = predictor.fights_df
        self.model_data_through = date_string(model_fights["date"].max()) if model_fights is not None else None
        self.profile_data_through = date_string(profiles["Date"].max()) if profiles is not None else None
        self.model_fight_counts = pd.concat([model_fights["red_fighter_id"], model_fights["blue_fighter_id"]]).value_counts().to_dict() if model_fights is not None else {}
        self.research_fights = None if self.explicit_bundle else model_fights
        self.data_through = None if self.explicit_bundle else self.model_data_through
        self.provenance = {"source": "UFCStats saved dataset", "sourceUrl": "http://ufcstats.com/statistics/events/completed?page=all", "retrievedAt": None}
        self.refresh_status = "not_available"
        self.refresh_error = None
        self.histories = {}
        self.round_stats = {}
        self.snapshot_by_id = {}
        self.registry_by_id = {}
        self.fighter_labels_by_id = {}
        self.ambiguous_names = set()
        self.career_histories = {}
        self.career_source = None
        self.career_cutoff = None
        self.career_manifest_sha256 = None
        self.career_bundle_directory = None
        self.career_load_error = None
        snapshots = None if self.explicit_bundle else predictor.snapshots
        if snapshots is not None:
            self.snapshot_by_id = snapshots.set_index("fighter_id").to_dict("index")
        directory = Path(bundle_directory) if self.explicit_bundle else artifact.parent / "research-data"
        refreshed = self._load_research_bundle(directory)
        registry = None
        rounds = None
        if refreshed is not None:
            self.research_fights, rounds, registry, manifest = refreshed
            self.data_through = manifest["dataThrough"]
            self.provenance = {field: manifest[field] for field in ("source", "sourceUrl", "retrievedAt")}
            self.refresh_status = "loaded"
            self.registry_by_id = registry.set_index("fighter_id").to_dict("index")
        self.fighters = self._fighter_registry(snapshots, registry, self.research_fights)
        fights = self.research_fights
        if fights is None:
            return
        self._load_round_stats(artifact.parent / "rounds.parquet", fights, rounds)
        for row in fights.sort_values(["date", "fight_id"], ascending=False).itertuples(index=False):
            for side, fighter_id, opponent, opponent_id in (
                ("red", row.red_fighter_id, row.blue_name, row.blue_fighter_id),
                ("blue", row.blue_fighter_id, row.red_name, row.red_fighter_id),
            ):
                if pd.isna(fighter_id):
                    continue
                result = "unknown"
                if row.winner in ("red", "blue"):
                    result = "win" if row.winner == side else "loss"
                elif row.winner in ("draw", "nc"):
                    result = "draw" if row.winner == "draw" else "no_contest"
                fight_id = text_value(row.fight_id)
                self.histories.setdefault(fighter_id, []).append({
                    "fightId": fight_id,
                    "date": date_string(row.date),
                    "opponent": self.fighter_labels_by_id.get(opponent_id, text_value(opponent)),
                    "opponentId": text_value(opponent_id),
                    "weightClass": text_value(row.weightclass),
                    "resultUncertainty": text_value(getattr(row, "result_uncertainty", None)),
                    "result": result,
                    "method": text_value(row.method),
                    "round": int(row.finish_round) if pd.notna(row.finish_round) else None,
                    "url": f"http://ufcstats.com/fight-details/{fight_id}"
                    if fight_id and re.fullmatch(r"[a-f0-9]{16}", fight_id) else None,
                })
        if career_directory is not None:
            self._load_career_bundle(Path(career_directory), career_manifest_sha256)

    def _load_career_bundle(self, directory: Path, expected_sha256: str | None) -> None:
        try:
            content = (directory / "manifest.json").read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if not isinstance(expected_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_sha256) or digest != expected_sha256:
                raise ValueError("Career manifest checksum mismatch")
            manifest = json.loads(content)
            if manifest.get("schemaVersion") != 1 or manifest.get("bundleType") != "career-supplement":
                raise ValueError("Unsupported career supplement")
            source = manifest["source"]
            source_url = urlparse(source["url"])
            retrieved = utc_datetime(source["retrievedAt"])
            if source_url.scheme != "https" or source_url.hostname != "github.com" or not source_url.path.casefold().startswith("/jadevit/fight-canon-db/releases/tag/") or not text_value(source.get("name")):
                raise ValueError("Career source provenance is invalid")
            if not re.fullmatch(r"[a-f0-9]{64}", source["databaseSha256"]) or retrieved is None or retrieved > datetime.now(timezone.utc) + timedelta(minutes=5):
                raise ValueError("Career source retrieval is invalid")
            cutoff = manifest["cutoffInclusive"]
            if not isinstance(cutoff, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cutoff) or date_string(cutoff) != cutoff or cutoff > datetime.now(timezone.utc).date().isoformat():
                raise ValueError("Career cutoff is invalid")
            files = manifest["files"]
            if not isinstance(files, dict) or "fighter_bouts.parquet" not in files:
                raise ValueError("Career history file is missing")
            histories = None
            for filename, checksum in files.items():
                if not isinstance(filename, str) or Path(filename).name != filename or filename in (".", "..", "manifest.json") or not isinstance(checksum, str) or not re.fullmatch(r"[a-f0-9]{64}", checksum):
                    raise ValueError("Career file identity is invalid")
                file_bytes = (directory / filename).read_bytes()
                if hashlib.sha256(file_bytes).hexdigest() != checksum:
                    raise ValueError("Career file checksum mismatch")
                if filename == "fighter_bouts.parquet":
                    histories = pd.read_parquet(BytesIO(file_bytes))
            columns = {"fighter_id", "opponent_id", "canonical_fighter_id", "canonical_opponent_id",
                       "canonical_fight_id", "canonical_event_id", "date", "result", "opponent_result",
                       "fighter_name", "opponent_name", "promotion", "event_name", "method",
                       "source_aliases_json", "source_database_retrieved_at"}
            if histories is None or not columns.issubset(histories.columns):
                raise ValueError("Career history columns are incomplete")
            valid_id = lambda value: isinstance(value, str) and bool(re.fullmatch(r"[a-f0-9]{16}", value))
            if not histories["fighter_id"].map(valid_id).all() or not histories["opponent_id"].dropna().map(valid_id).all():
                raise ValueError("Career fighter identities are invalid")
            for column in ("canonical_fighter_id", "canonical_opponent_id", "canonical_fight_id", "canonical_event_id"):
                numbers = pd.to_numeric(histories[column], errors="coerce")
                if not (numbers.gt(0) & numbers.lt(float("inf")) & numbers.mod(1).eq(0)).all():
                    raise ValueError("Canonical career identities are invalid")
            dates = pd.to_datetime(histories["date"], errors="coerce")
            if dates.isna().any() or not histories["date"].map(lambda value: isinstance(value, str) and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))).all() or histories["date"].gt(cutoff).any():
                raise ValueError("Career bout dates are invalid")
            if histories.duplicated(["fighter_id", "canonical_fight_id"]).any() or histories["canonical_fighter_id"].eq(histories["canonical_opponent_id"]).any() or histories["fighter_id"].eq(histories["opponent_id"]).any():
                raise ValueError("Career bout identities are duplicated or self-referential")
            valid_results = {("W", "L"), ("L", "W"), ("D", "D"), ("NC", "NC")}
            if any(pair not in valid_results for pair in histories[["result", "opponent_result"]].itertuples(index=False, name=None)):
                raise ValueError("Career outcomes are inconsistent")
            identity_pairs = pd.concat([
                histories[["fighter_id", "canonical_fighter_id"]].rename(columns={"fighter_id": "source", "canonical_fighter_id": "canonical"}),
                histories[["opponent_id", "canonical_opponent_id"]].rename(columns={"opponent_id": "source", "canonical_opponent_id": "canonical"}),
            ]).dropna()
            if identity_pairs.groupby("source").canonical.nunique().gt(1).any() or identity_pairs.groupby("canonical").source.nunique().gt(1).any():
                raise ValueError("Career identity mapping is ambiguous")
            pairs = [tuple(sorted((int(row.canonical_fighter_id), int(row.canonical_opponent_id)))) for row in histories.itertuples()]
            outcomes = [f"winner:{int(row.canonical_fighter_id if row.result == 'W' else row.canonical_opponent_id)}"
                        if row.result in ("W", "L") else row.result for row in histories.itertuples()]
            audit = histories.assign(pair=pairs, outcome=outcomes)
            if audit.groupby(["date", "pair"]).canonical_fight_id.nunique().gt(1).any():
                raise ValueError("Career same-day pairs are ambiguous")
            if audit.groupby("canonical_fight_id")[["date", "pair", "canonical_event_id", "method", "outcome"]].nunique(dropna=False).gt(1).any().any():
                raise ValueError("Career bout versions conflict")
            if audit.groupby("canonical_fight_id").size().gt(2).any():
                raise ValueError("Career bout has too many participants")
            known_fight_ids = set(self.research_fights.fight_id)
            known_pairs = {(date_string(row.date), tuple(sorted((row.red_fighter_id, row.blue_fighter_id))))
                           for row in self.research_fights.itertuples(index=False)
                           if valid_id(row.red_fighter_id) and valid_id(row.blue_fighter_id)}
            loaded = {}
            result_names = {"W": "win", "L": "loss", "D": "draw", "NC": "no_contest"}
            for row in histories.sort_values(["date", "canonical_fight_id"], ascending=False).itertuples(index=False):
                if valid_id(row.opponent_id) and (row.date, tuple(sorted((row.fighter_id, row.opponent_id)))) in known_pairs:
                    raise ValueError("Career supplement overlaps recorded UFC bouts")
                aliases = json.loads(row.source_aliases_json)
                if not isinstance(aliases, list) or not aliases or any(not isinstance(alias, dict) for alias in aliases):
                    raise ValueError("Career bout source aliases are invalid")
                urls = []
                for alias in aliases:
                    if alias.get("source") == "ufcstats" and alias.get("sourceID") in known_fight_ids:
                        raise ValueError("Career supplement repeats a recorded UFC source bout")
                    raw_url = alias.get("sourceURL")
                    parsed = urlparse(raw_url) if isinstance(raw_url, str) else None
                    if parsed and parsed.scheme in ("http", "https") and parsed.hostname in ("ufcstats.com", "www.ufcstats.com", "sherdog.com", "www.sherdog.com"):
                        urls.append(raw_url)
                if row.date > self.data_through:
                    continue
                loaded.setdefault(row.fighter_id, []).append({
                    "canonicalFightId": int(row.canonical_fight_id), "date": row.date,
                    "opponent": self.fighter_labels_by_id.get(row.opponent_id, text_value(row.opponent_name)),
                    "opponentId": text_value(row.opponent_id), "promotion": text_value(row.promotion),
                    "event": text_value(row.event_name), "result": result_names[row.result],
                    "method": text_value(row.method), "sourceUrl": sorted(urls)[0] if urls else None,
                })
            if (directory / "manifest.json").read_bytes() != content:
                raise ValueError("Career manifest changed during loading")
            self.career_histories = loaded
            self.career_cutoff = min(cutoff, self.data_through)
            self.career_manifest_sha256 = digest
            self.career_bundle_directory = directory.resolve()
            self.career_source = {"name": source["name"], "url": source["url"], "retrievedAt": source["retrievedAt"],
                                  "manifestSha256": digest, "databaseSha256": source["databaseSha256"], "knownAt": manifest.get("knownAt")}
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            self.career_load_error = "Additional professional results could not be verified. Recorded UFC research remains available."

    def career_history(self, fighter_id: str, as_of: str | None = None) -> dict:
        cutoff = self.career_cutoff or self.data_through
        if as_of is not None:
            if not isinstance(as_of, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", as_of) or date_string(as_of) != as_of:
                raise ValueError("Use a valid YYYY-MM-DD event date")
            previous_day = (datetime.fromisoformat(as_of) - timedelta(days=1)).date().isoformat()
            cutoff = min(cutoff, previous_day) if cutoff else previous_day
        fights = [fight for fight in self.career_histories.get(fighter_id, []) if cutoff and fight["date"] <= cutoff]
        promotions = {}
        for fight in fights:
            promotions[fight["promotion"]] = promotions.get(fight["promotion"], 0) + 1
        return {
            "available": self.career_source is not None, "scope": "additional_professional_results",
            "asOf": cutoff, "beforeEventDate": as_of, "source": self.career_source,
            "totalRecordedBouts": len(fights), "record": recorded_record(fights), "winMethods": win_methods(fights),
            "promotions": [{"name": name, "bouts": count} for name, count in sorted(promotions.items(), key=lambda item: (-item[1], item[0] or ""))],
            "recentBouts": fights[:12],
            "coverage": {"completeCareer": False, "historicalAvailabilityVerified": False},
            "loadError": self.career_load_error,
        }

    def _load_research_bundle(self, directory: Path) -> tuple | None:
        if not directory.exists() and not self.explicit_bundle:
            return None
        try:
            manifest_bytes = (directory / "manifest.json").read_bytes()
            manifest = json.loads(manifest_bytes)
            if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 1:
                raise ValueError("Unsupported research data manifest")
            files = manifest.get("files")
            names = ("fights.parquet", "rounds.parquet", "fighters.parquet")
            if not isinstance(files, dict) or set(files) != set(names):
                raise ValueError("Research data files are incomplete")
            source_url = manifest.get("sourceUrl")
            parsed_url = urlparse(source_url) if isinstance(source_url, str) else None
            if not text_value(manifest.get("source")) or parsed_url is None or parsed_url.scheme not in ("http", "https") or parsed_url.hostname not in ("ufcstats.com", "www.ufcstats.com", "github.com"):
                raise ValueError("Research source provenance is invalid")
            retrieved = utc_datetime(manifest.get("retrievedAt"))
            if retrieved is None or retrieved > datetime.now(timezone.utc) + timedelta(minutes=5):
                raise ValueError("Research retrieval timestamp is invalid")
            frames = {}
            for name in names:
                digest = files[name]
                if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                    raise ValueError("Research file checksum is invalid")
                content = (directory / name).read_bytes()
                if hashlib.sha256(content).hexdigest() != digest:
                    raise ValueError("Research data files could not be verified together")
                frames[name] = pd.read_parquet(BytesIO(content))
            fights = frames["fights.parquet"]
            rounds = frames["rounds.parquet"]
            fighters = frames["fighters.parquet"]
            fight_columns = {"fight_id", "date", "red_fighter_id", "blue_fighter_id", "red_name", "blue_name", "winner", "method", "finish_round", "finish_time", "weightclass"}
            round_columns = {"fight_id", "fighter_id", "round", "sig_str_landed", "sig_str_attempted", "td_landed", "td_attempted"}
            if not fight_columns.issubset(fights.columns) or not round_columns.issubset(rounds.columns) or not {"fighter_id", "name"}.issubset(fighters.columns):
                raise ValueError("Research data columns are incomplete")
            valid_ids = fighters["fighter_id"].map(lambda value: isinstance(value, str) and bool(re.fullmatch(r"[a-f0-9]{16}", value)))
            if fighters.empty or not valid_ids.all() or fighters["fighter_id"].duplicated().any() or not fighters["name"].map(text_value).notna().all():
                raise ValueError("Research fighter identities are invalid")
            for side in ("red", "blue"):
                known = fights[f"{side}_fighter_id"].dropna()
                if self.explicit_bundle and not known.isin(fighters["fighter_id"]).all():
                    raise ValueError("Research fight participants are missing from the registry")
            if (fights["red_fighter_id"].notna() & fights["red_fighter_id"].eq(fights["blue_fighter_id"])).any():
                raise ValueError("Research fight participants must be distinct")
            dates = pd.to_datetime(fights["date"], errors="coerce", utc=True)
            if fights.empty or dates.isna().any() or fights["fight_id"].duplicated().any() or not fights["fight_id"].map(lambda value: isinstance(value, str) and bool(re.fullmatch(r"[a-f0-9]{16}", value))).all():
                raise ValueError("Research fight identities or dates are invalid")
            if not fights["winner"].isin(["red", "blue", "draw", "nc"]).all():
                raise ValueError("Research fight outcomes are invalid")
            finish_rounds = pd.to_numeric(fights["finish_round"], errors="coerce")
            known_rounds = fights["finish_round"].notna()
            valid_rounds = finish_rounds.ge(1) & finish_rounds.lt(float("inf")) & finish_rounds.mod(1).eq(0)
            if (known_rounds & ~valid_rounds).any():
                raise ValueError("Research finish rounds are invalid")
            through = date_string(dates.max())
            if manifest.get("dataThrough") != through or dates.max().date() > datetime.now(timezone.utc).date():
                raise ValueError("Research coverage date does not match its data")
            if not rounds["fight_id"].isin(fights["fight_id"]).all():
                raise ValueError("Research round statistics refer to unknown fights")
            if "time_format" not in rounds:
                if "time_format" not in fights:
                    raise ValueError("Research bout time formats are unavailable")
                rounds = rounds.merge(fights[["fight_id", "time_format"]], on="fight_id", how="left", validate="many_to_one", indicator=True)
                if not rounds["_merge"].eq("both").all():
                    raise ValueError("Research round statistics have no matching bout format")
                rounds = rounds.drop(columns="_merge")
            if (directory / "manifest.json").read_bytes() != manifest_bytes:
                raise ValueError("Research manifest changed during loading")
            fights = fights.assign(date=dates.dt.tz_localize(None), finish_round=finish_rounds)
            self.research_bundle_directory = directory.resolve()
            self.research_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            return fights, rounds, fighters, manifest
        except (OSError, ValueError, TypeError, KeyError):
            self.refresh_status = "invalid"
            self.refresh_error = "The selected research dataset could not be verified. Research is unavailable." if self.explicit_bundle else "The refreshed research dataset could not be verified. Showing saved results instead."
            return None

    def _fighter_registry(self, snapshots: pd.DataFrame | None, registry: pd.DataFrame | None, fights: pd.DataFrame | None) -> dict:
        candidates = []
        for frame in (registry, snapshots):
            if frame is not None:
                candidates.extend((row.name, row.fighter_id) for row in frame[["name", "fighter_id"]].sort_values("fighter_id").itertuples(index=False))
        divisions = {}
        if fights is not None:
            for row in fights.sort_values(["date", "fight_id"], ascending=False).itertuples(index=False):
                candidates.extend(((row.red_name, row.red_fighter_id), (row.blue_name, row.blue_fighter_id)))
                for fighter_id in (row.red_fighter_id, row.blue_fighter_id):
                    if text_value(row.weightclass):
                        divisions.setdefault(fighter_id, row.weightclass.strip())
        aliases = {}
        names_by_id = {}
        for raw_name, fighter_id in candidates:
            name = text_value(raw_name)
            if not name or not isinstance(fighter_id, str) or not re.fullmatch(r"[a-f0-9]{16}", fighter_id):
                continue
            aliases.setdefault(name.casefold(), set()).add(fighter_id)
            names_by_id.setdefault(fighter_id, []).append(name)
        labels = {}
        for fighter_id, names in sorted(names_by_id.items()):
            name = names[0]
            if len(aliases[name.casefold()]) > 1:
                profile = self.registry_by_id.get(fighter_id, {})
                nickname = text_value(profile.get("nickname"))
                weight = pd.to_numeric(profile.get("weight_lbs"), errors="coerce")
                weight_label = f"{weight:g} lb" if pd.notna(weight) and math.isfinite(weight) and weight > 0 else None
                qualifier = nickname or divisions.get(fighter_id) or weight_label or fighter_id
                name = f"{name} ({qualifier})"
            labels[fighter_id] = name
        label_ids = {}
        for fighter_id, label in labels.items():
            label_ids.setdefault(label.casefold(), set()).add(fighter_id)
        for fighter_id, label in labels.items():
            if (label_ids[label.casefold()] | aliases.get(label.casefold(), set())) - {fighter_id}:
                labels[fighter_id] = f"{label} [{fighter_id}]"
        self.fighter_labels_by_id = labels
        self.ambiguous_names = {name for name, ids in aliases.items() if len(ids) > 1}
        resolved = {name: (labels[next(iter(ids))], next(iter(ids)))
                    for name, ids in aliases.items() if len(ids) == 1}
        resolved.update({label.casefold(): (label, fighter_id) for fighter_id, label in labels.items()})
        return resolved

    def get_fighter_names(self) -> list[str]:
        return sorted({name for name, _ in self.fighters.values()})

    def recent_fights(self) -> dict:
        rows = []
        if self.research_fights is not None:
            fights = self.research_fights.sort_values(["date", "fight_id"], ascending=False)
            for row in fights.drop_duplicates("fight_id").itertuples(index=False):
                first = self.fighter_labels_by_id.get(row.red_fighter_id)
                second = self.fighter_labels_by_id.get(row.blue_fighter_id)
                if not first or not second or first == second:
                    continue
                fight_id = text_value(row.fight_id)
                result = row.winner if row.winner in ("red", "blue", "draw", "nc") else "unknown"
                rows.append({
                    "fightId": fight_id, "date": date_string(row.date),
                    "firstFighter": first, "secondFighter": second,
                    "weightClass": text_value(row.weightclass), "method": text_value(row.method),
                    "winner": first if result == "red" else second if result == "blue" else None,
                    "result": result,
                    "url": f"http://ufcstats.com/fight-details/{fight_id}"
                    if fight_id and re.fullmatch(r"[a-f0-9]{16}", fight_id) else None,
                })
                if len(rows) == 8:
                    break
        return {"dataThrough": self.data_through, "source": self.provenance, "fights": rows}

    def research_status(self) -> dict:
        return {
            "available": self.research_fights is not None and bool(self.fighters),
            "dataThrough": self.data_through, "fighterCount": len(self.get_fighter_names()),
            "manifestSha256": self.research_manifest_sha256,
            "career": {"available": self.career_source is not None, "asOf": self.career_cutoff,
                       "manifestSha256": self.career_manifest_sha256, "source": self.career_source,
                       "loadError": self.career_load_error},
            **self.provenance, "refreshStatus": self.refresh_status, "loadError": self.refresh_error,
        }

    def _load_round_stats(self, path: Path, fights: pd.DataFrame, rounds: pd.DataFrame | None = None) -> None:
        if rounds is None:
            if not path.is_file():
                return
            try:
                rounds = pd.read_parquet(path)
            except (OSError, ValueError):
                return
        stat_columns = ["sig_str_landed", "sig_str_attempted", "td_landed", "td_attempted"]
        keys = ["fight_id", "fighter_id", "round"]
        if not set(keys + stat_columns + ["time_format"]).issubset(rounds.columns):
            return
        numbers = rounds[stat_columns + ["round"]].apply(pd.to_numeric, errors="coerce")
        valid = numbers.ge(0).all(axis=1) & numbers.lt(float("inf")).all(axis=1)
        valid &= numbers["sig_str_landed"].le(numbers["sig_str_attempted"])
        valid &= numbers["td_landed"].le(numbers["td_attempted"])
        valid &= numbers["round"].ge(1) & numbers["round"].mod(1).eq(0)
        invalid_ids = set(rounds.loc[~valid | rounds.duplicated(keys, keep=False), "fight_id"])
        rounds = rounds.assign(**numbers).loc[~rounds["fight_id"].isin(invalid_ids)]
        grouped = rounds.groupby(["fight_id", "fighter_id"], sort=False)
        totals = grouped[stat_columns].sum().join(grouped["round"].agg(["min", "max", "count"]))
        totals = totals.join(grouped["time_format"].agg(["first", "nunique"]))
        records = totals.to_dict("index")
        for fight in fights.itertuples(index=False):
            if text_value(getattr(fight, "result_uncertainty", None)):
                continue
            if pd.isna(fight.finish_round) or not float(fight.finish_round).is_integer():
                continue
            finish_round = int(fight.finish_round)
            red = records.get((fight.fight_id, fight.red_fighter_id))
            blue = records.get((fight.fight_id, fight.blue_fighter_id))
            if not red or not blue:
                continue
            if any(row["min"] != 1 or row["max"] != finish_round or row["count"] != finish_round or row["nunique"] != 1 for row in (red, blue)):
                continue
            if red["first"] != blue["first"]:
                continue
            seconds = fight_duration(finish_round, fight.finish_time, red["first"])
            if seconds is None:
                continue
            for fighter_id, own, opponent in ((fight.red_fighter_id, red, blue), (fight.blue_fighter_id, blue, red)):
                self.round_stats[(fight.fight_id, fighter_id)] = {
                    "seconds": seconds,
                    "sigLanded": own["sig_str_landed"],
                    "sigAttempted": own["sig_str_attempted"],
                    "sigAbsorbed": opponent["sig_str_landed"],
                    "opponentSigAttempted": opponent["sig_str_attempted"],
                    "tdLanded": own["td_landed"],
                    "tdAttempted": own["td_attempted"],
                    "opponentTdLanded": opponent["td_landed"],
                    "opponentTdAttempted": opponent["td_attempted"],
                }

    def _recent_metrics(self, fighter_id: str, recent: list[dict]) -> dict:
        covered = [(fight, self.round_stats[(fight["fightId"], fighter_id)])
                   for fight in recent if (fight["fightId"], fighter_id) in self.round_stats]
        fields = ["sigLanded", "sigAttempted", "sigAbsorbed", "opponentSigAttempted", "tdLanded", "tdAttempted", "opponentTdLanded", "opponentTdAttempted"]
        totals = {field: sum(stats[field] for _, stats in covered) for field in fields}
        seconds = sum(stats["seconds"] for _, stats in covered)
        dates = [fight["date"] for fight, _ in covered if fight["date"]]
        values = {
            "sigLandedPerMinute": totals["sigLanded"] * 60 / seconds if seconds else None,
            "sigAbsorbedPerMinute": totals["sigAbsorbed"] * 60 / seconds if seconds else None,
            "tdLandedPer15": totals["tdLanded"] * 900 / seconds if seconds else None,
            "sigAccuracy": totals["sigLanded"] / totals["sigAttempted"] if totals["sigAttempted"] else None,
            "sigDefense": 1 - totals["sigAbsorbed"] / totals["opponentSigAttempted"] if totals["opponentSigAttempted"] else None,
            "tdAccuracy": totals["tdLanded"] / totals["tdAttempted"] if totals["tdAttempted"] else None,
            "tdDefense": 1 - totals["opponentTdLanded"] / totals["opponentTdAttempted"] if totals["opponentTdAttempted"] else None,
        }
        return {
            "available": bool(covered), "windowBouts": len(recent),
            "coveredBouts": len(covered), "missingBouts": len(recent) - len(covered),
            "from": min(dates) if dates else None, "through": max(dates) if dates else None,
            "fightIds": [fight["fightId"] for fight, _ in covered],
            "totalSeconds": seconds, "totals": totals, "values": values,
        }

    def matchup_evidence(self, first: str, second: str) -> dict:
        if not all(isinstance(name, str) and name.strip() for name in (first, second)):
            raise ValueError("Two fighter names are required")
        if first.strip().casefold() == second.strip().casefold():
            raise ValueError("Choose two different fighters")
        if any(name.strip().casefold() in self.ambiguous_names for name in (first, second)):
            raise ValueError("That name matches multiple fighters. Select the fighter's full label from the list.")
        selected = [self.fighters.get(name.strip().casefold()) for name in (first, second)]
        if any(fighter is None for fighter in selected):
            raise ValueError("Fighter not available in this dataset")
        if selected[0][1] == selected[1][1]:
            raise ValueError("Choose two different fighters")
        if self.research_fights is None:
            raise ValueError("Recorded fight results are unavailable")
        fighters = []
        for name, fighter_id in selected:
            history = self.histories.get(fighter_id, [])
            recent = history[:5]
            snapshot = self.registry_by_id.get(fighter_id, {}) if self.explicit_bundle else self.snapshot_by_id.get(fighter_id, {})
            physical = {}
            for key, column in (("heightCm", "height_cm"), ("reachCm", "reach_cm")):
                value = pd.to_numeric(snapshot.get(column), errors="coerce")
                physical[key] = float(value) if pd.notna(value) and math.isfinite(value) and value > 0 else None
            physical.update({
                "stance": text_value(snapshot.get("stance")), "observedAt": None,
                "sourceUrl": f"http://ufcstats.com/fighter-details/{fighter_id}"
                if isinstance(fighter_id, str) and re.fullmatch(r"[a-f0-9]{16}", fighter_id) else None,
            })
            fighters.append({
                "name": name, "fighterId": fighter_id,
                "totalRecordedBouts": len(history), "record": recorded_record(history),
                "latestBout": history[0] if history else None,
                "recentFive": {"count": len(recent), "record": recorded_record(recent),
                               "winMethods": win_methods(recent), "fights": recent},
                "recentMetrics": self._recent_metrics(fighter_id, recent),
                "physical": physical, "featureAsOf": date_string(snapshot.get("date")),
                "careerHistory": self.career_history(fighter_id),
            })
        first_history = self.histories.get(selected[0][1], [])
        second_history = self.histories.get(selected[1][1], [])
        meetings = [fight for fight in first_history if fight["opponentId"] == selected[1][1]]
        common_ids = {fight["opponentId"] for fight in first_history if fight["opponentId"]} & {fight["opponentId"] for fight in second_history if fight["opponentId"]}
        common = []
        for opponent_id in common_ids:
            first_meetings = [fight for fight in first_history if fight["opponentId"] == opponent_id]
            second_meetings = [fight for fight in second_history if fight["opponentId"] == opponent_id]
            common.append({
                "name": first_meetings[0]["opponent"] or second_meetings[0]["opponent"],
                "fighterId": opponent_id,
                "firstFighter": {"total": len(first_meetings), "fights": first_meetings[:2]},
                "secondFighter": {"total": len(second_meetings), "fights": second_meetings[:2]},
            })
        common.sort(key=lambda opponent: (max(opponent["firstFighter"]["fights"][0]["date"] or "", opponent["secondFighter"]["fights"][0]["date"] or ""), opponent["fighterId"]), reverse=True)
        gaps = [
            {"code": "historical_data", "text": f"Recorded results end on {self.data_through or 'an unknown date'}. Later fights, injuries and camp updates are not included."},
            {"code": "unscheduled_matchup", "text": "No scheduled fight date, weight class or number of rounds has been supplied."},
        ]
        if self.refresh_error:
            gaps.append({"code": "research_refresh_unavailable", "text": self.refresh_error})
        for fighter in fighters:
            if any(fight["opponentId"] is None for fight in self.histories.get(fighter["fighterId"], [])):
                gaps.append({"code": "unresolved_opponent_identity", "text": f"Some recorded opponents for {fighter['name']} do not have verified source identities."})
            if any(fight["resultUncertainty"] for fight in fighter["recentFive"]["fights"]):
                gaps.append({"code": "conflicting_bout_details", "text": f"A recent bout for {fighter['name']} has conflicting source details and is excluded from the measured statistics."})
            if fighter["recentFive"]["count"] < 5:
                gaps.append({"code": "small_sample", "text": f"{fighter['name']} has only {fighter['recentFive']['count']} recorded UFC bouts in this dataset."})
            metrics = fighter["recentMetrics"]
            if metrics["missingBouts"]:
                gaps.append({"code": "incomplete_recent_stats", "text": f"Complete round statistics are unavailable for {metrics['missingBouts']} of {metrics['windowBouts']} recent bouts for {fighter['name']}."})
        return {
            "dataThrough": self.data_through, "generatedAt": iso_utc(datetime.now(timezone.utc)),
            "source": self.research_status(), "fighters": fighters,
            "headToHead": {"total": len(meetings), "fights": meetings[:5]},
            "commonOpponents": {"total": len(common), "opponents": common[:6]},
            "gaps": gaps,
        }

    def status(self) -> dict:
        configured = bool(odds_api_key())
        return {
            "model": {
                "ready": self.predictor.is_ready(),
                "validatedForBetting": False,
                "dataThrough": self.model_data_through,
                "snapshotDate": None,
                "profileDataThrough": self.profile_data_through,
                "modelVersion": self.model_version,
            },
            "research": self.research_status(),
            "odds": {
                "configured": configured,
                "provider": ODDS_PROVIDER,
                "mode": "connected" if configured else "manual",
            },
        }

    def history(self, name: str) -> dict | None:
        fighter = self.fighters.get(name.strip().casefold())
        if fighter is None:
            return None
        canonical_name, fighter_id = fighter
        fights = self.histories.get(fighter_id, [])
        return {
            "fighter": canonical_name,
            "fighterId": fighter_id,
            "totalFights": len(fights),
            "dataThrough": self.data_through,
            "fights": fights[:10],
            "careerHistory": self.career_history(fighter_id),
        }

    def prediction_context(self, result: dict) -> dict:
        context = {
            "modelVersion": self.model_version,
            "dataThrough": self.model_data_through,
            "profileDataThrough": self.profile_data_through,
            "validatedForBetting": False,
        }
        for number in (1, 2):
            stats = result[f"f{number}Stats"]
            fighter = self.fighters.get(result[f"f{number}Name"].strip().casefold())
            context[f"fighter{number}"] = {
                "featureAsOf": stats.get("FeatureAsOf"),
                "profileAsOf": stats.get("ProfileAsOf"),
                "recordedFightCount": int(self.model_fight_counts.get(fighter[1], 0)) if fighter else None,
            }
        return context


def normalize_odds(payload: object, now: datetime) -> list[dict]:
    if not isinstance(payload, list):
        raise ValueError("Unexpected provider response")
    events = []
    seen_events = set()
    for event in payload:
        if not isinstance(event, dict):
            continue
        event_id = text_value(event.get("id"))
        home = text_value(event.get("home_team"))
        away = text_value(event.get("away_team"))
        start = utc_datetime(event.get("commence_time"))
        books = event.get("bookmakers")
        if not event_id or event_id in seen_events or not home or not away or home.casefold() == away.casefold():
            continue
        if start is None or start <= now or not isinstance(books, list):
            continue
        bookmakers = []
        seen_books = set()
        for book in books:
            if not isinstance(book, dict):
                continue
            key = text_value(book.get("key"))
            title = text_value(book.get("title"))
            markets = book.get("markets")
            if not key or key in seen_books or not title or not isinstance(markets, list):
                continue
            h2h = [market for market in markets if isinstance(market, dict) and market.get("key") == "h2h"]
            if len(h2h) != 1:
                continue
            market = h2h[0]
            outcomes = market.get("outcomes")
            updated = utc_datetime(market.get("last_update")) or utc_datetime(book.get("last_update"))
            if not isinstance(outcomes, list) or len(outcomes) != 2 or updated is None or updated > now:
                continue
            paired = {}
            for outcome in outcomes:
                if not isinstance(outcome, dict):
                    break
                name = text_value(outcome.get("name"))
                price = outcome.get("price")
                if name not in (home, away) or name in paired:
                    break
                if isinstance(price, bool) or not isinstance(price, (int, float)):
                    break
                try:
                    decimal_price = float(price)
                except OverflowError:
                    break
                if not math.isfinite(decimal_price) or decimal_price <= 1 or decimal_price > 10000:
                    break
                paired[name] = decimal_price
            if len(paired) != 2:
                continue
            seen_books.add(key)
            bookmakers.append({
                "key": key,
                "title": title,
                "lastUpdate": iso_utc(updated),
                "outcomes": [{"name": name, "price": paired[name]} for name in (home, away)],
            })
        if not bookmakers:
            continue
        seen_events.add(event_id)
        events.append({
            "id": event_id,
            "startTime": iso_utc(start),
            "homeTeam": home,
            "awayTeam": away,
            "bookmakers": bookmakers,
        })
    return sorted(events, key=lambda event: event["startTime"])


class OddsFeed:
    def __init__(self):
        # The five-minute cache lives only in this warm server process.
        self.cached = None
        self.expires_at = 0
        self.key_hash = None
        self.lock = Lock()

    def fetch(self) -> tuple[dict, int]:
        api_key = odds_api_key()
        if not api_key:
            return {"configured": False, "events": [], "source": "manual"}, 200
        key_hash = hashlib.sha256(api_key.encode()).digest()
        with self.lock:
            now = datetime.now(timezone.utc)
            if self.cached is not None and self.key_hash == key_hash and time.monotonic() < self.expires_at:
                return {**self.cached, "events": [event for event in self.cached["events"] if utc_datetime(event["startTime"]) > now]}, 200
            query = urlencode({"regions": "us", "markets": "h2h", "oddsFormat": "decimal", "apiKey": api_key})
            url = f"https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds/?{query}"
            try:
                with urlopen(url, timeout=6) as response:
                    raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise ValueError("Provider response too large")
                events = normalize_odds(json.loads(raw), datetime.now(timezone.utc))
            except (URLError, OSError, HTTPException, ValueError):
                return {"configured": True, "provider": ODDS_PROVIDER, "events": [], "error": ODDS_UNAVAILABLE}, 503
            self.cached = {
                "configured": True,
                "provider": ODDS_PROVIDER,
                "fetchedAt": iso_utc(datetime.now(timezone.utc)),
                "events": events,
            }
            self.key_hash = key_hash
            self.expires_at = time.monotonic() + 300
            return self.cached, 200
