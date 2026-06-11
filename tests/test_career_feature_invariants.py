import hashlib
import json
import re
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import career_features as career
from scripts import evaluate_challenger as challenger
from scripts import evaluate_research as base


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"
SOURCE = DATA / "identity-repaired-source-v1"
SUPPLEMENT = DATA / "career-supplement-v1"
EVALUATION = DATA / "career-supplement-evaluation-v1"
SEED = "career-stress-v1-100"
FEATURE_SHA = "d601a14dc17572945ad4dafa7e05828442dda50b0feb0d2d7293135246aed562"
OLD_HASHES = {
    "manifest.json": "ddfa39cb228663c58654fb30bc20bfd77bc0665414652b9c8c81f232d7304782",
    "fights.parquet": "4f14f1a7114fb4e50d6626f542ced81252482d9aa5e33633de4206387e576b5c",
    "rounds.parquet": "7b1f2390317e867ed1589bb3acf16f2b3ea30a2ef640cf1b924453eb3df28a1a",
    "fighters.parquet": "bddd469986ec3e126f77262f7083b23595573f37d4f48459170487321c8c25ed",
}


def selected_contexts() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    original = pd.read_parquet(DATA / "identity-repair-evaluation-v1/development_and_calibration_features.parquet")
    replay = pd.read_parquet(EVALUATION / "replay_features.parquet")
    pool = pd.concat([original, replay[original.columns]], ignore_index=True)
    groups = [pool[pool.date < "2023-01-01"], pool[pool.date.between("2023-01-01", "2024-12-31")],
              pool[pool.date >= "2025-01-01"]]
    chosen = []
    for group, count in zip(groups, (40, 30, 30)):
        ranks = group.fight_id.map(lambda value: hashlib.sha256(f"{SEED}|{value}".encode()).hexdigest())
        chosen.append(group.loc[ranks.sort_values().index[:count]])
    sample = pd.concat(chosen).sort_values(["date", "fight_id"]).reset_index(drop=True)
    columns = json.loads((EVALUATION / "feature_columns.json").read_text())["original48"]
    frozen = pd.concat([pd.read_parquet(EVALUATION / "development_and_calibration_features.parquet"), replay])
    return sample, frozen.set_index("fight_id").loc[sample.fight_id].reset_index(), columns


def reference_state(fights: pd.DataFrame, extra: pd.DataFrame, fighter_id: str, date: pd.Timestamp) -> dict:
    rows = []
    outcomes = {"red": ("W", "L"), "blue": ("L", "W"), "draw": ("D", "D"), "nc": ("NC", "NC")}
    for side, index in (("red", 0), ("blue", 1)):
        for row in fights[(fights[f"{side}_fighter_id"] == fighter_id) & (fights.date < date)].itertuples():
            rows.append((row.date, outcomes[row.winner][index], row.method, False))
    for row in extra[(extra.fighter_id == fighter_id) & (extra.date < date)].itertuples():
        rows.append((row.date, row.result, row.method, True))

    def score(items):
        return (sum(row[1] == "W" for row in items) + .5 * sum(row[1] == "D" for row in items) + 1) / (
            sum(row[1] in ("W", "L", "D") for row in items) + 2)

    def finish_rate(outcome):
        values = []
        for _, result, method, _ in rows:
            if result != outcome or not isinstance(method, str):
                continue
            method = method.lower()
            if re.search(r"\b(?:ko|tko)\b|submission", method):
                values.append(1)
            elif re.search(r"decision|\bdq\b|disqualification|forfeit", method):
                values.append(0)
        return sum(values) / len(values) if values else np.nan

    recent = [row for row in rows if row[0] >= date - pd.Timedelta(days=365)]
    return {"bouts": len(rows), "wins": sum(row[1] == "W" for row in rows),
            "losses": sum(row[1] == "L" for row in rows), "smoothed_score": score(rows),
            "finish_win_rate": finish_rate("W"), "finish_loss_rate": finish_rate("L"),
            "recent365_score": score(recent), "recent365_bouts": len(recent),
            "layoff_days": (date - max(row[0] for row in rows)).days if rows else np.nan,
            "supplement_bouts": sum(row[3] for row in rows)}


@unittest.skipUnless((EVALUATION / "replay_features.parquet").exists(), "Frozen research artifacts are local only")
class CareerFeatureInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.requests, cls.frozen, cls.columns48 = selected_contexts()
        cls.fights = pd.read_parquet(SOURCE / "fights.parquet")
        cls.extra, _ = career.load_supplement(SUPPLEMENT)
        cls.crosswalk = pd.read_csv(SUPPLEMENT / "crosswalk.csv").set_index("fighter_id").canonical_fighter_id.to_dict()
        cls.history = career.prepare_history(cls.fights, cls.extra)
        cls.expected = career.add_career_features(cls.requests, cls.fights, cls.extra, cls.history)
        cls.before = cls.fingerprints()

    @staticmethod
    def fingerprints() -> dict:
        paths = [DATA / "inference" / name for name in OLD_HASHES]
        paths.append(ROOT / "scripts/career_features.py")
        for directory in (SOURCE, SUPPLEMENT):
            paths.extend([directory / "manifest.json", *[directory / name for name in json.loads((directory / "manifest.json").read_text())["files"]]])
        for directory in (DATA / "identity-repair-evaluation-v1", EVALUATION):
            paths.extend([directory / "artifact_hashes.json", *[directory / name for name in json.loads((directory / "artifact_hashes.json").read_text())]])
        return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    @classmethod
    def tearDownClass(cls):
        if cls.before != cls.fingerprints():
            raise AssertionError("A frozen input changed during the stress suite")

    def assert_features(self, actual: pd.DataFrame, expected: pd.DataFrame):
        pd.testing.assert_frame_equal(actual[expected.columns], expected, check_exact=True)

    def test_01_frozen_inputs_and_original_bundle_integrity(self):
        self.assertEqual(len(self.requests), 100)
        self.assertEqual(self.requests.fight_id.nunique(), 100)
        self.assertEqual(len(self.columns48), 48)
        self.assertEqual(len(career.ADDED_FEATURES), 12)
        self.assertEqual(self.before["scripts/career_features.py"], FEATURE_SHA)
        for name, expected in OLD_HASHES.items():
            self.assertEqual(self.before[f"data/evaluation/inference/{name}"], expected)
        for directory in (SOURCE, SUPPLEMENT):
            for name, expected in json.loads((directory / "manifest.json").read_text())["files"].items():
                self.assertEqual(self.before[str((directory / name).relative_to(ROOT))], expected)
        for directory in (DATA / "identity-repair-evaluation-v1", EVALUATION):
            for name, expected in json.loads((directory / "artifact_hashes.json").read_text()).items():
                self.assertEqual(self.before[str((directory / name).relative_to(ROOT))], expected)

    def test_02_reconstruction_and_frozen_cache_parity(self):
        rounds = pd.read_parquet(SOURCE / "rounds.parquet")
        fighters = pd.read_parquet(SOURCE / "fighters.parquet")
        totals, _ = base.prepare_round_stats(rounds)
        reconstructed = challenger.add_context(base.reconstruct(self.fights, totals), self.fights, fighters)
        actual = reconstructed.set_index("fight_id").loc[self.requests.fight_id].reset_index()
        pd.testing.assert_frame_equal(actual[self.columns48], self.requests[self.columns48], check_exact=True)
        pd.testing.assert_frame_equal(self.expected[self.requests.columns], self.requests, check_exact=True)
        # The full cache has missing layoff values; this sample's layoff values are all integers.
        pd.testing.assert_frame_equal(self.expected[career.ADDED_FEATURES], self.frozen[career.ADDED_FEATURES],
                                      check_exact=True, check_dtype=False)

    def test_03_strict_prior_filter_and_adversarial_future_rows(self):
        for index, row in self.requests.iterrows():
            with self.subTest(fight_id=row.fight_id):
                request = self.requests.loc[[index]]
                expected = self.expected.loc[[index]]
                sides = {row.red_fighter_id, row.blue_fighter_id}
                fights = self.fights[self.fights.red_fighter_id.isin(sides) | self.fights.blue_fighter_id.isin(sides)].copy()
                extra = self.extra[self.extra.fighter_id.isin(sides)].copy()
                self.assert_features(career.add_career_features(request, fights, extra), expected)
                prior_fights, prior_extra = fights[fights.date < row.date], extra[extra.date < row.date]
                self.assert_features(career.add_career_features(request, prior_fights, prior_extra), expected)
                future = extra.date >= row.date
                extra.loc[future, "result"] = extra.loc[future, "result"].map({"W": "L", "L": "W", "D": "D", "NC": "NC"})
                extra.loc[future, "opponent_result"] = extra.loc[future, "result"].map({"W": "L", "L": "W", "D": "D", "NC": "NC"})
                extra.loc[future, "method"] = "Submission"
                fights.loc[fights.date >= row.date, "winner"] = fights.loc[fights.date >= row.date, "winner"].map(
                    {"red": "blue", "blue": "red", "draw": "draw", "nc": "nc"})
                fights.loc[fights.date >= row.date, "method"] = "KO/TKO"
                self.assert_features(career.add_career_features(request, fights, extra), expected)
                additions = []
                for corner, fighter in enumerate(sorted(sides)):
                    for offset in (0, 137):
                        identity = 10_000_000 + index * 1000 + corner * 200 + offset
                        additions.append({"fighter_id": fighter, "opponent_id": None,
                            "canonical_fighter_id": self.crosswalk[fighter], "canonical_opponent_id": identity + 1_000_000,
                            "canonical_fight_id": identity, "date": row.date + pd.Timedelta(days=offset),
                            "result": "W" if corner else "L", "opponent_result": "L" if corner else "W",
                            "method": "KO/TKO", "source_aliases_json": json.dumps([{"source": "stress-test", "sourceID": str(identity)}])})
                injected = pd.concat([prior_extra, pd.DataFrame(additions)], ignore_index=True)
                self.assert_features(career.add_career_features(request, prior_fights, injected), expected)

    def test_04_source_order_and_source_corner_orientation(self):
        shuffled = self.fights.sample(frac=1, random_state=7283).copy()
        mask = np.random.default_rng(7283).random(len(shuffled)) < .5
        shuffled.loc[mask, ["red_fighter_id", "blue_fighter_id"]] = shuffled.loc[mask, ["blue_fighter_id", "red_fighter_id"]].to_numpy()
        shuffled.loc[mask, "winner"] = shuffled.loc[mask, "winner"].map({"red": "blue", "blue": "red", "draw": "draw", "nc": "nc"})
        extra = self.extra.sample(frac=1, random_state=9921)
        self.assert_features(career.add_career_features(self.requests, shuffled, extra), self.expected)

    def test_05_batch_singleton_and_duplicate_index_invariance(self):
        shuffled = self.requests.sample(frac=1, random_state=831)
        expected = self.expected.loc[shuffled.index].copy()
        shuffled.index = np.arange(len(shuffled)) // 2
        expected.index = shuffled.index
        self.assert_features(career.add_career_features(shuffled, self.fights, self.extra, self.history), expected)
        for size in (1, 7, 31):
            pieces = [career.add_career_features(shuffled.iloc[start:start + size], self.fights, self.extra, self.history)
                      for start in range(0, len(shuffled), size)]
            self.assert_features(pd.concat(pieces), expected)

    def test_06_request_corner_symmetry(self):
        mirrored = base.swapped(self.requests)
        mirrored[["red_fighter_id", "blue_fighter_id"]] = self.requests[["blue_fighter_id", "red_fighter_id"]].to_numpy()
        actual = career.add_career_features(mirrored, self.fights, self.extra, self.history)
        pd.testing.assert_frame_equal(actual[self.columns48], mirrored[self.columns48], check_exact=True)
        pd.testing.assert_frame_equal(actual[career.ADDED_FEATURES], base.swapped(self.expected)[career.ADDED_FEATURES], check_exact=True)

    def test_07_independent_boolean_cutoff_reference(self):
        expected = []
        for row in self.requests.itertuples():
            first = reference_state(self.fights, self.extra, row.red_fighter_id, row.date)
            second = reference_state(self.fights, self.extra, row.blue_fighter_id, row.date)
            expected.append([first[name] - second[name] for name in career.STATE_NAMES] + [
                min(first["bouts"], second["bouts"]), min(first["supplement_bouts"], second["supplement_bouts"])])
        np.testing.assert_allclose(self.expected[career.ADDED_FEATURES].to_numpy(), np.asarray(expected), rtol=0, atol=0, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
