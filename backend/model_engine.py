"""V3RSUS prediction engine — v2.

Loads:
  - data/model_v2.pkl              ensemble + scaler + imputer + feature names
  - data/fighter_snapshots.parquet  per-fighter ML features
  - data/fights.parquet             historical matchup lookup
  - data/ufc-master.csv                                career-aggregate stats for display

For arbitrary matchups: pull snapshots, compute diff vector, scale, predict.
The web app requires a compatible v2 artifact and its snapshot table.
"""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


SNAPSHOT_COLUMNS = {
    "rating": "post_rating", "rd": "post_rd",
    "height": "height_cm", "reach": "reach_cm", "weight": "weight_lbs",
    "age": "current_age", "layoff": "current_layoff_days",
    "career_fights": "career_fights_before", "career_wins": "career_wins_before",
    "career_losses": "career_losses_before", "win_streak": "win_streak_before",
    "lose_streak": "lose_streak_before",
}

FEATURE_DISPLAY = {
    "rating": "Glicko Rating",
    "rd": "Rating Uncertainty",
    "height": "Height",
    "reach": "Reach",
    "weight": "Weight",
    "age": "Age",
    "layoff": "Layoff (days)",
    "career_fights": "Career Fights",
    "career_wins": "Career Wins",
    "career_losses": "Career Losses",
    "win_streak": "Win Streak",
    "lose_streak": "Lose Streak",
    "sig_str_landed_last3": "Sig. Strikes (last 3)",
    "sig_str_attempted_last3": "Sig. Strikes Attempted (last 3)",
    "td_landed_last3": "Takedowns (last 3)",
    "td_attempted_last3": "Takedowns Attempted (last 3)",
    "head_landed_last3": "Head Strikes (last 3)",
    "body_landed_last3": "Body Strikes (last 3)",
    "leg_landed_last3": "Leg Strikes (last 3)",
    "distance_landed_last3": "Distance Strikes (last 3)",
    "clinch_landed_last3": "Clinch Strikes (last 3)",
    "ground_landed_last3": "Ground Strikes (last 3)",
    "sub_att_last3": "Submission Attempts (last 3)",
    "kd_last3": "Knockdowns (last 3)",
    "ctrl_seconds_last3": "Control Time (last 3)",
    "sig_str_landed_last5": "Sig. Strikes (last 5)",
    "sig_str_attempted_last5": "Sig. Strikes Attempted (last 5)",
    "td_landed_last5": "Takedowns (last 5)",
    "td_attempted_last5": "Takedowns Attempted (last 5)",
    "head_landed_last5": "Head Strikes (last 5)",
    "body_landed_last5": "Body Strikes (last 5)",
    "leg_landed_last5": "Leg Strikes (last 5)",
    "distance_landed_last5": "Distance Strikes (last 5)",
    "clinch_landed_last5": "Clinch Strikes (last 5)",
    "ground_landed_last5": "Ground Strikes (last 5)",
    "sub_att_last5": "Submission Attempts (last 5)",
    "kd_last5": "Knockdowns (last 5)",
    "ctrl_seconds_last5": "Control Time (last 5)",
}

FEATURE_CATEGORY = {
    "rating": "experience", "rd": "experience",
    "height": "physical", "reach": "physical", "weight": "physical", "age": "physical",
    "layoff": "experience",
    "career_fights": "experience", "career_wins": "experience", "career_losses": "experience",
    "win_streak": "experience", "lose_streak": "experience",
}
for base in ("sig_str_landed", "sig_str_attempted", "head_landed", "body_landed", "leg_landed",
             "distance_landed", "clinch_landed", "kd"):
    for n in (3, 5):
        FEATURE_CATEGORY[f"{base}_last{n}"] = "striking"
for base in ("td_landed", "td_attempted", "ground_landed", "sub_att", "ctrl_seconds"):
    for n in (3, 5):
        FEATURE_CATEGORY[f"{base}_last{n}"] = "grappling"

CATEGORY_LABELS = {
    "striking": "Striking",
    "grappling": "Grappling",
    "physical": "Physical",
    "experience": "Experience",
}


class FightPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.imputer = None
        self.features: list[str] = []
        self.model_metrics = {}
        self.lr_coefficients = None

        # Data sources
        self.snapshots: pd.DataFrame | None = None
        self.fights_df: pd.DataFrame | None = None   # ufc-master.csv format for display + fighters list
        self.norm_fights: pd.DataFrame | None = None  # normalized fights table for historical lookup

        # Stats
        self.fighter_count = 0
        self.fight_count = 0
        self.feature_count = 0
        self.trained_at = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_artifact(self, artifact_path: str, data_path: str) -> bool:
        artifact = joblib.load(artifact_path)
        if not artifact["features"] or not all(feature.startswith(("diff_", "wc_")) for feature in artifact["features"]):
            print("Model artifact uses unsupported legacy features; fetch the v2 model and its snapshots.")
            return False
        self.models = artifact["models"]
        self.scaler = artifact["scaler"]
        self.imputer = artifact.get("imputer")
        self.features = artifact["features"]
        self.model_metrics = artifact["model_metrics"]
        self.lr_coefficients = artifact["lr_coefficients"]
        self.feature_count = len(self.features)
        self.trained_at = artifact.get("trained_at")

        self.fights_df = pd.read_csv(data_path)
        self.fight_count = len(self.fights_df)

        # data_dir = the same folder we just loaded the artifact from
        data_dir = Path(artifact_path).resolve().parent
        repo_root = Path(__file__).resolve().parents[1]
        self._load_v2_sources(repo_root, data_dir)
        self.fighter_count = len(self.get_fighter_names())
        return self.is_ready()

    def _load_v2_sources(self, repo_root: Path, data_dir: Path | None = None) -> None:
        d = data_dir or (repo_root / "data")
        snap_path = d / "fighter_snapshots.parquet"
        fights_path = d / "fights.parquet"
        events_path = d / "events.parquet"

        if snap_path.exists():
            self.snapshots = pd.read_parquet(snap_path)
            self.fighter_count = len(self.snapshots)
            print(f"  loaded {len(self.snapshots)} fighter snapshots")

        if fights_path.exists() and events_path.exists():
            f = pd.read_parquet(fights_path)
            ev = pd.read_parquet(events_path)
            self.norm_fights = f.merge(ev[["event_id", "date"]], on="event_id", how="left")
            print(f"  loaded {len(self.norm_fights)} normalized fights")

    # ------------------------------------------------------------------
    # Backwards-compat fall-back train (used if no v2 artifact)
    # ------------------------------------------------------------------

    def train(self, data_path: str = "data/ufc-master.csv") -> bool:
        """Legacy training on ufc-master.csv. Kept as fallback only."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score

        if not os.path.exists(data_path):
            print(f"data not found at {data_path}")
            return False

        self.fights_df = pd.read_csv(data_path)
        fights = self.fights_df.copy()
        self.fight_count = len(fights)
        fights["target"] = (fights["Winner"] == "Red").astype(int)

        diff_cols = [c for c in fights.columns if c.endswith("Dif") and pd.api.types.is_numeric_dtype(fights[c])]
        X = fights[diff_cols].fillna(0)
        y = fights["target"]
        self.features = diff_cols
        self.feature_count = len(diff_cols)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        configs = [
            ("logistic_regression", LogisticRegression(max_iter=1000, random_state=42)),
            ("random_forest", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)),
            ("gradient_boosting", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)),
        ]
        for name, m in configs:
            m.fit(X_scaled, y)
            self.models[name] = m
            scores = cross_val_score(m, X_scaled, y, cv=5, scoring="accuracy")
            self.model_metrics[name] = {"accuracy": round(float(scores.mean()), 4), "std": round(float(scores.std()), 4)}

        self.lr_coefficients = self.models["logistic_regression"].coef_[0]
        return True

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        return bool(self.models) and self.scaler is not None and self.snapshots is not None and not self.snapshots.empty

    def get_fighter_names(self) -> list[str]:
        if not self.is_ready():
            return []
        names = self.snapshots["name"]
        return sorted(names.dropna().str.strip().unique().tolist())

    def get_fighter_stats(self, name: str) -> dict | None:
        """Returns display stats (ufc-master.csv format). Used by build_profile in app.py."""
        if self.fights_df is None:
            return None
        n = name.lower().strip()
        red = self.fights_df[self.fights_df["RedFighter"].str.lower().str.strip() == n]
        blue = self.fights_df[self.fights_df["BlueFighter"].str.lower().str.strip() == n]
        if red.empty and blue.empty:
            return None

        latest_red = red.sort_values("Date", ascending=False).iloc[0] if not red.empty else None
        latest_blue = blue.sort_values("Date", ascending=False).iloc[0] if not blue.empty else None

        if latest_red is None:
            row, prefix = latest_blue, "Blue"
        elif latest_blue is None:
            row, prefix = latest_red, "Red"
        else:
            if latest_red["Date"] >= latest_blue["Date"]:
                row, prefix = latest_red, "Red"
            else:
                row, prefix = latest_blue, "Blue"

        stats = {}
        for col in self.fights_df.columns:
            if col.startswith(prefix):
                stats[col.replace(prefix, "", 1)] = row[col]
        stats["ActualName"] = row[prefix + "Fighter"]
        stats["TotalFights"] = len(red) + len(blue)
        stats["ProfileAsOf"] = self._date(row["Date"])
        return stats

    def _lookup_snapshot(self, name: str) -> pd.Series | None:
        if self.snapshots is None:
            return None
        mask = self.snapshots["name"].str.lower().str.strip() == name.lower().strip()
        if not mask.any():
            return None
        return self.snapshots[mask].iloc[0]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @staticmethod
    def _safe(v):
        if v is None:
            return np.nan
        if isinstance(v, (int, np.integer)):
            return float(v)
        if isinstance(v, (float, np.floating)):
            return np.nan if (np.isnan(v) or np.isinf(v)) else float(v)
        return np.nan

    def _build_diff_vector(self, s1: pd.Series, s2: pd.Series) -> np.ndarray:
        """Map each training feature to a diff(s1, s2) value."""
        out = np.zeros(len(self.features), dtype=float)
        for i, fname in enumerate(self.features):
            if fname.startswith("wc_"):
                continue  # weight class dummies not used at prediction (unknown until fight)
            base = fname.replace("diff_", "", 1)
            col = SNAPSHOT_COLUMNS.get(base, base)
            v1 = self._safe(s1.get(col))
            v2 = self._safe(s2.get(col))
            out[i] = v1 - v2
        return out

    def predict_matchup(self, f1_name: str, f2_name: str) -> dict:
        s1 = self._lookup_snapshot(f1_name)
        s2 = self._lookup_snapshot(f2_name)
        if s1 is None:
            raise ValueError(f"Fighter not available for prediction: {f1_name}")
        if s2 is None:
            raise ValueError(f"Fighter not available for prediction: {f2_name}")

        diff_vec = np.vstack((self._build_diff_vector(s1, s2), self._build_diff_vector(s2, s1)))
        if self.imputer is not None:
            diff_vec = self.imputer.transform(pd.DataFrame(diff_vec, columns=self.features))
        else:
            diff_vec = np.nan_to_num(diff_vec)
        scaled = self.scaler.transform(diff_vec)
        symmetric_scaled = (scaled[0] - scaled[1]) / 2

        model_breakdown = {}
        ensemble_prob = np.zeros(2)
        for name, model in self.models.items():
            corner_probs = model.predict_proba(scaled)
            prob = (corner_probs[0] + corner_probs[1, ::-1]) / 2
            model_breakdown[name] = {
                "f1Prob": round(float(prob[1]), 4),
                "f2Prob": round(float(prob[0]), 4),
                "accuracy": self.model_metrics.get(name, {}).get("accuracy"),
                "ts_accuracy": self.model_metrics.get(name, {}).get("ts_accuracy"),
            }
            ensemble_prob += prob
        ensemble_prob /= len(self.models)
        f1_prob = float(ensemble_prob[1])
        f2_prob = float(ensemble_prob[0])

        f1_actual = str(s1["name"]).strip()
        f2_actual = str(s2["name"]).strip()
        f1_display = self.get_fighter_stats(f1_actual) or {"ActualName": f1_actual}
        f2_display = self.get_fighter_stats(f2_actual) or {"ActualName": f2_actual}
        f1_display["FeatureAsOf"] = self._date(s1.get("date"))
        f2_display["FeatureAsOf"] = self._date(s2.get("date"))

        winner = None if round(f1_prob, 4) == 0.5 else (f1_actual if f1_prob > 0.5 else f2_actual)
        confidence = max(f1_prob, f2_prob)
        model_breakdown["ensemble"] = {"f1Prob": round(f1_prob, 4), "f2Prob": round(f2_prob, 4)}

        return {
            "winner": winner,
            "confidence": round(confidence, 4),
            "f1Prob": round(f1_prob, 4),
            "f2Prob": round(f2_prob, 4),
            "f1Name": f1_actual,
            "f2Name": f2_actual,
            "f1Stats": f1_display,
            "f2Stats": f2_display,
            "keyFactors": self._key_factors(symmetric_scaled, f1_actual, f2_actual, s1, s2),
            "categoryAnalysis": self._category_scores(symmetric_scaled),
            "modelBreakdown": model_breakdown,
            "historicalMatchups": self._historical(f1_actual, f2_actual),
        }

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def _display(self, fname: str) -> str:
        base = fname.replace("diff_", "", 1)
        return FEATURE_DISPLAY.get(base, base.replace("_", " ").title())

    def _category(self, fname: str) -> str:
        base = fname.replace("diff_", "", 1)
        return FEATURE_CATEGORY.get(base, "other")

    def _key_factors(self, scaled: np.ndarray, f1_name: str, f2_name: str, s1: pd.Series, s2: pd.Series):
        if self.lr_coefficients is None:
            return []
        contribs = self.lr_coefficients * scaled
        raw = []
        for i, fname in enumerate(self.features):
            if fname.startswith("wc_"):
                continue
            c = contribs[i]
            if abs(c) < 0.005:
                continue
            base = fname.replace("diff_", "", 1)
            col = SNAPSHOT_COLUMNS.get(base, base)
            v1 = self._safe(s1.get(col))
            v2 = self._safe(s2.get(col))
            raw.append({
                "factor": self._display(fname),
                "category": self._category(fname),
                "advantage": f1_name if c > 0 else f2_name,
                "impact": round(abs(float(c)), 4),
                "f1Value": round(v1, 1) if np.isfinite(v1) else None,
                "f2Value": round(v2, 1) if np.isfinite(v2) else None,
                "imputed": not (np.isfinite(v1) and np.isfinite(v2)),
            })

        seen = {}
        for r in raw:
            if r["factor"] not in seen or r["impact"] > seen[r["factor"]]["impact"]:
                seen[r["factor"]] = r
        return sorted(seen.values(), key=lambda x: x["impact"], reverse=True)[:8]

    def _category_scores(self, scaled: np.ndarray) -> dict:
        if self.lr_coefficients is None:
            return {c: {"score": 50, "advantage": "even", "label": l} for c, l in CATEGORY_LABELS.items()}
        contribs = self.lr_coefficients * scaled
        buckets: dict[str, list[float]] = {}
        for i, fname in enumerate(self.features):
            cat = self._category(fname)
            if cat in CATEGORY_LABELS:
                buckets.setdefault(cat, []).append(contribs[i])

        result = {}
        for cat, label in CATEGORY_LABELS.items():
            vals = buckets.get(cat, [])
            if vals:
                avg = float(np.mean(vals))
                score = 50 + 50 * float(np.tanh(avg * 3))
                score = max(0, min(100, score))
                if avg > 0.01:
                    adv = "fighter1"
                elif avg < -0.01:
                    adv = "fighter2"
                else:
                    adv = "even"
                result[cat] = {"score": round(score), "advantage": adv, "label": label}
            else:
                result[cat] = {"score": 50, "advantage": "even", "label": label}
        return result

    def _historical(self, f1_actual: str, f2_actual: str):
        if self.norm_fights is None:
            return None
        f1 = f1_actual.lower().strip()
        f2 = f2_actual.lower().strip()
        nf = self.norm_fights
        mask = (
            ((nf["red_name"].str.lower() == f1) & (nf["blue_name"].str.lower() == f2)) |
            ((nf["red_name"].str.lower() == f2) & (nf["blue_name"].str.lower() == f1))
        )
        matches = nf[mask].sort_values("date", ascending=False)
        if matches.empty:
            return None
        results = []
        for _, row in matches.iterrows():
            if row["winner"] == "red":
                winner = row["red_name"]
            elif row["winner"] == "blue":
                winner = row["blue_name"]
            elif row["winner"] == "draw":
                winner = "Draw"
            else:
                winner = "No Contest"
            results.append({
                "date": str(row["date"].date()) if pd.notna(row.get("date")) else "",
                "winner": winner,
                "method": str(row["method"]) if pd.notna(row.get("method")) else None,
                "round": int(row["finish_round"]) if pd.notna(row.get("finish_round")) else None,
                "time": str(row["finish_time"]) if pd.notna(row.get("finish_time")) else None,
            })
        return results

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @staticmethod
    def _date(value) -> str | None:
        parsed = pd.to_datetime(value, errors="coerce")
        return parsed.date().isoformat() if pd.notna(parsed) else None

    def _date_range(self, frame: pd.DataFrame | None, column: str) -> dict:
        dates = pd.to_datetime(frame[column], errors="coerce").dropna() if frame is not None else pd.Series(dtype="datetime64[ns]")
        return {"start": self._date(dates.min()), "end": self._date(dates.max())}

    def get_model_info(self) -> dict:
        temporal_metrics = any("ts_accuracy" in metrics for metrics in self.model_metrics.values())
        return {
            "ready": self.is_ready(),
            "fightCount": len(self.norm_fights) if self.norm_fights is not None else self.fight_count,
            "profileFightCount": self.fight_count,
            "trainingFightCount": None,
            "fighterCount": len(self.get_fighter_names()),
            "featureCount": self.feature_count,
            "modelCount": len(self.models),
            "modelNames": list(self.models),
            "models": self.model_metrics,
            "dateRange": self._date_range(self.norm_fights, "date") if self.norm_fights is not None else self._date_range(self.fights_df, "Date"),
            "profileDateRange": self._date_range(self.fights_df, "Date"),
            "trainedAt": self._date(self.trained_at),
            "evaluation": {
                "random": {"method": "5-fold shuffled cross-validation" if temporal_metrics else "5-fold cross-validation", "metric": "accuracy"},
                "temporal": {"method": "5-fold chronological cross-validation", "metric": "ts_accuracy"} if temporal_metrics else None,
                "ensembleAccuracy": None,
                "limitation": "Imputation and scaling were fitted before cross-validation; these are not leakage-free holdout estimates. Scores evaluate the original corner-dependent models, not the current predictions averaged over both corner orders.",
            },
            "probabilityMethod": "both_corner_orders",
            "explanationMethod": "logistic_coefficients",
            "calibration": "none",
            "limitations": [
                "Historical coverage is not a recorded model-training count or training date.",
                "Display profiles and prediction snapshots come from different dated sources.",
                "Snapshot rolling statistics and career counts precede each fighter's last recorded bout; ratings include that bout.",
                "Snapshot age and layoff values are fixed at dataset generation, not updated for today.",
                "Missing differences use training medians; weight-class indicators default to zero for arbitrary matchups.",
                "Key factors use logistic coefficients times half the difference between standardized corner orders; they are feature effects, not an additive explanation of the ensemble probability.",
            ],
            "categories": list(CATEGORY_LABELS.keys()),
        }

    def save_artifact(self, artifact_path: str) -> None:
        joblib.dump({
            "models": self.models,
            "scaler": self.scaler,
            "imputer": self.imputer,
            "features": self.features,
            "model_metrics": self.model_metrics,
            "lr_coefficients": self.lr_coefficients,
        }, artifact_path, compress=3)
