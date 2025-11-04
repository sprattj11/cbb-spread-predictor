"""
Interactive Cover Predictor

Prompts the user for:
  - Home team
  - Away team
  - Spread (from home team's perspective, e.g. -24.5 means home favored)

Trains (if needed) or loads a simple logistic regression model on per-game data
from data/games.csv and predicts whether the **home team covers** the spread.

Expected CSV columns:
  date, home_team, away_team, home_score, away_score, spread, neutral, source_file
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

MODEL_PATH = "models/cover_model.joblib"
GAMES_CSV = "data/games.csv"


def load_games_two_perspective(path=GAMES_CSV):
    """Build two-perspective dataset: one row for home team, one for away team."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find games CSV at {path}")
    raw = pd.read_csv(path)

    required = ["home_team", "away_team", "home_score", "away_score", "spread"]
    for c in required:
        if c not in raw.columns:
            raise RuntimeError(f"Missing expected column in {path}: {c}")

    rows = []
    for _, r in raw.iterrows():
        h, a = r["home_team"], r["away_team"]
        hs, as_ = r["home_score"], r["away_score"]
        spr = float(r["spread"])
        # Home perspective
        rows.append({
            "Team": h, "Opponent": a,
            "Pts": hs, "OppPts": as_,
            "Spread": spr, "IsHome": 1
        })
        # Away perspective (invert spread)
        rows.append({
            "Team": a, "Opponent": h,
            "Pts": as_, "OppPts": hs,
            "Spread": -spr, "IsHome": 0
        })

    df = pd.DataFrame(rows)
    df["Margin"] = df["Pts"] - df["OppPts"]
    df["Covers"] = (df["Margin"] > df["Spread"]).astype(int)
    return df


def train_model_for_pair(df, home_team, away_team, retrain=False):
    """Train logistic regression using only rows where Team in {home_team, away_team}."""
    if os.path.exists(MODEL_PATH) and not retrain:
        try:
            obj = joblib.load(MODEL_PATH)
            return obj["pipeline"]
        except Exception:
            pass

    subset = df[df["Team"].isin([home_team, away_team])]
    if len(subset) < 10:
        print(f"Warning: only {len(subset)} training examples for these teams — model may be weak.")

    X = subset[["Spread", "IsHome"]]
    y = subset["Covers"]

    if y.nunique() < 2:
        print("Not enough variation in outcomes to train; using majority class fallback.")
        val = int(y.mode().iloc[0]) if len(y) else 0
        dummy = {"type": "constant", "value": val}
        os.makedirs("models", exist_ok=True)
        joblib.dump({"pipeline": dummy, "teams": [home_team, away_team]}, MODEL_PATH)
        return dummy

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)
    probs = pipe.predict_proba(Xte)[:, 1]
    acc = accuracy_score(yte, preds)
    auc = roc_auc_score(yte, probs) if len(set(yte)) > 1 else np.nan
    print(f"Trained logistic model. Validation acc={acc:.3f}, AUC={auc if not np.isnan(auc) else 'NA'}")

    os.makedirs("models", exist_ok=True)
    joblib.dump({"pipeline": pipe, "teams": [home_team, away_team]}, MODEL_PATH)
    print("Saved ->", MODEL_PATH)
    return pipe


def predict_cover(pipeline, spread, is_home=1):
    """Return (pred, prob) from model given spread and home flag."""
    if isinstance(pipeline, dict) and pipeline.get("type") == "constant":
        val = pipeline["value"]
        return int(val), float(1.0 if val == 1 else 0.0)

    X = pd.DataFrame([[spread, is_home]], columns=["Spread", "IsHome"])
    prob = pipeline.predict_proba(X)[0, 1]
    pred = int(pipeline.predict(X)[0])
    return pred, prob


def main():
    # Simple interactive prompt
    print("=== College Basketball Spread Predictor ===")
    home_team = input("Enter home team: ").strip()
    away_team = input("Enter away team: ").strip()
    spread_str = input("Enter spread (from home team perspective, e.g. -24.5 means home favored): ").strip()

    try:
        spread = float(spread_str)
    except ValueError:
        print("Invalid spread input. Please enter a numeric value like -24.5.")
        return

    games = load_games_two_perspective(GAMES_CSV)
    pipeline = train_model_for_pair(games, home_team, away_team, retrain=True)

    pred, prob = predict_cover(pipeline, spread, is_home=1)
    result = "COVERS" if pred == 1 else "DOES NOT COVER"
    print(f"\nPrediction: {home_team} {result} (probability of covering = {prob:.3f})")


if __name__ == "__main__":
    main()
