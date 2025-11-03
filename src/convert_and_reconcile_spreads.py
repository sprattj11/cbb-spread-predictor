#!/usr/bin/env python3
"""
Convert per-team CSVs (like data/auburn.csv, data/bethune.csv) into a master data/games.csv,
and attempt to reconcile/fill missing spreads by matching counterpart rows.

Input expected columns (per file):
  Date, Site, Opponent, Pts, Opp, Spread

Output columns:
  date, home_team, away_team, home_score, away_score, spread, neutral, source_file

Usage:
  Place team CSVs in `data/` and run this script.
"""
from pathlib import Path
import pandas as pd
import re
import sys
import numpy as np

DATA_DIR = Path("data")
OUT_FILE = DATA_DIR / "games.csv"
GLOB_PATTERN = "*.csv"

if not DATA_DIR.exists():
    print("Create data/ and put team CSVs (e.g., auburn.csv, bethune.csv) there.")
    sys.exit(1)

def clean_team_name_from_filename(p: Path) -> str:
    name = p.stem
    name = name.replace("_"," ").replace("-", " ").strip()
    # Basic title-casing; you can add exceptions if needed
    return name.title()

def clean_opponent_name(raw: str) -> str:
    if pd.isna(raw): return ""
    s = str(raw).strip()
    s = re.sub(r"\s*\(\d+\)", "", s)      # remove rankings like " (4)"
    s = re.sub(r"[*#]", "", s)           # remove stray chars
    return s.strip()

def parse_date(raw: str):
    if pd.isna(raw):
        return None
    try:
        dt = pd.to_datetime(str(raw), errors='coerce')
        if pd.isna(dt):
            s = re.sub(r"^[A-Za-z]{3}\s+", "", str(raw))
            dt = pd.to_datetime(s, errors='coerce')
        return dt.date().isoformat() if not pd.isna(dt) else None
    except Exception:
        return None

def find_col(df, candidates):
    for p in candidates:
        for c in df.columns:
            if c.strip().lower() == p.lower():
                return c
    return None

def process_file(path: Path):
    team_name = clean_team_name_from_filename(path)
    df = pd.read_csv(path)
    date_col = find_col(df, ["Date","date"])
    site_col = find_col(df, ["Site","site"])
    opp_col = find_col(df, ["Opponent","opponent","Opp","opp"])
    pts_col = find_col(df, ["Pts","Pta","Pts.","pts","pta"])
    oppscore_col = find_col(df, ["Opp","opp","OppScore","opp_score","Opp."])
    spread_col = find_col(df, ["Spread","spread","Line","line"])

    required = {"date": date_col, "site": site_col, "opp": opp_col, "pts": pts_col, "oppscore": oppscore_col}
    missing = [k for k,v in required.items() if v is None]
    if missing:
        print(f"[WARN] skipping {path.name} — missing columns: {missing}")
        return pd.DataFrame()

    rows = []
    for _, r in df.iterrows():
        d_raw = r[date_col]
        site_raw = r[site_col] if site_col in r.index else ""
        opp_raw = r[opp_col]
        pts_raw = r[pts_col]
        oppscore_raw = r[oppscore_col]
        spread_raw = r[spread_col] if spread_col in r.index else np.nan

        date_iso = parse_date(d_raw)
        if date_iso is None:
            print(f"[WARN] could not parse date '{d_raw}' in {path.name}; skipping row")
            continue

        site = str(site_raw).strip()
        opponent = clean_opponent_name(opp_raw)

        neutral = 0
        if site == "@" or site.lower() == "@":
            # team file's team is away
            home_team = opponent
            away_team = team_name
            # flip spread sign to make it from home team's perspective
            try:
                spread = None if pd.isna(spread_raw) else float(spread_raw) * -1.0
            except:
                spread = None
            try:
                home_score = int(oppscore_raw)
                away_score = int(pts_raw)
            except:
                home_score = oppscore_raw
                away_score = pts_raw
        else:
            # site blank or neutral
            if str(site).strip().upper() in ("N","NEUTRAL"):
                neutral = 1
            home_team = team_name
            away_team = opponent
            try:
                spread = None if pd.isna(spread_raw) else float(spread_raw)
            except:
                spread = None
            try:
                home_score = int(pts_raw)
                away_score = int(oppscore_raw)
            except:
                home_score = pts_raw
                away_score = oppscore_raw

        rows.append({
            "date": date_iso,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "spread": spread,
            "neutral": neutral,
            "source_file": path.name
        })

    out = pd.DataFrame(rows)
    out["home_team"] = out["home_team"].astype(str).str.strip()
    out["away_team"] = out["away_team"].astype(str).str.strip()
    return out

def reconcile_spreads(master_df):
    """
    For each game (date, home, away) where one row has a spread and the other doesn't,
    fill the missing spread.
    Also attempt to fix small sign inconsistencies.
    """
    df = master_df.copy()
    # Build an index to find counterpart rows (home/away swapped)
    df['game_key'] = df['date'].astype(str) + "|" + df['home_team'].str.lower() + "|" + df['away_team'].str.lower()

    # Create a mapping from game_key -> spreads present
    grouped = df.groupby('game_key')

    # For each group (normally size 1, but if both teams present you'll get 2 rows with same key),
    # if one has NaN and the other has a value, fill the NaN.
    filled = df.copy()
    for key, g in grouped:
        spreads = g['spread'].values
        has_value = [not pd.isna(s) for s in spreads]
        if len(spreads) == 2:
            # two records about the same game (unlikely small variations) - prefer non-NaN
            if has_value.count(True) == 1:
                val = spreads[has_value.index(True)]
                # set missing one to val
                idx_missing = g.index[~g['spread'].notna()][0]
                filled.at[idx_missing, 'spread'] = val
        # If group size == 1, we'll fill later by looking for swapped home/away key.

    # Now try matching swapped home/away: find rows where swapped counterpart exists and use its spread flipped.
    # Create a swapped-key column to find counterpart
    df2 = filled.copy()
    df2['swapped_key'] = df2['date'].astype(str) + "|" + df2['away_team'].str.lower() + "|" + df2['home_team'].str.lower()
    key_to_spread = {k: v for k, v in zip(df2['game_key'], df2['spread'])}

    for idx, row in df2.iterrows():
        if pd.isna(row['spread']):
            sk = row['swapped_key']
            if sk in key_to_spread and not pd.isna(key_to_spread[sk]):
                # counterpart has spread; since counterpart's spread is already from its home perspective,
                # and this row is the swapped pairing, use the counterpart spread but flip sign to represent this row's home perspective.
                counterpart_spread = key_to_spread[sk]
                # counterpart_spread is from counterpart_home perspective, which equals our away perspective.
                # We want the spread from our home perspective, which is -counterpart_spread
                filled.at[idx, 'spread'] = -1.0 * counterpart_spread

    # drop helper cols
    filled = filled.drop(columns=['game_key'] , errors='ignore')
    return filled

def main():
    files = sorted([p for p in DATA_DIR.glob(GLOB_PATTERN) if p.is_file() and p.name.lower() != OUT_FILE.name.lower()])
    if not files:
        print("No CSV files found in data/. Place team CSVs like auburn.csv and bethune.csv there.")
        return

    all_dfs = []
    for f in files:
        print("Processing", f.name)
        dfg = process_file(f)
        if not dfg.empty:
            all_dfs.append(dfg)

    if not all_dfs:
        print("No game rows processed.")
        return

    master = pd.concat(all_dfs, ignore_index=True)
    # Attempt to reconcile missing spreads using counterpart rows
    master_reconciled = reconcile_spreads(master)

    # Deduplicate exact duplicates
    before = len(master_reconciled)
    master_reconciled = master_reconciled.drop_duplicates(subset=["date","home_team","away_team","home_score","away_score","spread"])
    after = len(master_reconciled)
    print(f"Processed {before} -> {after} unique rows after deduplication.")

    # Save
    master_reconciled.sort_values(["date","home_team","away_team"]).to_csv(OUT_FILE, index=False)
    print(f"Saved master games to {OUT_FILE} ({len(master_reconciled)} rows).")

if __name__ == "__main__":
    main()
