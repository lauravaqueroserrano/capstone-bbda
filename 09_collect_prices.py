"""09_collect_prices.py — Fetch pre-resolution implied probabilities from CLOB API.

For each resolved market, retrieves the last traded price (Yes token) before
the resolution date. Uses chunked 7-day windows to stay within API limits.

Usage:
    python 09_collect_prices.py

Outputs:
    data/implied_prices.parquet — market_id, implied_prob, last_trade_ts, n_points
"""

import json
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import requests

import config as cfg
import thesis_utils as tu

# ── Config ───────────────────────────────────────────────────────────────────
CLOB_BASE = cfg.CLOB_BASE
CHUNK_DAYS = 7          # API limit is ~10 days per request
RATE_LIMIT = 0.2        # seconds between requests
FIDELITY = 60           # minutes between price points
SAVE_EVERY = 200        # checkpoint interval
OUTPUT_PATH = cfg.DATA_DIR / "implied_prices.parquet"
PROGRESS_PATH = cfg.RESULTS_DIR / "price_collection_progress.csv"


def fetch_price_history(token_id: str, start_ts: int, end_ts: int) -> list[dict]:
    """Fetch price history for a single token in one chunk."""
    url = f"{CLOB_BASE}/prices-history"
    params = {
        "market": token_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "fidelity": FIDELITY,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict) and "history" in data:
                return data["history"]
            if isinstance(data, list):
                return data
        return []
    except Exception:
        return []


def get_last_price_before(token_id: str, resolution_ts: int,
                          lookback_days: int = 30) -> dict:
    """Get the last traded price before the resolution timestamp.

    Searches backwards in CHUNK_DAYS windows up to lookback_days.
    """
    all_points = []
    end_ts = resolution_ts
    earliest_ts = resolution_ts - (lookback_days * 86400)

    while end_ts > earliest_ts:
        start_ts = max(end_ts - (CHUNK_DAYS * 86400), earliest_ts)
        points = fetch_price_history(token_id, start_ts, end_ts)
        time.sleep(RATE_LIMIT)

        if points:
            all_points.extend(points)
            # If we have points, check if the oldest one is near our target
            # We want the LAST price before resolution, so we can stop once
            # we have enough data
            if len(all_points) >= 2:
                break

        end_ts = start_ts

    if not all_points:
        return {"price": np.nan, "timestamp": None, "n_points": 0}

    # Filter to points AT or BEFORE resolution
    valid = [p for p in all_points if p["t"] <= resolution_ts]
    if not valid:
        # If all points are after resolution, take the earliest
        valid = all_points

    # Sort by timestamp descending, take last before resolution
    valid.sort(key=lambda x: x["t"], reverse=True)
    best = valid[0]

    return {
        "price": float(best["p"]),
        "timestamp": best["t"],
        "n_points": len(all_points),
    }


def main():
    data_dir, results_dir = tu.ensure_project_dirs(cfg.PROJECT_ROOT)

    # Load resolved markets
    df_res = tu.load_parquet("market_resolutions.parquet")
    print(f"Resolved markets: {len(df_res):,}")

    # Load market metadata for token IDs
    df_mkts = tu.load_parquet("polymarket_stocks_markets_raw.parquet")
    df_mkts["token_list"] = df_mkts["clobTokenIds"].apply(tu.parse_token_ids)
    df_mkts["has_tokens"] = df_mkts["token_list"].apply(lambda x: len(x) > 0)
    print(f"Markets with token IDs: {df_mkts['has_tokens'].sum():,}")

    # Merge token IDs into resolution data
    token_map = df_mkts[df_mkts["has_tokens"]][["market_id", "token_list"]].copy()
    df = df_res.merge(token_map, on="market_id", how="left")
    df["has_tokens"] = df["token_list"].apply(
        lambda x: isinstance(x, list) and len(x) > 0
    )
    print(f"Resolved markets with tokens: {df['has_tokens'].sum():,}")

    # Convert resolution_date to unix timestamp
    df["resolution_date"] = pd.to_datetime(df["resolution_date"])
    df["res_ts"] = df["resolution_date"].apply(
        lambda x: int(x.timestamp()) if pd.notna(x) else None
    )

    # Filter to fetchable markets
    fetchable = df[df["has_tokens"] & df["res_ts"].notna()].copy()
    print(f"Fetchable markets: {len(fetchable):,}")

    # Resume from checkpoint if it exists
    already_done = set()
    if PROGRESS_PATH.exists():
        prev = pd.read_csv(PROGRESS_PATH)
        already_done = set(prev["market_id"].values)
        print(f"Resuming — {len(already_done):,} already collected")

    # Collect prices
    results = []
    if PROGRESS_PATH.exists():
        results = pd.read_csv(PROGRESS_PATH).to_dict("records")

    to_fetch = fetchable[~fetchable["market_id"].isin(already_done)]
    total = len(to_fetch)
    print(f"\nFetching prices for {total:,} markets...")
    print(f"Estimated time: ~{total * RATE_LIMIT / 60:.0f} minutes\n")

    errors = 0
    for i, (_, row) in enumerate(to_fetch.iterrows()):
        token_yes = row["token_list"][0]  # First token = Yes outcome
        res_ts = int(row["res_ts"])

        result = get_last_price_before(token_yes, res_ts)

        results.append({
            "market_id": row["market_id"],
            "implied_prob": result["price"],
            "last_trade_ts": result["timestamp"],
            "n_points": result["n_points"],
        })

        if np.isnan(result["price"]):
            errors += 1

        if (i + 1) % 50 == 0 or i == total - 1:
            valid = sum(1 for r in results if not np.isnan(r["implied_prob"]))
            print(f"  [{i+1}/{total}] valid={valid:,} errors={errors}")

        # Checkpoint
        if (i + 1) % SAVE_EVERY == 0:
            pd.DataFrame(results).to_csv(PROGRESS_PATH, index=False)

    # Save final results
    df_prices = pd.DataFrame(results)
    df_prices.to_csv(PROGRESS_PATH, index=False)

    valid_count = df_prices["implied_prob"].notna().sum()
    print(f"\n=== Collection Complete ===")
    print(f"Total fetched: {len(df_prices):,}")
    print(f"Valid prices:  {valid_count:,} ({valid_count/len(df_prices)*100:.1f}%)")
    print(f"Missing:       {len(df_prices) - valid_count:,}")

    # Filter out post-resolution prices (near 0 or 1 are likely settled)
    # Keep prices between 0.01 and 0.99 as pre-resolution
    df_prices["is_pre_resolution"] = (
        df_prices["implied_prob"].between(0.01, 0.99, inclusive="both")
    )
    pre_res_count = df_prices["is_pre_resolution"].sum()
    print(f"Pre-resolution prices (0.01-0.99): {pre_res_count:,}")

    # Save
    tu.save_parquet(df_prices, OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")

    # Summary statistics
    valid_prices = df_prices[df_prices["implied_prob"].notna()]["implied_prob"]
    print(f"\nPrice distribution:")
    print(valid_prices.describe())


if __name__ == "__main__":
    main()
