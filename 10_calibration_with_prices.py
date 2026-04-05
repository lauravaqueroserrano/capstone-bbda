"""10_calibration_with_prices.py — Full calibration analysis using collected prices.

Reads implied_prices.parquet (from 09_collect_prices.py), merges with
market_resolutions.parquet, and computes all calibration metrics needed
for the thesis rewrite.

Usage:
    python 10_calibration_with_prices.py

Outputs:
    results/thesis_figures/calibration_overall_v2.png
    results/thesis_figures/calibration_by_type_v2.png
    results/thesis_figures/calibration_by_volume_v2.png
    results/thesis_figures/table_calibration_full.csv
    results/calibration_full_summary.csv
    data/market_resolutions_with_prices.parquet
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

import config as cfg
import thesis_utils as tu

data_dir, results_dir = tu.ensure_project_dirs(cfg.PROJECT_ROOT)
figures_dir = cfg.FIGURES_DIR
figures_dir.mkdir(parents=True, exist_ok=True)

try:
    plt.style.use(cfg.PLOT_STYLE)
except Exception:
    plt.style.use("seaborn-v0_8")
matplotlib.rcParams.update({"font.size": 11, "axes.titlesize": 13})


# ── 1. Load and merge ────────────────────────────────────────────────────────

df_res = tu.load_parquet("market_resolutions.parquet")
df_prices = tu.load_parquet("implied_prices.parquet")

print(f"Resolved markets: {len(df_res):,}")
print(f"Collected prices: {len(df_prices):,}")
print(f"Valid prices:     {df_prices['implied_prob'].notna().sum():,}")

# Merge prices into resolutions
df = df_res.merge(
    df_prices[["market_id", "implied_prob", "last_trade_ts", "n_points"]],
    on="market_id",
    how="left",
    suffixes=("_old", ""),
)

# Use new implied_prob; fall back to old if new is missing
if "implied_prob_old" in df.columns:
    df["implied_prob"] = df["implied_prob"].fillna(df["implied_prob_old"])
    df.drop(columns=["implied_prob_old"], inplace=True)

# Save merged dataset
tu.save_parquet(df, data_dir / "market_resolutions_with_prices.parquet")

# Filter to calibration-ready markets
df_cal = df[
    df["implied_prob"].notna()
    & df["outcome_int"].notna()
    & df["implied_prob"].between(0.01, 0.99)
].copy()

print(f"\nCalibration-ready markets (price in [0.01, 0.99]): {len(df_cal):,}")
print(f"Base rate: {df_cal['outcome_int'].mean():.3f}")


# ── 2. Helper functions ──────────────────────────────────────────────────────

def compute_brier_decomposition(predicted, observed, n_bins=10):
    """Murphy-Winkler decomposition: BS = REL - RES + UNC."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(predicted, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    o_bar = observed.mean()
    unc = o_bar * (1 - o_bar)

    rel = 0.0
    res = 0.0
    details = []
    for k in range(n_bins):
        mask = bin_idx == k
        nk = mask.sum()
        if nk == 0:
            continue
        pk = predicted[mask].mean()
        ok = observed[mask].mean()
        rel += nk * (pk - ok) ** 2
        res += nk * (ok - o_bar) ** 2
        details.append({
            "bin": f"{bins[k]:.1f}-{bins[k+1]:.1f}",
            "n": int(nk),
            "mean_pred": round(pk, 4),
            "mean_obs": round(ok, 4),
            "gap": round(abs(pk - ok), 4),
        })

    n = len(predicted)
    return {
        "reliability": rel / n,
        "resolution": res / n,
        "uncertainty": unc,
        "brier": rel / n - res / n + unc,
        "details": details,
    }


def compute_ece(predicted, observed, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(predicted, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    ece = 0.0
    n = len(predicted)
    for k in range(n_bins):
        mask = bin_idx == k
        nk = mask.sum()
        if nk == 0:
            continue
        pk = predicted[mask].mean()
        ok = observed[mask].mean()
        ece += (nk / n) * abs(pk - ok)
    return ece


def plot_reliability(predicted, observed, title, ax, n_bins=10, color="steelblue"):
    """Plot reliability diagram with histogram."""
    prob_true, prob_pred = calibration_curve(
        observed, predicted, n_bins=n_bins, strategy="uniform"
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.plot(prob_pred, prob_true, "s-", color=color, markersize=8, label="Observed")
    ax2 = ax.twinx()
    ax2.hist(predicted, bins=n_bins, range=(0, 1), alpha=0.12, color=color)
    ax2.set_ylabel("Count", color="gray", alpha=0.6)
    ax2.tick_params(axis="y", labelcolor="gray")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)


# ── 3. Overall calibration ───────────────────────────────────────────────────

predicted = df_cal["implied_prob"].values
observed = df_cal["outcome_int"].values.astype(float)

bs = brier_score_loss(observed, predicted)
decomp = compute_brier_decomposition(predicted, observed)
ece = compute_ece(predicted, observed)
ll = log_loss(observed, np.clip(predicted, 1e-6, 1 - 1e-6))

print(f"\n{'='*50}")
print(f"OVERALL CALIBRATION (N={len(df_cal):,})")
print(f"{'='*50}")
print(f"Brier Score:  {bs:.4f}")
print(f"  Reliability:  {decomp['reliability']:.4f}")
print(f"  Resolution:   {decomp['resolution']:.4f}")
print(f"  Uncertainty:  {decomp['uncertainty']:.4f}")
print(f"ECE:          {ece:.4f}")
print(f"Log-Loss:     {ll:.4f}")
print(f"Base rate:    {observed.mean():.4f}")

# Reliability diagram
fig, ax = plt.subplots(figsize=(7, 7))
plot_reliability(
    predicted, observed,
    title=f"Overall Calibration (N={len(df_cal):,}, BS={bs:.3f}, ECE={ece:.3f})",
    ax=ax,
)
plt.tight_layout()
plt.savefig(figures_dir / "calibration_overall_v2.png", dpi=cfg.FIGURE_DPI, bbox_inches="tight")
plt.close()
print(f"Saved calibration_overall_v2.png")


# ── 4. Calibration by market type ────────────────────────────────────────────

type_metrics = []
market_types = df_cal["market_type"].value_counts()
plot_types = [mt for mt, ct in market_types.items() if ct >= 30]

n_types = len(plot_types)
ncols = min(3, n_types)
nrows = (n_types + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
axes = np.array(axes).flatten() if n_types > 1 else [axes]

for i, mtype in enumerate(plot_types):
    subset = df_cal[df_cal["market_type"] == mtype]
    pred_s = subset["implied_prob"].values
    obs_s = subset["outcome_int"].values.astype(float)

    bs_s = brier_score_loss(obs_s, pred_s)
    ece_s = compute_ece(pred_s, obs_s)
    decomp_s = compute_brier_decomposition(pred_s, obs_s)

    type_metrics.append({
        "market_type": mtype,
        "n": len(subset),
        "brier": round(bs_s, 4),
        "reliability": round(decomp_s["reliability"], 4),
        "resolution": round(decomp_s["resolution"], 4),
        "ece": round(ece_s, 4),
        "base_rate": round(obs_s.mean(), 4),
    })

    plot_reliability(
        pred_s, obs_s,
        title=f"{mtype} (N={len(subset)}, BS={bs_s:.3f})",
        ax=axes[i],
    )

for j in range(n_types, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Calibration by Market Type", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(figures_dir / "calibration_by_type_v2.png", dpi=cfg.FIGURE_DPI, bbox_inches="tight")
plt.close()

type_df = pd.DataFrame(type_metrics).sort_values("n", ascending=False)
print(f"\n{'='*50}")
print("CALIBRATION BY MARKET TYPE")
print(f"{'='*50}")
print(type_df.to_string(index=False))
type_df.to_csv(figures_dir / "table_calibration_by_type.csv", index=False)


# ── 5. Calibration by volume ─────────────────────────────────────────────────

df_cal["volume_group"] = pd.qcut(
    df_cal["volume"].fillna(0), q=3,
    labels=["Low Volume", "Medium Volume", "High Volume"],
)

vol_metrics = []
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors = ["#e74c3c", "#f39c12", "#27ae60"]

for i, vgroup in enumerate(["Low Volume", "Medium Volume", "High Volume"]):
    subset = df_cal[df_cal["volume_group"] == vgroup]
    if len(subset) < 30:
        continue
    pred_s = subset["implied_prob"].values
    obs_s = subset["outcome_int"].values.astype(float)

    bs_s = brier_score_loss(obs_s, pred_s)
    ece_s = compute_ece(pred_s, obs_s)
    vol_range = f"${subset['volume'].min():,.0f}–${subset['volume'].max():,.0f}"

    vol_metrics.append({
        "volume_group": vgroup,
        "n": len(subset),
        "brier": round(bs_s, 4),
        "ece": round(ece_s, 4),
        "volume_range": vol_range,
        "base_rate": round(obs_s.mean(), 4),
    })

    plot_reliability(
        pred_s, obs_s,
        title=f"{vgroup} (N={len(subset)}, BS={bs_s:.3f})",
        ax=axes[i], color=colors[i],
    )

plt.suptitle("Calibration by Volume Tercile", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(figures_dir / "calibration_by_volume_v2.png", dpi=cfg.FIGURE_DPI, bbox_inches="tight")
plt.close()

vol_df = pd.DataFrame(vol_metrics)
print(f"\n{'='*50}")
print("CALIBRATION BY VOLUME")
print(f"{'='*50}")
print(vol_df.to_string(index=False))
vol_df.to_csv(figures_dir / "table_calibration_by_volume.csv", index=False)


# ── 6. Combined summary table ────────────────────────────────────────────────

summary_rows = [{
    "group": "Overall",
    "n": len(df_cal),
    "brier": round(bs, 4),
    "reliability": round(decomp["reliability"], 4),
    "resolution": round(decomp["resolution"], 4),
    "ece": round(ece, 4),
    "log_loss": round(ll, 4),
    "base_rate": round(observed.mean(), 4),
}]

for _, row in type_df.iterrows():
    summary_rows.append({
        "group": f"Type: {row['market_type']}",
        "n": row["n"],
        "brier": row["brier"],
        "reliability": row["reliability"],
        "resolution": row["resolution"],
        "ece": row["ece"],
        "base_rate": row["base_rate"],
    })

for _, row in vol_df.iterrows():
    summary_rows.append({
        "group": f"Vol: {row['volume_group']}",
        "n": row["n"],
        "brier": row["brier"],
        "ece": row["ece"],
        "base_rate": row["base_rate"],
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(results_dir / "calibration_full_summary.csv", index=False)
summary.to_csv(figures_dir / "table_calibration_full.csv", index=False)

print(f"\n{'='*50}")
print("FULL CALIBRATION SUMMARY")
print(f"{'='*50}")
print(summary.to_string(index=False))

# ── 7. Bin details for the overall reliability diagram ────────────────────────

print(f"\n{'='*50}")
print("CALIBRATION BIN DETAILS (Overall)")
print(f"{'='*50}")
bin_df = pd.DataFrame(decomp["details"])
print(bin_df.to_string(index=False))
bin_df.to_csv(figures_dir / "table_calibration_bins.csv", index=False)

print(f"\n✓ All outputs saved to {figures_dir}")
