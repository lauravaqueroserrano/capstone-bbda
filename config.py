"""config.py — Centralized thesis configuration."""

import os
from pathlib import Path

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "thesis_figures"
ARCHIVE_DIR = PROJECT_ROOT / "archive"

# ── Data window ──────────────────────────────────────────────────────────────
DATA_START = "2024-10-01"  # Polymarket stock markets launched Oct 2025
DATA_END = "2026-03-20"

# ── Tickers ──────────────────────────────────────────────────────────────────
PRIMARY_TICKERS = ["AAPL", "NVDA", "AMZN"]
ALL_TICKERS = ["AAPL", "NVDA", "AMZN", "GOOGL", "TSLA", "MSFT", "META", "NFLX", "PLTR"]

# ── Polymarket APIs ──────────────────────────────────────────────────────────
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
STRAPI_BASE = "https://strapi-matic.polymarket.com"
TAGS = {"stocks": 604, "politics": 2, "sports": 100639}

# ── Market type patterns (regex) ─────────────────────────────────────────────
# Used for classifying market questions into types
MARKET_TYPES = {
    "daily_updown": r"Up or Down on (\w+ \d+)",
    "daily_close_above": r"close above \$[\d,.]+ on (\w+ \d+)",
    "daily_close_range": r"close at \$[\d,.]+-\$?[\d,.]+ on the final day.*?of (\w+ \d+)",
    "weekly_above": r"finish week of (\w+ \d+) above",
    "weekly_range": r"close at \$[\d,.]+-\$?[\d,.]+ on the final day of trading of the week",
    "monthly_above": r"close above \$[\d,.]+ end of (\w+)",
    "monthly_hit": r"(?:hit|reach) \$[\d,.]+ (?:in|before|during) (\w+)",
    "yearly_range": r"close at \$[\d,.]+-\$?[\d,.]+ in \d{4}",
}

# ── Calibration parameters ───────────────────────────────────────────────────
CALIBRATION_BINS = 10  # Number of bins for reliability diagrams
CALIBRATION_BIN_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MIN_BIN_COUNT = 10  # Minimum markets per bin to report

# ── Volume thresholds ────────────────────────────────────────────────────────
LOW_VOLUME_THRESHOLD = 2500
HIGH_VOLUME_THRESHOLD = 10000

# ── Event study parameters ───────────────────────────────────────────────────
EVENT_WINDOW = (-5, 5)  # Trading days around event
ESTIMATION_WINDOW = (-260, -11)  # 250 trading days before event window
FF5_FACTORS_FILE = "ff5_factors.parquet"

# ── APIs ─────────────────────────────────────────────────────────────────────
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")

# ── Plot style ───────────────────────────────────────────────────────────────
PLOT_STYLE = "seaborn-v0_8-whitegrid"
FIGURE_DPI = 150
FIGURE_FORMAT = "png"
TICKER_COLORS = {
    "AAPL": "#555555",
    "NVDA": "#76B900",
    "AMZN": "#FF9900",
    "GOOGL": "#4285F4",
    "TSLA": "#CC0000",
    "MSFT": "#00A4EF",
}
