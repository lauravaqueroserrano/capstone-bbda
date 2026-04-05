
from __future__ import annotations

import importlib.util
import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


def parquet_engine() -> str | None:
    for mod in ('pyarrow', 'fastparquet'):
        if importlib.util.find_spec(mod) is not None:
            return mod
    return None


def require_parquet_engine() -> str:
    engine = parquet_engine()
    if engine is None:
        raise RuntimeError(
            'No hay motor de Parquet disponible. Instala uno de estos paquetes en tu entorno:\n'
            '  pip install pyarrow\n'
            'o\n'
            '  pip install fastparquet'
        )
    return engine


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen = set()
    out = []
    for p in paths:
        p = Path(p)
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def candidate_roots(extra: Iterable[Path] | None = None) -> list[Path]:
    roots = [
        Path.cwd(),
        Path.cwd() / 'data',
        Path.cwd() / 'results',
        Path('/mnt/data'),
        Path('/mnt/data/data'),
        Path('/mnt/data/results'),
    ]
    if extra:
        roots.extend(Path(x) for x in extra)
    return [p for p in unique_paths(roots)]


def ensure_project_dirs(base: Path | None = None) -> tuple[Path, Path]:
    base = Path.cwd() if base is None else Path(base)
    data_dir = base / 'data'
    results_dir = base / 'results'
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, results_dir


def find_file(name: str, roots: Iterable[Path] | None = None) -> Path | None:
    p = Path(name)
    if p.exists():
        return p
    for root in candidate_roots(roots):
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def find_files(pattern: str, roots: Iterable[Path] | None = None) -> list[Path]:
    hits: list[Path] = []
    for root in candidate_roots(roots):
        if root.exists():
            hits.extend(root.glob(pattern))
    return unique_paths(sorted(hits))


def load_parquet(name_or_path: str | Path, roots: Iterable[Path] | None = None, required: bool = True) -> pd.DataFrame | None:
    engine = require_parquet_engine()
    path = find_file(str(name_or_path), roots=roots)
    if path is None:
        if required:
            raise FileNotFoundError(f'No se encontró el archivo: {name_or_path}')
        return None
    return pd.read_parquet(path, engine=engine)


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    engine = require_parquet_engine()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine=engine)
    return path


def load_csv(name_or_path: str | Path, roots: Iterable[Path] | None = None, required: bool = True, **kwargs) -> pd.DataFrame | None:
    path = find_file(str(name_or_path), roots=roots)
    if path is None:
        if required:
            raise FileNotFoundError(f'No se encontró el archivo: {name_or_path}')
        return None
    return pd.read_csv(path, **kwargs)


def copy_into_data(filenames: Iterable[str], source_roots: Iterable[Path] | None = None, overwrite: bool = False) -> pd.DataFrame:
    data_dir, _ = ensure_project_dirs()
    rows = []
    for name in filenames:
        src = find_file(name, roots=source_roots)
        dst = data_dir / name
        if src is None:
            rows.append({'filename': name, 'status': 'missing', 'src': None, 'dst': str(dst)})
            continue
        if dst.exists() and not overwrite:
            rows.append({'filename': name, 'status': 'already_exists', 'src': str(src), 'dst': str(dst)})
            continue
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
            status = 'copied'
        else:
            status = 'same_path'
        rows.append({'filename': name, 'status': status, 'src': str(src), 'dst': str(dst)})
    return pd.DataFrame(rows)


def slugify(text: str, max_len: int = 80) -> str:
    safe = ''.join(ch if ch.isalnum() else '_' for ch in str(text))
    while '__' in safe:
        safe = safe.replace('__', '_')
    safe = safe.strip('_')
    return safe[:max_len] if len(safe) > max_len else safe


def parse_token_ids(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, float) and math.isnan(raw):
        return []
    token_ids = raw
    if isinstance(token_ids, str):
        token_ids = token_ids.strip()
        if token_ids in {'', '[]', 'nan', 'None'}:
            return []
        try:
            token_ids = json.loads(token_ids)
        except Exception:
            return []
    if not isinstance(token_ids, list):
        return []
    while token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    cleaned = []
    for item in token_ids:
        if isinstance(item, list):
            if item:
                item = item[0]
            else:
                continue
        if item is None:
            continue
        s = str(item).strip()
        if not s or s in {'nan', 'None'} or '[' in s or ']' in s:
            continue
        cleaned.append(s)
    return cleaned


def describe_paths(paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        p = Path(p)
        rows.append({
            'path': str(p),
            'exists': p.exists(),
            'size_bytes': p.stat().st_size if p.exists() else None,
        })
    return pd.DataFrame(rows)


def discover_prob_files(ticker: str | None = None, roots: Iterable[Path] | None = None) -> list[Path]:
    patterns = ['prob_*.parquet', 'prob_policy_*.parquet', 'prob_nba_*.parquet']
    hits: list[Path] = []
    for pattern in patterns:
        hits.extend(find_files(pattern, roots=roots))
    hits = unique_paths(hits)
    if ticker is None:
        return hits
    ticker_upper = ticker.upper()
    filtered = [p for p in hits if ticker_upper in p.name.upper()]
    return filtered


def first_prob_file_for_ticker(ticker: str, roots: Iterable[Path] | None = None) -> Path | None:
    files = discover_prob_files(ticker=ticker, roots=roots)
    return files[0] if files else None


# ═══════════════════════════════════════════════════════════════════════════════
# Market Title Parsing — extract ticker, strike, date, market type from questions
# ═══════════════════════════════════════════════════════════════════════════════

import re
from datetime import datetime, date
from dateutil import parser as dateutil_parser
import numpy as np

# Ticker extraction pattern — matches (TICKER) or standalone well-known tickers
_TICKER_PAREN = re.compile(r'\(([A-Z]{1,5})\)')
_KNOWN_TICKERS = {
    'AAPL', 'NVDA', 'AMZN', 'GOOGL', 'TSLA', 'MSFT', 'META', 'NFLX', 'PLTR',
    'AMD', 'INTC', 'BABA', 'PYPL', 'UBER', 'SNAP', 'COIN', 'SOFI', 'RIVN',
    'SPY', 'QQQ', 'OPEN', 'RKLB', 'ABNB', 'LULU', 'SMCI', 'MSTR', 'AI',
    'ARM', 'CRWD', 'DDOG', 'SNOW', 'NET', 'SQ', 'SHOP', 'ROKU', 'PINS',
    'HOOD', 'MARA', 'RIOT', 'DIS', 'GOOG', 'WMT', 'JPM', 'BAC', 'GS',
    'V', 'MA', 'UNH', 'JNJ', 'PFE', 'MRK', 'ABBV', 'LLY', 'COST', 'HD',
    'CRM', 'ORCL', 'ADBE', 'AVGO', 'MU', 'QCOM', 'TXN', 'PANW', 'ZS',
}

# Price extraction — $XXX or $XXX.XX
_PRICE = re.compile(r'\$\s*([\d,]+(?:\.\d+)?)')
_PRICE_RANGE = re.compile(r'\$([\d,]+(?:\.\d+)?)\s*[-–]\s*\$?([\d,]+(?:\.\d+)?)')

# Month names for date parsing
_MONTHS = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12,
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
}


def _parse_price(s: str) -> float | None:
    """Parse a dollar string like '$130' or '$1,234.56' to float."""
    s = s.replace(',', '').replace('$', '').strip()
    try:
        return float(s)
    except ValueError:
        return None


def extract_ticker(question: str) -> str | None:
    """Extract ticker symbol from a market question."""
    # First try parenthesized tickers — e.g., "(AAPL)", "(OPEN)"
    m = _TICKER_PAREN.search(question)
    if m:
        ticker = m.group(1)
        # Accept any 1-5 char uppercase alphabetic string in parens as ticker
        if ticker.isalpha() and len(ticker) <= 5:
            return ticker
    # Fallback: look for known tickers in text
    upper = question.upper()
    if 'S&P 500' in question or 'S&P500' in question:
        return 'SPY'
    # Match "Will NVIDIA", "Will Apple", etc. — map common names to tickers
    _NAME_MAP = {
        'NVIDIA': 'NVDA', 'APPLE': 'AAPL', 'AMAZON': 'AMZN', 'GOOGLE': 'GOOGL',
        'TESLA': 'TSLA', 'MICROSOFT': 'MSFT', 'NETFLIX': 'NFLX', 'PALANTIR': 'PLTR',
        'OPENDOOR': 'OPEN', 'AIRBNB': 'ABNB', 'LULULEMON': 'LULU',
    }
    for name, ticker in _NAME_MAP.items():
        if name in upper:
            return ticker
    for t in _KNOWN_TICKERS:
        if t in upper.split():
            return t
    return None


def classify_market_type(question: str) -> str:
    """Classify a market question into one of the 7+ market types."""
    q = question.strip()
    ql = q.lower()

    # Daily up/down
    if re.search(r'up or down on', ql):
        return 'daily_updown'

    # Weekly range — "close at $X-$Y on the final day of trading of the week of"
    # Also: "close at >$X" or "close at <$X" on the final day of the week
    if re.search(r'close at .*?on the final day.*?of the week', ql):
        return 'weekly_range'

    # Daily close range — "close at $X-$Y on <date>?" (no "week" keyword)
    if re.search(r'close at \$.*?on\b', ql) and 'week' not in ql:
        return 'daily_close_range'

    # Weekly above — "finish week of ... above"
    if re.search(r'finish week of.*above', ql):
        return 'weekly_above'

    # Monthly dip — "dip to $X in March" (stock must close below target)
    if re.search(r'dip to \$[\d,.]+ (?:in|before|during)', ql):
        return 'monthly_dip'

    # Monthly hit — "hit $X in March" or "reach $X in March"
    if re.search(r'(?:hit|reach) \$[\d,.]+ (?:in|before|during)', ql):
        return 'monthly_hit'

    # Monthly above — "close above $X end of March"
    if re.search(r'close above \$[\d,.]+ end of', ql):
        return 'monthly_above'

    # Daily close above — "close above $X on <date>"
    if re.search(r'close above \$[\d,.]+ on \w+', ql):
        return 'daily_close_above'

    # Yearly range — "close at $X-$Y in 2025"
    if re.search(r'close at \$.*?in \d{4}', ql):
        return 'yearly_range'

    # S&P all time high
    if 'all time high' in ql:
        return 'sp500_ath'

    # Worst/best performing
    if re.search(r'(?:worst|best) performing', ql):
        return 'performance_rank'

    return 'other'


def _infer_year(month: int, end_date_str: str | None) -> int:
    """Infer the year for a parsed month based on market end date.

    Logic: if parsed month is after the end_date's month, it must be in
    the previous year (e.g., October in a market ending March 2026 → Oct 2025).
    """
    if end_date_str:
        try:
            end_dt = pd.Timestamp(end_date_str)
            if end_dt.month >= month:
                return end_dt.year
            return end_dt.year - 1
        except Exception:
            pass
    # Default: use current context (2025-2026 range)
    return 2026 if month <= 3 else 2025


def extract_resolution_date(question: str, market_type: str, end_date: str | None = None) -> date | None:
    """Extract the resolution date from a market question.

    Returns the date on which we should check the stock price.
    """
    q = question.strip()
    ql = q.lower()

    # Daily types — look for "on March 18" or "on October 17"
    if market_type in ('daily_updown', 'daily_close_above', 'daily_close_range'):
        m = re.search(r'on (\w+ \d{1,2})\b', q)
        if m:
            try:
                dt = dateutil_parser.parse(m.group(1), fuzzy=True)
                year = _infer_year(dt.month, end_date)
                return date(year, dt.month, dt.day)
            except Exception:
                pass

    # Weekly types — "week of March 2" or "week of Feb 2 – Feb 6"
    if market_type in ('weekly_above', 'weekly_range'):
        # Try to find the end of week date
        m = re.search(r'week of (\w+ \d{1,2})', q)
        if m:
            try:
                dt = dateutil_parser.parse(m.group(1), fuzzy=True)
                year = _infer_year(dt.month, end_date)
                start = date(year, dt.month, dt.day)
                # Resolution date = Friday of that week
                days_until_friday = (4 - start.weekday()) % 7
                return start + pd.Timedelta(days=days_until_friday)
            except Exception:
                pass

    # Monthly above — "end of March"
    if market_type == 'monthly_above':
        m = re.search(r'end of (\w+)', ql)
        if m:
            month_name = m.group(1).lower()
            if month_name in _MONTHS:
                month_num = _MONTHS[month_name]
                year = _infer_year(month_num, end_date)
                # Last day of month
                next_month = month_num + 1 if month_num < 12 else 1
                next_year = year if month_num < 12 else year + 1
                last_day = date(next_year, next_month, 1) - pd.Timedelta(days=1)
                return last_day

    # Monthly dip — same date logic as monthly hit
    if market_type == 'monthly_dip':
        m = re.search(r'(?:in|during) (\w+)', ql)
        if m:
            month_name = m.group(1).lower()
            if month_name in _MONTHS:
                month_num = _MONTHS[month_name]
                year = _infer_year(month_num, end_date)
                next_month = month_num + 1 if month_num < 12 else 1
                next_year = year if month_num < 12 else year + 1
                return date(next_year, next_month, 1) - pd.Timedelta(days=1)

    # Monthly hit — "in March" or "before 2026"
    if market_type == 'monthly_hit':
        m = re.search(r'(?:in|during) (\w+)', ql)
        if m:
            month_name = m.group(1).lower()
            if month_name in _MONTHS:
                month_num = _MONTHS[month_name]
                year = _infer_year(month_num, end_date)
                next_month = month_num + 1 if month_num < 12 else 1
                next_year = year if month_num < 12 else year + 1
                return date(next_year, next_month, 1) - pd.Timedelta(days=1)
        # "before 2026"
        m = re.search(r'before (\d{4})', ql)
        if m:
            return date(int(m.group(1)) - 1, 12, 31)

    # Yearly range — "in 2025"
    if market_type == 'yearly_range':
        m = re.search(r'in (\d{4})', ql)
        if m:
            return date(int(m.group(1)), 12, 31)

    # Fallback: use end_date
    if end_date:
        try:
            return pd.Timestamp(end_date).date()
        except Exception:
            pass

    return None


def extract_strike_price(question: str, market_type: str) -> dict:
    """Extract strike price or range from a market question.

    Returns dict with keys: 'strike', 'strike_low', 'strike_high' as applicable.
    """
    result = {'strike': None, 'strike_low': None, 'strike_high': None}
    q = question.strip()

    if market_type == 'daily_updown':
        # No strike price — resolution is relative to previous close
        return result

    # Range types — $X-$Y
    m = _PRICE_RANGE.search(q)
    if m and market_type in ('weekly_range', 'daily_close_range', 'yearly_range'):
        result['strike_low'] = _parse_price(m.group(1))
        result['strike_high'] = _parse_price(m.group(2))
        return result

    # Special: "close at <$X" (less than)
    m = re.search(r'close at <\$([\d,]+(?:\.\d+)?)', q)
    if m:
        result['strike_high'] = _parse_price(m.group(1))
        return result

    # Special: "close at >$X" (greater than)
    m = re.search(r'close at >\$([\d,]+(?:\.\d+)?)', q)
    if m:
        result['strike'] = _parse_price(m.group(1))
        return result

    # Single strike — "above $130", "hit $140", "reach $280", "dip to $130"
    m = re.search(r'(?:above|hit|reach|dip to) \$([\d,]+(?:\.\d+)?)', q)
    if m:
        result['strike'] = _parse_price(m.group(1))
        return result

    return result


def parse_market_title(question: str, end_date: str | None = None) -> dict:
    """Parse a Polymarket question into structured components.

    Returns dict with: ticker, market_type, resolution_date, strike, strike_low, strike_high
    """
    ticker = extract_ticker(question)
    mtype = classify_market_type(question)
    res_date = extract_resolution_date(question, mtype, end_date)
    strikes = extract_strike_price(question, mtype)

    return {
        'ticker': ticker,
        'market_type': mtype,
        'resolution_date': res_date,
        **strikes,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Resolution Logic — determine if "Yes" outcome occurred
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_market(row: pd.Series, stock_prices: pd.DataFrame) -> bool | None:
    """Determine the actual outcome of a market given stock price data.

    Args:
        row: Series with keys: ticker, market_type, resolution_date, strike, strike_low, strike_high
        stock_prices: DataFrame indexed by date with columns for each ticker (Close prices).
                     Must also have '{ticker}_High' columns for monthly_hit type.

    Returns:
        True if "Yes" outcome, False if "No", None if cannot determine.
    """
    ticker = row.get('ticker')
    mtype = row.get('market_type')
    res_date = row.get('resolution_date')
    strike = row.get('strike')
    strike_low = row.get('strike_low')
    strike_high = row.get('strike_high')

    if ticker is None or res_date is None or mtype in ('other', 'sp500_ath', 'performance_rank'):
        return None

    res_date = pd.Timestamp(res_date)

    # Get close price on resolution date (or nearest prior trading day)
    close_col = f'{ticker}_Close' if f'{ticker}_Close' in stock_prices.columns else ticker
    if close_col not in stock_prices.columns:
        return None

    # Find the closest trading day on or before resolution date
    valid_dates = stock_prices.index[stock_prices.index <= res_date]
    if len(valid_dates) == 0:
        return None
    actual_date = valid_dates[-1]

    # Don't resolve if actual trading date is more than 3 days from resolution date
    if (res_date - actual_date).days > 3:
        return None

    close_price = stock_prices.loc[actual_date, close_col]

    if mtype == 'daily_updown':
        # Close > previous close?
        prev_dates = stock_prices.index[stock_prices.index < actual_date]
        if len(prev_dates) == 0:
            return None
        prev_close = stock_prices.loc[prev_dates[-1], close_col]
        # "Up" = close > prev_close (question asks "Up or Down?" — Yes = Up)
        return bool(close_price > prev_close)

    if mtype in ('daily_close_above', 'weekly_above', 'monthly_above'):
        if strike is None:
            return None
        return bool(close_price > strike)

    if mtype in ('daily_close_range', 'weekly_range', 'yearly_range'):
        if strike_low is None or strike_high is None:
            return None
        return bool(strike_low <= close_price <= strike_high)

    if mtype == 'monthly_hit':
        # Did the stock hit the price at any point during the month?
        if strike is None:
            return None
        high_col = f'{ticker}_High' if f'{ticker}_High' in stock_prices.columns else None
        if high_col is None:
            return bool(close_price >= strike)
        month_start = res_date.replace(day=1)
        mask = (stock_prices.index >= month_start) & (stock_prices.index <= res_date)
        month_highs = stock_prices.loc[mask, high_col]
        if len(month_highs) == 0:
            return None
        return bool(month_highs.max() >= strike)

    if mtype == 'monthly_dip':
        # Did the stock dip to (close at or below) the target at any point?
        if strike is None:
            return None
        low_col = f'{ticker}_Low' if f'{ticker}_Low' in stock_prices.columns else None
        if low_col is None:
            return bool(close_price <= strike)
        month_start = res_date.replace(day=1)
        mask = (stock_prices.index >= month_start) & (stock_prices.index <= res_date)
        month_lows = stock_prices.loc[mask, low_col]
        if len(month_lows) == 0:
            return None
        return bool(month_lows.min() <= strike)

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def brier_score(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Compute Brier score: mean squared error between predicted probs and outcomes."""
    return float(np.mean((predicted - observed) ** 2))


def brier_decomposition(predicted: np.ndarray, observed: np.ndarray,
                        n_bins: int = 10) -> dict:
    """Decompose Brier score into reliability, resolution, and uncertainty.

    Returns dict with: brier, reliability, resolution, uncertainty, bin_details
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_details = []

    reliability = 0.0
    resolution = 0.0
    base_rate = observed.mean()
    n = len(predicted)

    for i in range(n_bins):
        mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = mask | (predicted == bin_edges[i + 1])
        n_k = mask.sum()
        if n_k == 0:
            continue

        pred_k = predicted[mask].mean()
        obs_k = observed[mask].mean()

        reliability += n_k * (pred_k - obs_k) ** 2
        resolution += n_k * (obs_k - base_rate) ** 2

        bin_details.append({
            'bin_center': (bin_edges[i] + bin_edges[i + 1]) / 2,
            'bin_low': bin_edges[i],
            'bin_high': bin_edges[i + 1],
            'n': int(n_k),
            'mean_predicted': float(pred_k),
            'mean_observed': float(obs_k),
        })

    reliability /= n
    resolution /= n
    uncertainty = base_rate * (1 - base_rate)

    return {
        'brier': float(brier_score(predicted, observed)),
        'reliability': float(reliability),
        'resolution': float(resolution),
        'uncertainty': float(uncertainty),
        'bin_details': bin_details,
    }


def expected_calibration_error(predicted: np.ndarray, observed: np.ndarray,
                               n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(predicted)

    for i in range(n_bins):
        mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (predicted == bin_edges[i + 1])
        n_k = mask.sum()
        if n_k == 0:
            continue
        pred_k = predicted[mask].mean()
        obs_k = observed[mask].mean()
        ece += (n_k / n) * abs(pred_k - obs_k)

    return float(ece)


def log_loss(predicted: np.ndarray, observed: np.ndarray, eps: float = 1e-15) -> float:
    """Compute log-loss (binary cross-entropy)."""
    p = np.clip(predicted, eps, 1 - eps)
    return float(-np.mean(observed * np.log(p) + (1 - observed) * np.log(1 - p)))
