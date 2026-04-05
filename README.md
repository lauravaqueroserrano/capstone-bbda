# Calibration of Polymarket Equity Prediction Markets

**BBDA Capstone Project - IE University**
**Author:** Laura Vaquero
**Date:** March 2026

## Overview

This project analyzes the calibration quality of Polymarket's equity prediction markets. It examines whether market prices correspond to actual outcome probabilities, and how calibration varies across market structures and liquidity levels.

The analysis covers 9,090 equity prediction contracts across 10 primary tickers (AAPL, NVDA, AMZN, GOOGL, TSLA, MSFT, META, PLTR, NFLX, OPEN) over six months (October 2025 - March 2026).

## Key Findings

- **Binary calibration is strong**: reliability of 0.004, ECE of 0.042
- **Market type matters**: weekly above contracts are best calibrated; weekly range contracts are worst
- **Liquidity improves calibration**: high-volume ECE 0.035 vs low-volume 0.066
- **Distributional calibration is imperfect**: implied distributions underestimate tail probabilities (KS = 0.101, p < 0.001)

## Repository Structure

```
├── 00_data_audit.ipynb           # Initial data exploration
├── 01_data_collection.ipynb      # Polymarket API data collection
├── 02_resolution_matching.ipynb  # Market resolution against stock prices
├── 03_calibration.ipynb          # Binary calibration analysis
├── 04_distributions.ipynb        # Implied distribution analysis (PIT)
├── 07_thesis_output.ipynb        # Generate thesis figures and tables
├── 09_collect_prices.py          # CLOB API price collection
├── 10_calibration_with_prices.py # Final calibration with pre-resolution prices
├── config.py                     # Project configuration
├── config_template.py            # Configuration template
├── thesis_utils.py               # Shared utility functions
├── data/                         # Datasets (parquet format)
└── results/                      # Output figures, tables, and CSVs
    └── thesis_figures/           # Final figures used in the thesis
```

## Data Sources

- **Polymarket Gamma API**: market metadata, resolution outcomes (9,090 markets)
- **Polymarket CLOB API**: historical price data for Yes/No tokens
- **Yahoo Finance**: daily OHLCV stock price data

## Methodology

1. **Data Collection**: markets collected via Gamma API, prices via CLOB API
2. **Market Classification**: regex-based taxonomy into 8 structural types (98.8% extraction rate)
3. **Resolution Matching**: type-specific logic validated against Polymarket settlement data (98.8% agreement)
4. **Price Collection**: pre-resolution implied probabilities from CLOB API (99.2% retrieval)
5. **Calibration Analysis**: Brier score, Murphy-Winkler decomposition, ECE, reliability diagrams
6. **Distributional Analysis**: Probability Integral Transform (PIT) on 478 multi-strike groups

## Requirements

```
pip install -r requirements_tesis_notebooks.txt
```
