# config.py — Configuración centralizada de la tesis
# Copia este archivo a tu carpeta notebooks_corregidos/ como config.py

# ── Ventana de datos ──────────────────────────────────────────────────────────
DATA_START = '2024-01-01'
DATA_END   = '2026-02-28'   # Último cierre verificado de Yahoo Finance

# ── Tickers analizados ────────────────────────────────────────────────────────
CASE_A_TICKERS = ['AAPL', 'NVDA', 'AMZN']

# ── APIs ──────────────────────────────────────────────────────────────────────
# SerpApi: pon tu clave aquí o en la variable de entorno SERPAPI_KEY
# No subas este archivo con la clave real a un repositorio público.
SERPAPI_KEY = ''   # o lee de os.environ.get('SERPAPI_KEY', '')

GAMMA_BASE = 'https://gamma-api.polymarket.com'
CLOB_BASE  = 'https://clob.polymarket.com'
TAGS       = {'stocks': 604, 'politics': 2, 'sports': 100639}

# ── Polymarket ────────────────────────────────────────────────────────────────
EVENT_ALIGN_TOL_DAYS       = 45
TOP_EVENT_ALIGNED_MARKETS  = 25
