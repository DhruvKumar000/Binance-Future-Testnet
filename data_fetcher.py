"""
bot/data_fetcher.py
───────────────────
Fetches historical OHLCV (Open, High, Low, Close, Volume) candlestick data
from the Binance Futures Testnet REST API.

Endpoint : GET /fapi/v1/klines  (unsigned — no auth required)
Returns  : pandas DataFrame indexed by open_time
"""

import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://testnet.binancefuture.com"

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def fetch_ohlcv(symbol: str, interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    """
    Fetch historical candlestick data from Binance Futures Testnet.

    Args:
        symbol   : Trading pair  e.g. 'BTCUSDT'
        interval : Candle size   e.g. '1m','5m','15m','1h','4h','1d'
        limit    : Number of candles to fetch (max 1500)

    Returns:
        pd.DataFrame with columns: open, high, low, close, volume
        Index: open_time (datetime, UTC)
    """
    url    = f"{BASE_URL}/fapi/v1/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}

    logger.info(f"Fetching {limit} x {interval} candles for {symbol}...")

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        raw  = resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch OHLCV data: {e}")
        raise RuntimeError(f"Data fetch failed: {e}")

    df = pd.DataFrame(raw, columns=KLINE_COLS)

    # Keep only the useful columns and cast to float
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)

    logger.info(f"Fetched {len(df)} candles | {df.index[0]} → {df.index[-1]}")
    return df
