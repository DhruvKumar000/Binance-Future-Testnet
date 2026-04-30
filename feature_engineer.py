"""
bot/feature_engineer.py
───────────────────────
Transforms raw OHLCV data into a rich set of technical indicator features
that the LSTM model uses to learn market patterns.

Features computed (22 total):
  Price    : close, open, high, low, volume
  Returns  : returns, log_returns
  Trend    : ema_9, ema_21, ema_50, ema_200
  Momentum : rsi, macd, macd_signal, macd_hist
  Volatility: bb_upper, bb_middle, bb_lower, bb_width, atr
  Volume   : volume_change, vol_ma_20

All features are normalized to [0, 1] using MinMaxScaler.
A separate close_scaler is kept to inverse-transform predictions.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "close", "open", "high", "low", "volume",
    "returns", "log_returns",
    "ema_9", "ema_21", "ema_50", "ema_200",
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower", "bb_width",
    "volume_change", "vol_ma_20", "atr",
]


# ── Individual indicator functions ────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (0–100). <30 = oversold, >70 = overbought."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    """MACD line, signal line, and histogram."""
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(series: pd.Series, period=20, std_dev=2):
    """Bollinger Bands: upper, middle (SMA), lower."""
    sma   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — measures market volatility."""
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ── Main feature builder ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> tuple:
    """
    Compute all technical indicator features from an OHLCV DataFrame.

    Returns:
        X_scaled     : np.ndarray  shape (n_samples, 22) — normalized feature matrix
        scaler       : MinMaxScaler fitted on all 22 features
        close_scaler : MinMaxScaler fitted on 'close' only (for inverse-transform)
        df           : pd.DataFrame with all indicator columns added (NaN rows dropped)
    """
    df = df.copy()

    # ── Price returns ─────────────────────────────────────────────────────────
    df["returns"]     = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # ── Trend: Exponential Moving Averages ───────────────────────────────────
    for span in [9, 21, 50, 200]:
        df[f"ema_{span}"] = df["close"].ewm(span=span, adjust=False).mean()

    # ── Momentum: RSI ────────────────────────────────────────────────────────
    df["rsi"] = _rsi(df["close"], 14)

    # ── Momentum: MACD ───────────────────────────────────────────────────────
    macd, sig, hist   = _macd(df["close"])
    df["macd"]        = macd
    df["macd_signal"] = sig
    df["macd_hist"]   = hist

    # ── Volatility: Bollinger Bands ──────────────────────────────────────────
    bb_up, bb_mid, bb_low = _bollinger(df["close"])
    df["bb_upper"]  = bb_up
    df["bb_middle"] = bb_mid
    df["bb_lower"]  = bb_low
    df["bb_width"]  = (bb_up - bb_low) / (bb_mid + 1e-10)

    # ── Volatility: ATR ──────────────────────────────────────────────────────
    df["atr"] = _atr(df, 14)

    # ── Volume indicators ────────────────────────────────────────────────────
    df["volume_change"] = df["volume"].pct_change()
    df["vol_ma_20"]     = df["volume"].rolling(20).mean()

    # ── Drop warm-up NaN rows (first ~200 rows for EMA-200) ─────────────────
    df.dropna(inplace=True)
    logger.info(f"Feature matrix: {len(df)} rows x {len(FEATURE_COLS)} features (after NaN drop)")

    # ── Normalize all features to [0, 1] ─────────────────────────────────────
    scaler       = MinMaxScaler()
    close_scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(df[FEATURE_COLS])
    close_scaler.fit(df[["close"]])

    return X_scaled, scaler, close_scaler, df
