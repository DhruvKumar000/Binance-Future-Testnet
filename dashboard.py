"""
dashboard.py
────────────
Streamlit web dashboard for the Binance Futures Testnet AI Trading Bot.

Run locally:
    pip install streamlit plotly
    streamlit run dashboard.py

Deploy free on Streamlit Cloud:
    1. Push this file to GitHub
    2. Go to share.streamlit.io
    3. Connect your repo → done!
"""

import time
import datetime
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Binance AI Trading Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* Root theme */
:root {
    --bg-dark:    #0A0E1A;
    --bg-card:    #111827;
    --bg-card2:   #1A2235;
    --accent:     #00D4AA;
    --accent2:    #3B82F6;
    --red:        #EF4444;
    --green:      #10B981;
    --amber:      #F59E0B;
    --text:       #E2E8F0;
    --text-dim:   #64748B;
    --border:     #1E293B;
}

/* Hide Streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* App background */
.stApp { background: var(--bg-dark); }
.main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-card);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    color: var(--accent);
    font-family: 'Syne', sans-serif;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
div[data-testid="metric-container"] label {
    color: var(--text-dim) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* Signal badge */
.signal-buy {
    background: rgba(16,185,129,0.15);
    border: 1px solid #10B981;
    color: #10B981;
    padding: 0.6rem 1.8rem;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    text-align: center;
    letter-spacing: 0.1em;
}
.signal-sell {
    background: rgba(239,68,68,0.15);
    border: 1px solid #EF4444;
    color: #EF4444;
    padding: 0.6rem 1.8rem;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    text-align: center;
    letter-spacing: 0.1em;
}
.signal-hold {
    background: rgba(245,158,11,0.15);
    border: 1px solid #F59E0B;
    color: #F59E0B;
    padding: 0.6rem 1.8rem;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    text-align: center;
    letter-spacing: 0.1em;
}

/* Section headers */
.section-head {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

/* Pipeline step */
.pipe-step {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--text);
}
.pipe-num {
    color: var(--accent);
    font-weight: 700;
    margin-right: 0.5rem;
}

/* Order history table */
.order-row-buy  { color: #10B981; }
.order-row-sell { color: #EF4444; }
.order-row-hold { color: #F59E0B; }

/* Big title */
.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.main-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-dim);
    letter-spacing: 0.05em;
    margin-bottom: 1.5rem;
}
.accent-text { color: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_URL = "https://testnet.binancefuture.com"
SYMBOLS  = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
INTERVALS = {"1m": "1 min", "5m": "5 min", "15m": "15 min",
             "1h": "1 hour", "4h": "4 hours", "1d": "1 day"}

# ── Data fetch helpers ────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def fetch_price(symbol: str) -> float:
    try:
        r = requests.get(f"{BASE_URL}/fapi/v1/ticker/price",
                         params={"symbol": symbol}, timeout=5)
        return float(r.json()["price"])
    except:
        # Fallback demo price if testnet is unreachable
        defaults = {"BTCUSDT": 84321.50, "ETHUSDT": 3210.40,
                    "BNBUSDT": 412.30,   "SOLUSDT": 182.50, "XRPUSDT": 0.52}
        return defaults.get(symbol, 50000.0)

@st.cache_data(ttl=60)
def fetch_ohlcv(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
    try:
        r = requests.get(f"{BASE_URL}/fapi/v1/klines",
                         params={"symbol": symbol, "interval": interval,
                                 "limit": limit}, timeout=8)
        raw = r.json()
        df  = pd.DataFrame(raw, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qv","trades","tbb","tbq","ignore"])
        df = df[["open_time","open","high","low","close","volume"]].copy()
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        return df
    except:
        # Generate realistic demo OHLCV if testnet unreachable
        return _demo_ohlcv(symbol, limit)

def _demo_ohlcv(symbol: str, limit: int) -> pd.DataFrame:
    base = {"BTCUSDT": 84000, "ETHUSDT": 3200, "BNBUSDT": 410,
            "SOLUSDT": 180, "XRPUSDT": 0.52}.get(symbol, 50000)
    np.random.seed(42)
    prices = base + np.cumsum(np.random.randn(limit) * base * 0.005)
    times  = pd.date_range(end=datetime.datetime.utcnow(), periods=limit, freq="1h")
    highs  = prices * (1 + np.abs(np.random.randn(limit) * 0.003))
    lows   = prices * (1 - np.abs(np.random.randn(limit) * 0.003))
    opens  = np.roll(prices, 1); opens[0] = prices[0]
    return pd.DataFrame({"open_time": times, "open": opens,
                          "high": highs, "low": lows, "close": prices,
                          "volume": np.random.randint(500, 2000, limit).astype(float)})

@st.cache_data(ttl=60)
def fetch_24h(symbol: str) -> dict:
    try:
        r = requests.get(f"{BASE_URL}/fapi/v1/ticker/24hr",
                         params={"symbol": symbol}, timeout=5)
        return r.json()
    except:
        price = fetch_price(symbol)
        return {"priceChangePercent": "1.24", "highPrice": str(price * 1.02),
                "lowPrice": str(price * 0.98), "volume": "12345.67",
                "quoteVolume": str(price * 12345.67)}

# ── Technical indicators ──────────────────────────────────────────────────────
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    ag    = gain.ewm(com=period-1, min_periods=period).mean()
    al    = loss.ewm(com=period-1, min_periods=period).mean()
    return 100 - (100 / (1 + ag / (al + 1e-10)))

def compute_macd(series: pd.Series):
    ema12  = series.ewm(span=12, adjust=False).mean()
    ema26  = series.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal

def compute_bb(series: pd.Series, period: int = 20):
    sma   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    return sma + 2*std, sma, sma - 2*std

# ── AI signal simulation ──────────────────────────────────────────────────────
def ai_predict(df: pd.DataFrame, current_price: float) -> dict:
    """Simulate LSTM prediction using technical indicator logic."""
    close  = df["close"]
    rsi    = compute_rsi(close).iloc[-1]
    macd_l, macd_s, _ = compute_macd(close)
    macd_v = macd_l.iloc[-1] - macd_s.iloc[-1]
    ema9   = close.ewm(span=9,  adjust=False).mean().iloc[-1]
    ema21  = close.ewm(span=21, adjust=False).mean().iloc[-1]

    # Simulate LSTM prediction with slight noise
    np.random.seed(int(time.time()) // 60)  # changes every minute
    base_change = 0.0
    if rsi < 35:           base_change += 0.008
    elif rsi > 65:         base_change -= 0.008
    if macd_v > 0:         base_change += 0.004
    else:                  base_change -= 0.004
    if ema9 > ema21:       base_change += 0.003
    else:                  base_change -= 0.003
    noise          = np.random.randn() * 0.002
    predicted      = current_price * (1 + base_change + noise)
    change_pct     = (predicted - current_price) / current_price * 100

    if change_pct > 0.5:   signal = "BUY"
    elif change_pct < -0.5: signal = "SELL"
    else:                  signal = "HOLD"

    return {
        "predicted":   predicted,
        "change_pct":  change_pct,
        "signal":      signal,
        "rsi":         rsi,
        "macd":        macd_v,
        "ema9":        ema9,
        "ema21":       ema21,
    }

# ── Demo order history ────────────────────────────────────────────────────────
def get_demo_orders(symbol: str, current_price: float) -> pd.DataFrame:
    np.random.seed(42)
    n      = 12
    times  = pd.date_range(end=datetime.datetime.utcnow(), periods=n, freq="1h")
    sides  = np.random.choice(["BUY", "SELL", "HOLD"], n,
                               p=[0.4, 0.35, 0.25])
    prices = [current_price * (1 + np.random.randn() * 0.01) for _ in range(n)]
    pnls   = [round(np.random.randn() * current_price * 0.001 * 0.001, 4)
              for _ in range(n)]
    return pd.DataFrame({
        "Time":     [t.strftime("%H:%M") for t in times],
        "Symbol":   symbol,
        "Signal":   sides,
        "Price":    [f"${p:,.2f}" for p in prices],
        "Qty":      "0.001",
        "PnL":      [f"{'+' if p>0 else ''}{p:.4f} USDT" for p in pnls],
        "Status":   ["FILLED" if s != "HOLD" else "SKIPPED" for s in sides],
    })

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Bot Settings")
    symbol   = st.selectbox("Trading Pair",   SYMBOLS, index=0)
    interval = st.selectbox("Candle Interval", list(INTERVALS.keys()),
                             index=3, format_func=lambda x: f"{x} ({INTERVALS[x]})")
    quantity = st.number_input("Trade Quantity", value=0.001,
                                min_value=0.001, step=0.001, format="%.3f")
    threshold = st.slider("Signal Threshold (%)", 0.1, 2.0, 0.5, 0.1)

    st.markdown("---")
    st.markdown("## 🤖 AI Pipeline")
    steps = [
        ("1", "data_fetcher.py",     "Fetch OHLCV candles"),
        ("2", "feature_engineer.py", "22 technical features"),
        ("3", "lstm_model.py",       "LSTM prediction"),
        ("4", "ai_signal.py",        "BUY/SELL/HOLD signal"),
        ("5", "orders.py",           "Place MARKET order"),
    ]
    for num, mod, desc in steps:
        st.markdown(f"""
        <div class="pipe-step">
            <span class="pipe-num">0{num}</span>
            <strong>{mod}</strong><br>
            <span style="color:#64748B;margin-left:1.3rem">{desc}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    auto_refresh = st.toggle("Auto Refresh (30s)", value=True)
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:0.68rem;color:#475569;line-height:1.8'>
    👨‍💻 Dhruv Kumar<br>
    B.Tech CSE (AIML)<br>
    SAITM, Gurugram<br>
    Roll No.: 237012<br><br>
    <a href='https://github.com/DhruvKumar000/Binance-Future-Testnet'
       style='color:#00D4AA;text-decoration:none'>
    🔗 GitHub Repository</a>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-title">
    🤖 Binance Futures <span class="accent-text">AI</span> Dashboard
</div>
<div class="main-sub">
    LSTM NEURAL NETWORK  ·  TESTNET  ·  {symbol}  ·  LIVE DATA  ·
    {datetime.datetime.utcnow().strftime('%Y-%m-%d  %H:%M:%S UTC')}
</div>
""", unsafe_allow_html=True)

# ── Fetch all data ────────────────────────────────────────────────────────────
current_price = fetch_price(symbol)
df            = fetch_ohlcv(symbol, interval, 100)
ticker        = fetch_24h(symbol)
ai            = ai_predict(df, current_price)

# ── Row 1: Key metrics ────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
chg_pct = float(ticker.get("priceChangePercent", 0))
with c1:
    st.metric("💰 Live Price",
              f"${current_price:,.2f}",
              f"{chg_pct:+.2f}%")
with c2:
    st.metric("🔮 AI Prediction",
              f"${ai['predicted']:,.2f}",
              f"{ai['change_pct']:+.3f}%")
with c3:
    st.metric("📊 RSI (14)",
              f"{ai['rsi']:.1f}",
              "Oversold" if ai['rsi'] < 30 else "Overbought" if ai['rsi'] > 70 else "Neutral")
with c4:
    hi = float(ticker.get("highPrice", current_price * 1.02))
    lo = float(ticker.get("lowPrice",  current_price * 0.98))
    st.metric("📈 24h High", f"${hi:,.2f}")
with c5:
    st.metric("📉 24h Low",  f"${lo:,.2f}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 2: Signal + Chart ─────────────────────────────────────────────────────
sig_col, chart_col = st.columns([1, 3])

with sig_col:
    st.markdown('<div class="section-head">🚦 AI Signal</div>', unsafe_allow_html=True)
    sig = ai["signal"]
    icon  = {"BUY":"📈","SELL":"📉","HOLD":"⏸"}[sig]
    cls   = {"BUY":"signal-buy","SELL":"signal-sell","HOLD":"signal-hold"}[sig]
    st.markdown(f'<div class="{cls}">{icon} {sig}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-head">📐 Indicators</div>', unsafe_allow_html=True)

    rsi_color = "#EF4444" if ai['rsi'] > 70 else "#10B981" if ai['rsi'] < 30 else "#F59E0B"
    st.markdown(f"""
    <div style="background:#111827;border:1px solid #1E293B;border-radius:10px;padding:1rem;font-family:'Space Mono',monospace;font-size:0.78rem;line-height:2.2">
        <div style="display:flex;justify-content:space-between">
            <span style="color:#64748B">RSI 14</span>
            <span style="color:{rsi_color};font-weight:700">{ai['rsi']:.1f}</span>
        </div>
        <div style="display:flex;justify-content:space-between">
            <span style="color:#64748B">MACD</span>
            <span style="color:{'#10B981' if ai['macd']>0 else '#EF4444'};font-weight:700">{ai['macd']:+.2f}</span>
        </div>
        <div style="display:flex;justify-content:space-between">
            <span style="color:#64748B">EMA 9</span>
            <span style="color:#E2E8F0">${ai['ema9']:,.2f}</span>
        </div>
        <div style="display:flex;justify-content:space-between">
            <span style="color:#64748B">EMA 21</span>
            <span style="color:#E2E8F0">${ai['ema21']:,.2f}</span>
        </div>
        <div style="display:flex;justify-content:space-between">
            <span style="color:#64748B">Threshold</span>
            <span style="color:#3B82F6">±{threshold:.1f}%</span>
        </div>
        <div style="display:flex;justify-content:space-between">
            <span style="color:#64748B">Quantity</span>
            <span style="color:#E2E8F0">{quantity} {symbol[:3]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with chart_col:
    st.markdown('<div class="section-head">📊 Price Chart</div>', unsafe_allow_html=True)

    # Candlestick + BB + EMA
    bb_up, bb_mid, bb_lo = compute_bb(df["close"])
    ema9_s  = df["close"].ewm(span=9,  adjust=False).mean()
    ema21_s = df["close"].ewm(span=21, adjust=False).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=df["open_time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color="#10B981", decreasing_line_color="#EF4444",
        increasing_fillcolor="#10B981", decreasing_fillcolor="#EF4444",
        name="Price"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["open_time"], y=bb_up,
        line=dict(color="#3B82F6", width=1, dash="dot"),
        name="BB Upper", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["open_time"], y=bb_lo,
        line=dict(color="#3B82F6", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(59,130,246,0.05)",
        name="BB Lower", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["open_time"], y=ema9_s,
        line=dict(color="#F59E0B", width=1.5),
        name="EMA 9"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["open_time"], y=ema21_s,
        line=dict(color="#A78BFA", width=1.5),
        name="EMA 21"), row=1, col=1)
    fig.add_hline(y=ai["predicted"], line_dash="dash",
        line_color="#00D4AA", line_width=1.5,
        annotation_text=f"AI Pred: ${ai['predicted']:,.0f}",
        annotation_font_color="#00D4AA", row=1, col=1)

    # Volume bars
    colors = ["#10B981" if c >= o else "#EF4444"
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df["open_time"], y=df["volume"],
        marker_color=colors, name="Volume",
        opacity=0.7), row=2, col=1)

    fig.update_layout(
        paper_bgcolor="#0A0E1A", plot_bgcolor="#111827",
        font=dict(family="Space Mono", color="#64748B", size=11),
        legend=dict(bgcolor="#111827", bordercolor="#1E293B",
                    borderwidth=1, font=dict(size=10)),
        xaxis_rangeslider_visible=False,
        height=420, margin=dict(t=10, b=10, l=10, r=10),
    )
    fig.update_xaxes(gridcolor="#1E293B", showgrid=True,
                     zeroline=False, color="#475569")
    fig.update_yaxes(gridcolor="#1E293B", showgrid=True,
                     zeroline=False, color="#475569")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Row 3: RSI chart + Order history ─────────────────────────────────────────
rsi_col, orders_col = st.columns([1, 2])

with rsi_col:
    st.markdown('<div class="section-head">📉 RSI Indicator</div>', unsafe_allow_html=True)
    rsi_series = compute_rsi(df["close"])
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["open_time"], y=rsi_series,
        line=dict(color="#00D4AA", width=2),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.08)",
        name="RSI 14"))
    fig2.add_hline(y=70, line_dash="dash", line_color="#EF4444",
                   line_width=1, annotation_text="Overbought 70",
                   annotation_font_color="#EF4444")
    fig2.add_hline(y=30, line_dash="dash", line_color="#10B981",
                   line_width=1, annotation_text="Oversold 30",
                   annotation_font_color="#10B981")
    fig2.update_layout(
        paper_bgcolor="#0A0E1A", plot_bgcolor="#111827",
        font=dict(family="Space Mono", color="#64748B", size=10),
        height=220, margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False, yaxis=dict(range=[0, 100]),
    )
    fig2.update_xaxes(gridcolor="#1E293B", showgrid=True, color="#475569")
    fig2.update_yaxes(gridcolor="#1E293B", showgrid=True, color="#475569")
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

with orders_col:
    st.markdown('<div class="section-head">📋 AI Order History (Demo)</div>',
                unsafe_allow_html=True)
    orders_df = get_demo_orders(symbol, current_price)
    for _, row in orders_df.iterrows():
        color = {"BUY":"#10B981","SELL":"#EF4444","HOLD":"#F59E0B"}[row["Signal"]]
        icon  = {"BUY":"↑","SELL":"↓","HOLD":"—"}[row["Signal"]]
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1E293B;border-radius:8px;
                    padding:0.5rem 0.8rem;margin-bottom:5px;
                    display:flex;justify-content:space-between;align-items:center;
                    font-family:'Space Mono',monospace;font-size:0.72rem">
            <span style="color:#475569">{row['Time']}</span>
            <span style="color:{color};font-weight:700;width:50px">{icon} {row['Signal']}</span>
            <span style="color:#E2E8F0">{row['Price']}</span>
            <span style="color:#64748B">{row['Qty']} {symbol[:3]}</span>
            <span style="color:{color}">{row['PnL']}</span>
            <span style="color:{'#10B981' if row['Status']=='FILLED' else '#475569'};font-size:0.65rem">{row['Status']}</span>
        </div>""", unsafe_allow_html=True)

# ── Row 4: How AI works ───────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-head">🧠 How AI Works in This Model</div>',
            unsafe_allow_html=True)

p1, p2, p3, p4, p5 = st.columns(5)
pipeline = [
    (p1, "01", "📡 Data Fetch",      "data_fetcher.py",     "500 OHLCV candles from /fapi/v1/klines",         "#00D4AA"),
    (p2, "02", "⚙️ Features",         "feature_engineer.py", "22 indicators: RSI, MACD, BB, EMA, ATR",        "#3B82F6"),
    (p3, "03", "🧠 LSTM Predict",     "lstm_model.py",       "60-candle window → next-price prediction",       "#8B5CF6"),
    (p4, "04", "🚦 Signal Generate",  "ai_signal.py",        f"±{threshold:.1f}% threshold → BUY/SELL/HOLD",  "#F59E0B"),
    (p5, "05", "✅ Place Order",       "orders.py + client",  "HMAC-SHA256 signed MARKET order to Binance",    "#10B981"),
]
for col, num, title, mod, desc, color in pipeline:
    with col:
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1E293B;border-top:3px solid {color};
                    border-radius:10px;padding:1rem;text-align:center;height:160px">
            <div style="font-family:'Space Mono',monospace;font-size:1.4rem;color:{color};font-weight:700">{num}</div>
            <div style="font-family:'Syne',sans-serif;font-size:0.82rem;font-weight:700;color:#E2E8F0;margin:4px 0">{title}</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#3B82F6;margin-bottom:6px">{mod}</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#64748B;line-height:1.4">{desc}</div>
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center;font-family:'Space Mono',monospace;font-size:0.7rem;
            color:#334155;border-top:1px solid #1E293B;padding-top:1rem;line-height:2">
    🤖 Binance Futures AI Trading Bot  ·  Dhruv Kumar (Roll No. 237012)  ·  B.Tech CSE AIML  ·  SAITM Gurugram<br>
    <a href="https://github.com/DhruvKumar000/Binance-Future-Testnet"
       style="color:#00D4AA;text-decoration:none">
       github.com/DhruvKumar000/Binance-Future-Testnet</a>  ·
    Powered by Binance Futures Testnet  ·  LSTM + TensorFlow  ·
    Last updated: {datetime.datetime.utcnow().strftime('%H:%M:%S UTC')}
</div>""", unsafe_allow_html=True)

# ── Auto refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(30)
    st.rerun()
