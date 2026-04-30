# 🤖 Binance Futures Testnet Trading Bot (AI-Powered)

A clean, modular, production-quality **CLI + AI trading bot** for the
**Binance USDT-M Futures Testnet** built in Python.

Supports manual order placement via CLI **and** fully autonomous trading
powered by an **LSTM neural network** that predicts the next candle's
closing price and generates BUY / SELL / HOLD signals automatically.

---

## 📁 Project Structure

```
trading_bot/
├── bot/
│   ├── __init__.py           # Package marker
│   ├── client.py             # BinanceClient – HMAC-SHA256 auth & HTTP
│   ├── orders.py             # Order placement logic & output formatting
│   ├── validators.py         # Input validation for all CLI parameters
│   ├── logging_config.py     # RotatingFileHandler + console logging
│   ├── data_fetcher.py       # 🆕 AI: Fetch OHLCV from /fapi/v1/klines
│   ├── feature_engineer.py   # 🆕 AI: RSI, MACD, Bollinger, EMA, ATR
│   ├── lstm_model.py         # 🆕 AI: LSTM architecture, train, predict
│   └── ai_signal.py          # 🆕 AI: Signal generator → place_order()
├── models/
│   └── lstm_model.keras      # Auto-created after training
├── logs/
│   └── trading_bot.log       # Rotating log (auto-created on first run)
├── cli.py                    # Manual CLI entry point
├── ai_trader.py              # 🆕 Autonomous AI trading loop
├── .env                      # Your API credentials (NOT committed to Git)
├── .env.example              # Credentials template
├── requirements.txt          # All dependencies (core + AI)
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/dhruv-kumar/binance-future-testnet
cd binance-future-testnet/trading_bot
```

### 2. Create virtual environment & install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Get Binance Futures Testnet API keys

1. Visit **https://testnet.binancefuture.com**
2. Register / login with your **GitHub account**
3. Go to **API Key** tab → click **Generate Key**
4. Copy your **API Key** and **Secret Key**

### 4. Configure `.env`

```bash
cp .env.example .env
```

Edit `.env`:

```env
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

---

## 🖥️ Manual CLI Usage

Place orders manually by specifying all parameters:

```bash
# Market BUY — execute immediately at market price
python cli.py --symbol BTCUSDT --side BUY --type MARKET --quantity 0.001

# Limit SELL — place in order book at specified price
python cli.py --symbol BTCUSDT --side SELL --type LIMIT --quantity 0.001 --price 86000

# Stop SELL — triggers at stop price, executes at limit price
python cli.py --symbol BTCUSDT --side SELL --type STOP \
              --quantity 0.001 --price 68000 --stop-price 68500

# Help
python cli.py --help
```

### CLI Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--symbol` | ✅ Yes | Trading pair e.g. `BTCUSDT`, `ETHUSDT` |
| `--side` | ✅ Yes | `BUY` or `SELL` |
| `--type` | ✅ Yes | `MARKET`, `LIMIT`, or `STOP` |
| `--quantity` | ✅ Yes | Order quantity e.g. `0.001` |
| `--price` | ⚠️ LIMIT/STOP | Limit price e.g. `86000` |
| `--stop-price` | ⚠️ STOP only | Stop trigger price e.g. `68500` |

---

## 🧠 AI / LSTM Autonomous Trading

The AI mode uses a trained **LSTM neural network** to automatically:
1. Fetch 800 hourly candles from Binance
2. Compute 22 technical indicator features (RSI, MACD, Bollinger Bands, EMA, ATR)
3. Predict the next candle's closing price
4. Generate a **BUY / SELL / HOLD** signal
5. Place a MARKET order automatically — no human intervention needed

### Step 1 — Train the model (run once)

```bash
python ai_trader.py --train
```

```
[Step 1/3] Fetching 800 x 1h candles for BTCUSDT...
           ✅ 800 candles fetched
[Step 2/3] Computing 22 technical indicator features...
           ✅ Feature matrix: 592 rows x 22 features
[Step 3/3] Training LSTM neural network...
Epoch 1/50  | loss: 0.0124 | val_loss: 0.0089
...
Epoch 28/50 | Early stopping triggered (best val_loss: 0.00147)
✅  Training complete! Model saved → models/lstm_model.keras
```

### Step 2 — Run the live AI trading loop

```bash
python ai_trader.py --live
```

```
── Cycle #1 ────────────────────────────────────────
[1/5] Fetching market data...
[2/5] Computing features...
[3/5] Running LSTM prediction...
[4/5] Fetching current price...

  📊  Current Price  :  $84,321.50 USDT
  🔮  Predicted Next :  $85,104.30 USDT
  📈  Change         :       +0.928%

[5/5] Generating signal & placing order...
  🚦  Signal : BUY

--- Order Request Summary ---
  Symbol     : BTCUSDT
  Side       : BUY
  Type       : MARKET
  Quantity   : 0.001
-----------------------------

--- Order Response ---
  Order ID     : 3991245
  Status       : FILLED
  Executed Qty : 0.001
  Avg Price    : 84329.10
----------------------
✅ Order placed successfully!

  ⏰  Next cycle in 1h (60 min). Press Ctrl+C to stop.
```

### Advanced options

```bash
# Train on ETHUSDT with 15-minute candles
python ai_trader.py --train --symbol ETHUSDT --interval 15m --limit 1000 --epochs 100

# Live trade ETHUSDT with 0.01 ETH per trade
python ai_trader.py --live --symbol ETHUSDT --interval 15m --quantity 0.01
```

---

## 🤖 How AI Works in This Model

```
Binance API (/fapi/v1/klines)
        │
        ▼
  Raw OHLCV Data (500 candles × 5 columns)
        │   data_fetcher.py
        ▼
  22 Feature Engineering
  RSI · MACD · Bollinger · EMA(9/21/50/200) · ATR · Volume
        │   feature_engineer.py + MinMaxScaler
        ▼
  LSTM Neural Network
  Input  : last 60 candles × 22 features
  Layer 1: LSTM(128) → Dropout → BatchNorm
  Layer 2: LSTM(64)  → Dropout → BatchNorm
  Output : predicted next-candle USDT close price
        │   lstm_model.py
        ▼
  Signal Generator
  predicted > current + 0.5% → BUY
  predicted < current - 0.5% → SELL
  within threshold            → HOLD
        │   ai_signal.py
        ▼
  place_order() → MARKET order on Binance Futures Testnet
        │   orders.py → client.py (HMAC-SHA256 signed)
        ▼
  Sleep 1h → repeat cycle  [ai_trader.py loop]
```

---

## 📋 Order Types Supported

| Type | Description | Required Params |
|------|-------------|-----------------|
| `MARKET` | Execute immediately at market price | symbol, side, quantity |
| `LIMIT` | Place in order book at specified price | + price |
| `STOP` | Trigger at stop price → limit at price | + price, stop-price |

---

## 📝 Logging

All API interactions are logged to `logs/trading_bot.log`:
- **File**: DEBUG level and above (every request, response, prediction)
- **Console**: WARNING level and above only (clean terminal output)
- **Rotation**: Max 5 MB per file, 3 backup files kept automatically

---

## 🔐 Security

- API credentials stored in `.env` file (never hardcoded in source)
- `.env` is listed in `.gitignore` — never committed to Git
- All signed API requests use **HMAC-SHA256** authentication
- Testnet keys are separate from production keys

---

## 🚀 Future Scope

- [ ] WebSocket real-time price streaming
- [ ] Reinforcement Learning (PPO) strategy
- [ ] Streamlit web dashboard
- [ ] Stop-loss / take-profit position management
- [ ] Multi-symbol parallel trading
- [ ] Cloud deployment (AWS / GCP / Heroku)

---

## 👨‍💻 Developer

**Dhruv Kumar**
B.Tech Computer Science & Engineering (AIML)
St. Andrews Institute of Technology & Management, Gurugram
Roll No.: 237012 | Academic Year: 2025–2026

---

## 📄 License

MIT License — free to use, modify, and distribute.
