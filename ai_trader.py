"""
ai_trader.py
────────────
Autonomous AI trading loop for Binance Futures Testnet.

Full pipeline every cycle:
  1. Fetch OHLCV  →  data_fetcher.py
  2. Engineer features  →  feature_engineer.py
  3. LSTM predict next close  →  lstm_model.py
  4. Generate BUY/SELL/HOLD signal  →  ai_signal.py
  5. Place MARKET order if signal ≠ HOLD  →  orders.py → client.py

Usage:
  # Step 1 – Train the LSTM model on historical data (run once)
  python ai_trader.py --train

  # Step 2 – Run the live autonomous AI trading loop
  python ai_trader.py --live

  # Optional flags
  python ai_trader.py --train --symbol ETHUSDT --interval 15m --limit 1000
  python ai_trader.py --live  --symbol BTCUSDT --interval 1h
"""

import argparse
import logging
import os
import sys
import time

from dotenv import load_dotenv

from bot.logging_config   import setup_logging
from bot.client           import BinanceClient
from bot.data_fetcher     import fetch_ohlcv
from bot.feature_engineer import build_features
from bot.lstm_model       import train_model, predict_next_close
from bot.ai_signal        import get_current_price, generate_signal, execute_signal

logger = logging.getLogger(__name__)

# ── Default config (overridable via CLI flags) ────────────────────────────────
DEFAULT_SYMBOL   = "BTCUSDT"
DEFAULT_INTERVAL = "1h"
DEFAULT_LIMIT    = 800
DEFAULT_QUANTITY = 0.001
SLEEP_SECONDS    = 3600   # 1 hour between prediction cycles (matches 1h candles)


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="🤖 Binance Futures Testnet – AI-Powered LSTM Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--train", action="store_true",
        help="Fetch historical data and train the LSTM model",
    )
    mode.add_argument(
        "--live", action="store_true",
        help="Run the autonomous AI trading loop (requires trained model)",
    )
    parser.add_argument("--symbol",   default=DEFAULT_SYMBOL,   help="Trading pair (default: BTCUSDT)")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="Candle interval (default: 1h)")
    parser.add_argument("--limit",    default=DEFAULT_LIMIT,    type=int, help="Candles to fetch (default: 800)")
    parser.add_argument("--quantity", default=DEFAULT_QUANTITY, type=float, help="Trade quantity (default: 0.001)")
    parser.add_argument("--epochs",   default=50,               type=int, help="Training epochs (default: 50)")
    return parser.parse_args()


# ── Training mode ─────────────────────────────────────────────────────────────

def run_training(symbol: str, interval: str, limit: int, epochs: int) -> None:
    """Fetch historical OHLCV data and train the LSTM model."""
    print(f"\n{'='*55}")
    print(f"  🧠  LSTM Training Mode")
    print(f"  Symbol   : {symbol}")
    print(f"  Interval : {interval}")
    print(f"  Candles  : {limit}")
    print(f"  Epochs   : {epochs}")
    print(f"{'='*55}\n")

    # 1. Fetch data
    print(f"[Step 1/3] Fetching {limit} x {interval} candles for {symbol}...")
    df = fetch_ohlcv(symbol, interval, limit)
    print(f"           ✅ {len(df)} candles fetched ({df.index[0]} → {df.index[-1]})")

    # 2. Engineer features
    print(f"\n[Step 2/3] Computing 22 technical indicator features...")
    X_scaled, _, _, df_feat = build_features(df)
    print(f"           ✅ Feature matrix: {X_scaled.shape[0]} rows x {X_scaled.shape[1]} features")

    # 3. Train LSTM
    print(f"\n[Step 3/3] Training LSTM neural network...\n")
    train_model(X_scaled, epochs=epochs)

    print(f"\n{'='*55}")
    print(f"  ✅  Training complete!")
    print(f"  Model saved → models/lstm_model.keras")
    print(f"  Run 'python ai_trader.py --live' to start trading.")
    print(f"{'='*55}\n")


# ── Live trading mode ─────────────────────────────────────────────────────────

def run_live(client: BinanceClient, symbol: str, interval: str,
             limit: int, quantity: float) -> None:
    """Autonomous AI trading loop — runs indefinitely."""
    print(f"\n{'='*55}")
    print(f"  🤖  AI Live Trading Mode")
    print(f"  Symbol   : {symbol}")
    print(f"  Interval : {interval}")
    print(f"  Quantity : {quantity} per trade")
    print(f"  Cycle    : every {SLEEP_SECONDS // 60} minutes")
    print(f"{'='*55}\n")

    cycle = 0
    while True:
        cycle += 1
        print(f"\n── Cycle #{cycle} {'─'*40}")

        try:
            # Step 1: Fetch latest OHLCV data
            print("[1/5] Fetching market data...")
            df = fetch_ohlcv(symbol, interval, limit)

            # Step 2: Engineer features
            print("[2/5] Computing features...")
            X_scaled, _, close_scaler, _ = build_features(df)

            # Step 3: LSTM prediction
            print("[3/5] Running LSTM prediction...")
            predicted = predict_next_close(X_scaled, close_scaler)

            # Step 4: Get current live price
            print("[4/5] Fetching current price...")
            current = get_current_price(symbol)

            change_pct = (predicted - current) / current * 100
            print(f"\n  📊  Current Price  : ${current:>12,.2f} USDT")
            print(f"  🔮  Predicted Next : ${predicted:>12,.2f} USDT")
            print(f"  📈  Change         : {change_pct:>+.3f}%")

            # Step 5: Signal & order
            print("[5/5] Generating signal & placing order...")
            signal = generate_signal(predicted, current)
            print(f"\n  🚦  Signal : {signal}")
            execute_signal(signal, symbol, client, quantity)

        except RuntimeError as e:
            print(f"\n  ❌  Runtime error: {e}")
            logger.error(f"Cycle #{cycle} error: {e}")

        except Exception as e:
            print(f"\n  ❌  Unexpected error: {e}")
            logger.exception(f"Cycle #{cycle} unexpected error")

        print(f"\n  ⏰  Next cycle in {SLEEP_SECONDS // 3600}h "
              f"({SLEEP_SECONDS // 60} min). Press Ctrl+C to stop.")
        time.sleep(SLEEP_SECONDS)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging()
    load_dotenv()

    args = parse_args()

    if args.train:
        run_training(
            symbol   = args.symbol.upper(),
            interval = args.interval,
            limit    = args.limit,
            epochs   = args.epochs,
        )

    elif args.live:
        api_key    = os.getenv("BINANCE_API_KEY",    "").strip()
        api_secret = os.getenv("BINANCE_API_SECRET", "").strip()

        if not api_key or not api_secret:
            print("❌  Error: BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env")
            sys.exit(1)

        client = BinanceClient(api_key=api_key, api_secret=api_secret)
        try:
            run_live(
                client   = client,
                symbol   = args.symbol.upper(),
                interval = args.interval,
                limit    = args.limit,
                quantity = args.quantity,
            )
        except KeyboardInterrupt:
            print("\n\n  👋  AI Trader stopped by user. Goodbye!\n")
            sys.exit(0)


if __name__ == "__main__":
    main()
