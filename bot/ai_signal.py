"""
bot/ai_signal.py
────────────────
Bridges LSTM price prediction → trading signal → order placement.

Signal logic:
    predicted > current + BUY_THRESHOLD  → BUY  (MARKET order)
    predicted < current - SELL_THRESHOLD → SELL (MARKET order)
    otherwise                            → HOLD (no order)
"""

import logging
import requests
from bot.client import BinanceClient
from bot.orders import place_order

logger = logging.getLogger(__name__)

BASE_URL       = "https://testnet.binancefuture.com"
BUY_THRESHOLD  = 0.005   # 0.5% predicted upside  → BUY
SELL_THRESHOLD = 0.005   # 0.5% predicted downside → SELL
TRADE_QUANTITY = 0.001   # BTC quantity per trade (adjust per symbol)


def get_current_price(symbol: str) -> float:
    """Fetch current mark price from Binance Futures Testnet ticker."""
    url  = f"{BASE_URL}/fapi/v1/ticker/price"
    resp = requests.get(url, params={"symbol": symbol.upper()}, timeout=5)
    resp.raise_for_status()
    price = float(resp.json()["price"])
    logger.info(f"Current price {symbol}: {price:.2f} USDT")
    return price


def generate_signal(predicted_price: float, current_price: float) -> str:
    """
    Compare LSTM prediction vs live price and return BUY / SELL / HOLD.
    """
    change_pct = (predicted_price - current_price) / current_price

    if change_pct > BUY_THRESHOLD:
        logger.info(
            f"BUY signal | predicted={predicted_price:.2f} "
            f"current={current_price:.2f} change={change_pct*100:+.3f}%"
        )
        return "BUY"
    elif change_pct < -SELL_THRESHOLD:
        logger.info(
            f"SELL signal | predicted={predicted_price:.2f} "
            f"current={current_price:.2f} change={change_pct*100:+.3f}%"
        )
        return "SELL"
    else:
        logger.info(f"HOLD | change={change_pct*100:+.3f}% within threshold")
        return "HOLD"


def execute_signal(
    signal: str,
    symbol: str,
    client: BinanceClient,
    quantity: float = TRADE_QUANTITY,
) -> None:
    """Place MARKET order on BUY/SELL signal; skip on HOLD."""
    if signal == "HOLD":
        print("  ⏸  HOLD — no order placed.\n")
        return

    side = "BUY" if signal == "BUY" else "SELL"
    place_order(
        client=client,
        symbol=symbol,
        side=side,
        order_type="MARKET",
        quantity=quantity,
    )
