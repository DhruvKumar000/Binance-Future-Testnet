"""
CLI entry point for the Binance Futures Testnet trading bot.

Usage examples:
    # Market BUY
    python cli.py --symbol BTCUSDT --side BUY --type MARKET --quantity 0.001

    # Limit SELL
    python cli.py --symbol BTCUSDT --side SELL --type LIMIT --quantity 0.001 --price 70000

    # Stop order
    python cli.py --symbol BTCUSDT --side SELL --type STOP --quantity 0.001 --price 68000 --stop-price 68500
"""

import argparse
import os
import sys
from dotenv import load_dotenv

from bot.logging_config import setup_logging
from bot.client import BinanceClient
from bot.validators import (
    validate_symbol,
    validate_side,
    validate_order_type,
    validate_quantity,
    validate_price,
)
from bot.orders import place_order


def parse_args():
    parser = argparse.ArgumentParser(
        description="Binance Futures Testnet Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--symbol", required=True, help="Trading pair, e.g. BTCUSDT")
    parser.add_argument("--side", required=True, choices=["BUY", "SELL"], help="BUY or SELL")
    parser.add_argument("--type", dest="order_type", required=True,
                        choices=["MARKET", "LIMIT", "STOP"], help="Order type")
    parser.add_argument("--quantity", required=True, help="Order quantity")
    parser.add_argument("--price", default=None, help="Limit price (required for LIMIT/STOP)")
    parser.add_argument("--stop-price", dest="stop_price", default=None,
                        help="Stop trigger price (required for STOP orders)")
    return parser.parse_args()


def main():
    setup_logging()
    load_dotenv()

    api_key = os.getenv("BINANCE_API_KEY", "").strip()
    api_secret = os.getenv("BINANCE_API_SECRET", "").strip()

    if not api_key or not api_secret:
        print("❌ Error: BINANCE_API_KEY and BINANCE_API_SECRET must be set in your .env file.")
        sys.exit(1)

    args = parse_args()

    # --- Validate inputs ---
    try:
        symbol = validate_symbol(args.symbol)
        side = validate_side(args.side)
        order_type = validate_order_type(args.order_type)
        quantity = validate_quantity(args.quantity)
        price = validate_price(args.price, required=(order_type in ("LIMIT", "STOP")))
        stop_price = validate_price(args.stop_price, required=(order_type == "STOP"))
    except ValueError as e:
        print(f"❌ Validation error: {e}")
        sys.exit(1)

    # --- Execute order ---
    client = BinanceClient(api_key=api_key, api_secret=api_secret)
    try:
        place_order(
            client=client,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
        )
    except RuntimeError:
        sys.exit(1)


if __name__ == "__main__":
    main()
