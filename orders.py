"""
Order placement logic — wraps client calls and formats results.
"""

import logging
from bot.client import BinanceClient

logger = logging.getLogger(__name__)


def place_order(
    client: BinanceClient,
    symbol: str,
    side: str,
    order_type: str,
    quantity: float,
    price: float = None,
    stop_price: float = None,
) -> dict:
    """
    Place a MARKET, LIMIT, or STOP order on Binance Futures Testnet.
    Returns a formatted result dict.
    """
    print("\n--- Order Request Summary ---")
    print(f"  Symbol     : {symbol}")
    print(f"  Side       : {side}")
    print(f"  Type       : {order_type}")
    print(f"  Quantity   : {quantity}")
    if price:
        print(f"  Price      : {price}")
    if stop_price:
        print(f"  Stop Price : {stop_price}")
    print("-----------------------------\n")

    logger.info(
        f"Order request | symbol={symbol} side={side} type={order_type} "
        f"qty={quantity} price={price} stop_price={stop_price}"
    )

    try:
        response = client.place_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
        )
        _print_response(response)
        logger.info(f"Order success | response={response}")
        return response

    except RuntimeError as e:
        msg = f"Order failed: {e}"
        print(f"\n❌ {msg}")
        logger.error(msg)
        raise


def _print_response(response: dict):
    print("--- Order Response ---")
    print(f"  Order ID     : {response.get('orderId', 'N/A')}")
    print(f"  Status       : {response.get('status', 'N/A')}")
    print(f"  Executed Qty : {response.get('executedQty', 'N/A')}")
    avg_price = response.get('avgPrice') or response.get('price', 'N/A')
    print(f"  Avg Price    : {avg_price}")
    print(f"  Symbol       : {response.get('symbol', 'N/A')}")
    print(f"  Side         : {response.get('side', 'N/A')}")
    print(f"  Type         : {response.get('type', 'N/A')}")
    print("----------------------")
    print("✅ Order placed successfully!\n")
