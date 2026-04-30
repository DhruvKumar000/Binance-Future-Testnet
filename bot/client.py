"""
Binance Futures Testnet client wrapper.
Handles authentication and raw API communication.
"""

import hmac
import hashlib
import time
import logging
import requests
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

BASE_URL = "https://testnet.binancefuture.com"


class BinanceClient:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        })

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def _request(self, method: str, endpoint: str, params: dict = None, signed: bool = True):
        params = params or {}
        if signed:
            params = self._sign(params)

        url = BASE_URL + endpoint
        logger.debug(f"Request: {method.upper()} {url} | params={params}")

        try:
            response = self.session.request(method, url, params=params if method == "GET" else None,
                                            data=params if method == "POST" else None)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Response: {data}")
            return data
        except requests.exceptions.HTTPError as e:
            error_body = e.response.json() if e.response else {}
            logger.error(f"HTTP error {e.response.status_code}: {error_body}")
            raise RuntimeError(f"API error {e.response.status_code}: {error_body.get('msg', str(e))}")
        except requests.exceptions.ConnectionError:
            logger.error("Network connection failed.")
            raise RuntimeError("Network connection failed. Check your internet connection.")
        except requests.exceptions.Timeout:
            logger.error("Request timed out.")
            raise RuntimeError("Request timed out.")

    def get_exchange_info(self):
        return self._request("GET", "/fapi/v1/exchangeInfo", signed=False)

    def place_order(self, symbol: str, side: str, order_type: str,
                    quantity: float, price: float = None,
                    stop_price: float = None, time_in_force: str = "GTC") -> dict:
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity,
        }
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("Price is required for LIMIT orders.")
            params["price"] = price
            params["timeInForce"] = time_in_force

        if order_type.upper() == "STOP":
            if stop_price is None:
                raise ValueError("Stop price is required for STOP orders.")
            params["stopPrice"] = stop_price
            if price:
                params["price"] = price
            params["timeInForce"] = time_in_force

        logger.info(f"Placing order: {params}")
        return self._request("POST", "/fapi/v1/order", params=params)
