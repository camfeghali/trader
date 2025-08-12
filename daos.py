from typing import Optional, Callable, Dict, Any
import requests
from models import BinanceKlineData
import time

BASE = "https://api.binance.com/api/v3/"
ENDPOINT = "klines"

class BinanceDataProcessor:
    def fetch_klines(self, start_ms: int, end_ms: int, symbol="btcusdt", interval="1m", limit=1000):
        """Generator yielding kline batches [list[list]] between start_ms and end_ms."""
        curr = start_ms
        while curr < end_ms:
            params = {
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": limit,
                "startTime": curr,
                "endTime": end_ms,
            }
            r = requests.get(BASE + ENDPOINT, params=params, timeout=30)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            yield batch
            # advance: next start is last closeTime + 1
            last_close_ms = batch[-1][6]
            next_start = last_close_ms + 1
            # guard against no progress (rare)
            if next_start <= curr:
                break
            curr = next_start
            # be nice to rate limits
            time.sleep(0.2)

    def parse_kline_data(self, data: Dict[str, Any]) -> BinanceKlineData:
        """Parse kline data from Binance WebSocket message"""
        k = data['k']
        
        return BinanceKlineData(
            symbol=k['s'].lower(),
            interval=k['i'],
            open_time=k['t'],
            close_time=k['T'],
            open_price=float(k['o']),
            high_price=float(k['h']),
            low_price=float(k['l']),
            close_price=float(k['c']),
            volume=float(k['v']),
            quote_volume=float(k['q']),
            trades_count=k['n'],
            taker_buy_volume=float(k['V']),
            taker_buy_quote_volume=float(k['Q']),
            is_closed=k['x']
        )
