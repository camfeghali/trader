"""
Constants and shared definitions for the trading system
"""

# Standard kline dataframe columns
KLINE_COLUMNS = [
    "symbol", "interval", 
    "open_time", "close_time", 
    "open_price", "high_price", "low_price", "close_price", 
    "volume", "quote_volume", 
    "trades_count", "taker_buy_volume", "taker_buy_quote_volume", "is_closed"
]

# Supported time intervals in minutes
SUPPORTED_INTERVALS = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440
} 