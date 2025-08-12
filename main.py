from typing import Union
from fastapi import FastAPI
from binance_websocket import BinanceWebSocketClient
from dataframe import KlineDataFrame
from ta_calculator import TaCalculator
from daos import BinanceDataProcessor
import time

app = FastAPI()

symbol = "btcusdt"
interval = "1m"

# Use proper datetime calculation like in get_historical_data.py
from datetime import datetime, timezone, timedelta

def to_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

start_dt = datetime(2025, 8, 1, tzinfo=timezone.utc)
end_dt   = datetime.now(timezone.utc)
start_ms, end_ms = to_ms(start_dt), to_ms(end_dt)

@app.on_event("startup")
async def startup_event():
    """Start Binance WebSocket connection when app starts"""
    import asyncio

    # Create Binance WebSocket client instance
    binance_client = BinanceWebSocketClient(symbol=symbol, interval=interval)
    kline_df = KlineDataFrame(TaCalculator(), BinanceDataProcessor())
    
    # Override the default process_kline method
    binance_client.process_kline = kline_df.process_kline 

    asyncio.create_task(binance_client.connect())
    await kline_df.get_historical_data(symbol=symbol, interval="1m", start_ms=start_ms, end_ms=end_ms)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/binance/kline-history")
async def get_kline_history(symbol: str = "btcusdt", interval: str = "1m", limit: int = 10):
    """Get recent kline history from the custom processor"""
    # Import the custom processor from binance_websocket
    from binance_websocket import custom_processor
    
    # Normalize symbol and interval
    symbol = symbol.lower()
    interval = interval.lower()
    
    print(f"Debug - kline_history: {custom_processor.kline_history}")
    print(f"Debug - symbol: {symbol}")
    print(f"Debug - interval: {interval}")

    if (hasattr(custom_processor, 'kline_history') and 
        custom_processor.kline_history and 
        symbol in custom_processor.kline_history and 
        interval in custom_processor.kline_history[symbol]):
        
        history = custom_processor.kline_history[symbol][interval]
        
        return {
            "symbol": symbol.upper(),
            "interval": interval,
            "count": len(history),
            "klines": [
                {
                    "time": kline.close_time,
                    "open": kline.open_price,
                    "high": kline.high_price,
                    "low": kline.low_price,
                    "close": kline.close_price,
                    "volume": kline.volume
                }
                for kline in history[-limit:]  # Last N klines based on limit
            ]
        }
    
    return {
        "message": f"No kline history available for {symbol}/{interval}",
        "available_data": list(custom_processor.kline_history.keys()) if hasattr(custom_processor, 'kline_history') else []
    }

@app.get("/binance/current-price")
async def get_current_price(symbol: str = "btcusdt", interval: str = "1m"):
    """Get current price from the latest kline"""
    from binance_websocket import custom_processor
    
    # Normalize symbol and interval
    symbol = symbol.lower()
    interval = interval.lower()
    
    print(f"Debug - kline_history: {custom_processor.kline_history}")
    print(f"Debug - symbol: {symbol}")
    print(f"Debug - interval: {interval}")

    if (hasattr(custom_processor, 'kline_history') and 
        custom_processor.kline_history and 
        symbol in custom_processor.kline_history and 
        interval in custom_processor.kline_history[symbol]):
        
        history = custom_processor.kline_history[symbol][interval]
        
        if history:
            latest_kline = history[-1]
            return {
                "symbol": latest_kline.symbol,
                "price": latest_kline.close_price,
                "time": latest_kline.close_time,
                "interval": interval
            }
    
    return {
        "message": f"No price data available for {symbol.upper()}/{interval}",
        "available_data": list(custom_processor.kline_history.keys()) if hasattr(custom_processor, 'kline_history') else []
    }

@app.get("/websocket-status")
async def get_websocket_status():
    """Check if Binance WebSocket is connected"""
    return {
        "connected": binance_client.is_connected,
        "symbol": binance_client.symbol.upper(),
        "interval": binance_client.interval
    }

@app.get("/binance/available-data")
async def get_available_data():
    """Get all available symbols and intervals with data"""
    from binance_websocket import custom_processor
    
    if hasattr(custom_processor, 'kline_history') and custom_processor.kline_history:
        available_data = {}
        for symbol in custom_processor.kline_history:
            available_data[symbol.upper()] = {
                interval: len(custom_processor.kline_history[symbol][interval])
                for interval in custom_processor.kline_history[symbol]
            }
        return {
            "available_data": available_data,
            "total_symbols": len(available_data)
        }
    
    return {
        "message": "No data available yet",
        "available_data": {}
    }

