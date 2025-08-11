from typing import Union
from fastapi import FastAPI
from binance_websocket import binance_client

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Start Binance WebSocket connection when app starts"""
    import asyncio
    asyncio.create_task(binance_client.connect())

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
        "message": f"No kline history available for {symbol.upper()}/{interval}",
        "available_data": list(custom_processor.kline_history.keys()) if hasattr(custom_processor, 'kline_history') else []
    }

@app.get("/binance/current-price")
async def get_current_price(symbol: str = "btcusdt", interval: str = "1m"):
    """Get current price from the latest kline"""
    from binance_websocket import custom_processor
    
    # Normalize symbol and interval
    symbol = symbol.lower()
    interval = interval.lower()
    
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
