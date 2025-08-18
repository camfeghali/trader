from typing import Union
from fastapi import FastAPI
from binance_websocket import BinanceWebSocketClient
from dataframe import KlineDataFrame
from ta_calculator import TaCalculator
from daos import BinanceDataProcessor
import time
from aggregated_dataframe import AggregateDataFrame

app = FastAPI()

symbol = "btcusdc"
interval = "1m"

# Global variables to store instances
kline_df = None
binance_client = None
agg_3m_df = None

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
    global kline_df, binance_client, agg_3m_df
    ta_calculator = TaCalculator()  

    # Create aggregated dataframe for 3m
    agg_3m_df = AggregateDataFrame(interval_in_minutes=3, ta_calculator=ta_calculator)
    
    # Create Binance WebSocket client instance
    binance_client = BinanceWebSocketClient(symbol=symbol, interval=interval)
    kline_df = KlineDataFrame(ta_calculator, BinanceDataProcessor(), [agg_3m_df])
    
    # Override the default process_kline method
    binance_client.process_kline = kline_df.process_kline 

    asyncio.create_task(binance_client.connect())
    await kline_df.get_historical_data(symbol=symbol, interval="1m", start_ms=start_ms, end_ms=end_ms)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/websocket-status")
async def get_websocket_status():
    """Check if Binance WebSocket is connected"""
    if binance_client is None:
        return {"error": "WebSocket client not initialized"}
    
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

@app.get("/data/latest")
async def get_latest_data():
    """
    Get the latest rows from both 1m and 3m dataframes.
    Returns 15 rows from 1m dataframe and 5 rows from 3m dataframe.
    """
    if kline_df is None or agg_3m_df is None:
        return {"error": "Dataframes not initialized"}
    
    try:
        # Get latest 15 rows from 1m dataframe
        latest_1m = kline_df.get_latest(15)
        latest_3m = agg_3m_df.get_latest(5)
        
        # Convert DataFrames to dictionaries for JSON serialization
        latest_1m_dict = latest_1m.to_dict('records') if not latest_1m.empty else []
        latest_3m_dict = latest_3m.to_dict('records') if not latest_3m.empty else []
        
        return {
            "1m_data": {
                "count": len(latest_1m_dict),
                "rows": latest_1m_dict
            },
            "3m_data": {
                "count": len(latest_3m_dict),
                "rows": latest_3m_dict
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {"error": f"Failed to retrieve data: {str(e)}"}

