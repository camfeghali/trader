from typing import Union
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from binance_websocket import BinanceWebSocketClient
from dataframe import KlineDataFrame
from ta_calculator import TaCalculator
from daos import BinanceDataProcessor
import time
from aggregated_dataframe import AggregateDataFrame
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# Pydantic models for API documentation
class KlineData(BaseModel):
    """Individual kline data point"""
    symbol: str
    interval: str
    open_time: int
    close_time: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    quote_volume: float
    trades_count: int
    taker_buy_volume: float
    taker_buy_quote_volume: float
    is_closed: bool

class DataframeData(BaseModel):
    """Dataframe data structure"""
    count: int
    rows: List[KlineData]

class LatestDataResponse(BaseModel):
    """Response model for latest data endpoint"""
    data_1m: DataframeData = Field(alias="1m_data")
    data_3m: DataframeData = Field(alias="3m_data")
    timestamp: str

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str

app = FastAPI(
    title="Trading Data API",
    description="Real-time trading data API with WebSocket support for 1-minute and 3-minute kline data",
    version="1.0.0"
)

symbol = "btcusdc"
interval = "1m"

# Global variables to store instances
kline_df = None
binance_client = None

# Use proper datetime calculation like in get_historical_data.py
from datetime import datetime, timezone, timedelta

def to_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

start_dt = datetime(2025, 8, 19, tzinfo=timezone.utc)
end_dt   = datetime.now(timezone.utc)
start_ms, end_ms = to_ms(start_dt), to_ms(end_dt)

@app.on_event("startup")
async def startup_event():
    """Start Binance WebSocket connection when app starts"""
    import asyncio
    global kline_df, binance_client
    ta_calculator = TaCalculator()  
    
    # Create Binance WebSocket client instance
    binance_client = BinanceWebSocketClient(symbol=symbol, interval=interval)
    kline_df = KlineDataFrame(ta_calculator, BinanceDataProcessor(), [
        AggregateDataFrame(interval_in_minutes=3, ta_calculator=ta_calculator), 
        AggregateDataFrame(interval_in_minutes=5, ta_calculator=ta_calculator)])
    
    # Override the default process_kline method
    binance_client.process_kline = kline_df.process_kline 

    asyncio.create_task(binance_client.connect())
    await kline_df.get_historical_data(symbol=symbol, interval="1m", start_ms=start_ms, end_ms=end_ms)

@app.get("/", 
         summary="Health check",
         description="Simple health check endpoint")
def read_root():
    return {"Hello": "World"}

@app.get("/websocket-status",
         summary="Get WebSocket connection status",
         description="Check if the Binance WebSocket connection is active")
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

def get_latest_data_dict():
    """
    Get the latest data from 1m dataframe and all aggregate dataframes.
    Returns a dictionary with the data that can be used by both HTTP and WebSocket endpoints.
    """
    if kline_df is None:
        return {"error": "Dataframes not initialized"}
    
    try:
        # Build response with all dataframes
        response = {
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Add 1m dataframe
        latest_1m = kline_df.get_latest(15)
        latest_1m_dict = latest_1m.to_dict('records') if not latest_1m.empty else []
        response["1m_data"] = {
            "count": len(latest_1m_dict),
            "rows": latest_1m_dict
        }
        
        # Add all aggregate dataframes
        for agg_df in kline_df.get_aggregated_dataframes():
            # Calculate sensible row count: more rows for longer intervals
            # 3m = 15 rows, 5m = 12 rows, 15m = 8 rows, 1h = 6 rows, etc.
            row_count = max(5, 20 - (agg_df.interval_in_minutes // 5))
            
            latest_agg = agg_df.get_latest(row_count)
            latest_agg_dict = latest_agg.to_dict('records') if not latest_agg.empty else []
            
            # Use interval as key (e.g., "3m_data", "5m_data", "15m_data")
            interval_key = f"{agg_df.interval_in_minutes}m_data"
            response[interval_key] = {
                "count": len(latest_agg_dict),
                "rows": latest_agg_dict
            }
        
        return response
        
    except Exception as e:
        return {"error": f"Failed to retrieve data: {str(e)}"}

@app.get("/data/latest",
         summary="Get latest trading data",
         description="Retrieve the latest 15 rows from 1-minute dataframe and 5 rows from 3-minute dataframe",
         response_model=LatestDataResponse,
         responses={
             200: {"description": "Latest trading data retrieved successfully"},
             500: {"description": "Dataframes not initialized", "model": ErrorResponse}
         })
async def get_latest_data():
    """
    Get the latest rows from both 1m and 3m dataframes.
    Returns 15 rows from 1m dataframe and 5 rows from 3m dataframe.
    """
    return get_latest_data_dict()

@app.get("/test-ws",
         summary="WebSocket test page",
         description="Returns an HTML page that connects to the WebSocket endpoint and requests data every 60 seconds")
async def get_test_ws():
    """
    Returns a minimal HTML file that connects to the WebSocket and requests data every second.
    """
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>WebSocket Test</title>
</head>
<body>
    <h1>WebSocket Test</h1>
    <p>This page connects to a WebSocket endpoint and requests data every second.</p>
    <script>
        const ws = new WebSocket('ws://localhost:8000/ws');

        ws.onopen = () => {
            console.log('WebSocket connection opened.');
            ws.send('Hello! You are connected to the trading server.');
        };

        ws.onmessage = (event) => {
            console.log('Received message from server:', event.data);
            // You can parse JSON here if the server sends JSON
            // const data = JSON.parse(event.data);
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        ws.onclose = () => {
            console.log('WebSocket connection closed.');
        };

        // Send a message to the server every second
        setInterval(() => {
            ws.send('get data');
        }, 60000);
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time data.
    
    Connect to this WebSocket endpoint to receive real-time trading data. 
    Send 'get data' to request latest data.
    """
    await websocket.accept()
    print("🔌 Client connected to WebSocket")
    
    try:
        # Send a welcome message
        await websocket.send_text("Hello! You are connected to the trading server.")
        
        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_text()
            print(f"📨 Received: {data}")
            
            # Check if client is requesting latest data
            if data.lower() == "get data":
                import json
                latest_data = get_latest_data_dict()
                await websocket.send_text(json.dumps(latest_data))
            else:
                # Echo back the message
                await websocket.send_text(f"Server received: {data}")
            
    except WebSocketDisconnect:
        print("🔌 Client disconnected from WebSocket")


