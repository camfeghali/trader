import asyncio
import websockets
import json
from typing import Optional, Callable, Dict, Any
from datetime import datetime
from models import BinanceKlineData
from dataframe import KlineDataFrame
from ta_calculator import TaCalculator
from daos import BinanceDataProcessor

class BinanceWebSocketClient:
    def __init__(self, symbol: str = "btcusdc", interval: str = "1m"):
        self.symbol = symbol.lower()
        self.interval = interval
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.is_connected = False
        self.should_reconnect = True
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_interval = 5
        
        # Build the WebSocket URL for Binance
        self.uri = f"wss://stream.binance.com:9443/ws/{self.symbol}@kline_{self.interval}"
        
    async def connect(self):
        """Connect to Binance WebSocket with automatic reconnection"""
        while self.should_reconnect and self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                async with websockets.connect(self.uri) as websocket:
                    self.websocket = websocket
                    self.is_connected = True
                    self.reconnect_attempts = 0
                    print(f"✅ Connected to Binance WebSocket: {self.symbol.upper()}/{self.interval}")
                    
                    async for message in websocket:
                        await self.handle_message(message)
                        
            except Exception as e:
                self.is_connected = False
                self.reconnect_attempts += 1
                print(f"❌ WebSocket connection error: {e}")
                print(f"🔄 Reconnecting in {self.reconnect_interval} seconds... (attempt {self.reconnect_attempts})")
                
                if self.should_reconnect:
                    await asyncio.sleep(self.reconnect_interval)
    
    async def handle_message(self, message: str):
        """Handle incoming WebSocket messages from Binance"""
        try:
            data = json.loads(message)
            
            # Check if it's kline data
            if 'k' in data:
                kline_data = self.parse_kline_data(data)
                await self.process_kline(kline_data)
            else:
                print(f"📨 Other message: {data}")
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
        except Exception as e:
            print(f"❌ Error processing message: {e}")
    
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
    
    async def process_kline(self, kline: BinanceKlineData):
        """Process kline data - override this method for custom logic"""
        if kline.is_closed:
            print(f"📊 {kline.symbol} {kline.interval} CLOSED:")
            print(f"   Open: {kline.open_price}")
            print(f"   High: {kline.high_price}")
            print(f"   Low: {kline.low_price}")
            print(f"   Close: {kline.close_price}")
            print(f"   Volume: {kline.volume}")
            print(f"   Time: {datetime.fromtimestamp(kline.close_time / 1000)}")
            print("─" * 50)
        else:
            # Real-time updates for current kline
            print(f"🔄 {kline.symbol} {kline.interval} UPDATE: Close={kline.close_price}, Volume={kline.volume}")
    
    def disconnect(self):
        """Stop reconnection attempts"""
        self.should_reconnect = False
        self.is_connected = False
        print("🔌 Disconnected from Binance WebSocket")

# Example of custom kline processor
class CustomKlineProcessor:
    def __init__(self):
        self.kline_history = {}
    
    async def process_kline(self, kline: BinanceKlineData):
        """Custom kline processing logic"""
        print(f"Processing kline: {kline}")
        if kline.is_closed:
            # Initialize nested structure if it doesn't exist
            if kline.symbol not in self.kline_history:
                self.kline_history[kline.symbol] = {}
            if kline.interval not in self.kline_history[kline.symbol]:
                self.kline_history[kline.symbol][kline.interval] = []
            
            self.kline_history[kline.symbol][kline.interval].append(kline)
            
            if len(self.kline_history[kline.symbol][kline.interval]) > 100:
                self.kline_history[kline.symbol][kline.interval].pop(0)
            
            # Calculate simple moving average
            klines = self.kline_history[kline.symbol][kline.interval]
            if len(klines) >= 20:
                sma_20 = sum(k.close_price for k in klines[-20:]) / 20
                print(f"📈 20-period SMA: {sma_20:.2f}")
            
            # Example trading logic
            if kline.close_price > kline.open_price:
                print("🟢 Bullish candle")
                print(f"📈 20-period SMA: {sma_20:.2f}")
            else:
                print("🔴 Bearish candle")
                print(f"📉 20-period SMA: {sma_20:.2f}")