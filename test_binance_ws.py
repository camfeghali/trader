#!/usr/bin/env python3
"""
Test script for Binance WebSocket connection
Run this to test the WebSocket connection independently
"""

import asyncio
import uvicorn
from binance_websocket import binance_client

async def test_connection():
    """Test the Binance WebSocket connection"""
    print("🚀 Starting Binance WebSocket test...")
    print(f"📡 Connecting to: {binance_client.uri}")
    
    # Start the connection
    await binance_client.connect()

if __name__ == "__main__":
    print("🧪 Testing Binance WebSocket Connection")
    print("Press Ctrl+C to stop")
    
    try:
        asyncio.run(test_connection())
    except KeyboardInterrupt:
        print("\n🛑 Stopping WebSocket connection...")
        binance_client.disconnect()
        print("✅ Test completed") 