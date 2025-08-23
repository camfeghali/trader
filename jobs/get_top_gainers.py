#!/usr/bin/env python3
"""
Binance Volume Spike Detector

This script fetches all tradeable stablecoin pairs from Binance and detects volume spikes
by comparing the last 15-minute candle's volume to the average of the previous 4 candles.

Usage:
    uv run jobs/get_top_gainers.py                    # Default: 5 candles, 100% threshold
    uv run jobs/get_top_gainers.py 10 200            # 10 candles, 200% threshold
    uv run jobs/get_top_gainers.py BTCUSDT           # Analyze single symbol
    uv run jobs/get_top_gainers.py BTCUSDT 10 200    # Single symbol with custom params
    uv run jobs/get_top_gainers.py symbol:BTCUSDT    # Analyze single symbol (explicit)
    uv run jobs/get_top_gainers.py debug:SYMBOL      # Debug specific symbol
    uv run jobs/get_top_gainers.py status:SYMBOL     # Check symbol status
"""

import requests
import pandas as pd
import numpy as np
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Dict, Any
import talib


class BinanceVolumeSpikeDetector:
    def __init__(self, max_workers: int = 10):
        self.base_url = "https://api.binance.com/api/v3"
        self._volume_cache = {}
        self._cache_lock = threading.Lock()
        self.max_workers = max_workers
        # Create a session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; BinanceVolumeSpikeDetector/1.0)'
        })
        
    def get_all_tickers(self) -> List[Dict[str, Any]]:
        """Fetch all 24hr ticker data from Binance."""
        try:
            response = self.session.get(f"{self.base_url}/ticker/24hr", timeout=10)
            response.raise_for_status()
            tickers = response.json()
            print(f"✅ Fetched {len(tickers)} trading pairs from Binance")
            return tickers
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching tickers: {e}")
            return []
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """Fetch exchange information with retry mechanism."""
        for attempt in range(3):
            try:
                response = self.session.get(f"{self.base_url}/exchangeInfo", timeout=10)
                response.raise_for_status()
                exchange_info = response.json()
                print(f"✅ Fetched exchange info with {len(exchange_info.get('symbols', []))} symbols")
                return exchange_info
            except requests.exceptions.RequestException as e:
                if attempt < 2:
                    print(f"⚠️  Attempt {attempt + 1} failed, retrying...")
                    time.sleep(1)
                else:
                    print(f"❌ Error fetching exchange info: {e}")
                    return {}
    
    def filter_stablecoin_pairs(self, tickers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter to stablecoin and BNB pairs only."""
        stablecoins = ["USDC", "USDT"]
        quote_assets = stablecoins + ["BNB"]
        
        stablecoin_pairs = []
        for ticker in tickers:
            for asset in quote_assets:
                if ticker['symbol'].endswith(asset):
                    stablecoin_pairs.append(ticker)
                    break
        
        print(f"📊 Found {len(stablecoin_pairs)} stablecoin/BNB pairs")
        return stablecoin_pairs
    
    def filter_listed_pairs(self, tickers: List[Dict[str, Any]], exchange_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter to listed pairs and add trading venue info."""
        symbols_info = {s['symbol']: s for s in exchange_info.get('symbols', [])}
        
        listed_pairs = []
        for ticker in tickers:
            symbol = ticker['symbol']
            if symbol in symbols_info:
                symbol_info = symbols_info[symbol]
                status = symbol_info.get('status', 'UNKNOWN')
                
                if status == 'TRADING':
                    # Determine trading venue
                    is_spot = symbol_info.get('isSpotTradingAllowed', False)
                    is_futures = symbol_info.get('isFuturesTradingAllowed', False)
                    
                    if is_spot and is_futures:
                        trading_venue = "Spot & Futures"
                    elif is_spot:
                        trading_venue = "Spot Only"
                    elif is_futures:
                        trading_venue = "Futures Only"
                    else:
                        trading_venue = "Other"
                    
                    ticker['trading_venue'] = trading_venue
                    listed_pairs.append(ticker)
        
        print(f"✅ Filtered to {len(listed_pairs)} tradeable pairs")
        return listed_pairs
    
    def filter_active_pairs(self, tickers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter to active pairs when exchange info is unavailable."""
        active_pairs = []
        for ticker in tickers:
            if ticker.get('status') == 'TRADING':
                ticker['trading_venue'] = "Active (Exchange Info Unavailable)"
                active_pairs.append(ticker)
        
        print(f"✅ Filtered to {len(active_pairs)} active pairs")
        return active_pairs
    
    def create_dataframe(self, tickers: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create DataFrame from ticker data."""
        data = []
        for ticker in tickers:
            data.append({
                'symbol': ticker['symbol'],
                'price_change_percent': float(ticker['priceChangePercent']),
                'last_price': float(ticker['lastPrice']),
                'quote_volume': float(ticker['quoteVolume']),
                'count': int(ticker['count']),
                'trading_venue': ticker.get('trading_venue', 'Unknown'),
                'high_24h': float(ticker['highPrice']),
                'low_24h': float(ticker['lowPrice'])
            })
        
        return pd.DataFrame(data)
    
    def _get_server_time_ms(self) -> int:
        """Get Binance server time in milliseconds."""
        try:
            response = self.session.get(f"{self.base_url}/time", timeout=5)
            response.raise_for_status()
            return response.json()['serverTime']
        except:
            # Fallback to local time
            return int(time.time() * 1000)
    
    def get_historical_quote_volumes(self, symbol: str, periods: int = 5) -> List[float]:
        """Get historical quote volumes for the last N 15-minute periods."""
        cache_key = f"{symbol}_15m_{periods}"
        
        with self._cache_lock:
            if cache_key in self._volume_cache:
                return self._volume_cache[cache_key]
        
        # Retry logic for network requests
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Calculate time range for closed candles
                server_now_ms = self._get_server_time_ms()
                period_ms = 15 * 60 * 1000  # 15 minutes in milliseconds
                current_open = (server_now_ms // period_ms) * period_ms
                end_time = current_open - 1
                start_time = end_time - (periods * period_ms) + 1
                
                params = {
                    "symbol": symbol,
                    "interval": "15m",
                    "startTime": start_time,
                    "endTime": end_time,
                    "limit": periods
                }
                
                response = self.session.get(f"{self.base_url}/klines", params=params, timeout=15)
                response.raise_for_status()
                klines = response.json()
                
                if not isinstance(klines, list) or len(klines) != periods:
                    return []
                
                volumes = [float(k[7]) for k in klines]  # Quote volume is at index 7
                
                with self._cache_lock:
                    self._volume_cache[cache_key] = volumes
                
                return volumes
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Brief delay before retry
                    continue
                else:
                    # Only print error on final attempt to reduce noise
                    print(f"❌ Error fetching historical data for {symbol} after {max_retries} attempts: {e}")
                    return []
    
    def get_historical_quote_volumes_with_timestamps(self, symbol: str, periods: int = 5) -> List[Dict[str, Any]]:
        """Get historical quote volumes with timestamps for the last N 15-minute periods."""
        # Retry logic for network requests
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Calculate time range for closed candles
                server_now_ms = self._get_server_time_ms()
                period_ms = 15 * 60 * 1000  # 15 minutes in milliseconds
                current_open = (server_now_ms // period_ms) * period_ms
                end_time = current_open - 1
                start_time = end_time - (periods * period_ms) + 1
                
                params = {
                    "symbol": symbol,
                    "interval": "15m",
                    "startTime": start_time,
                    "endTime": end_time,
                    "limit": periods
                }
                
                response = self.session.get(f"{self.base_url}/klines", params=params, timeout=15)
                response.raise_for_status()
                klines = response.json()
                
                if not isinstance(klines, list) or len(klines) != periods:
                    return []
                
                result = []
                for k in klines:
                    timestamp = int(k[0])  # Open time
                    volume = float(k[7])   # Quote volume
                    result.append({
                        'timestamp': timestamp,
                        'datetime': datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S UTC'),
                        'volume': volume
                    })
                return result
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Brief delay before retry
                    continue
                else:
                    print(f"❌ Error fetching historical data for {symbol} after {max_retries} attempts: {e}")
                    return []
        
        return []
    
    def get_historical_price_data(self, symbol: str, periods: int = 5) -> List[Dict[str, float]]:
        """Get historical price data (open, close) for the last N 15-minute periods."""
        cache_key = f"{symbol}_price_15m_{periods}"
        
        with self._cache_lock:
            if cache_key in self._volume_cache:
                return self._volume_cache[cache_key]
        
        # Retry logic for network requests
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Calculate time range for closed candles
                server_now_ms = self._get_server_time_ms()
                period_ms = 15 * 60 * 1000  # 15 minutes in milliseconds
                current_open = (server_now_ms // period_ms) * period_ms
                end_time = current_open - 1
                start_time = end_time - (periods * period_ms) + 1
                
                params = {
                    "symbol": symbol,
                    "interval": "15m",
                    "startTime": start_time,
                    "endTime": end_time,
                    "limit": periods
                }
                
                response = self.session.get(f"{self.base_url}/klines", params=params, timeout=15)
                response.raise_for_status()
                klines = response.json()
                
                if not isinstance(klines, list) or len(klines) != periods:
                    return []
                
                price_data = []
                for k in klines:
                    price_data.append({
                        'open': float(k[1]),   # Open price
                        'close': float(k[4])   # Close price
                    })
                
                with self._cache_lock:
                    self._volume_cache[cache_key] = price_data
                
                return price_data
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Brief delay before retry
                    continue
                else:
                    return []
    
    def calculate_volume_emas(self, volumes: List[float], ema_periods: List[int] = [9, 21, 50]) -> Dict[str, float]:
        """Calculate volume EMAs for the given periods."""
        if not volumes or len(volumes) < max(ema_periods):
            # Return NaN for all requested EMAs if insufficient data
            return {f'ema_{period}': np.nan for period in ema_periods}
        
        # Convert to numpy array for talib
        volume_array = np.array(volumes, dtype=float)
        
        # Calculate EMAs for all requested periods
        ema_results = {}
        for period in ema_periods:
            if len(volumes) >= period:
                ema_values = talib.EMA(volume_array, timeperiod=period)
                ema_results[f'ema_{period}'] = ema_values[-1] if not np.isnan(ema_values[-1]) else np.nan
            else:
                ema_results[f'ema_{period}'] = np.nan
        
        return ema_results
    
    def calculate_price_emas(self, prices: List[float], ema_periods: List[int] = [9, 21, 50]) -> Dict[str, float]:
        """Calculate price EMAs for the given periods."""
        if not prices or len(prices) < max(ema_periods):
            # Return NaN for all requested EMAs if insufficient data
            return {f'price_ema_{period}': np.nan for period in ema_periods}
        
        # Convert to numpy array for talib
        price_array = np.array(prices, dtype=float)
        
        # Calculate EMAs for all requested periods
        ema_results = {}
        for period in ema_periods:
            if len(prices) >= period:
                ema_values = talib.EMA(price_array, timeperiod=period)
                ema_results[f'price_ema_{period}'] = ema_values[-1] if not np.isnan(ema_values[-1]) else np.nan
            else:
                ema_results[f'price_ema_{period}'] = np.nan
        
        return ema_results
    
    def detect_volume_spikes(self, df: pd.DataFrame, periods: int = 50, spike_periods: int = 5, threshold_pct: float = 100.0, ema_periods: List[int] = [9, 21, 50]) -> pd.DataFrame:
        """Detect volume spikes by comparing last candle to previous N-1 candles."""
        print(f"🔍 Detecting volume spikes using {periods} periods of data")
        print(f"📊 Comparing last candle to previous {spike_periods} candles")
        print(f"🎯 Threshold: {threshold_pct}%")
        print(f"🚀 Using {self.max_workers} parallel workers")
        print(f"📈 Calculating volume EMAs for periods: {ema_periods}")
        
        symbols = df['symbol'].tolist()
        
        def calc(symbol: str) -> tuple:
            vols = self.get_historical_quote_volumes(symbol, periods)
            if len(vols) != periods:
                # Return tuple with dynamic EMA values
                ema_values = [np.nan] * len(ema_periods)
                return (symbol, np.nan, np.nan, np.nan, np.nan, np.nan) + tuple(ema_values)
            
            last_volume = vols[-1]  # Most recent candle (N)
            # Use spike_periods to determine how many previous candles to compare against
            # We want to compare N to N-1, N-2, N-3, etc.
            if len(vols) >= spike_periods + 1:  # Need at least spike_periods + 1 candles
                prev_volumes = vols[-(spike_periods + 1):-1]  # Candles N-1, N-2, N-3, etc.
            else:
                prev_volumes = vols[:-1]  # All previous candles if not enough data
            avg_prev_volume = sum(prev_volumes) / len(prev_volumes) if prev_volumes else np.nan
            
            # Calculate volume EMAs
            volume_emas = self.calculate_volume_emas(vols, ema_periods)
            
            # Get price data for the last candle
            price_data = self.get_historical_price_data(symbol, periods)
            if len(price_data) != periods:
                # Return tuple with dynamic EMA values
                ema_values = [volume_emas.get(f'ema_{period}', np.nan) for period in ema_periods]
                return (symbol, np.nan, last_volume, avg_prev_volume, len(prev_volumes), np.nan) + tuple(ema_values)
            
            last_open = price_data[-1]['open']
            last_close = price_data[-1]['close']
            is_bullish = last_close > last_open
            
            # Extract close prices for price EMA calculation
            close_prices = [p['close'] for p in price_data]
            price_emas = self.calculate_price_emas(close_prices, ema_periods)
            
            # Get EMA values in order (volume + price)
            volume_ema_values = [volume_emas.get(f'ema_{period}', np.nan) for period in ema_periods]
            price_ema_values = [price_emas.get(f'price_ema_{period}', np.nan) for period in ema_periods]
            ema_values = volume_ema_values + price_ema_values
            
            # Calculate VEMA multiplier (using the longest EMA period available)
            longest_ema_period = max(ema_periods) if ema_periods else 50
            longest_ema_key = f'ema_{longest_ema_period}'
            longest_ema_value = volume_emas.get(longest_ema_key, np.nan)
            vema_multiplier = (last_volume / longest_ema_value) if longest_ema_value and longest_ema_value > 0 else np.nan
            
            if avg_prev_volume and avg_prev_volume > 0:
                spike_pct = ((last_volume - avg_prev_volume) / avg_prev_volume) * 100
                return (symbol, round(spike_pct, 1), last_volume, avg_prev_volume, len(prev_volumes), is_bullish, vema_multiplier) + tuple(ema_values)
            else:
                return (symbol, np.nan, last_volume, avg_prev_volume, len(prev_volumes), is_bullish, vema_multiplier) + tuple(ema_values)
        
        spike_results = {}
        last_volume_results = {}
        avg_volume_results = {}
        periods_used_results = {}
        bullish_results = {}
        vema_multiplier_results = {}
        volume_ema_results = {period: {} for period in ema_periods}
        price_ema_results = {period: {} for period in ema_periods}
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(calc, s): s for s in symbols}
            for f in as_completed(futures):
                result = f.result()
                if result is None or len(result) < 7 + 2 * len(ema_periods):
                    continue  # Skip failed results
                    
                sym = result[0]
                spike_pct = result[1]
                last_vol = result[2]
                avg_vol = result[3]
                periods_used = result[4]
                is_bullish = result[5]
                vema_multiplier = result[6]
                
                spike_results[sym] = spike_pct
                last_volume_results[sym] = last_vol
                avg_volume_results[sym] = avg_vol
                periods_used_results[sym] = periods_used
                bullish_results[sym] = is_bullish
                vema_multiplier_results[sym] = vema_multiplier
                
                # Store EMA results (volume + price)
                for i, period in enumerate(ema_periods):
                    if 7 + i < len(result):
                        volume_ema_results[period][sym] = result[7 + i]
                    if 7 + len(ema_periods) + i < len(result):
                        price_ema_results[period][sym] = result[7 + len(ema_periods) + i]
                
                completed += 1
                if completed % 20 == 0 or completed == len(symbols):
                    print(f"   Progress: {completed}/{len(symbols)}")
        
        df["volume_spike_pct"] = [spike_results.get(s, np.nan) for s in symbols]
        df["last_volume"] = [last_volume_results.get(s, np.nan) for s in symbols]
        df["avg_prev_volume"] = [avg_volume_results.get(s, np.nan) for s in symbols]
        df["periods_used"] = [periods_used_results.get(s, np.nan) for s in symbols]
        df["is_bullish"] = [bullish_results.get(s, np.nan) for s in symbols]
        df["vema_multiplier"] = [vema_multiplier_results.get(s, np.nan) for s in symbols]
        
        # Add volume EMA columns dynamically
        for period in ema_periods:
            df[f"volume_ema_{period}"] = [volume_ema_results[period].get(s, np.nan) for s in symbols]
        
        # Add price EMA columns dynamically
        for period in ema_periods:
            df[f"price_ema_{period}"] = [price_ema_results[period].get(s, np.nan) for s in symbols]
        
        return df
    
    def get_volume_spikes(self, periods: int = 50, spike_periods: int = 5, threshold_pct: float = 100.0, 
                         min_volume_usdt: float = 100000, limit: int = 50, ema_periods: List[int] = [9, 21, 50],
                         filters: List[Dict[str, Any]] = None, sort_by: str = 'volume_spike_pct', 
                         sort_ascending: bool = False) -> pd.DataFrame:
        """
        Get trading pairs with volume spikes above threshold.
        
        Args:
            periods: Number of 15-minute periods to fetch data (default: 50)
            spike_periods: Number of previous periods to compare last candle against (default: 5)
            threshold_pct: Minimum spike percentage to include (default: 100%)
            min_volume_usdt: Minimum 24h volume in USDT to filter by
            limit: Maximum number of results to return
            ema_periods: List of EMA periods to calculate
            filters: List of filter dictionaries with format:
                    [{'column': 'column_name', 'operator': '>=', 'value': 100}]
                    Supported operators: '==', '!=', '>=', '<=', '>', '<'
            sort_by: Column name to sort by (default: 'volume_spike_pct')
            sort_ascending: Sort in ascending order if True, descending if False (default: False)
            
        Returns:
            DataFrame sorted by specified column
        """
        print("🚀 Fetching all trading pairs from Binance...")
        
        # Get all tickers
        all_tickers = self.get_all_tickers()
        if not all_tickers:
            return pd.DataFrame()
        
        # Get exchange info for better filtering
        exchange_info = self.get_exchange_info()
        
        # Filter to stablecoin/BNB pairs only
        stablecoin_pairs = self.filter_stablecoin_pairs(all_tickers)
        
        # Filter to listed pairs if exchange info is available
        if exchange_info:
            listed_pairs = self.filter_listed_pairs(stablecoin_pairs, exchange_info)
        else:
            listed_pairs = self.filter_active_pairs(stablecoin_pairs)
        
        # Create DataFrame
        df = self.create_dataframe(listed_pairs)
        
        # Filter by minimum volume
        df = df[df['quote_volume'] >= min_volume_usdt]
        print(f"📈 Filtered to {len(df)} pairs with volume >= ${min_volume_usdt:,.0f}")
        
        # Detect volume spikes
        df = self.detect_volume_spikes(df, periods=periods, spike_periods=spike_periods, threshold_pct=threshold_pct, ema_periods=ema_periods)
        
        # Filter for positive spikes above threshold only
        df_spikes = df[(df['volume_spike_pct'] >= threshold_pct) & (df['volume_spike_pct'] > 0)].copy()
        print(f"📈 Found {len(df_spikes)} pairs with positive volume spikes >= {threshold_pct}%")
        
        # Filter by minimum average volume (liquidity filter)
        min_avg_volume = 10000  # $10,000 minimum average volume
        df_liquid = df_spikes[df_spikes['avg_prev_volume'] >= min_avg_volume].copy()
        print(f"💧 Filtered to {len(df_liquid)} pairs with avg volume >= ${min_avg_volume:,.0f}")
        
        # Apply custom filters if provided
        if filters:
            df_filtered = df_liquid.copy()
            for filter_rule in filters:
                column = filter_rule.get('column')
                operator = filter_rule.get('operator')
                value = filter_rule.get('value')
                
                if column and operator and column in df_filtered.columns:
                    if operator == '==':
                        df_filtered = df_filtered[df_filtered[column] == value]
                    elif operator == '!=':
                        df_filtered = df_filtered[df_filtered[column] != value]
                    elif operator == '>=':
                        df_filtered = df_filtered[df_filtered[column] >= value]
                    elif operator == '<=':
                        df_filtered = df_filtered[df_filtered[column] <= value]
                    elif operator == '>':
                        df_filtered = df_filtered[df_filtered[column] > value]
                    elif operator == '<':
                        df_filtered = df_filtered[df_filtered[column] < value]
                    elif operator == 'endswith':
                        df_filtered = df_filtered[df_filtered[column].astype(str).str.endswith(str(value))]
                    elif operator == 'startswith':
                        df_filtered = df_filtered[df_filtered[column].astype(str).str.startswith(str(value))]
                    elif operator == 'contains':
                        df_filtered = df_filtered[df_filtered[column].astype(str).str.contains(str(value), case=False)]
                    
                    print(f"🔍 Applied filter: {column} {operator} {value} -> {len(df_filtered)} pairs remaining")
                else:
                    print(f"⚠️  Invalid filter rule: {filter_rule}")
            
            df_liquid = df_filtered
        
        # Sort by specified column
        if sort_by in df_liquid.columns:
            df_sorted = df_liquid.sort_values(sort_by, ascending=sort_ascending)
            print(f"📊 Sorted by {sort_by} ({'ascending' if sort_ascending else 'descending'})")
        else:
            print(f"⚠️  Sort column '{sort_by}' not found, using default sort by volume_spike_pct")
            df_sorted = df_liquid.sort_values('volume_spike_pct', ascending=False)
        
        # Select top N
        top_spikes = df_sorted.head(limit)
        
        return top_spikes
    
    def get_volume_spikes_single_symbol(self, symbol: str, periods: int = 50, spike_periods: int = 5, threshold_pct: float = 100.0, ema_periods: List[int] = [9, 21, 50], sort_by: str = 'volume_spike_pct', sort_ascending: bool = False) -> pd.DataFrame:
        """
        Get volume spike analysis for a single symbol.
        
        Args:
            symbol: The trading pair symbol to analyze
            periods: Number of 15-minute periods to analyze (default: 5)
            threshold_pct: Minimum spike percentage to include (default: 100%)
            
        Returns:
            DataFrame with volume spike analysis for the single symbol
        """
        print(f"🔍 Analyzing volume spikes for {symbol}...")
        
        # Get ticker data for the specific symbol
        all_tickers = self.get_all_tickers()
        if not all_tickers:
            return pd.DataFrame()
        
        ticker_data = next((t for t in all_tickers if t['symbol'] == symbol), None)
        if not ticker_data:
            print(f"❌ Symbol {symbol} not found in ticker data")
            return pd.DataFrame()
        
        # Get exchange info for trading venue
        exchange_info = self.get_exchange_info()
        
        # Create ticker with trading venue info
        if exchange_info:
            symbol_info = next((s for s in exchange_info.get('symbols', []) if s['symbol'] == symbol), None)
            if symbol_info:
                is_spot = symbol_info.get('isSpotTradingAllowed', False)
                is_futures = symbol_info.get('isFuturesTradingAllowed', False)
                
                if is_spot and is_futures:
                    trading_venue = "Spot & Futures"
                elif is_spot:
                    trading_venue = "Spot Only"
                elif is_futures:
                    trading_venue = "Futures Only"
                else:
                    trading_venue = "Other"
                
                ticker_data['trading_venue'] = trading_venue
            else:
                ticker_data['trading_venue'] = "Unknown"
        else:
            ticker_data['trading_venue'] = "Active (Exchange Info Unavailable)"
        
        # Create DataFrame
        df = self.create_dataframe([ticker_data])
        
        # Detect volume spikes
        df = self.detect_volume_spikes(df, periods=periods, spike_periods=spike_periods, threshold_pct=threshold_pct, ema_periods=ema_periods)
        
        # Filter for spikes above threshold (if any)
        if not df.empty and 'volume_spike_pct' in df.columns:
            df_filtered = df[df['volume_spike_pct'] >= threshold_pct].copy()
            if df_filtered.empty:
                print(f"📉 No volume spikes >= {threshold_pct}% found for {symbol}")
                # Return the original data even if no spike
                return df
            
            # Filter by minimum average volume (liquidity filter)
            min_avg_volume = 10000  # $10,000 minimum average volume
            df_liquid = df_filtered[df_filtered['avg_prev_volume'] >= min_avg_volume].copy()
            if df_liquid.empty:
                print(f"⚠️  {symbol} has low average volume (${df_filtered['avg_prev_volume'].iloc[0]:,.0f}) - below ${min_avg_volume:,.0f} threshold")
                return df_filtered  # Return unfiltered for single symbol analysis
            
            print(f"💧 {symbol} meets liquidity threshold (avg volume: ${df_liquid['avg_prev_volume'].iloc[0]:,.0f})")
            return df_liquid
        
        return df
    
    def print_results(self, df: pd.DataFrame, title: str, ema_periods: List[int] = [9, 21, 50]):
        """Print results in a formatted table."""
        if df.empty:
            print(f"❌ No results found for {title}")
            return
        
        print(f"\n{title}")
        print("=" * 120)
        
        # Format the DataFrame for display
        display_df = df.copy()
        
        # Format percentage columns
        display_df['price_change_percent'] = display_df['price_change_percent'].apply(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
        )
        display_df['volume_spike_pct'] = display_df['volume_spike_pct'].apply(
            lambda x: f"🔥 +{x:.0f}%" if pd.notna(x) and x > 0 else f"📉 {x:.0f}%" if pd.notna(x) else "N/A"
        )
        
        # Format VEMA multiplier
        if 'vema_multiplier' in display_df.columns:
            display_df['vema_multiplier'] = display_df['vema_multiplier'].apply(
                lambda x: f"×{x:.1f}" if pd.notna(x) else "N/A"
            )
        
        # Format bullish/bearish column
        display_df['is_bullish'] = display_df['is_bullish'].apply(
            lambda x: "🟢 Bullish" if x == True else "🔴 Bearish" if x == False else "❓ Unknown"
        )
        
        # Format price columns
        display_df['last_price'] = display_df['last_price'].apply(
            lambda x: f"${x:.6f}" if pd.notna(x) else "N/A"
        )
        display_df['high_24h'] = display_df['high_24h'].apply(
            lambda x: f"${x:.6f}" if pd.notna(x) else "N/A"
        )
        display_df['low_24h'] = display_df['low_24h'].apply(
            lambda x: f"${x:.6f}" if pd.notna(x) else "N/A"
        )
        
        # Format volume columns
        display_df['quote_volume'] = display_df['quote_volume'].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
        )
        display_df['last_volume'] = display_df['last_volume'].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
        )
        display_df['avg_prev_volume'] = display_df['avg_prev_volume'].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
        )
        
        # Format volume EMA columns dynamically
        for period in ema_periods:
            ema_col = f'volume_ema_{period}'
            if ema_col in display_df.columns:
                display_df[ema_col] = display_df[ema_col].apply(
                    lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
                )
        
        # Format price EMA columns dynamically
        for period in ema_periods:
            ema_col = f'price_ema_{period}'
            if ema_col in display_df.columns:
                display_df[ema_col] = display_df[ema_col].apply(
                    lambda x: f"${x:.4f}" if pd.notna(x) else "N/A"
                )
        
        # Select and order columns dynamically
        base_columns = ['symbol', 'price_change_percent', 'last_price', 'quote_volume', 
                       'volume_spike_pct', 'last_volume', 'avg_prev_volume', 'vema_multiplier']
        
        # Add volume EMA columns dynamically
        volume_ema_columns = [f'volume_ema_{period}' for period in ema_periods]
        
        # Add price EMA columns dynamically
        price_ema_columns = [f'price_ema_{period}' for period in ema_periods]
        
        remaining_columns = ['is_bullish', 'trading_venue', 'high_24h', 'low_24h']
        columns_to_show = base_columns + volume_ema_columns + price_ema_columns + remaining_columns
        
        # Filter to only include columns that exist
        columns_to_show = [col for col in columns_to_show if col in display_df.columns]
        display_df = display_df[columns_to_show]
        
        # Create column headers dynamically
        base_headers = ['Symbol', '24h %', 'Price', '24h Vol', 
                       'Spike %', '15m Vol', 'Avg Vol', f'V{max(ema_periods)}×']
        
        # Add volume EMA headers dynamically
        volume_ema_headers = [f'V{period}' for period in ema_periods]
        
        # Add price EMA headers dynamically
        price_ema_headers = [f'P{period}' for period in ema_periods]
        
        remaining_headers = ['Candle', 'Venue', 'High', 'Low']
        column_headers = base_headers + volume_ema_headers + price_ema_headers + remaining_headers
        
        # Filter headers to match the actual columns
        column_headers = column_headers[:len(columns_to_show)]
        display_df.columns = column_headers
        
        # Print the table
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 15)
        print(display_df.to_string(index=False))
        
        # Print summary statistics
        print(f"\n📊 Summary:")
        print(f"   Total pairs analyzed: {len(df)}")
        print(f"   Pairs with spikes: {len(df[df['volume_spike_pct'] >= 0])}")
        print(f"   Average spike: {df['volume_spike_pct'].mean():.1f}%")
        print(f"   Max spike: {df['volume_spike_pct'].max():.1f}%")
    
    def debug_volume_data(self, symbol: str) -> Dict[str, Any]:
        """Debug volume data for a specific symbol."""
        print(f"🔍 Debugging volume data for {symbol}...")
        
        # Get ticker data
        all_tickers = self.get_all_tickers()
        if not all_tickers:
            return {'error': 'Could not fetch ticker data'}
        
        ticker_data = next((t for t in all_tickers if t['symbol'] == symbol), None)
        if not ticker_data:
            return {'error': f'Symbol {symbol} not found in ticker data'}
        
        # Get historical volumes with timestamps
        volume_data = self.get_historical_quote_volumes_with_timestamps(symbol, periods=5)
        if not volume_data:
            return {'error': f'Could not fetch historical data for {symbol}'}
        
        # Get price data with timestamps
        price_data = self.get_historical_price_data(symbol, periods=5)
        
        # Extract volumes and timestamps
        volumes = [v['volume'] for v in volume_data]
        timestamps = [v['datetime'] for v in volume_data]
        
        # Calculate spike
        last_volume = volumes[-1]
        prev_volumes = volumes[:-1]
        avg_prev_volume = sum(prev_volumes) / len(prev_volumes) if prev_volumes else 0
        spike_pct = ((last_volume - avg_prev_volume) / avg_prev_volume * 100) if avg_prev_volume > 0 else 0
        
        # Get current server time
        try:
            server_time_response = self.session.get('https://api.binance.com/api/v3/time')
            server_time = server_time_response.json()['serverTime']
            server_time_str = datetime.fromtimestamp(server_time / 1000).strftime('%Y-%m-%d %H:%M:%S UTC')
        except:
            server_time_str = "Could not fetch"
        
        return {
            'symbol': symbol,
            'ticker_24h_volume': float(ticker_data['quoteVolume']),
            'last_15m_volume': last_volume,
            'avg_prev_4_volumes': avg_prev_volume,
            'volume_spike_pct': round(spike_pct, 1),
            'volumes': volumes,
            'timestamps': timestamps,
            'volume_data_with_timestamps': volume_data,
            'price_data': price_data,
            'server_time': server_time_str,
            'last_volume_formatted': f"${last_volume:,.0f}",
            'avg_prev_formatted': f"${avg_prev_volume:,.0f}",
            'last_candle_time': timestamps[-1] if timestamps else "Unknown"
        }
    
    def check_symbol_status(self, symbol: str) -> Dict[str, Any]:
        """Check the status of a specific symbol."""
        print(f"🔍 Checking status for {symbol}...")
        
        # Get exchange info
        exchange_info = self.get_exchange_info()
        if not exchange_info:
            return {'error': 'Could not fetch exchange info'}
        
        # Find symbol info
        symbol_info = next((s for s in exchange_info.get('symbols', []) if s['symbol'] == symbol), None)
        if not symbol_info:
            return {'error': f'Symbol {symbol} not found in exchange info'}
        
        return {
            'symbol': symbol,
            'status': symbol_info.get('status'),
            'trading_venue': 'Spot & Futures' if symbol_info.get('isSpotTradingAllowed') and symbol_info.get('isFuturesTradingAllowed') else
                           'Spot Only' if symbol_info.get('isSpotTradingAllowed') else
                           'Futures Only' if symbol_info.get('isFuturesTradingAllowed') else 'Other',
            'isSpotTradingAllowed': symbol_info.get('isSpotTradingAllowed'),
            'isSuspended': symbol_info.get('isSuspended'),
            'isMarginTradingAllowed': symbol_info.get('isMarginTradingAllowed'),
            'isFuturesTradingAllowed': symbol_info.get('isFuturesTradingAllowed'),
            'permissions': symbol_info.get('permissions', [])
        }


def main():
    """Main function with command line argument parsing."""
    print("🚀 Binance Volume Spike Detector")
    print("=" * 50)
    
    # Parse command line arguments
    args = sys.argv[1:]
    
    # Default parameters
    periods = 50  # Need at least 50 periods for EMA 50 calculation
    spike_periods = 1  # Compare last candle to previous 4 candles
    threshold_pct = 100.0
    max_workers = 10  # Reduced to avoid API rate limits
    ema_periods = [9, 21, 50]  # Default EMA periods
    sort_by = 'volume_spike_pct'  # Default sort column
    sort_ascending = False  # Default sort order (descending)
    
    # ============================================================================
    # DEFAULT FILTERS - Define your filters here
    # ============================================================================
    # Example filters (uncomment and modify as needed):
    # filters = [
    #     {'column': 'vema_multiplier', 'operator': '>=', 'value': 2.0},
    #     {'column': 'volume_spike_pct', 'operator': '>=', 'value': 200.0},
    #     {'column': 'is_bullish', 'operator': '==', 'value': True},
    #     {'column': 'quote_volume', 'operator': '>=', 'value': 1000000},  # $1M+ 24h volume
    # ]
    
    # Available columns for filtering:
    # - vema_multiplier: VEMA multiplier (float)
    # - volume_spike_pct: Volume spike percentage (float)
    # - last_volume: Last 15m volume (float)
    # - avg_prev_volume: Average previous volume (float)
    # - quote_volume: 24h volume (float)
    # - price_change_percent: 24h price change (float)
    # - last_price: Current price (float)
    # - is_bullish: Bullish/bearish candle (bool)
    # - symbol: Trading pair symbol (string)
    
    # Available operators: '==', '!=', '>=', '<=', '>', '<', 'endswith', 'startswith', 'contains'
    
    # ============================================================================
    # DEFAULT SORTING - Define your sorting preferences here
    # ============================================================================
    # Available columns for sorting:
    # - volume_spike_pct: Volume spike percentage (default)
    # - vema_multiplier: VEMA multiplier
    # - last_volume: Last 15m volume
    # - avg_prev_volume: Average previous volume
    # - quote_volume: 24h volume
    # - price_change_percent: 24h price change
    # - last_price: Current price
    # - symbol: Trading pair symbol
    
    # Script-level sorting (uncomment and modify as needed):
    sort_by = 'vema_multiplier'  # Sort by VEMA multiplier
    sort_ascending = False       # False = descending (highest first), True = ascending (lowest first)
    
    # Example: Enable these filters by uncommenting the line below
    # filters = [
    #     {'column': 'vema_multiplier', 'operator': '>=', 'value': 2.0},
    #     {'column': 'volume_spike_pct', 'operator': '>=', 'value': 200.0},
    # ]
    
    # 🔥 ENABLE FILTERS HERE - Uncomment and modify the filters below:
    
    # Example 1: High volume spikes with strong VEMA multiplier
    filters = [
        {'column': 'vema_multiplier', 'operator': '>=', 'value': 2.0},
        {'column': 'volume_spike_pct', 'operator': '>=', 'value': 200.0},
        {'column': 'is_bullish', 'operator': '==', 'value': True},
    ]
    
    # Example 2: Only bullish candles with high volume
    # filters = [
    #     {'column': 'is_bullish', 'operator': '==', 'value': True},
    #     {'column': 'volume_spike_pct', 'operator': '>=', 'value': 100.0},
    #     {'column': 'quote_volume', 'operator': '>=', 'value': 1000000},  # $1M+ 24h volume
    # ]
    
    filters = filters  # Disable filters temporarily due to API rate limits
    
    # Example 3: High liquidity pairs with moderate spikes
    # filters = [
    #     {'column': 'quote_volume', 'operator': '>=', 'value': 5000000},  # $5M+ 24h volume
    #     {'column': 'volume_spike_pct', 'operator': '>=', 'value': 50.0},
    #     {'column': 'vema_multiplier', 'operator': '>=', 'value': 1.5},
    # ]
    
    # Example 4: Only USDT pairs with high spikes
    # filters = [
    #     {'column': 'symbol', 'operator': 'endswith', 'value': 'USDT'},
    #     {'column': 'volume_spike_pct', 'operator': '>=', 'value': 300.0},
    # ]
    
    # filters = None  # Set to None to disable default filters
    
    # Parse arguments
    single_symbol = None
    numeric_args = []
    
    for arg in args:
        if arg.startswith('workers:'):
            try:
                max_workers = int(arg.split(':')[1])
            except (ValueError, IndexError):
                pass
        elif arg.startswith('emas:'):
            try:
                ema_str = arg.split(':')[1]
                ema_periods = [int(x.strip()) for x in ema_str.split(',')]
            except (ValueError, IndexError):
                pass
        elif arg.startswith('filter:'):
            try:
                filter_str = arg.split(':', 1)[1]  # Split only on first colon
                # Format: filter:column:operator:value
                parts = filter_str.split(':')
                if len(parts) >= 3:
                    column = parts[0]
                    operator = parts[1]
                    value_str = ':'.join(parts[2:])  # Handle values that might contain colons
                    
                    # Try to convert value to appropriate type
                    try:
                        if '.' in value_str:
                            value = float(value_str)
                        else:
                            value = int(value_str)
                    except ValueError:
                        value = value_str  # Keep as string if conversion fails
                    
                    if filters is None:
                        filters = []
                    filters.append({'column': column, 'operator': operator, 'value': value})
            except (ValueError, IndexError):
                pass
        elif arg.startswith('sort:'):
            try:
                sort_str = arg.split(':', 1)[1]  # Split only on first colon
                # Format: sort:column:asc or sort:column:desc
                parts = sort_str.split(':')
                if len(parts) >= 2:
                    sort_by = parts[0]
                    sort_order = parts[1].lower() if len(parts) > 1 else 'desc'
                    sort_ascending = sort_order in ['asc', 'ascending', 'true', '1']
            except (ValueError, IndexError):
                pass
        elif arg.startswith('debug:'):
            symbol = arg.split(':')[1]
            detector = BinanceVolumeSpikeDetector(max_workers=max_workers)
            debug_info = detector.debug_volume_data(symbol)
            print(f"Debug info: {debug_info}")
            return
        elif arg.startswith('status:'):
            symbol = arg.split(':')[1]
            detector = BinanceVolumeSpikeDetector(max_workers=max_workers)
            status_info = detector.check_symbol_status(symbol)
            print(f"Status info: {status_info}")
            return
        elif arg.startswith('symbol:'):
            single_symbol = arg.split(':')[1]
        else:
            try:
                # Try to parse as number
                numeric_args.append(float(arg))
            except ValueError:
                # If it's not a number, it might be a symbol without the symbol: prefix
                if not arg.startswith('symbol:') and not arg.startswith('debug:') and not arg.startswith('status:') and not arg.startswith('workers:'):
                    single_symbol = arg
    
    # Apply numeric arguments
    if len(numeric_args) >= 1:
        periods = int(numeric_args[0])
    if len(numeric_args) >= 2:
        spike_periods = int(numeric_args[1])
    if len(numeric_args) >= 3:
        threshold_pct = numeric_args[2]
    
    print(f"📊 Parameters: {periods} data periods, {spike_periods} spike periods, {threshold_pct}% threshold, {max_workers} workers")
    print(f"📈 EMA periods: {ema_periods}")
    print(f"📊 Sorting: {sort_by} ({'ascending' if sort_ascending else 'descending'})")
    
    # Show active filters
    if filters:
        print(f"🔍 Active filters: {len(filters)} filter(s)")
        for i, filter_rule in enumerate(filters, 1):
            print(f"   {i}. {filter_rule['column']} {filter_rule['operator']} {filter_rule['value']}")
    else:
        print("🔍 No filters applied")
    
    # Create detector and get results
    detector = BinanceVolumeSpikeDetector(max_workers=max_workers)
    
    if single_symbol:
        print(f"🔍 Analyzing single symbol: {single_symbol}")
        results = detector.get_volume_spikes_single_symbol(
            symbol=single_symbol,
            periods=periods,
            spike_periods=spike_periods,
            threshold_pct=threshold_pct,
            ema_periods=ema_periods,
            sort_by=sort_by,
            sort_ascending=sort_ascending
        )
        title = f"🔥 Volume Spike Analysis for {single_symbol} (Last {periods} periods, spike: {spike_periods}, >= {threshold_pct}% threshold)"
    else:
        results = detector.get_volume_spikes(
            periods=periods,
            spike_periods=spike_periods,
            threshold_pct=threshold_pct,
            min_volume_usdt=100000,
            limit=50,
            ema_periods=ema_periods,
            filters=filters,
            sort_by=sort_by,
            sort_ascending=sort_ascending
        )
        title = f"🔥 Top Volume Spikes (Last {periods} periods, spike: {spike_periods}, >= {threshold_pct}% threshold)"
    
    # Print results
    detector.print_results(results, title, ema_periods=ema_periods)
    
    # Save to CSV
    if not results.empty:
        if single_symbol:
            filename = f"volume_spikes_{single_symbol}_{periods}periods_{threshold_pct}pct.csv"
        else:
            filename = f"volume_spikes_{periods}periods_{threshold_pct}pct.csv"
        results.to_csv(filename, index=False)
        print(f"\n💾 Results saved to {filename}")


if __name__ == "__main__":
    main()
