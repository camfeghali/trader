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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Dict, Any, Optional, Tuple


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
        stablecoins = ["USDC"]
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
    
    def detect_volume_spikes(self, df: pd.DataFrame, periods: int = 5, threshold_pct: float = 100.0) -> pd.DataFrame:
        """Detect volume spikes by comparing last candle to previous N-1 candles."""
        print(f"🔍 Detecting volume spikes using {periods} periods with {threshold_pct}% threshold")
        print(f"🚀 Using {self.max_workers} parallel workers")
        
        symbols = df['symbol'].tolist()
        
        def calc(symbol: str) -> tuple:
            vols = self.get_historical_quote_volumes(symbol, periods)
            if len(vols) != periods:
                return symbol, np.nan, np.nan, np.nan, np.nan, np.nan
            
            last_volume = vols[-1]  # Most recent candle
            prev_volumes = vols[:-1]  # Previous N-1 candles
            avg_prev_volume = sum(prev_volumes) / len(prev_volumes) if prev_volumes else np.nan
            
            # Get price data for the last candle
            price_data = self.get_historical_price_data(symbol, periods)
            if len(price_data) != periods:
                return symbol, np.nan, last_volume, avg_prev_volume, len(prev_volumes), np.nan
            
            last_open = price_data[-1]['open']
            last_close = price_data[-1]['close']
            is_bullish = last_close > last_open
            
            if avg_prev_volume and avg_prev_volume > 0:
                spike_pct = ((last_volume - avg_prev_volume) / avg_prev_volume) * 100
                return symbol, round(spike_pct, 1), last_volume, avg_prev_volume, len(prev_volumes), is_bullish
            else:
                return symbol, np.nan, last_volume, avg_prev_volume, len(prev_volumes), is_bullish
        
        spike_results = {}
        last_volume_results = {}
        avg_volume_results = {}
        periods_used_results = {}
        bullish_results = {}
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(calc, s): s for s in symbols}
            for f in as_completed(futures):
                sym, spike_pct, last_vol, avg_vol, periods_used, is_bullish = f.result()
                spike_results[sym] = spike_pct
                last_volume_results[sym] = last_vol
                avg_volume_results[sym] = avg_vol
                periods_used_results[sym] = periods_used
                bullish_results[sym] = is_bullish
                completed += 1
                if completed % 20 == 0 or completed == len(symbols):
                    print(f"   Progress: {completed}/{len(symbols)}")
        
        df["volume_spike_pct"] = [spike_results.get(s, np.nan) for s in symbols]
        df["last_volume"] = [last_volume_results.get(s, np.nan) for s in symbols]
        df["avg_prev_volume"] = [avg_volume_results.get(s, np.nan) for s in symbols]
        df["periods_used"] = [periods_used_results.get(s, np.nan) for s in symbols]
        df["is_bullish"] = [bullish_results.get(s, np.nan) for s in symbols]
        
        return df
    
    def get_volume_spikes(self, periods: int = 5, threshold_pct: float = 100.0, 
                         min_volume_usdt: float = 100000, limit: int = 50) -> pd.DataFrame:
        """
        Get trading pairs with volume spikes above threshold.
        
        Args:
            periods: Number of 15-minute periods to analyze (default: 5)
            threshold_pct: Minimum spike percentage to include (default: 100%)
            min_volume_usdt: Minimum 24h volume in USDT to filter by
            limit: Maximum number of results to return
            
        Returns:
            DataFrame sorted by volume spike percentage (descending)
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
        df = self.detect_volume_spikes(df, periods=periods, threshold_pct=threshold_pct)
        
        # Filter for positive spikes above threshold only
        df_spikes = df[(df['volume_spike_pct'] >= threshold_pct) & (df['volume_spike_pct'] > 0)].copy()
        print(f"📈 Found {len(df_spikes)} pairs with positive volume spikes >= {threshold_pct}%")
        
        # Filter by minimum average volume (liquidity filter)
        min_avg_volume = 10000  # $10,000 minimum average volume
        df_liquid = df_spikes[df_spikes['avg_prev_volume'] >= min_avg_volume].copy()
        print(f"💧 Filtered to {len(df_liquid)} pairs with avg volume >= ${min_avg_volume:,.0f}")
        
        # Sort by spike percentage
        df_sorted = df_liquid.sort_values('volume_spike_pct', ascending=False)
        print("📊 Sorted by volume spike percentage (highest spikes first)")
        
        # Select top N
        top_spikes = df_sorted.head(limit)
        
        return top_spikes
    
    def get_volume_spikes_single_symbol(self, symbol: str, periods: int = 5, threshold_pct: float = 100.0) -> pd.DataFrame:
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
        df = self.detect_volume_spikes(df, periods=periods, threshold_pct=threshold_pct)
        
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
    
    def print_results(self, df: pd.DataFrame, title: str):
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
        
        # Select and order columns
        columns_to_show = ['symbol', 'price_change_percent', 'last_price', 'quote_volume', 
                          'volume_spike_pct', 'last_volume', 'avg_prev_volume', 'trading_venue', 
                          'high_24h', 'low_24h']
        display_df = display_df[columns_to_show]
        display_df.columns = ['Symbol', '24h Change %', 'Last Price', '24h Volume', 
                             'Volume Spike %', 'Last 15m Vol', 'Avg Prev Vol', 'Trading Venue', 
                             '24h High', '24h Low']
        
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
        
        # Get historical volumes
        volumes = self.get_historical_quote_volumes(symbol, periods=5)
        if not volumes:
            return {'error': f'Could not fetch historical data for {symbol}'}
        
        # Calculate spike
        last_volume = volumes[-1]
        prev_volumes = volumes[:-1]
        avg_prev_volume = sum(prev_volumes) / len(prev_volumes) if prev_volumes else 0
        spike_pct = ((last_volume - avg_prev_volume) / avg_prev_volume * 100) if avg_prev_volume > 0 else 0
        
        return {
            'symbol': symbol,
            'ticker_24h_volume': float(ticker_data['quoteVolume']),
            'last_15m_volume': last_volume,
            'avg_prev_4_volumes': avg_prev_volume,
            'volume_spike_pct': round(spike_pct, 1),
            'volumes': volumes
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
    periods = 3
    threshold_pct = 100.0
    max_workers = 10
    
    # Parse arguments
    single_symbol = None
    numeric_args = []
    
    for arg in args:
        if arg.startswith('workers:'):
            try:
                max_workers = int(arg.split(':')[1])
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
        threshold_pct = numeric_args[1]
    
    print(f"📊 Parameters: {periods} periods, {threshold_pct}% threshold, {max_workers} workers")
    
    # Create detector and get results
    detector = BinanceVolumeSpikeDetector(max_workers=max_workers)
    
    if single_symbol:
        print(f"🔍 Analyzing single symbol: {single_symbol}")
        results = detector.get_volume_spikes_single_symbol(
            symbol=single_symbol,
            periods=periods,
            threshold_pct=threshold_pct
        )
        title = f"🔥 Volume Spike Analysis for {single_symbol} (Last {periods} periods, >= {threshold_pct}% threshold)"
    else:
        results = detector.get_volume_spikes(
            periods=periods,
            threshold_pct=threshold_pct,
            min_volume_usdt=100000,
            limit=50
        )
        title = f"🔥 Top Volume Spikes (Last {periods} periods, >= {threshold_pct}% threshold)"
    
    # Print results
    detector.print_results(results, title)
    
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
