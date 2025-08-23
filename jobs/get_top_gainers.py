#!/usr/bin/env python3
"""
Script to fetch all trading pairs from Binance and sort them by 24-hour price change.
This helps identify the best performing cryptocurrencies in the last 24 hours.

Features:
- Parallel processing for RVOL calculations (configurable workers)
- Support for all major stablecoins and BNB pairs
- Caching for historical volume data
- Thread-safe operations
- Rate limit respecting API calls

Usage:
    python get_top_gainers.py                    # Default: 10 workers
    python get_top_gainers.py workers:20        # Use 20 parallel workers
    python get_top_gainers.py BTCUSDT           # Search specific symbol
    python get_top_gainers.py status:BTCUSDT    # Check symbol status
"""

import requests
import pandas as pd
from typing import List, Dict, Any
import time
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class BinanceTopGainers:
    def __init__(self, max_workers: int = 10):
        self.base_url = "https://api.binance.com/api/v3"
        self._volume_cache = {}  # Cache for historical volume data
        self._cache_lock = threading.Lock()  # Thread-safe cache access
        self.max_workers = max_workers  # Number of parallel workers for RVOL calculation
        
    def get_all_tickers(self) -> List[Dict[str, Any]]:
        """
        Fetch 24hr ticker price change statistics for all symbols.
        
        Returns:
            List of dictionaries containing ticker information
        """
        try:
            url = f"{self.base_url}/ticker/24hr"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            tickers = response.json()
            print(f"✅ Fetched {len(tickers)} trading pairs from Binance")
            return tickers
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching tickers: {e}")
            return []
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """
        Fetch exchange information to get detailed status of all symbols.
        
        Returns:
            Dictionary containing exchange information
        """
        try:
            url = f"{self.base_url}/exchangeInfo"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            exchange_info = response.json()
            print(f"✅ Fetched exchange info with {len(exchange_info.get('symbols', []))} symbols")
            return exchange_info
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching exchange info: {e}")
            return {}
    
    def filter_listed_pairs(self, tickers: List[Dict[str, Any]], exchange_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter to only pairs that are currently tradeable on spot or futures.
        
        Args:
            tickers: List of ticker data
            exchange_info: Exchange information from Binance
            
        Returns:
            List of tradeable pairs with trading venue information
        """
        if not exchange_info or 'symbols' not in exchange_info:
            print("⚠️ No exchange info available, using basic filtering")
            return self.filter_active_pairs(tickers)
        
        # Create dictionaries to store trading venue info
        symbol_venue_info = {}
        tradeable_symbols = set()
        spot_only_symbols = set()
        futures_only_symbols = set()
        suspended_symbols = set()
        other_status_symbols = set()
        
        for symbol_info in exchange_info['symbols']:
            symbol = symbol_info['symbol']
            status = symbol_info['status']
            is_spot_allowed = symbol_info.get('isSpotTradingAllowed', False)
            is_futures_allowed = symbol_info.get('isFuturesTradingAllowed', False)
            is_suspended = symbol_info.get('isSuspended', False)
            
            # Check if symbol is tradeable (either spot or futures)
            if status == 'TRADING' and not is_suspended:
                if is_spot_allowed and is_futures_allowed:
                    tradeable_symbols.add(symbol)
                    symbol_venue_info[symbol] = "Spot & Futures"
                elif is_spot_allowed:
                    spot_only_symbols.add(symbol)
                    symbol_venue_info[symbol] = "Spot Only"
                elif is_futures_allowed:
                    futures_only_symbols.add(symbol)
                    symbol_venue_info[symbol] = "Futures Only"
            elif status == 'SUSPENDED' or is_suspended:
                suspended_symbols.add(symbol)
                symbol_venue_info[symbol] = "Suspended"
            else:
                other_status_symbols.add(symbol)
                symbol_venue_info[symbol] = "Other"
        
        # Combine all tradeable symbols
        all_tradeable = tradeable_symbols | spot_only_symbols | futures_only_symbols
        
        print(f"📋 Found {len(all_tradeable)} tradeable symbols:")
        print(f"   - {len(tradeable_symbols)} tradeable on both spot & futures")
        print(f"   - {len(spot_only_symbols)} tradeable on spot only")
        print(f"   - {len(futures_only_symbols)} tradeable on futures only")
        print(f"🚫 Found {len(suspended_symbols)} suspended symbols")
        print(f"❓ Found {len(other_status_symbols)} symbols with other status")        
        
        # Filter tickers to only include tradeable symbols and add venue info
        tradeable_pairs = []
        for ticker in tickers:
            if ticker['symbol'] in all_tradeable:
                ticker['trading_venue'] = symbol_venue_info.get(ticker['symbol'], "Unknown")
                tradeable_pairs.append(ticker)
        
        print(f"✅ Filtered to {len(tradeable_pairs)} tradeable pairs")
        return tradeable_pairs
    
    def filter_stablecoin_pairs(self, tickers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter to trading pairs with stablecoins and BNB as quote assets.
        
        Args:
            tickers: List of all ticker data
            
        Returns:
            List of stablecoin and BNB trading pairs
        """
        stablecoins = [
            "USDT",   # Tether
            "USDC",   # USD Coin
            "FDUSD",  # First Digital USD
            "DAI",    # Dai
            "TUSD",   # TrueUSD
            "USD1",   # World Liberty Financial USD
            "XUSD",   # StraitsX USD
            "EURI"    # Eurite (EUR-backed)
        ]
        
        # Add BNB to the list
        quote_assets = stablecoins + ["BNB"]
        
        stablecoin_pairs = []
        for ticker in tickers:
            for asset in quote_assets:
                if ticker['symbol'].endswith(asset):
                    stablecoin_pairs.append(ticker)
                    break
        
        # Count pairs by quote asset
        asset_counts = {}
        for ticker in stablecoin_pairs:
            for asset in quote_assets:
                if ticker['symbol'].endswith(asset):
                    asset_counts[asset] = asset_counts.get(asset, 0) + 1
                    break
        
        print(f"📊 Found {len(stablecoin_pairs)} stablecoin/BNB trading pairs:")
        for asset, count in sorted(asset_counts.items()):
            print(f"   - {asset}: {count} pairs")
        
        return stablecoin_pairs
    
    def filter_active_pairs(self, tickers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter to only active trading pairs (volume > 0 and status is TRADING).
        
        Args:
            tickers: List of ticker data
            
        Returns:
            List of active trading pairs only
        """
        active_pairs = [
            ticker for ticker in tickers 
            if (float(ticker['volume']) > 0 and 
                float(ticker['quoteVolume']) > 0 and
                ticker.get('status', 'TRADING') == 'TRADING')
        ]
        print(f"🔄 Found {len(active_pairs)} active trading pairs")
        return active_pairs
    
    def create_dataframe(self, tickers: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert ticker data to a pandas DataFrame for easier analysis.
        
        Args:
            tickers: List of ticker data
            
        Returns:
            DataFrame with ticker information including trading venue
        """
        # Extract relevant fields
        data = []
        for ticker in tickers:
            data.append({
                'symbol': ticker['symbol'],
                'price_change': float(ticker['priceChange']),
                'price_change_percent': float(ticker['priceChangePercent']),
                'weighted_avg_price': float(ticker['weightedAvgPrice']),
                'prev_close_price': float(ticker['prevClosePrice']),
                'last_price': float(ticker['lastPrice']),
                'volume': float(ticker['volume']),
                'quote_volume': float(ticker['quoteVolume']),
                'high_24h': float(ticker['highPrice']),
                'low_24h': float(ticker['lowPrice']),
                'count': int(ticker['count']),
                'trading_venue': ticker.get('trading_venue', 'Unknown')
            })
        
        df = pd.DataFrame(data)
        return df
    
    def get_historical_quote_volumes(self, symbol: str, days: int = 20) -> List[float]:
        """
        Return the last `days` *closed* daily candles' quote-asset volumes (Binance kline[7]).
        Uses caching to avoid redundant API calls. Thread-safe.
        """
        # Check cache first (thread-safe)
        cache_key = f"{symbol}_{days}"
        with self._cache_lock:
            if cache_key in self._volume_cache:
                return self._volume_cache[cache_key]
        
        try:
            now_ms = int(time.time() * 1000)
            day_ms = 24 * 60 * 60 * 1000
            # end at today's 00:00:00 UTC to exclude today's still-open candle
            today_open_ms = (now_ms // day_ms) * day_ms

            url = f"{self.base_url}/klines"
            params = {
                "symbol": symbol,
                "interval": "1d",
                "endTime": today_open_ms,   # exclude today's candle
                "limit": days               # get exactly the last `days` closed candles
            }
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            klines = resp.json()

            # kline[7] = quote asset volume
            volumes = [float(k[7]) for k in klines]
            
            # Cache the result (thread-safe)
            with self._cache_lock:
                self._volume_cache[cache_key] = volumes
            return volumes
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching historical data for {symbol}: {e}")
            return []

    def calculate_rvol(self, df: pd.DataFrame, days: int = 20) -> pd.DataFrame:
        """
        RVOL = today's *daily-candle* quote volume / avg(quote volume of last `days` closed candles)
        NOTE: Requires `df` to contain today's *daily candle* quote volume, not rolling 24h.
        Uses parallel processing for faster execution.
        """
        import numpy as np

        print(f"📊 Calculating RVOL for {len(df)} pairs using {days}-day average (daily candles)...")
        print(f"🚀 Using {self.max_workers} parallel workers")
        
        # Prepare data for parallel processing
        symbols = df['symbol'].tolist()
        current_volumes = df['quote_volume'].tolist()
        
        # Function to calculate RVOL for a single symbol
        def calculate_single_rvol(symbol: str, current_volume: float) -> tuple:
            vols = self.get_historical_quote_volumes(symbol, days)
            if vols and len(vols) > 0:
                avg_qv = sum(vols) / len(vols)
                rvol = (current_volume / avg_qv) if avg_qv > 0 else np.nan
                return symbol, round(rvol, 2)
            else:
                return symbol, np.nan
        
        # Process in parallel
        rvol_results = {}
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(calculate_single_rvol, symbol, current_volume): symbol 
                for symbol, current_volume in zip(symbols, current_volumes)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol, rvol = future.result()
                rvol_results[symbol] = rvol
                completed += 1
                
                # Progress indicator
                if completed % 10 == 0 or completed == len(symbols):
                    print(f"   Progress: {completed}/{len(symbols)} pairs processed")
        
        # Add RVOL values to DataFrame in correct order
        df["rvol"] = [rvol_results.get(symbol, np.nan) for symbol in symbols]
        return df

    
    def get_top_gainers(self, limit: int = 20, min_volume_usdt: float = 100000, use_exchange_info: bool = True, sort_by_rvol: bool = False, rvol_days: int = 20) -> pd.DataFrame:
        """
        Get the top gaining trading pairs in the last 24 hours.
        
        Args:
            limit: Number of top gainers to return
            min_volume_usdt: Minimum 24h volume in USDT to filter by
            use_exchange_info: Whether to use exchange info for better filtering
            sort_by_rvol: Whether to sort by RVOL instead of price change
            rvol_days: Number of days to use for RVOL calculation (default: 20)
            
        Returns:
            DataFrame sorted by price change percentage (descending) or RVOL (descending)
        """
        print("🚀 Fetching all trading pairs from Binance...")
        
        # Get all tickers
        all_tickers = self.get_all_tickers()
        if not all_tickers:
            return pd.DataFrame()
        
        # Get exchange info for better filtering
        exchange_info = {}
        if use_exchange_info:
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
        
        # Calculate RVOL
        df = self.calculate_rvol(df, days=rvol_days)
        
        # Sort by price change percentage or RVOL
        if sort_by_rvol:
            df_sorted = df.sort_values('rvol', ascending=False)
            print("📊 Sorted by RVOL (highest relative volume first)")
        else:
            df_sorted = df.sort_values('price_change_percent', ascending=False)
            print("📈 Sorted by price change percentage (highest gainers first)")
        
        # Select top N
        top_gainers = df_sorted.head(limit)
        
        return top_gainers
    
    def get_top_volume_movers(self, limit: int = 20, min_volume_usdt: float = 100000, use_exchange_info: bool = True, rvol_days: int = 20) -> pd.DataFrame:
        """
        Get the trading pairs with the highest relative volume (RVOL).
        
        Args:
            limit: Number of top volume movers to return
            min_volume_usdt: Minimum 24h volume in USDT to filter by
            use_exchange_info: Whether to use exchange info for better filtering
            rvol_days: Number of days to use for RVOL calculation (default: 20)
            
        Returns:
            DataFrame sorted by RVOL (descending)
        """
        print("📊 Fetching pairs with highest relative volume...")
        
        # Get all tickers
        all_tickers = self.get_all_tickers()
        if not all_tickers:
            return pd.DataFrame()
        
        # Get exchange info for better filtering
        exchange_info = {}
        if use_exchange_info:
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
        
        # Calculate RVOL
        df = self.calculate_rvol(df, days=rvol_days)
        
        # Sort by RVOL (descending)
        df_sorted = df.sort_values('rvol', ascending=False)
        
        # Select top N
        top_volume_movers = df_sorted.head(limit)
        
        return top_volume_movers
    
    def get_top_losers(self, limit: int = 20, min_volume_usdt: float = 100000, use_exchange_info: bool = True, rvol_days: int = 20) -> pd.DataFrame:
        """
        Get the top losing trading pairs in the last 24 hours.
        
        Args:
            limit: Number of top losers to return
            min_volume_usdt: Minimum 24h volume in USDT to filter by
            use_exchange_info: Whether to use exchange info for better filtering
            rvol_days: Number of days to use for RVOL calculation (default: 20)
            
        Returns:
            DataFrame sorted by price change percentage (ascending)
        """
        print("📉 Fetching all trading pairs from Binance...")
        
        # Get all tickers
        all_tickers = self.get_all_tickers()
        if not all_tickers:
            return pd.DataFrame()
        
        # Get exchange info for better filtering
        exchange_info = {}
        if use_exchange_info:
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
        
        # Calculate RVOL
        df = self.calculate_rvol(df, days=rvol_days)
        
        # Sort by price change percentage (ascending for losers)
        df_sorted = df.sort_values('price_change_percent', ascending=True)
        
        # Select top N
        top_losers = df_sorted.head(limit)
        
        return top_losers
    
    def print_results(self, df: pd.DataFrame, title: str):
        """
        Print the results in a formatted table.
        
        Args:
            df: DataFrame with results
            title: Title for the output
        """
        if df.empty:
            print(f"❌ No data found for {title}")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 {title}")
        print(f"{'='*80}")
        print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"{'='*80}")
        
        # Format the DataFrame for display
        display_df = df.copy()
        display_df['price_change_percent'] = display_df['price_change_percent'].apply(lambda x: f"{x:+.2f}%")
        display_df['last_price'] = display_df['last_price'].apply(lambda x: f"${x:.6f}")
        display_df['quote_volume'] = display_df['quote_volume'].apply(lambda x: f"${x:,.0f}")
        display_df['high_24h'] = display_df['high_24h'].apply(lambda x: f"${x:.6f}")
        display_df['low_24h'] = display_df['low_24h'].apply(lambda x: f"${x:.6f}")
        
        # Format RVOL with emphasis for high values
        def format_rvol(x):
            if x >= 5.0:
                return f"🔥 {x:.1f}X"  # Very high volume
            elif x >= 3.0:
                return f"⚡ {x:.1f}X"  # High volume
            elif x >= 2.0:
                return f"📈 {x:.1f}X"  # Above average volume
            elif x >= 1.5:
                return f"📊 {x:.1f}X"  # Slightly above average
            else:
                return f"{x:.1f}X"     # Normal or below average
        
        display_df['rvol'] = display_df['rvol'].apply(format_rvol)
        
        # Select columns to display
        columns_to_show = ['symbol', 'price_change_percent', 'last_price', 'quote_volume', 'rvol', 'trading_venue', 'high_24h', 'low_24h']
        display_df = display_df[columns_to_show]
        
        # Rename columns for better display
        display_df.columns = ['Symbol', '24h Change %', 'Last Price', '24h Volume', 'RVOL (vs 20d avg)', 'Trading Venue', '24h High', '24h Low']
        
        print(display_df.to_string(index=False))
        print(f"{'='*80}")
        
        # Add RVOL statistics summary
        if 'rvol' in df.columns:
            rvol_values = df['rvol'].dropna()
            if len(rvol_values) > 0:
                print(f"\n📊 RVOL Statistics:")
                print(f"   Highest RVOL: {rvol_values.max():.1f}X")
                print(f"   Average RVOL: {rvol_values.mean():.1f}X")
                print(f"   Median RVOL: {rvol_values.median():.1f}X")
                print(f"   Pairs with RVOL > 2X: {len(rvol_values[rvol_values > 2.0])}")
                print(f"   Pairs with RVOL > 3X: {len(rvol_values[rvol_values > 3.0])}")
                print(f"   Pairs with RVOL > 5X: {len(rvol_values[rvol_values > 5.0])}")
    
    def save_results(self, df: pd.DataFrame, filename: str):
        """
        Save results to a CSV file.
        
        Args:
            df: DataFrame with results
            filename: Name of the file to save
        """
        if df.empty:
            print(f"❌ No data to save for {filename}")
            return
        
        try:
            df.to_csv(filename, index=False)
            print(f"💾 Results saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def check_symbol_status(self, symbol: str) -> Dict[str, Any]:
        """
        Check the detailed status of a specific symbol.
        
        Args:
            symbol: Symbol to check (e.g., 'TKOUSDT')
            
        Returns:
            Dictionary with symbol status information
        """
        print(f"🔍 Checking status for {symbol}...")
        
        # Get exchange info
        exchange_info = self.get_exchange_info()
        
        if not exchange_info or 'symbols' not in exchange_info:
            return {"error": "Could not fetch exchange info"}
        
        # Find the symbol in exchange info
        for symbol_info in exchange_info['symbols']:
            if symbol_info['symbol'] == symbol.upper():
                # Determine trading venue
                is_spot_allowed = symbol_info.get('isSpotTradingAllowed', False)
                is_futures_allowed = symbol_info.get('isFuturesTradingAllowed', False)
                
                if is_spot_allowed and is_futures_allowed:
                    trading_venue = "Spot & Futures"
                elif is_spot_allowed:
                    trading_venue = "Spot Only"
                elif is_futures_allowed:
                    trading_venue = "Futures Only"
                else:
                    trading_venue = "Not Tradeable"
                
                return {
                    "symbol": symbol_info['symbol'],
                    "status": symbol_info['status'],
                    "trading_venue": trading_venue,
                    "isSpotTradingAllowed": symbol_info.get('isSpotTradingAllowed', False),
                    "isSuspended": symbol_info.get('isSuspended', False),
                    "isMarginTradingAllowed": symbol_info.get('isMarginTradingAllowed', False),
                    "isFuturesTradingAllowed": symbol_info.get('isFuturesTradingAllowed', False),
                    "permissions": symbol_info.get('permissions', [])
                }
        
        return {"error": f"Symbol {symbol} not found in exchange info"}
    
    def search_symbol(self, symbol: str, use_exchange_info: bool = True) -> pd.DataFrame:
        """
        Search for a specific symbol and show its 24h performance.
        
        Args:
            symbol: Symbol to search for (e.g., 'CREAMUSDT')
            use_exchange_info: Whether to use exchange info for better filtering
            
        Returns:
            DataFrame with the symbol's data if found
        """
        print(f"🔍 Searching for {symbol}...")
        
        # Get all tickers
        all_tickers = self.get_all_tickers()
        if not all_tickers:
            return pd.DataFrame()
        
        # Get exchange info for better filtering
        exchange_info = {}
        if use_exchange_info:
            exchange_info = self.get_exchange_info()
        
        # Filter to stablecoin/BNB pairs only
        stablecoin_pairs = self.filter_stablecoin_pairs(all_tickers)
        
        # Filter to listed pairs if exchange info is available
        if exchange_info:
            listed_pairs = self.filter_listed_pairs(stablecoin_pairs, exchange_info)
        else:
            listed_pairs = self.filter_active_pairs(stablecoin_pairs)
        
        # Search for the specific symbol
        symbol_data = None
        for ticker in listed_pairs:
            if ticker['symbol'] == symbol.upper():
                symbol_data = ticker
                break
        
        if symbol_data:
            print(f"✅ Found {symbol.upper()} in listed pairs")
            df = self.create_dataframe([symbol_data])
            return df
        else:
            print(f"❌ {symbol.upper()} not found in listed pairs")
            
            # Check if it exists in all tickers but not in listed pairs
            for ticker in all_tickers:
                if ticker['symbol'] == symbol.upper():
                    print(f"⚠️ {symbol.upper()} exists but is not currently listed/trading")
                    print(f"   Status: {ticker.get('status', 'Unknown')}")
                    break
            else:
                print(f"❌ {symbol.upper()} not found in any Binance pairs")
            
            return pd.DataFrame()

def main():
    """Main function to run the top gainers analysis."""
    import sys
    
    print("🚀 Binance Top Gainers Analysis")
    print("=" * 50)
    
    # Configuration
    RVOL_DAYS = 15  # Number of days for RVOL calculation
    LIMIT = 15      # Number of results to show
    MIN_VOLUME = 100000  # Minimum volume filter
    MAX_WORKERS = 10  # Number of parallel workers for RVOL calculation
    
    # Parse command line arguments for workers
    if len(sys.argv) > 1 and sys.argv[1].startswith("workers:"):
        try:
            MAX_WORKERS = int(sys.argv[1].split(":")[1])
            print(f"🔧 Using {MAX_WORKERS} parallel workers")
            # Remove the workers argument from sys.argv
            sys.argv.pop(1)
        except (ValueError, IndexError):
            print("⚠️ Invalid workers format. Using default: 10")
    
    # Initialize the analyzer with parallel processing
    analyzer = BinanceTopGainers(max_workers=MAX_WORKERS)
    
    # Check if a symbol was provided as command line argument
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
        
        # Check if it's a status check (symbol starts with "status:")
        if symbol.startswith("status:"):
            symbol_to_check = symbol[7:]  # Remove "status:" prefix
            print(f"🔍 Checking status for: {symbol_to_check}")
            status_info = analyzer.check_symbol_status(symbol_to_check)
            print(f"Status info: {status_info}")
            return
        
        print(f"🔍 Searching for specific symbol: {symbol}")
        
        # Search for the specific symbol
        symbol_data = analyzer.search_symbol(symbol)
        if not symbol_data.empty:
            analyzer.print_results(symbol_data, f"{symbol.upper()} PERFORMANCE (24h)")
        return
    
    # Get top gainers (only listed pairs)
    print("📈 Getting top gainers (listed pairs only)...")
    top_gainers = analyzer.get_top_gainers(
        limit=LIMIT, 
        min_volume_usdt=MIN_VOLUME, 
        use_exchange_info=True, 
        sort_by_rvol=True,
        rvol_days=RVOL_DAYS
    )
    analyzer.print_results(top_gainers, f"TOP {LIMIT} GAINERS (24h) - LISTED PAIRS ONLY")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analyzer.save_results(top_gainers, f"top_gainers_listed_{timestamp}.csv")
    
    # Get top volume movers (highest RVOL)
    print("\n" + "="*50)
    print("📊 Getting top volume movers (highest RVOL)...")
    top_volume_movers = analyzer.get_top_volume_movers(
        limit=LIMIT, 
        min_volume_usdt=MIN_VOLUME, 
        use_exchange_info=True,
        rvol_days=RVOL_DAYS
    )
    analyzer.print_results(top_volume_movers, f"TOP {LIMIT} VOLUME MOVERS (HIGHEST RVOL) - LISTED PAIRS ONLY")
    
    # Save results
    analyzer.save_results(top_volume_movers, f"top_volume_movers_{timestamp}.csv")
    
    # Get top losers (only listed pairs)
    print("\n" + "="*50)
    print("📉 Getting top losers (listed pairs only)...")
    top_losers = analyzer.get_top_losers(
        limit=LIMIT, 
        min_volume_usdt=MIN_VOLUME, 
        use_exchange_info=True,
        rvol_days=RVOL_DAYS
    )
    analyzer.print_results(top_losers, f"TOP {LIMIT} LOSERS (24h) - LISTED PAIRS ONLY")
    
    # Save results
    analyzer.save_results(top_losers, f"top_losers_listed_{timestamp}.csv")
    
    print(f"\n✅ Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"\n📊 Analysis Configuration:")
    print(f"   - RVOL Period: {RVOL_DAYS} days")
    print(f"   - Results Limit: {LIMIT} pairs")
    print(f"   - Min Volume: ${MIN_VOLUME:,.0f}")
    print(f"\n💡 To search for a specific symbol, run:")
    print(f"   python {sys.argv[0]} CREAMUSDT")
    print(f"\n💡 To check symbol status, run:")
    print(f"   python {sys.argv[0]} status:CREAMUSDT")

if __name__ == "__main__":
    main() 