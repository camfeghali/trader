import pandas as pd
import time
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dataframe import KlineDataFrame

from models import BinanceKlineData

class AggregateDataFrame:
    def __init__(self, interval_in_minutes: int):
        self.interval_in_minutes = interval_in_minutes
        
        # Initialize columns for aggregated dataframe
        init_cols = [
            "symbol", "interval", 
            "open_time", "close_time", 
            "open_price", "high_price", "low_price", "close_price", 
            "volume", "quote_volume", 
            "trades_count", "taker_buy_volume", "taker_buy_quote_volume", "is_closed"
        ]
        self.df = pd.DataFrame(columns=init_cols)

    def fill_from_1m_dataframe(self, kline_df: 'KlineDataFrame') -> None:
        """
        Fill the aggregated dataframe with historical data from a 1-minute dataframe.
        This function processes all available 1-minute data and creates aggregated candles
        for all complete intervals, aligning with standard time boundaries.
        
        Args:
            kline_df: The KlineDataFrame containing 1-minute data
        """
        if len(kline_df.df) == 0:
            print(f"⚠️ No data available in 1-minute dataframe")
            return
        
        print(f"🔄 Filling {self.interval_in_minutes}-min aggregated dataframe from {len(kline_df.df)} 1-minute candles...")
        
        # Clear existing aggregated data
        self.df = pd.DataFrame(columns=self.df.columns)
        
        # Calculate how many 1-minute candles we need for each aggregated candle
        candles_needed = self.interval_in_minutes
        
        # Find the first row that starts at a proper interval boundary
        start_idx = 0
        for i in range(len(kline_df.df)):
            first_candle = kline_df.df.iloc[i]
            if self._is_interval_boundary(first_candle['open_time']):
                # Check if we have enough data to create a complete interval
                if i + candles_needed <= len(kline_df.df):
                    start_idx = i
                    print(f"Starting aggregation from row {i} (timestamp: {first_candle['open_time']})")
                    break
                else:
                    print(f"Found boundary at row {i} but not enough data for complete interval")
        
        # Calculate how many complete intervals we can create from the aligned start
        remaining_candles = len(kline_df.df) - start_idx
        complete_intervals = remaining_candles // candles_needed
        
        print(f"Starting from row {start_idx}, {remaining_candles} remaining candles, {complete_intervals} complete intervals")
        
        # Process complete intervals only, starting from the aligned position
        for interval_idx in range(complete_intervals):
            current_start_idx = start_idx + (interval_idx * candles_needed)
            current_end_idx = current_start_idx + candles_needed
            
            # Get the chunk of data for this interval
            chunk = kline_df.df.iloc[current_start_idx:current_end_idx]
            
            print(f"Debug: Interval {interval_idx}: selecting rows {current_start_idx} to {current_end_idx}")
            print(f"Debug: First candle in chunk: open_time={chunk.iloc[0]['open_time']}, close_time={chunk.iloc[0]['close_time']}")
            print(f"Debug: Last candle in chunk: open_time={chunk.iloc[-1]['open_time']}, close_time={chunk.iloc[-1]['close_time']}")
            
            # Verify we have a complete chunk
            if len(chunk) == candles_needed:
                # The first candle should already be at a boundary (we checked this above)
                first_candle = chunk.iloc[0]
                
                # Create aggregated kline using existing function - pass the first candle as reference
                aggregated_kline = self._create_aggregated_kline(chunk, first_candle)
                
                # Add to dataframe using existing function
                self.add_aggregated_kline(aggregated_kline)
            else:
                print(f"⚠️ Interval {interval_idx}: Incomplete chunk, got {len(chunk)} candles, need {candles_needed}")
        
        print(f"✅ Filled {self.interval_in_minutes}-min aggregated dataframe with {len(self.df)} candles")

    def aggregate(self, kline_df: 'KlineDataFrame') -> Optional[BinanceKlineData]:
        """
        Check the last row of kline_df and if its close_time matches the interval boundary,
        compute a new aggregated row by aggregating the necessary rows.
        
        Args:
            kline_df: The KlineDataFrame containing base interval data
            
        Returns:
            Aggregated BinanceKlineData if interval boundary is reached, None otherwise
        """
        # Get the last row from kline_df
        if len(kline_df.df) == 0:
            return None
            
        last_row = kline_df.df.iloc[-1]
        last_close_time = last_row['close_time']
        
        # Check if the close_time matches our interval boundary
        if not self._is_interval_boundary(last_close_time):
            return None
        
        # Calculate how many base candles we need for this interval
        # Assuming base interval is 1 minute
        candles_needed = self.interval_in_minutes
        
        # Get the required number of recent candles
        recent_data = kline_df.df.tail(candles_needed)
        
        # Check if we have enough data
        if len(recent_data) < candles_needed:
            print(f"⚠️ Insufficient data for {self.interval_in_minutes}-min aggregation. Need {candles_needed}, have {len(recent_data)}")
            return None
        
        # Verify the data spans the correct time range
        first_candle_time = recent_data.iloc[0]['open_time']
        last_candle_time = recent_data.iloc[-1]['close_time']
        
        # Check if the time range matches our expected interval
        time_span_ms = last_candle_time - first_candle_time
        expected_span_ms = (self.interval_in_minutes * 60 * 1000) - 1000  # Full interval minus 1ms for inclusive range
        
        # Allow for some tolerance in time span (1 second)
        tolerance_ms = 1000
        if abs(time_span_ms - expected_span_ms) > tolerance_ms:
            print(f"⚠️ Time span mismatch for {self.interval_in_minutes}-min aggregation. Expected {expected_span_ms}ms, got {time_span_ms}ms")
            print(f"   First candle: {first_candle_time}, Last candle: {last_candle_time}")
            return None
        
        # Create the aggregated kline
        aggregated_kline = self._create_aggregated_kline(recent_data, last_row)
        
        # Add to our aggregated dataframe
        self.add_aggregated_kline(aggregated_kline)
        
        print(f"📊 Aggregated {self.interval_in_minutes}-min candle: O:{aggregated_kline.open_price:.2f} H:{aggregated_kline.high_price:.2f} L:{aggregated_kline.low_price:.2f} C:{aggregated_kline.close_price:.2f}")
        
        return aggregated_kline

    def _is_interval_boundary(self, timestamp_ms: int) -> bool:
        """
        Check if a timestamp aligns with the target interval boundary.
        
        Args:
            timestamp_ms: Timestamp in milliseconds
            
        Returns:
            True if the timestamp aligns with the interval boundary
        """
        # Convert to minutes since epoch
        minutes_since_epoch = timestamp_ms // (1000 * 60)
        
        # Check if it's divisible by the target interval
        is_boundary = minutes_since_epoch % self.interval_in_minutes == 0
        
        return is_boundary

    def _create_aggregated_kline(self, candles_df: pd.DataFrame, last_row: pd.Series) -> BinanceKlineData:
        """
        Create an aggregated kline from multiple base candles.
        
        Args:
            candles_df: DataFrame containing the base candles to aggregate
            last_row: The last row from the base dataframe
            
        Returns:
            Aggregated BinanceKlineData
        """
        # Aggregate the OHLCV data
        open_price = candles_df.iloc[0]['open_price']
        high_price = candles_df['high_price'].max()
        low_price = candles_df['low_price'].min()
        close_price = candles_df.iloc[-1]['close_price']
        volume = candles_df['volume'].sum()
        quote_volume = candles_df['quote_volume'].sum()
        trades_count = candles_df['trades_count'].sum()
        taker_buy_volume = candles_df['taker_buy_volume'].sum()
        taker_buy_quote_volume = candles_df['taker_buy_quote_volume'].sum()
        
        first_candle_time = candles_df.iloc[0]['open_time']
        last_candle_time = candles_df.iloc[-1]['close_time']
        
        print(f"Debug: Aggregated result: open_time={first_candle_time}, close_time={last_candle_time}")
        
        # Create the aggregated kline
        aggregated_kline = BinanceKlineData(
            symbol=last_row['symbol'],
            interval=f"{self.interval_in_minutes}m",
            open_time=first_candle_time,
            close_time=last_candle_time,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume=volume,
            quote_volume=quote_volume,
            trades_count=trades_count,
            taker_buy_volume=taker_buy_volume,
            taker_buy_quote_volume=taker_buy_quote_volume,
            is_closed=True
        )
        
        return aggregated_kline

    def add_aggregated_kline(self, kline: BinanceKlineData):
        """
        Add an aggregated kline to the dataframe.
        
        Args:
            kline: The aggregated kline to add
        """
        # Check if this open_time already exists
        if len(self.df) > 0 and kline.open_time in self.df['open_time'].values:
            # Update existing row instead of adding duplicate
            mask = self.df['open_time'] == kline.open_time
            self.df.loc[mask, 'close_price'] = kline.close_price
            self.df.loc[mask, 'high_price'] = kline.high_price
            self.df.loc[mask, 'low_price'] = kline.low_price
            self.df.loc[mask, 'volume'] = kline.volume
            self.df.loc[mask, 'quote_volume'] = kline.quote_volume
            self.df.loc[mask, 'trades_count'] = kline.trades_count
            self.df.loc[mask, 'taker_buy_volume'] = kline.taker_buy_volume
            self.df.loc[mask, 'taker_buy_quote_volume'] = kline.taker_buy_quote_volume
            self.df.loc[mask, 'is_closed'] = kline.is_closed
            return
        
        # Convert the dataclass to a dictionary
        kline_dict = {
            'symbol': kline.symbol,
            'interval': kline.interval,
            'open_time': kline.open_time,
            'close_time': kline.close_time,
            'open_price': kline.open_price,
            'high_price': kline.high_price,
            'low_price': kline.low_price,
            'close_price': kline.close_price,
            'volume': kline.volume,
            'quote_volume': kline.quote_volume,
            'trades_count': kline.trades_count,
            'taker_buy_volume': kline.taker_buy_volume,
            'taker_buy_quote_volume': kline.taker_buy_quote_volume,
            'is_closed': kline.is_closed
        }
        
        # Add the new row to the dataframe
        new_row = pd.DataFrame([kline_dict])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        
        # Keep only last 100 rows (you can make this configurable)
        if len(self.df) > 100:
            self.df = self.df.tail(100)

    def get_dataframe(self) -> pd.DataFrame:
        """Get the underlying pandas DataFrame"""
        return self.df

    def get_latest(self, n: int = 5) -> pd.DataFrame:
        """Get the latest n rows"""
        return self.df.tail(n)

    def get_latest_price(self) -> Optional[float]:
        """Get the latest close price"""
        if len(self.df) > 0:
            return self.df.iloc[-1]['close_price']
        return None

    def is_interval_boundary(self, timestamp_ms: int) -> bool:
        """
        Public method to check if a timestamp is an interval boundary.
        
        Args:
            timestamp_ms: Timestamp in milliseconds
            
        Returns:
            True if the timestamp aligns with the interval boundary
        """
        return self._is_interval_boundary(timestamp_ms)

    def get_next_interval_close(self) -> int:
        """
        Get the timestamp of the next interval close.
        
        Returns:
            Timestamp in milliseconds of the next interval close
        """
        current_time = int(time.time() * 1000)
        current_minutes = current_time // (1000 * 60)
        
        # Find the next interval boundary
        next_interval_minutes = ((current_minutes // self.interval_in_minutes) + 1) * self.interval_in_minutes
        
        return next_interval_minutes * 60 * 1000

    def __str__(self):
        return f"AggregateDataFrame({self.interval_in_minutes}m) with {len(self.df)} rows:\n{self.df.to_string()}"

    def __repr__(self):
        return self.__str__()


    