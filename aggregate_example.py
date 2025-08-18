"""
Example showing how to use the aggregate function in AggregateDataFrame
"""
import time
from datetime import datetime, timezone
from aggregated_dataframe import AggregateDataFrame
from dataframe import KlineDataFrame
from ta_calculator import TaCalculator
from daos import BinanceDataProcessor
from models import BinanceKlineData

def timestamp_to_readable(ts_ms: int) -> str:
    """Convert timestamp to readable format"""
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

def create_sample_klines():
    """Create sample 1-minute klines for testing"""
    base_time = int(time.time() * 1000) // (1000 * 60) * (1000 * 60)  # Round to nearest minute
    symbol = "btcusdt"
    
    klines = []
    for i in range(10):  # Create 10 minutes of data
        kline_time = base_time + (i * 60 * 1000)
        kline = BinanceKlineData(
            symbol=symbol,
            interval="1m",
            open_time=kline_time,
            close_time=kline_time + (60 * 1000) - 1,
            open_price=50000.0 + (i * 10),  # Increasing price
            high_price=50000.0 + (i * 10) + 50,
            low_price=50000.0 + (i * 10) - 20,
            close_price=50000.0 + (i * 10) + 25,
            volume=100.0 + (i * 5),
            quote_volume=5000000.0 + (i * 50000),
            trades_count=1000 + (i * 100),
            taker_buy_volume=50.0 + (i * 2.5),
            taker_buy_quote_volume=2500000.0 + (i * 25000),
            is_closed=True
        )
        klines.append(kline)
    
    return klines

def main():
    # Initialize base dataframe (1-minute data)
    base_df = KlineDataFrame(TaCalculator(), BinanceDataProcessor())
    
    # Create aggregated dataframes for different intervals
    agg_3m = AggregateDataFrame(interval_in_minutes=3)
    agg_5m = AggregateDataFrame(interval_in_minutes=5)
    
    # Create sample klines
    sample_klines = create_sample_klines()
    
    print("Adding sample 1-minute klines to base dataframe...")
    for i, kline in enumerate(sample_klines):
        print(f"\n[{i+1}/10] Adding kline: {timestamp_to_readable(kline.open_time)} - O:{kline.open_price:.2f} C:{kline.close_price:.2f}")
        base_df.add_row(kline)
        
        # Check if this triggers any aggregations
        print(f"  Checking 3-minute aggregation...")
        agg_3m_result = agg_3m.aggregate(base_df)
        if agg_3m_result:
            print(f"  ✅ 3-minute aggregated: O:{agg_3m_result.open_price:.2f} C:{agg_3m_result.close_price:.2f}")
        
        print(f"  Checking 5-minute aggregation...")
        agg_5m_result = agg_5m.aggregate(base_df)
        if agg_5m_result:
            print(f"  ✅ 5-minute aggregated: O:{agg_5m_result.open_price:.2f} C:{agg_5m_result.close_price:.2f}")
    
    print(f"\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    print(f"Base dataframe (1m): {len(base_df.df)} rows")
    print(f"3-minute aggregated: {len(agg_3m.df)} rows")
    print(f"5-minute aggregated: {len(agg_5m.df)} rows")
    
    # Show the aggregated data
    if len(agg_3m.df) > 0:
        print(f"\n3-minute aggregated data:")
        for _, row in agg_3m.df.iterrows():
            print(f"  {timestamp_to_readable(row['open_time'])}: O:{row['open_price']:.2f} H:{row['high_price']:.2f} L:{row['low_price']:.2f} C:{row['close_price']:.2f}")
    
    if len(agg_5m.df) > 0:
        print(f"\n5-minute aggregated data:")
        for _, row in agg_5m.df.iterrows():
            print(f"  {timestamp_to_readable(row['open_time'])}: O:{row['open_price']:.2f} H:{row['high_price']:.2f} L:{row['low_price']:.2f} C:{row['close_price']:.2f}")
    
    # Test interval boundary detection
    print(f"\n" + "="*60)
    print("INTERVAL BOUNDARY TESTING")
    print("="*60)
    
    current_time = int(time.time() * 1000)
    print(f"Current time: {timestamp_to_readable(current_time)}")
    print(f"Is 3-minute boundary: {agg_3m.is_interval_boundary(current_time)}")
    print(f"Is 5-minute boundary: {agg_5m.is_interval_boundary(current_time)}")
    
    # Get next interval closes
    next_3min = agg_3m.get_next_interval_close()
    next_5min = agg_5m.get_next_interval_close()
    
    print(f"Next 3-minute close: {timestamp_to_readable(next_3min)}")
    print(f"Next 5-minute close: {timestamp_to_readable(next_5min)}")

if __name__ == "__main__":
    main() 