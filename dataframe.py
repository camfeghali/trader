import pandas as pd
import pandas_ta as ta
from models import BinanceKlineData
from ta_calculator import TaCalculator
from daos import BinanceDataProcessor
import time
from typing import List
from aggregated_dataframe import AggregateDataFrame
from base_dataframe import BaseDataFrame

class KlineDataFrame(BaseDataFrame):
    def __init__(self, ta_calculator: TaCalculator, data_fetcher: BinanceDataProcessor, aggregate_dataframes: List[AggregateDataFrame]):
        super().__init__()
        self.ta_calculator = ta_calculator
        self.data_fetcher = data_fetcher
        self.aggregate_dataframes = aggregate_dataframes

    async def get_historical_data(self, symbol: str, interval: str, start_ms: int, end_ms: int):
        all_rows = []
        print(f"Fetching historical data from {start_ms} to {end_ms}, symbol: {symbol}, interval: {interval}")
        for batch in self.data_fetcher.fetch_klines(start_ms, end_ms, symbol, interval):
            for raw_kline in batch:
                # Create BinanceKlineData from raw kline array
                kline_data = BinanceKlineData(
                    symbol=symbol.lower(),
                    interval=interval,
                    open_time=raw_kline[0],      # open_time
                    close_time=raw_kline[6],     # close_time
                    open_price=float(raw_kline[1]),   # open
                    high_price=float(raw_kline[2]),   # high
                    low_price=float(raw_kline[3]),    # low
                    close_price=float(raw_kline[4]),  # close
                    volume=float(raw_kline[5]),       # volume
                    quote_volume=float(raw_kline[7]), # quote_volume
                    trades_count=raw_kline[8],        # trades_count
                    taker_buy_volume=float(raw_kline[9]),    # taker_buy_volume
                    taker_buy_quote_volume=float(raw_kline[10]), # taker_buy_quote_volume
                    is_closed=True  # Historical data is always closed
                )
                
                self.add_kline(kline_data)
        print(f"Added {len(self.df)} historical klines to dataframe")
        for agg_df in self.aggregate_dataframes:
            agg_df.fill_from_1m_dataframe(self)

    async def process_kline(self, kline: BinanceKlineData):
        if kline.is_closed:
            print(f"Processing kline: {kline}")
            self.add_kline(kline)
            print(f"📊 Processed 1-min candle: O:{kline.open_price:.2f} H:{kline.high_price:.2f} L:{kline.low_price:.2f} C:{kline.close_price:.2f}")

            self.calculate_tas()
            
            # Check aggregations for all registered aggregate dataframes
            for agg_df in self.aggregate_dataframes:
                agg_result = agg_df.aggregate(self)
                if agg_result:
                    print(f"✅ {agg_df.interval_in_minutes}-min aggregation completed")

    def calculate_tas(self):
        start_time = time.time()
        self.df = self.ta_calculator.calculate_all_indicators(self.df)
        execution_time = time.time() - start_time
        print(f"⏱️ Technical Analysis calculation completed in {execution_time:.4f} seconds")

    def add_row(self, candle: BinanceKlineData):
        """Alias for add_kline for backward compatibility"""
        self.add_kline(candle)