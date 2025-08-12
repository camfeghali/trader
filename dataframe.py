import pandas as pd
import pandas_ta as ta
from models import BinanceKlineData
from ta_calculator import TaCalculator
from daos import BinanceDataProcessor

class KlineDataFrame:
    def __init__(self, ta_calculator: TaCalculator, data_fetcher: BinanceDataProcessor):
        init_cols = [
            "symbol", "interval", 
            "open_time", "close_time", 
            "open_price", "high_price", "low_price", "close_price", 
            "volume", "quote_volume", 
            "trades_count", "taker_buy_volume", "taker_buy_quote_volume", "is_closed"
        ]
        self.df = pd.DataFrame(columns=init_cols)
        self.ta_calculator = ta_calculator
        self.data_fetcher = data_fetcher

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
                
                self.add_row(kline_data)
        
        print(f"Added {len(self.df)} historical klines to dataframe")

    async def process_kline(self, kline: BinanceKlineData):
        if kline.is_closed:
            self.add_row(kline)
            # Calculate RSI and update the DataFrame
            self.df = self.ta_calculator.calculate_rsi(self.df)
        print(self.get_dataframe())

    def add_row(self, candle: BinanceKlineData):
        # Check if this open_time already exists (O(1) average case with hash lookup)
        if len(self.df) > 0 and candle.open_time in self.df['open_time'].values:
            # Update existing row instead of adding duplicate
            mask = self.df['open_time'] == candle.open_time
            self.df.loc[mask, 'close_price'] = candle.close_price
            self.df.loc[mask, 'high_price'] = candle.high_price
            self.df.loc[mask, 'low_price'] = candle.low_price
            self.df.loc[mask, 'volume'] = candle.volume
            self.df.loc[mask, 'quote_volume'] = candle.quote_volume
            self.df.loc[mask, 'trades_count'] = candle.trades_count
            self.df.loc[mask, 'taker_buy_volume'] = candle.taker_buy_volume
            self.df.loc[mask, 'taker_buy_quote_volume'] = candle.taker_buy_quote_volume
            self.df.loc[mask, 'is_closed'] = candle.is_closed
            return
        
        # Convert the dataclass to a dictionary
        candle_dict = {
            'symbol': candle.symbol,
            'interval': candle.interval,
            'open_time': candle.open_time,
            'close_time': candle.close_time,
            'open_price': candle.open_price,
            'high_price': candle.high_price,
            'low_price': candle.low_price,
            'close_price': candle.close_price,
            'volume': candle.volume,
            'quote_volume': candle.quote_volume,
            'trades_count': candle.trades_count,
            'taker_buy_volume': candle.taker_buy_volume,
            'taker_buy_quote_volume': candle.taker_buy_quote_volume,
            'is_closed': candle.is_closed
        }
        
        # Add the new row to the dataframe
        new_row = pd.DataFrame([candle_dict])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        
        # Keep only last 100 rows
        if len(self.df) > 100:
            self.df = self.df.tail(100)         
    
    def __str__(self):
        return f"KlineDataFrame with {len(self.df)} rows:\n{self.df.to_string()}"
    
    def __repr__(self):
        return self.__str__()
    
    def get_dataframe(self):
        """Get the underlying pandas DataFrame"""
        return self.df
    
    def get_latest(self, n: int = 5):
        """Get the latest n rows"""
        return self.df.tail(n)