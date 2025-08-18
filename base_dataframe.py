"""
Base class for dataframe operations to eliminate code duplication
"""
import pandas as pd
from typing import Optional
from models import BinanceKlineData
from constants import KLINE_COLUMNS

class BaseDataFrame:
    """
    Base class providing common dataframe operations for kline data.
    """
    
    def __init__(self):
        self.df = pd.DataFrame(columns=KLINE_COLUMNS)
    
    def add_kline(self, kline: BinanceKlineData):
        """
        Add a kline to the dataframe with deduplication logic.
        
        Args:
            kline: The kline data to add
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
    
    def limit_rows(self, max_rows: int):
        """
        Keep only the latest N rows in the dataframe.
        
        Args:
            max_rows: Maximum number of rows to keep
        """
        if len(self.df) > max_rows:
            self.df = self.df.tail(max_rows)
    
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
    
    def __str__(self):
        return f"{self.__class__.__name__} with {len(self.df)} rows:\n{self.df.to_string()}"
    
    def __repr__(self):
        return self.__str__() 