import pandas as pd
import talib

class TaCalculator:
    def __init__(self):
        pass

    def calculate_rsi(self, df: pd.DataFrame, column: str = 'close_price', period: int = 14):
        if len(df) < period:
            return df
        df['rsi'] = talib.RSI(df[column], timeperiod=period)
        return df