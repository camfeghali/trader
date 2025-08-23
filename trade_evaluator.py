import pandas as pd
from dataframe import KlineDataFrame

class TradeEvaluator:
    def __init__(self, name: str):
        self.name = name

    def show(self, data: KlineDataFrame):
        indicators = ['ema_21', 'rsi_14', 'macd']
        
        print(f"columns:{data.df.columns}")
        print(f"data length:{len(data.df)}")
        # Get the last values of the indicator columns
        if len(data.df) > 0:
            last_values = {}
            for indicator in indicators:
                if indicator in data.df.columns:
                    last_values[indicator] = data.df.iloc[-1][indicator]
                else:
                    last_values[indicator] = None
            
            return last_values
        else:
            return {indicator: None for indicator in indicators}
        