import pandas as pd
import talib
import numpy as np

class TaCalculator:
    def __init__(self):
        pass

    def calculate_rsi(self, df: pd.DataFrame, column: str = 'close_price', period: int = 14):
        """
        Calculate Relative Strength Index (RSI)
        
        RSI measures the speed and magnitude of price changes to identify overbought/oversold conditions.
        Values above 70 indicate overbought, below 30 indicate oversold.
        
        Args:
            df: DataFrame with price data
            column: Price column to use (default: 'close_price')
            period: RSI period (default: 14)
        """
        if len(df) < period:
            return df
        df[f'rsi_{period}'] = talib.RSI(df[column], timeperiod=period)
        return df

    def calculate_ema(self, df: pd.DataFrame, column: str = 'close_price', period: int = 9):
        """
        Calculate Exponential Moving Average (EMA)
        
        EMA gives more weight to recent prices compared to SMA.
        Useful for identifying trends and potential support/resistance levels.
        
        Args:
            df: DataFrame with price data
            column: Price column to use (default: 'close_price')
            period: EMA period (default: 9)
        """
        if len(df) < period:
            return df
        df[f'ema_{period}'] = talib.EMA(df[column], timeperiod=period)
        return df

    def calculate_sma(self, df: pd.DataFrame, column: str = 'close_price', period: int = 50):
        """
        Calculate Simple Moving Average (SMA)
        
        SMA is the average of prices over a specified period.
        Used to identify trends and potential support/resistance levels.
        
        Args:
            df: DataFrame with price data
            column: Price column to use (default: 'close_price')
            period: SMA period (default: 50)
        """
        if len(df) < period:
            return df
        df[f'sma_{period}'] = talib.SMA(df[column], timeperiod=period)
        return df

    def calculate_macd(self, df: pd.DataFrame, column: str = 'close_price', fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        MACD is a trend-following momentum indicator that shows the relationship between two moving averages.
        MACD line = Fast EMA - Slow EMA
        Signal line = EMA of MACD line
        Histogram = MACD line - Signal line
        
        Args:
            df: DataFrame with price data
            column: Price column to use (default: 'close_price')
            fastperiod: Fast EMA period (default: 12)
            slowperiod: Slow EMA period (default: 26)
            signalperiod: Signal line period (default: 9)
        """
        if len(df) < slowperiod:
            return df
        macd, signal, hist = talib.MACD(df[column], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        df[f'macd_fast_{fastperiod}'] = macd
        df[f'macd_signal_{signalperiod}'] = signal
        df[f'macd_hist_{signalperiod}'] = hist
        return df

    def calculate_stochastic(self, df: pd.DataFrame, fastk_period: int = 14, slowk_period: int = 3, slowd_period: int = 3):
        """
        Calculate Stochastic Oscillator
        
        Stochastic measures momentum by comparing a closing price to its price range over time.
        %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
        %D = SMA of %K
        
        Args:
            df: DataFrame with OHLC data
            fastk_period: %K period (default: 14)
            slowk_period: %K smoothing period (default: 3)
            slowd_period: %D period (default: 3)
        """
        if len(df) < fastk_period:
            return df
        slowk, slowd = talib.STOCH(df['high_price'], df['low_price'], df['close_price'], 
                                  fastk_period=fastk_period, slowk_period=slowk_period, slowd_period=slowd_period)
        df[f'stoch_slowk_{slowk_period}'] = slowk
        df[f'stoch_slowd_{slowd_period}'] = slowd
        return df

    def calculate_williams_r(self, df: pd.DataFrame, period: int = 14):
        """
        Calculate Williams %R
        
        Williams %R is a momentum indicator that measures overbought/oversold levels.
        Values range from 0 to -100, with readings above -20 indicating overbought and below -80 indicating oversold.
        
        Args:
            df: DataFrame with OHLC data
            period: Lookback period (default: 14)
        """
        if len(df) < period:
            return df
        df[f'williams_r_{period}'] = talib.WILLR(df['high_price'], df['low_price'], df['close_price'], timeperiod=period)
        return df

    def calculate_bollinger_bands(self, df: pd.DataFrame, column: str = 'close_price', period: int = 20, nbdevup: float = 2, nbdevdn: float = 2):
        """
        Calculate Bollinger Bands
        
        Bollinger Bands consist of a middle band (SMA) and upper/lower bands that are standard deviations away.
        Used to identify volatility and potential overbought/oversold conditions.
        
        Args:
            df: DataFrame with price data
            column: Price column to use (default: 'close_price')
            period: SMA period (default: 20)
            nbdevup: Upper band standard deviation (default: 2)
            nbdevdn: Lower band standard deviation (default: 2)
        """
        if len(df) < period:
            return df
        upper, middle, lower = talib.BBANDS(df[column], timeperiod=period, nbdevup=nbdevup, nbdevdn=nbdevdn)
        df[f'bb_upper_{period}'] = upper
        df[f'bb_middle_{period}'] = middle
        df[f'bb_lower_{period}'] = lower
        return df

    def calculate_atr(self, df: pd.DataFrame, period: int = 14):
        """
        Calculate Average True Range (ATR)
        
        ATR measures market volatility by considering the true range of price movements.
        Higher ATR indicates higher volatility, lower ATR indicates lower volatility.
        
        Args:
            df: DataFrame with OHLC data
            period: ATR period (default: 14)
        """
        if len(df) < period:
            return df
        df[f'atr_{period}'] = talib.ATR(df['high_price'], df['low_price'], df['close_price'], timeperiod=period)
        return df

    def calculate_donchian_channel(self, df: pd.DataFrame, period: int = 20):
        """
        Calculate Donchian Channel
        
        Donchian Channel consists of upper (highest high) and lower (lowest low) bands.
        Used to identify breakouts and potential support/resistance levels.
        
        Args:
            df: DataFrame with OHLC data
            period: Lookback period (default: 20)
        """
        if len(df) < period:
            return df
        df[f'donchian_high_{period}'] = df['high_price'].rolling(window=period).max()
        df[f'donchian_low_{period}'] = df['low_price'].rolling(window=period).min()
        return df

    def calculate_obv(self, df: pd.DataFrame):
        """
        Calculate On-Balance Volume (OBV)
        
        OBV measures buying and selling pressure by adding volume on up days and subtracting on down days.
        Used to confirm price trends and identify potential reversals.
        
        Args:
            df: DataFrame with price and volume data
        """
        if len(df) < 2:
            return df
        df['obv'] = talib.OBV(df['close_price'], df['volume'])
        return df

    def calculate_volume_sma(self, df: pd.DataFrame, period: int = 20):
        """
        Calculate Volume Simple Moving Average
        
        Volume SMA helps identify unusual volume activity compared to average volume.
        High volume often confirms price movements.
        
        Args:
            df: DataFrame with volume data
            period: SMA period (default: 20)
        """
        if len(df) < period:
            return df
        df[f'vol_sma_{period}'] = talib.SMA(df['volume'], timeperiod=period)
        return df

    def calculate_cmf(self, df: pd.DataFrame, period: int = 20):
        """
        Calculate Chaikin Money Flow (CMF)
        
        CMF measures buying and selling pressure over a specified period.
        Values range from -1 to +1, with positive values indicating buying pressure.
        
        Args:
            df: DataFrame with OHLCV data
            period: Lookback period (default: 20)
        """
        if len(df) < period:
            return df
        
        # Calculate Money Flow Multiplier
        mfm = ((df['close_price'] - df['low_price']) - (df['high_price'] - df['close_price'])) / (df['high_price'] - df['low_price'])
        mfm = mfm.replace([np.inf, -np.inf], 0)
        
        # Calculate Money Flow Volume
        mfv = mfm * df['volume']
        
        # Calculate CMF
        df[f'cmf_{period}'] = mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        return df

    def calculate_all_indicators(self, df: pd.DataFrame):
        """
        Calculate all technical indicators at once
        
        This method calculates all the requested indicators in the optimal order.
        
        Args:
            df: DataFrame with OHLCV data
        """
        # Moving Averages
        df = self.calculate_ema(df, 'close_price', 9)
        df = self.calculate_ema(df, 'close_price', 21)
        df = self.calculate_ema(df, 'close_price', 50)
        df = self.calculate_ema(df, 'close_price', 200)
        
        # Momentum Indicators
        df = self.calculate_rsi(df, 'close_price', 14)
        df = self.calculate_rsi(df, 'close_price', 21)
        df = self.calculate_macd(df, 'close_price')
        df = self.calculate_stochastic(df)
        df = self.calculate_williams_r(df)
        
        # Volatility Indicators
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_atr(df)
        df = self.calculate_donchian_channel(df)
        
        # Volume Indicators
        df = self.calculate_obv(df)
        df = self.calculate_volume_sma(df)
        df = self.calculate_cmf(df)
        
        return df