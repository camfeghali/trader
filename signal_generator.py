class SignalGenerator:
    def signals_trend(df: pd.DataFrame) -> pd.DataFrame:
        ema9, ema21 = df['ema_9'], df['ema_21']
        rsi14, vol, vol_sma = df['rsi'], df['volume'], df['vol_sma']
        atr = df['atr_14']
        price = df['close_price']

        long_entry  = cross_above(ema9, ema21) & (rsi14.between(40, 70)) & vol_spike(vol, vol_sma, 1.2) & atr_ok(atr, 0.0015, price)
        long_exit   = cross_below(ema9, ema21) | (rsi14 > 80) | (price < ema21 - 0.5 * atr)

        short_entry = cross_below(ema9, ema21) & (rsi14.between(30, 60)) & vol_spike(vol, vol_sma, 1.2) & atr_ok(atr, 0.0015, price)
        short_exit  = cross_above(ema9, ema21) | (rsi14 < 20) | (price > ema21 + 0.5 * atr)

        out = {
            'trend_long': long_entry.astype(int).tolist(),
            'trend_short': (-short_entry.astype(int)).tolist(),
            'trend_exit_long': long_exit.tolist(),
            'trend_exit_short': short_exit.tolist()
        }
        return out

    def signals_mean_reversion(df: pd.DataFrame) -> dict:
        price = df['close_price']
        rsi14 = df['rsi']  # your 14-period
        wr = df['williams_r']
        bb_upper, bb_mid, bb_lower = df['bb_upper'], df['bb_middle'], df['bb_lower']

        long_entry  = (price <= bb_lower) & (rsi14 < 30) & (wr < -80)
        long_exit   = (price >= bb_mid) | (rsi14 > 50)

        short_entry = (price >= bb_upper) & (rsi14 > 70) & (wr > -20)
        short_exit  = (price <= bb_mid) | (rsi14 < 50)

        out = {
            'mr_long': long_entry.astype(int).tolist(),
            'mr_short': (-short_entry.astype(int)).tolist(),
            'mr_exit_long': long_exit.tolist(),
            'mr_exit_short': short_exit.tolist()
        }
        return out

    def signals_breakout(df: pd.DataFrame) -> dict:
        price = df['close_price']
        dc_high = df['donchian_high'].shift(1)  # yesterday’s channel to avoid look-ahead
        dc_low  = df['donchian_low'].shift(1)
        macd_hist = df['macd_hist']
        cmf = df['cmf']
        vol, vol_sma = df['volume'], df['vol_sma']
        atr = df['atr_14']

        long_entry  = (price > dc_high) & (macd_hist > 0) & (cmf > 0) & vol_spike(vol, vol_sma, 1.2)
        long_exit   = (price < dc_low) | (macd_hist < 0) | (atr.pct_change().abs() > 0.5)  # safety exit

        short_entry = (price < dc_low) & (macd_hist < 0) & (cmf < 0) & vol_spike(vol, vol_sma, 1.2)
        short_exit  = (price > dc_high) | (macd_hist > 0)

        out = {
            'bo_long': long_entry.astype(int).tolist(),
            'bo_short': (-short_entry.astype(int)).tolist(),
            'bo_exit_long': long_exit.tolist(),
            'bo_exit_short': short_exit.tolist()
        }
        return out          

    def cross_above(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a.shift(1) <= b.shift(1)) & (a > b)

    def cross_below(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a.shift(1) >= b.shift(1)) & (a < b)

    def vol_spike(volume, vol_sma, multiple=1.2):
        return volume > (multiple * vol_sma)

    def atr_ok(atr, min_mult_of_price=0.002, price=None):
        # Require ATR to be at least X% of price so we don’t trade dead markets
        if price is None:
            return pd.Series(False, index=atr.index)
        return (atr / price) > min_mult_of_price