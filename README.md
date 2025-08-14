# Automatic trader

# Run the ap

`uv run fastapi dev main.py`

# Dependency & tools

- Package manager [uv](https://github.com/astral-sh/uv?tab=readme-ov-file)
- To run a single file `uv run <file-path>`

# Documentation

- [Binance](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints)
- Binance Api endpoint: <https://api4.binance.com>
- [Technical Analysis lib](https://ta-lib.github.io/ta-lib-python/doc_index.html)
- [Thinking about strategy](https://chatgpt.com/share/689b65a1-eba8-800f-a2e0-ab89267cbee3)
- [Multi-layered strategy](https://chatgpt.com/s/t_689b70c49f508191b42f6c32fa4e1051)

# Refs

- <https://www.youtube.com/watch?v=TT_D-4z-4zY&ab_channel=CodeTrading>
- [Point Forecast vs Probabilistic Forecast](https://chatgpt.com/share/689a50fa-2dd0-800f-8020-fee90fcf2de6)
- [Rolling Pandas DataFrame to maintain TAs](https://chatgpt.com/share/689a5171-2b10-800f-a4f4-7dd51f536d20)

## Pro tip

💡 Pro tip: Crypto is extremely noisy intraday. The most robust systems:

Use multi-timeframe confirmation (e.g., 5m for entry, 1h for trend)

Keep rules simple, risk small, and let winners run.

## Example signal generation

```
import numpy as np
import pandas as pd

df = df.copy()

# --- 1) Filters ---
uptrend   = (df['ema_50'] > df['ema_200']) & (df['close'] > df['ema_21'])
downtrend = (df['ema_50'] < df['ema_200']) & (df['close'] < df['ema_21'])

# Momentum flips (no look-ahead: compare to previous bar)
rsi_cross_up   = (df['rsi_14'].shift(1) <= 30) & (df['rsi_14'] > 30)
rsi_cross_down = (df['rsi_14'].shift(1) >= 70) & (df['rsi_14'] < 70)

macd_flip_up   = (df['macd_hist'].shift(1) <= 0) & (df['macd_hist'] > 0)
macd_flip_down = (df['macd_hist'].shift(1) >= 0) & (df['macd_hist'] < 0)

# Breakouts (use previous Donchian/Bollinger to avoid peeking)
breakout_up   = (df['close'] > df['donchian_high'].shift(1)) | (df['close'] > df['bb_upper'].shift(1))
breakout_down = (df['close'] < df['donchian_low'].shift(1))  | (df['close'] < df['bb_lower'].shift(1))

# Volume confirmation
obv_trending_up   = df['obv'].diff(5) > 0
obv_trending_down = df['obv'].diff(5) < 0

vol_confirm_up   = (df['volume'] > 1.2 * df['vol_sma']) | (df['cmf'] > 0)  | obv_trending_up
vol_confirm_down = (df['volume'] > 1.2 * df['vol_sma']) | (df['cmf'] < 0)  | obv_trending_down

# --- 2) Entries ---
entry_long  = uptrend   & (rsi_cross_up | macd_flip_up)   & breakout_up   & vol_confirm_up
entry_short = downtrend & (rsi_cross_down | macd_flip_down) & breakout_down & vol_confirm_down

# --- 3) Exits (flat signals) ---
# Momentum/Trend violation; you can add ATR stops/TP externally per trade
exit_long  = (df['close'] < df['ema_21']) | (df['macd_hist'] < 0) | rsi_cross_down
exit_short = (df['close'] > df['ema_21']) | (df['macd_hist'] > 0) | rsi_cross_up

# --- 4) Export signal columns ---
df['entry_long']  = entry_long.astype(int)
df['entry_short'] = entry_short.astype(int)
df['exit_long']   = exit_long.astype(int)
df['exit_short']  = exit_short.astype(int)

# Optional: a one-bar "direction" signal for execution engines
# +1 = open long, -1 = open short, 0 = do nothing; exits are separate events
df['signal'] = 0
df.loc[df['entry_long'],  'signal'] = 1
df.loc[df['entry_short'], 'signal'] = -1
```
