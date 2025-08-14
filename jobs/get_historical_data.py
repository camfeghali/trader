import time
import math
import requests
import pandas as pd
from datetime import datetime, timezone

BASE = "https://api.binance.com"      # Spot. For USDT-M futures use: https://fapi.binance.com
ENDPOINT = "/api/v3/klines"
SYMBOL = "BTCUSDC"
INTERVAL = "1m"
LIMIT = 1000

def to_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

def fetch_klines(start_ms: int, end_ms: int):
    """Generator yielding kline batches [list[list]] between start_ms and end_ms."""
    curr = start_ms
    while curr < end_ms:
        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "limit": LIMIT,
            "startTime": curr,
            "endTime": end_ms,
        }
        r = requests.get(BASE + ENDPOINT, params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        yield batch
        # advance: next start is last closeTime + 1
        last_close_ms = batch[-1][6]
        next_start = last_close_ms + 1
        # guard against no progress (rare)
        if next_start <= curr:
            break
        curr = next_start
        # be nice to rate limits
        time.sleep(0.2)

def klines_to_df(rows):
    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trade_count",
        "taker_buy_base","taker_buy_quote","ignore"
    ]
    df = pd.DataFrame(rows, columns=cols)
    # types
    for c in ["open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote"]:
        df[c] = df[c].astype(float)
    df["trade_count"] = df["trade_count"].astype(int)
    df["open_time"]  = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df

# ---- choose your time window here ----
start_dt = datetime(2025, 8, 1, tzinfo=timezone.utc)
end_dt   = datetime(2025, 8, 11, tzinfo=timezone.utc)
start_ms, end_ms = to_ms(start_dt), to_ms(end_dt)

all_rows = []
for batch in fetch_klines(start_ms, end_ms):
    all_rows.extend(batch)

df = klines_to_df(all_rows)
print(df.head())
print(df.tail())
print("Total rows:", len(df))
# Save to CSV if you want
df.to_csv(f"{SYMBOL}_{INTERVAL}.csv", index=False)
