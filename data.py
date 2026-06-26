"""OHLCV data via ccxt, with on-disk parquet caching.

Reaches the exchange, so run on YOUR machine, not in restricted sandboxes.
No API key is needed for public market data.

Cache key is independent of `limit`: we store the deepest pull for a
symbol+timeframe and slice from it, so one big download serves many backtests.
"""
from __future__ import annotations
import os
import time
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


def _cache_path(exchange_id, symbol, timeframe):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"{exchange_id}_{symbol.replace('/', '')}_{timeframe}.parquet"
    return os.path.join(CACHE_DIR, key)


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "5m",
    exchange_id: str = "binance",
    limit: int = 5000,
    use_cache: bool = True,
) -> pd.DataFrame:
    path = _cache_path(exchange_id, symbol, timeframe)
    if use_cache and os.path.exists(path):
        cached = pd.read_parquet(path)
        if len(cached) >= limit:
            return cached.tail(limit)

    import ccxt  # imported lazily so the rest of the package works offline
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    tf_ms = exchange.parse_timeframe(timeframe) * 1000

    # Page BACKWARD from now so we actually collect deep history.
    since = exchange.milliseconds() - limit * tf_ms
    all_rows = []
    while len(all_rows) < limit:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        all_rows += batch
        since = batch[-1][0] + tf_ms
        time.sleep(exchange.rateLimit / 1000)
        if len(batch) < 1000:
            break

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").astype(float)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    if use_cache and len(df):
        # merge with any existing cache so we keep the deepest history
        if os.path.exists(path):
            old = pd.read_parquet(path)
            df = pd.concat([old, df])
            df = df[~df.index.duplicated(keep="first")].sort_index()
        df.to_parquet(path)
    return df.tail(limit)


def synthetic_ohlcv(n: int = 6000, seed: int = 7, timeframe_minutes: int = 5,
                    drift: float = 0.00002) -> pd.DataFrame:
    """Geometric-brownian-motion candles for offline testing. `seed` lets you
    fake several 'coins'; `drift` lets you fake bull/bear/chop regimes."""
    import numpy as np
    rng = np.random.default_rng(seed)
    vol = 0.004
    shocks = rng.normal(drift, vol, n)
    for i in range(1, n):
        shocks[i] += 0.05 * shocks[i - 1]  # mild momentum so regimes exist
    price = 30000 * np.exp(np.cumsum(shocks))
    idx = pd.date_range("2024-01-01", periods=n, freq=f"{timeframe_minutes}min", tz="UTC")
    close = pd.Series(price, index=idx)
    high = close * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = close * (1 - np.abs(rng.normal(0, vol / 2, n)))
    open_ = close.shift(1).fillna(close.iloc[0])
    vol_s = pd.Series(rng.uniform(10, 100, n), index=idx)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol_s})
