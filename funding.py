"""Funding-rate data for the cash-and-carry (basis) trade.

The trade: long spot + short perpetual of equal size = delta-neutral. You don't
care where price goes; you collect the funding that longs pay shorts (positive
funding) every interval. Real, mechanical edge — but small and competitive, and
it INVERTS when funding goes negative. We measure all of that honestly.

Runs on YOUR machine (reaches the exchange). Perp symbols look like 'BTC/USDT:USDT'.
For Binance perps use exchange_id='binanceusdm'; Bybit uses 'bybit'.
"""
from __future__ import annotations
import os
import time
import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


def fetch_funding_history(symbol="BTC/USDT:USDT", exchange_id="binanceusdm",
                          limit=4000, use_cache=True) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"funding_{exchange_id}_{symbol.replace('/', '').replace(':', '')}.parquet"
    path = os.path.join(CACHE_DIR, key)
    if use_cache and os.path.exists(path):
        cached = pd.read_parquet(path)
        if len(cached) >= limit:
            return cached.tail(limit)

    import ccxt
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    rows, since = [], ex.milliseconds() - limit * 8 * 3600 * 1000  # assume ~8h to seed
    while len(rows) < limit:
        batch = ex.fetch_funding_rate_history(symbol, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1]["timestamp"] + 1
        time.sleep(ex.rateLimit / 1000)
        if len(batch) < 1000:
            break

    df = pd.DataFrame([{"ts": r["timestamp"], "fundingRate": r["fundingRate"]} for r in rows])
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if use_cache:
        df.to_parquet(path)
    return df.tail(limit)


def fetch_current_funding(symbols, exchange_id="binanceusdm") -> pd.DataFrame:
    import ccxt
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    out = []
    for s in symbols:
        try:
            fr = ex.fetch_funding_rate(s)
            out.append({"symbol": s, "fundingRate": fr.get("fundingRate"),
                        "interval_h": _interval_hours(fr)})
        except Exception as e:
            out.append({"symbol": s, "fundingRate": None, "interval_h": None, "error": str(e)})
        time.sleep(ex.rateLimit / 1000)
    return pd.DataFrame(out)


def _interval_hours(fr_dict) -> float:
    info = fr_dict.get("info", {}) or {}
    for k in ("fundingIntervalHours", "funding_interval_hours"):
        if k in info:
            try:
                return float(info[k])
            except Exception:
                pass
    return 8.0  # exchange default for most majors


def synthetic_funding(n=2000, seed=11, mean_bps=1.0, neg_prob=0.15) -> pd.DataFrame:
    """Fake 8h funding: mostly small positive, with occasional negative spikes —
    so the backtest has to deal with the regime where you PAY."""
    rng = np.random.default_rng(seed)
    base = rng.normal(mean_bps / 10000, 0.0004, n)
    flip = rng.random(n) < neg_prob
    base[flip] = -np.abs(rng.normal(0.0006, 0.0005, flip.sum()))
    idx = pd.date_range("2024-01-01", periods=n, freq="8h", tz="UTC")
    return pd.DataFrame({"fundingRate": base}, index=idx)
