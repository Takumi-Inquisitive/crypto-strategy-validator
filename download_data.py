"""Bulk history downloader. Pull deep history for several coins/timeframes once,
then every backtest reads from cache instantly.

Examples:
  # ~2 years of hourly for the majors (17520 = 24*730):
  python download_data.py --coins BTC/USDT ETH/USDT SOL/USDT BNB/USDT XRP/USDT \
      --timeframes 1h --limit 17520

  # if Binance is geo-blocked, add --exchange bybit
"""
from __future__ import annotations
import argparse
from data import fetch_ohlcv


def main():
    p = argparse.ArgumentParser(description="Bulk OHLCV downloader")
    p.add_argument("--coins", nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    p.add_argument("--timeframes", nargs="+", default=["1h"])
    p.add_argument("--exchange", default="binance")
    p.add_argument("--limit", type=int, default=17520)
    args = p.parse_args()

    for tf in args.timeframes:
        for coin in args.coins:
            try:
                df = fetch_ohlcv(coin, tf, args.exchange, args.limit, use_cache=True)
                span = f"{df.index[0].date()} -> {df.index[-1].date()}" if len(df) else "empty"
                print(f"  {coin:<12} {tf:<4} {len(df):>6} bars  {span}")
            except Exception as e:
                print(f"  {coin:<12} {tf:<4} FAILED: {e}")
    print("\nDone. Cached under .cache/ — backtests and compare.py now read instantly.")


if __name__ == "__main__":
    main()
