"""Live funding scanner: rank coins by current annualised funding yield so you
can see where the carry is right now. Positive = longs pay shorts = you (short
the perp) get paid. Run on your machine.

  python funding_scan.py --coins BTC ETH SOL BNB XRP DOGE
"""
from __future__ import annotations
import argparse
from funding import fetch_current_funding


def main():
    p = argparse.ArgumentParser(description="Current funding-rate scanner")
    p.add_argument("--coins", nargs="+",
                   default=["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK"])
    p.add_argument("--exchange", default="binanceusdm")
    p.add_argument("--quote", default="USDT")
    args = p.parse_args()

    symbols = [f"{c}/{args.quote}:{args.quote}" for c in args.coins]
    df = fetch_current_funding(symbols, args.exchange)
    df = df.dropna(subset=["fundingRate"])
    if df.empty:
        raise SystemExit("No funding data (check exchange/coins).")

    df["per_year"] = (365 * 24) / df["interval_h"].fillna(8.0)
    df["ann_pct"] = df["fundingRate"] * df["per_year"] * 100
    df = df.sort_values("ann_pct", ascending=False)

    print(f"Current funding on {args.exchange} (annualised; + = you get paid to short the perp)\n")
    print(f"  {'coin':<8}{'funding/interval':>18}{'annualised':>14}")
    print("  " + "-" * 40)
    for _, r in df.iterrows():
        print(f"  {r['symbol'].split('/')[0]:<8}{r['fundingRate']*100:>16.4f}%{r['ann_pct']:>13.1f}%")
    print("\nNote: current snapshot only. Backtest the HISTORY before trusting any of these:")
    print("  python funding_backtest.py --symbol BTC/USDT:USDT")


if __name__ == "__main__":
    main()
