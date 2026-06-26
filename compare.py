"""The generalisation test. Run ONE strategy across MANY coins and see whether an
edge holds up everywhere or was a one-coin fluke. A strategy that only works on
BTC in one window is overfit to that window — this is how you catch it.

Examples:
  python compare.py --strategy rsi_meanrev --timeframe 1h \
      --coins BTC/USDT ETH/USDT SOL/USDT BNB/USDT XRP/USDT
  python compare.py --strategy rsi_meanrev --timeframe 1h --chop-filter
"""
from __future__ import annotations
import argparse
import numpy as np

from data import fetch_ohlcv
from strategies import REGISTRY
from backtest import run_backtest, CostModel
from walkforward import walk_forward


def main():
    p = argparse.ArgumentParser(description="Compare a strategy across coins")
    p.add_argument("--strategy", choices=REGISTRY, default="rsi_meanrev")
    p.add_argument("--coins", nargs="+",
                   default=["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"])
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--exchange", default="binance")
    p.add_argument("--limit", type=int, default=17520)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--taker-bps", type=float, default=6.0)
    p.add_argument("--slippage-bps", type=float, default=3.0)
    p.add_argument("--chop-filter", action="store_true")
    p.add_argument("--trend-filter", action="store_true")
    p.add_argument("--stop-atr", type=float, default=0.0,
                   help="stop-loss distance in ATRs (e.g. 2.0). 0 = no stop")
    p.add_argument("--tp-atr", type=float, default=0.0,
                   help="take-profit distance in ATRs. 0 = none")
    p.add_argument("--risk", type=float, default=0.01,
                   help="fraction of equity risked per trade (with --stop-atr)")
    args = p.parse_args()

    costs = CostModel(args.taker_bps, args.slippage_bps)
    strat_cls = REGISTRY[args.strategy]
    kw = {}
    if args.chop_filter:
        kw["chop_filter"] = True
    if args.trend_filter:
        kw["trend_filter"] = True
    stops = None
    if args.stop_atr > 0:
        stops = {"stop_atr_mult": args.stop_atr, "tp_atr_mult": args.tp_atr,
                 "risk_per_trade": args.risk}

    print(f"Strategy: {args.strategy}  tf={args.timeframe}  filters={kw or 'none'}  "
          f"stops={stops or 'none'}\n")
    header = f"  {'coin':<11}{'net':>9}{'B&H':>9}{'OOS Sharpe':>12}{'expect':>9}  verdict"
    print(header)
    print("  " + "-" * (len(header) - 2))

    oos_sharpes, passes = [], 0
    for coin in args.coins:
        try:
            df = fetch_ohlcv(coin, args.timeframe, args.exchange, args.limit)
            res = run_backtest(df, strat_cls(**kw).positions(df), args.timeframe,
                               costs, **(stops or {}))
            wf = walk_forward(df, strat_cls, args.timeframe, args.folds, costs, stops)
            tag = wf.verdict.split(" ")[0]
            oos_sharpes.append(wf.oos_sharpe)
            if tag in ("ROBUST", "MODERATE"):
                passes += 1
            print(f"  {coin:<11}{res.metrics['total_return']:>+8.1%}"
                  f"{res.metrics['buy_hold_return']:>+9.1%}{wf.oos_sharpe:>12.2f}"
                  f"{res.metrics['expectancy']:>+9.2%}  {tag}")
        except Exception as e:
            print(f"  {coin:<11}  FAILED: {e}")

    if oos_sharpes:
        med = float(np.median(oos_sharpes))
        print("\n  " + "-" * (len(header) - 2))
        print(f"  Coins with a real edge (ROBUST/MODERATE): {passes}/{len(oos_sharpes)}")
        print(f"  Median out-of-sample Sharpe across coins: {med:.2f}")
        if passes >= max(3, len(oos_sharpes) * 0.6) and med > 0.5:
            print("  >> GENERALISES. Worth paper-trading. (Still paper-trade first.)")
        elif passes >= 2:
            print("  >> MIXED. Works on some coins, not others — fragile, keep digging.")
        else:
            print("  >> DOES NOT GENERALISE. One-window mirage. Do not trade this.")


if __name__ == "__main__":
    main()
