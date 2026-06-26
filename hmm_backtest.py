"""Honest HMM regime backtest: expanding-window walk-forward, no leverage.

For each fold we fit the HMM ONLY on past data, then trade the unseen block:
long while the live (causally-filtered) regime is bullish, flat otherwise. We
stitch the out-of-sample blocks — that is the closest thing to how it would
really have traded — and compare to buy-and-hold. Run across coins to see if any
edge generalises or was a one-window mirage.

  python hmm_backtest.py --coins BTC/USDT ETH/USDT SOL/USDT --timeframe 1h --states 4
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from data import fetch_ohlcv
from backtest import run_backtest, CostModel, BARS_PER_YEAR
from hmm_regime import CausalHMM


def walk_forward_hmm(df, timeframe, n_states=4, n_folds=5, costs=None,
                     min_train=2000, min_hold=6):
    costs = costs or CostModel()
    from backtest import apply_min_hold
    n = len(df)
    fold = n // (n_folds + 1)
    if fold < 50:
        return None
    oos_rets = []
    for k in range(n_folds):
        tr = df.iloc[: fold * (k + 1)]
        te = df.iloc[fold * (k + 1): fold * (k + 2)]
        if len(tr) < min_train or len(te) < 20:
            continue
        try:
            hmm = CausalHMM(n_states=n_states).fit(tr)
            dirs = hmm.filter_states(te)
        except Exception:
            continue
        pos = (dirs > 0).astype(float)  # long in bull regime, else flat. no leverage.
        if min_hold > 1:
            pos = apply_min_hold(pos, min_hold)   # hysteresis: curb regime flip-flop
        res = run_backtest(te, pos, timeframe, costs)
        oos_rets.append(res.returns)
    if not oos_rets:
        return None
    stitched = pd.concat(oos_rets)
    bpy = BARS_PER_YEAR.get(timeframe, 8760)
    vol = stitched.std()
    sharpe = stitched.mean() / vol * np.sqrt(bpy) if vol > 0 else 0.0
    eq = (1 + stitched).cumprod()
    return {"oos_sharpe": sharpe, "oos_return": eq.iloc[-1] - 1, "n": len(stitched)}


def main():
    p = argparse.ArgumentParser(description="Honest causal HMM regime backtest")
    p.add_argument("--coins", nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--exchange", default="binance")
    p.add_argument("--limit", type=int, default=17520)
    p.add_argument("--states", type=int, default=4)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--min-hold", type=int, default=6,
                   help="bars to hold a regime before flipping (hysteresis)")
    args = p.parse_args()
    costs = CostModel()

    print(f"Causal HMM ({args.states} states), walk-forward, no leverage, "
          f"tf={args.timeframe}, min-hold={args.min_hold}\n")
    print(f"  {'coin':<11}{'OOS ret':>10}{'B&H':>10}{'OOS Sharpe':>12}  verdict")
    print("  " + "-" * 52)
    sharpes, beats = [], 0
    for coin in args.coins:
        try:
            df = fetch_ohlcv(coin, args.timeframe, args.exchange, args.limit)
            bh = df["close"].iloc[-1] / df["close"].iloc[0] - 1
            r = walk_forward_hmm(df, args.timeframe, args.states, args.folds,
                                 costs, min_hold=args.min_hold)
            if not r:
                print(f"  {coin:<11}  insufficient data")
                continue
            v = "EDGE" if r["oos_sharpe"] > 0.5 else ("WEAK" if r["oos_sharpe"] > 0 else "NO EDGE")
            sharpes.append(r["oos_sharpe"])
            if r["oos_return"] > bh:
                beats += 1
            print(f"  {coin:<11}{r['oos_return']:>+9.1%}{bh:>+10.1%}{r['oos_sharpe']:>12.2f}  {v}")
        except Exception as e:
            print(f"  {coin:<11}  FAILED: {e}")
    if sharpes:
        print("\n  " + "-" * 52)
        print(f"  Median OOS Sharpe: {np.median(sharpes):.2f}   "
              f"Beat buy-and-hold: {beats}/{len(sharpes)} coins")
        if np.median(sharpes) > 0.5 and beats >= max(2, len(sharpes) * 0.6):
            print("  >> HMM regimes add real, generalising edge. Worth paper-trading.")
        else:
            print("  >> No generalising edge once lookahead is removed. The demo was a mirage.")


if __name__ == "__main__":
    main()
