"""CLI entry point. Examples:

    # offline self-test on synthetic data (works anywhere, no network):
    python run_backtest.py --strategy ema_cross --synthetic

    # real data on your machine:
    python run_backtest.py --strategy donchian --symbol BTC/USDT --timeframe 5m --limit 5000
"""
from __future__ import annotations
import argparse

from data import fetch_ohlcv, synthetic_ohlcv
from strategies import REGISTRY
from backtest import run_backtest, CostModel
from walkforward import walk_forward
from risk import atr_position_size


def main():
    p = argparse.ArgumentParser(description="Crypto intraday backtest + walk-forward")
    p.add_argument("--strategy", choices=REGISTRY, default="ema_cross")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--exchange", default="binance")
    p.add_argument("--limit", type=int, default=6000)
    p.add_argument("--synthetic", action="store_true", help="use offline synthetic data")
    p.add_argument("--taker-bps", type=float, default=6.0)
    p.add_argument("--slippage-bps", type=float, default=3.0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--risk-sizing", action="store_true", help="ATR-based position sizing")
    p.add_argument("--regimes", action="store_true", help="show per-regime attribution")
    p.add_argument("--min-hold", type=int, default=0,
                   help="min bars to hold before flipping (curbs overtrading)")
    args = p.parse_args()

    if args.synthetic:
        tf_min = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}.get(args.timeframe, 5)
        df = synthetic_ohlcv(n=args.limit, timeframe_minutes=tf_min)
        print(f"[synthetic data] {len(df)} bars\n")
    else:
        df = fetch_ohlcv(args.symbol, args.timeframe, args.exchange, args.limit)
        print(f"[{args.exchange} {args.symbol} {args.timeframe}] {len(df)} bars "
              f"({df.index[0]} -> {df.index[-1]})\n")

    costs = CostModel(taker_fee_bps=args.taker_bps, slippage_bps=args.slippage_bps)
    strat_cls = REGISTRY[args.strategy]
    strat = strat_cls()

    size = atr_position_size(df) if args.risk_sizing else None
    res = run_backtest(df, strat.positions(df), args.timeframe, costs,
                       size=size, min_hold=args.min_hold)
    print(f"=== Single-pass backtest: {strat.name} (default params) ===")
    print(res.summary())

    if args.regimes:
        from regime import regime_report
        print(f"\n=== Regime stress-test ===")
        print(regime_report(df, res, args.timeframe))

    print(f"\n=== Walk-forward validation ===")
    wf = walk_forward(df, strat_cls, args.timeframe, args.folds, costs)
    print(wf.report())
    for f in wf.folds:
        print(f"    fold {f['fold']}: {f['params']}  IS={f['is_sharpe']}  "
              f"OOS={f['oos_sharpe']}  ret={f['oos_return']:+.2%}")

    print("\nReminder: a good single-pass number means nothing on its own. "
          "Trust the walk-forward verdict, then paper-trade before risking a cent.")


if __name__ == "__main__":
    main()
