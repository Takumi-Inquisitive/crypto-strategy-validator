"""Cash-and-carry (basis) backtest: long spot + short perp, collect funding.

Honest accounting:
  * You RECEIVE funding when it's positive (you're short the perp), PAY when neg.
  * Round-trip costs: 4 taker fills (buy spot+sell perp to open, reverse to close).
  * Capital efficiency: you tie up spot notional PLUS margin for the short. Yield
    on *deployed capital* is lower than yield on notional. We report both.
  * We surface the ugly bits: % of intervals you paid, worst drawdown of the
    funding stream, and the annualised net — because a thin edge dies to fees and
    negative regimes, and you need to see that before trusting it.

This is delta-neutral, so spot price moves are hedged out and don't appear here
(in reality basis slippage on entry/exit adds noise — modelled as round-trip fee).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class CarryResult:
    equity: pd.Series
    metrics: dict

    def summary(self) -> str:
        m = self.metrics
        return "\n".join([
            f"  Intervals             {m['n']}  (~{m['interval_h']:.0f}h each)",
            f"  Gross funding (sum)   {m['gross']:+.2%} of notional",
            f"  Annualised gross      {m['ann_gross']:+.2%} on notional",
            f"  Annualised NET        {m['ann_net']:+.2%} on capital (after fees+margin)",
            f"  Intervals you PAID    {m['pct_negative']:.1%}",
            f"  Worst 30d window      {m['worst_30d']:+.2%}",
            f"  Max drawdown          {m['max_drawdown']:+.2%}",
            f"  Verdict: {m['verdict']}",
        ])


def carry_backtest(funding: pd.DataFrame, fee_bps=5.0, leverage=5.0,
                   roundtrips_per_year=1) -> CarryResult:
    """funding: DataFrame indexed by time with a 'fundingRate' column.
    fee_bps: taker fee per fill. leverage: margin = notional/leverage on the short.
    roundtrips_per_year: how often you re-open the position (rebalance churn)."""
    fr = funding["fundingRate"].astype(float)
    n = len(fr)
    if n < 2:
        return CarryResult(pd.Series(dtype=float), {"verdict": "INSUFFICIENT DATA"})

    interval_h = np.median(np.diff(fr.index.values).astype("timedelta64[h]").astype(float))
    per_year = (365 * 24) / interval_h

    # Short perp: receive funding when positive. Funding compounds on notional.
    equity = (1 + fr).cumprod()
    gross = equity.iloc[-1] - 1
    ann_gross = (equity.iloc[-1]) ** (per_year / n) - 1

    # Costs: 4 fills per round trip, amortised over the period at the chosen churn.
    years = n / per_year
    fee_frac = fee_bps / 10000
    total_roundtrips = max(1, roundtrips_per_year * years)
    fee_drag = total_roundtrips * 4 * fee_frac
    net_notional = gross - fee_drag

    # Capital efficiency: capital = spot notional + short margin = 1 + 1/leverage.
    capital_factor = 1 + 1 / leverage
    ann_net = ((1 + net_notional) ** (1 / max(years, 1e-9)) - 1) / capital_factor

    roll = equity.cummax()
    max_dd = (equity / roll - 1).min()
    win = int(round(per_year / 12))  # ~1 month of intervals
    worst_30d = (fr.rolling(win).sum().min()) if n > win else fr.sum()
    pct_neg = float((fr < 0).mean())

    verdict = _verdict(ann_net, pct_neg, max_dd)
    metrics = {
        "n": n, "interval_h": interval_h, "gross": gross, "ann_gross": ann_gross,
        "ann_net": ann_net, "pct_negative": pct_neg, "worst_30d": float(worst_30d),
        "max_drawdown": float(max_dd), "verdict": verdict,
    }
    return CarryResult(equity, metrics)


def _verdict(ann_net, pct_neg, max_dd):
    if ann_net <= 0:
        return "NOT WORTH IT — net yield <= 0 after costs. Skip."
    if ann_net < 0.04:
        return f"THIN — ~{ann_net:.1%}/yr net. Below most stablecoin lending; not worth the risk."
    if pct_neg > 0.35:
        return f"~{ann_net:.1%}/yr but funding negative {pct_neg:.0%} of the time — unstable carry."
    return f"VIABLE carry ~{ann_net:.1%}/yr net. Still: liquidation + exchange risk are real."


if __name__ == "__main__":
    import argparse
    from funding import fetch_funding_history, synthetic_funding
    p = argparse.ArgumentParser(description="Cash-and-carry funding backtest")
    p.add_argument("--symbol", default="BTC/USDT:USDT")
    p.add_argument("--exchange", default="binanceusdm")
    p.add_argument("--limit", type=int, default=4000)
    p.add_argument("--fee-bps", type=float, default=5.0)
    p.add_argument("--leverage", type=float, default=5.0)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    if args.synthetic:
        f = synthetic_funding(n=args.limit)
        print(f"[synthetic funding] {len(f)} intervals\n")
    else:
        f = fetch_funding_history(args.symbol, args.exchange, args.limit)
        if f.empty:
            raise SystemExit("No funding data returned (check symbol/exchange).")
        print(f"[{args.exchange} {args.symbol}] {len(f)} intervals "
              f"({f.index[0].date()} -> {f.index[-1].date()})\n")

    res = carry_backtest(f, args.fee_bps, args.leverage)
    print(res.summary())
