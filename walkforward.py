"""Walk-forward validation — the single most important guardrail here.

The trap: you grid-search parameters on all your history, find the combo with
the best Sharpe, and declare victory. That number is a lie; you fit the noise.

Walk-forward instead: roll a window through time. On each IN-SAMPLE block pick
the best params, then measure them on the NEXT, UNSEEN block. Stitch the
out-of-sample (OOS) results together — THAT is the closest thing to how the
strategy would actually have traded. We then compare OOS vs in-sample to flag
overfitting.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from backtest import run_backtest, CostModel


@dataclass
class WalkForwardResult:
    oos_equity: pd.Series
    oos_sharpe: float
    is_sharpe: float
    oos_return: float
    verdict: str
    folds: list

    def report(self) -> str:
        return (
            f"Walk-forward ({len(self.folds)} folds)\n"
            f"  In-sample Sharpe (avg)   {self.is_sharpe:.2f}\n"
            f"  Out-of-sample Sharpe     {self.oos_sharpe:.2f}\n"
            f"  Out-of-sample return     {self.oos_return:+.2%}\n"
            f"  VERDICT: {self.verdict}"
        )


def _score(df, strat_cls, params, timeframe, costs, stops=None):
    strat = strat_cls(**params)
    res = run_backtest(df, strat.positions(df), timeframe, costs, **(stops or {}))
    return res.metrics["sharpe"], res


def walk_forward(
    df: pd.DataFrame,
    strat_cls,
    timeframe: str = "5m",
    n_folds: int = 5,
    costs: CostModel | None = None,
    stops: dict | None = None,
) -> WalkForwardResult:
    costs = costs or CostModel()
    grid = list(strat_cls.param_grid())
    n = len(df)
    fold = n // (n_folds + 1)  # first block is pure training
    oos_returns, is_sharpes, folds = [], [], []

    for k in range(n_folds):
        is_slice = df.iloc[: fold * (k + 1)]
        oos_slice = df.iloc[fold * (k + 1) : fold * (k + 2)]
        if len(oos_slice) < 10:
            continue

        # Pick best params on in-sample, score them out-of-sample.
        best_params, best_is = None, -np.inf
        for params in grid:
            s, _ = _score(is_slice, strat_cls, params, timeframe, costs, stops)
            if np.isfinite(s) and s > best_is:
                best_is, best_params = s, params

        oos_sharpe, oos_res = _score(oos_slice, strat_cls, best_params, timeframe, costs, stops)
        is_sharpes.append(best_is)
        oos_returns.append(oos_res.returns)
        folds.append({
            "fold": k + 1, "params": best_params,
            "is_sharpe": round(best_is, 2), "oos_sharpe": round(oos_sharpe, 2),
            "oos_return": round(oos_res.metrics["total_return"], 4),
        })

    if not folds:
        return WalkForwardResult(pd.Series(dtype=float), 0, 0, 0, "INSUFFICIENT DATA", [])

    stitched = pd.concat(oos_returns)
    oos_equity = (1 + stitched).cumprod()
    vol = stitched.std()
    from backtest import BARS_PER_YEAR
    bpy = BARS_PER_YEAR.get(timeframe, 8_760)
    oos_sharpe = stitched.mean() / vol * np.sqrt(bpy) if vol > 0 else 0.0
    is_sharpe = float(np.mean(is_sharpes))
    oos_return = oos_equity.iloc[-1] - 1

    verdict = _verdict(is_sharpe, oos_sharpe)
    return WalkForwardResult(oos_equity, oos_sharpe, is_sharpe, oos_return, verdict, folds)


def _verdict(is_sharpe: float, oos_sharpe: float) -> str:
    """Compare honest OOS performance to the optimistic in-sample number."""
    if oos_sharpe <= 0:
        return "OVERFITTED — no real edge out-of-sample. Do NOT trade this."
    ratio = oos_sharpe / is_sharpe if is_sharpe > 0 else 0
    if ratio >= 0.6 and oos_sharpe >= 1.0:
        return "ROBUST — edge survives unseen data. Still paper-trade it first."
    if ratio >= 0.4:
        return "MODERATE — partial decay out-of-sample. Treat with suspicion."
    return "WEAK — most of the edge was curve-fit. Not ready for capital."
