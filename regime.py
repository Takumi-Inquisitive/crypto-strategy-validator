"""Regime stress-testing — directly implements the advice to test "chop vs
trend, low vs high volatility" instead of trusting one blended number.

The failure this catches: a strategy that looks fine overall but earns ALL its
money in trending markets and bleeds in chop. Time-based walk-forward can miss
this if your sample happens to be mostly trending. Here we label every bar's
regime (causally) and attribute the strategy's realised returns to the regime
that was active, so you can see exactly where the edge lives — and where it dies.

Regime axes:
  * trend vs chop  — Kaufman Efficiency Ratio (directional move / total path)
  * high vs low vol — ATR%, split on its own rolling median
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from indicators import atr
from backtest import BARS_PER_YEAR, BacktestResult


def efficiency_ratio(close: pd.Series, period: int = 20) -> pd.Series:
    direction = (close - close.shift(period)).abs()
    path = close.diff().abs().rolling(period).sum()
    return (direction / path.replace(0, np.nan)).fillna(0.0)


def smoothed_efficiency_ratio(close: pd.Series, period: int = 20,
                              smooth: int = 10) -> pd.Series:
    """EMA-smoothed ER. The raw ER flickers across any threshold bar-to-bar,
    which turns an on/off regime gate into an overtrading machine. Smoothing
    makes the regime label stable enough to actually gate trades."""
    return efficiency_ratio(close, period).ewm(span=smooth, adjust=False).mean()


def label_regimes(df: pd.DataFrame, er_period: int = 20, er_trend: float = 0.3,
                  atr_period: int = 14, vol_lookback: int = 200) -> pd.DataFrame:
    er = efficiency_ratio(df["close"], er_period)
    trend = np.where(er >= er_trend, "trend", "chop")

    atr_pct = atr(df, atr_period) / df["close"]
    vol_med = atr_pct.rolling(vol_lookback, min_periods=atr_period).median()
    vol = np.where(atr_pct >= vol_med, "high_vol", "low_vol")

    out = pd.DataFrame({"trend": trend, "vol": vol}, index=df.index)
    out["combined"] = out["trend"] + " / " + out["vol"]
    return out


def regime_report(df: pd.DataFrame, result: BacktestResult, timeframe: str = "5m") -> str:
    bpy = BARS_PER_YEAR.get(timeframe, 8_760)
    labels = label_regimes(df)
    ret = result.returns.reindex(df.index).fillna(0.0)
    held = result.position.reindex(df.index).fillna(0.0)

    def bucket(mask, name):
        r = ret[mask]
        if len(r) < 5:
            return None
        sharpe = r.mean() / r.std() * np.sqrt(bpy) if r.std() > 0 else 0.0
        net = (1 + r).prod() - 1
        return {
            "name": name, "time": mask.mean(), "net": net,
            "sharpe": sharpe, "exposure": (held[mask] != 0).mean(),
        }

    rows = []
    for ax in ("trend", "vol", "combined"):
        for val in sorted(labels[ax].unique()):
            b = bucket(labels[ax] == val, val)
            if b:
                rows.append(b)

    lines = ["Regime attribution (where the P&L actually comes from)",
             f"  {'regime':<22}{'%time':>7}{'net':>10}{'Sharpe':>9}{'expo':>7}"]
    for r in rows:
        lines.append(f"  {r['name']:<22}{r['time']:>6.0%}{r['net']:>+10.2%}"
                     f"{r['sharpe']:>9.2f}{r['exposure']:>7.0%}")

    lines.append("  " + "-" * 52)
    lines.append("  " + _regime_verdict(rows))
    return "\n".join(lines)


def _regime_verdict(rows: list) -> str:
    """Flag strategies whose edge is concentrated in one regime."""
    base = {r["name"]: r for r in rows if r["name"] in ("trend", "chop", "high_vol", "low_vol")}
    notes = []
    if "trend" in base and "chop" in base:
        if base["trend"]["sharpe"] > 0.5 and base["chop"]["sharpe"] < 0:
            notes.append("edge is TREND-only and loses in CHOP -> regime-dependent, fragile")
        elif min(base["trend"]["sharpe"], base["chop"]["sharpe"]) > 0:
            notes.append("positive in BOTH trend and chop -> genuinely more robust")
    if "high_vol" in base and "low_vol" in base:
        if base["high_vol"]["sharpe"] < 0 and base["low_vol"]["sharpe"] > 0:
            notes.append("breaks down in HIGH volatility")
    if not notes:
        notes.append("no single regime dominates strongly (inspect the table yourself)")
    return "VERDICT: " + "; ".join(notes)
