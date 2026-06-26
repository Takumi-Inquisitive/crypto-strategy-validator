"""Strategies produce a target position series in {-1, 0, +1} aligned to bars.
The backtester shifts positions +1 bar, so a signal on bar t's close acts at
t+1 (no lookahead, given causal indicators).

Regime filters use a SMOOTHED efficiency ratio. The raw ER flickers across any
threshold and causes overtrading (we learned this the hard way). Filters are
OFF by default because on a single bad window the raw signals tested better;
turn them on to A/B test whether a stable regime gate helps on broad data.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from typing import Iterator
import pandas as pd

from indicators import ema, rsi, atr, donchian
from regime import smoothed_efficiency_ratio


class Strategy:
    name = "base"

    def positions(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    @classmethod
    def param_grid(cls) -> Iterator[dict]:
        yield {}


@dataclass
class EmaCross(Strategy):
    fast: int = 12
    slow: int = 26
    allow_short: bool = False
    name: str = field(default="ema_cross", init=False)

    def positions(self, df: pd.DataFrame) -> pd.Series:
        f, s = ema(df["close"], self.fast), ema(df["close"], self.slow)
        pos = pd.Series(0.0, index=df.index)
        pos[f > s] = 1.0
        if self.allow_short:
            pos[f < s] = -1.0
        return pos

    @classmethod
    def param_grid(cls):
        for fast, slow in product((8, 12, 21), (26, 50, 100)):
            if fast < slow:
                yield {"fast": fast, "slow": slow}


@dataclass
class RsiMeanReversion(Strategy):
    """Buy oversold, exit at neutral. Long/flat. chop_filter (smoothed) keeps it
    flat during trends, where mean-reversion gets run over."""
    period: int = 14
    oversold: int = 30
    exit_level: int = 50
    chop_filter: bool = False
    er_period: int = 20
    er_smooth: int = 10
    er_max: float = 0.30
    name: str = field(default="rsi_meanrev", init=False)

    def positions(self, df: pd.DataFrame) -> pd.Series:
        r = rsi(df["close"], self.period)
        pos = pd.Series(float("nan"), index=df.index)
        pos[r < self.oversold] = 1.0
        pos[r > self.exit_level] = 0.0
        pos = pos.ffill().fillna(0.0)
        if self.chop_filter:
            ranging = smoothed_efficiency_ratio(df["close"], self.er_period, self.er_smooth) < self.er_max
            pos = pos.where(ranging, 0.0)
        return pos

    @classmethod
    def param_grid(cls):
        for ov, ex in product((20, 25, 30), (50, 55, 60)):
            yield {"oversold": ov, "exit_level": ex}


@dataclass
class DonchianBreakout(Strategy):
    """N-bar high breakout, exit on M-bar low. trend_filter (smoothed) only
    trades while trending. Channels shifted +1 bar -> no lookahead."""
    entry: int = 20
    exit: int = 10
    trend_filter: bool = False
    er_period: int = 20
    er_smooth: int = 10
    er_min: float = 0.30
    name: str = field(default="donchian", init=False)

    def positions(self, df: pd.DataFrame) -> pd.Series:
        upper, _ = donchian(df, self.entry)
        _, lower = donchian(df, self.exit)
        pos = pd.Series(float("nan"), index=df.index)
        pos[df["close"] > upper] = 1.0
        pos[df["close"] < lower] = 0.0
        pos = pos.ffill().fillna(0.0)
        if self.trend_filter:
            trending = smoothed_efficiency_ratio(df["close"], self.er_period, self.er_smooth) >= self.er_min
            pos = pos.where(trending, 0.0)
        return pos

    @classmethod
    def param_grid(cls):
        for entry, exit in product((20, 30, 55), (10, 15, 20)):
            if exit < entry:
                yield {"entry": entry, "exit": exit}


from gaussian_channel import GaussianChannel

REGISTRY = {
    "gaussian": GaussianChannel,
    "ema_cross": EmaCross,
    "rsi_meanrev": RsiMeanReversion,
    "donchian": DonchianBreakout,
}
