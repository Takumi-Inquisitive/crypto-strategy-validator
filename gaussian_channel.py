"""Gaussian Channel strategy (the public DonovanWall/Ehlers indicator behind the
'Trend Radar' claim). A Gaussian-filtered midline with volatility bands; go long
when the filter is rising and price breaks the upper band, short the mirror.

The famous '7000% since 2018' backtests of this are almost entirely BULL-MARKET
BETA: a long-biased trend follower that happened to ride 2018-2021. The honest
question is whether it adds anything once tested out-of-sample and across coins.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


def gaussian_filter(src: pd.Series, period: int = 144, poles: int = 4) -> pd.Series:
    beta = (1 - np.cos(2 * np.pi / period)) / (2 ** (1.0 / poles) - 1)
    alpha = -beta + np.sqrt(beta ** 2 + 2 * beta)
    f = src.to_numpy(dtype=float)
    for _ in range(poles):  # cascade single-pole filter `poles` times
        out = np.empty_like(f)
        out[0] = f[0]
        for i in range(1, len(f)):
            out[i] = alpha * f[i] + (1 - alpha) * out[i - 1]
        f = out
    return pd.Series(f, index=src.index)


@dataclass
class GaussianChannel:
    period: int = 144
    poles: int = 4
    mult: float = 1.414
    allow_short: bool = True
    name: str = field(default="gaussian", init=False)

    def positions(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        prev = close.shift(1)
        tr = pd.concat([(df["high"] - df["low"]),
                        (df["high"] - prev).abs(),
                        (df["low"] - prev).abs()], axis=1).max(axis=1)
        filt = gaussian_filter(close, self.period, self.poles)
        ftr = gaussian_filter(tr.fillna(0), self.period, self.poles)
        upper = filt + ftr * self.mult
        lower = filt - ftr * self.mult
        rising = filt > filt.shift(1)

        pos = pd.Series(np.nan, index=df.index)
        pos[(close > upper) & rising] = 1.0
        pos[close < upper] = 0.0
        if self.allow_short:
            pos[(close < lower) & (~rising)] = -1.0
            pos[close > lower] = pos[close > lower]  # leave longs/flats intact
        return pos.ffill().fillna(0.0)

    @classmethod
    def param_grid(cls):
        for period in (100, 144, 200):
            for mult in (1.0, 1.414, 2.0):
                yield {"period": period, "mult": mult}
