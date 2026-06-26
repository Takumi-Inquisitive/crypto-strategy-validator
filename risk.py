"""Risk management. In live trading this matters far more than your entry
signal. Survive first; optimise returns second.

Two layers:
  * sizing  — turn a 0/1/-1 signal into a *fraction* of equity to deploy,
              based on volatility (ATR) so you risk a constant % per trade.
  * RiskGuard — a stateful gate for the live/paper loop: enforces a max daily
              loss and a hard kill switch. Returns False -> do not trade.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd

from indicators import atr


def atr_position_size(
    df: pd.DataFrame,
    risk_per_trade: float = 0.01,   # risk 1% of equity per trade
    atr_period: int = 14,
    atr_stop_mult: float = 2.0,     # stop placed 2*ATR away
    max_fraction: float = 1.0,
) -> pd.Series:
    """Fraction of equity to deploy per bar so that a stop at atr_stop_mult*ATR
    loses ~risk_per_trade of equity. Capped at max_fraction (no leverage by
    default). Returns a series in [0, max_fraction]."""
    a = atr(df, atr_period)
    stop_distance = atr_stop_mult * a
    frac = (risk_per_trade * df["close"]) / stop_distance.replace(0, float("nan"))
    return frac.clip(0, max_fraction).fillna(0.0)


@dataclass
class RiskGuard:
    starting_equity: float
    max_daily_loss: float = 0.03      # halt for the day after -3%
    max_total_drawdown: float = 0.20  # kill switch at -20% from peak
    _day: object = field(default=None, init=False)
    _day_start_equity: float = field(default=0.0, init=False)
    _peak: float = field(default=0.0, init=False)
    halted_today: bool = field(default=False, init=False)
    killed: bool = field(default=False, init=False)

    def __post_init__(self):
        self._peak = self.starting_equity
        self._day_start_equity = self.starting_equity

    def update(self, equity: float, timestamp) -> tuple[bool, str]:
        """Call before every potential entry. Returns (can_trade, reason)."""
        day = pd.Timestamp(timestamp).date()
        if self._day is None or day != self._day:
            self._day, self._day_start_equity, self.halted_today = day, equity, False

        self._peak = max(self._peak, equity)

        if self.killed:
            return False, "KILL SWITCH active — total drawdown breached"
        if equity <= self._peak * (1 - self.max_total_drawdown):
            self.killed = True
            return False, "KILL SWITCH tripped — max total drawdown hit"
        if equity <= self._day_start_equity * (1 - self.max_daily_loss):
            self.halted_today = True
        if self.halted_today:
            return False, "Daily loss limit hit — no new trades until tomorrow"
        return True, "ok"
