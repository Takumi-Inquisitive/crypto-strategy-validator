"""Backtest engine.

Design choices that keep it honest:
  * No lookahead: the target position from bar t is shifted to t+1 before it
    earns any return. You decide on close, you're filled next bar.
  * Costs are charged on every position CHANGE: taker fee + slippage, in bps,
    proportional to the traded fraction of equity.
  * Metrics are annualised using the real bar frequency, and always compared
    against buy-and-hold over the same window.

This is single-asset, fully-invested-or-flat (or short) sizing by default. The
risk module can scale the position fraction; pass a `size` series in [0,1].
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


# Annualisation factor: how many bars per year, by timeframe string.
BARS_PER_YEAR = {
    "1m": 525_600, "3m": 175_200, "5m": 105_120, "15m": 35_040,
    "30m": 17_520, "1h": 8_760, "2h": 4_380, "4h": 2_190,
    "6h": 1_460, "12h": 730, "1d": 365,
}


@dataclass
class CostModel:
    taker_fee_bps: float = 6.0    # 0.06%: a realistic crypto taker fee
    slippage_bps: float = 3.0     # 0.03%: conservative for liquid pairs intraday

    @property
    def per_turn(self) -> float:
        # cost as a fraction of notional traded, one side
        return (self.taker_fee_bps + self.slippage_bps) / 10_000.0


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    position: pd.Series
    trades: int
    metrics: dict
    buy_hold_equity: pd.Series

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"  Net return        {m['total_return']:+.2%}",
            f"  Buy & hold        {m['buy_hold_return']:+.2%}",
            f"  Ann. return       {m['ann_return']:+.2%}",
            f"  Ann. volatility   {m['ann_vol']:.2%}",
            f"  Sharpe            {m['sharpe']:.2f}",
            f"  Sortino           {m['sortino']:.2f}",
            f"  Max drawdown      {m['max_drawdown']:.2%}",
            f"  Calmar            {m['calmar']:.2f}",
            f"  Win rate          {m['win_rate']:.2%}",
            f"  Profit factor     {m['profit_factor']:.2f}",
            f"  Expectancy/trade  {m['expectancy']:+.3%}",
            f"  Trades            {m['trades']}",
            f"  Exposure          {m['exposure']:.1%}",
            f"  Total fees paid   {m['total_cost']:.2%} of equity",
        ]
        return "\n".join(lines)


def apply_min_hold(pos: pd.Series, min_hold: int) -> pd.Series:
    """Overtrading control: once a position is taken, keep it for at least
    `min_hold` bars before allowing a change. Directly curbs the fee-bleed from
    flip-flopping that the experienced traders call 'the silent killer'."""
    if min_hold <= 1:
        return pos
    vals = pos.to_numpy(copy=True)
    last_change, current = 0, vals[0]
    for i in range(1, len(vals)):
        if vals[i] != current:
            if i - last_change < min_hold:
                vals[i] = current          # too soon: ignore the new signal
            else:
                current, last_change = vals[i], i
        else:
            current = vals[i]
    return pd.Series(vals, index=pos.index)


def _run_with_stops(df, pos, timeframe, costs, stop_atr_mult, tp_atr_mult,
                    risk_per_trade, atr_period):
    """Event-driven engine with intrabar stop-loss / take-profit and Van Tharp
    style risk-based sizing: the stop defines risk, so position size is set so
    EVERY trade risks the same fixed % of equity (capped at 1x, no leverage).
    Long/flat only. Conservative fills (stop fills at the stop price)."""
    from indicators import atr as _atr
    a = _atr(df, atr_period).to_numpy()
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()
    sig = pos.shift(1).fillna(0.0).to_numpy()  # decide on prior close, act now

    n = len(df)
    rets = np.zeros(n)
    in_pos = False
    entry = stop = tp = size = 0.0
    trade_pnls, n_trades = [], 0
    cur_trade = 0.0

    for t in range(n):
        bar = 0.0
        if not in_pos:
            if sig[t] > 0 and a[t] > 0 and c[t] > 0:
                entry = o[t]
                stop_dist = stop_atr_mult * a[t]
                stop = entry - stop_dist
                tp = entry + tp_atr_mult * a[t] if tp_atr_mult > 0 else 0.0
                stop_pct = stop_dist / entry
                size = min(1.0, risk_per_trade / stop_pct) if stop_pct > 0 else 0.0
                in_pos = True
                bar -= size * costs.per_turn  # entry cost
                cur_trade = -costs.per_turn
                # intrabar resolution on the entry bar itself
                if l[t] <= stop:
                    r = size * (stop / entry - 1) - size * costs.per_turn
                    bar += r; cur_trade += (stop / entry - 1) - costs.per_turn
                    in_pos = False; trade_pnls.append(cur_trade); n_trades += 1
                elif tp and h[t] >= tp:
                    r = size * (tp / entry - 1) - size * costs.per_turn
                    bar += r; cur_trade += (tp / entry - 1) - costs.per_turn
                    in_pos = False; trade_pnls.append(cur_trade); n_trades += 1
                else:
                    bar += size * (c[t] / entry - 1)
                    cur_trade += (c[t] / entry - 1)
        else:
            pc = c[t - 1]
            if l[t] <= stop:
                bar += size * (stop / pc - 1) - size * costs.per_turn
                cur_trade += (stop / pc - 1) - costs.per_turn
                in_pos = False; trade_pnls.append(cur_trade); n_trades += 1
            elif tp and h[t] >= tp:
                bar += size * (tp / pc - 1) - size * costs.per_turn
                cur_trade += (tp / pc - 1) - costs.per_turn
                in_pos = False; trade_pnls.append(cur_trade); n_trades += 1
            elif sig[t] <= 0:
                bar += size * (c[t] / pc - 1) - size * costs.per_turn
                cur_trade += (c[t] / pc - 1) - costs.per_turn
                in_pos = False; trade_pnls.append(cur_trade); n_trades += 1
            else:
                bar += size * (c[t] / pc - 1)
                cur_trade += (c[t] / pc - 1)
        rets[t] = bar

    strat_ret = pd.Series(rets, index=df.index)
    equity = (1 + strat_ret).cumprod()
    bh = df["close"].pct_change().fillna(0.0)
    bh_equity = (1 + bh).cumprod()
    held = pd.Series(np.where(strat_ret != 0, 1.0, 0.0), index=df.index)

    m = _metrics(strat_ret, equity, bh_equity, held, pd.Series(0, index=df.index),
                 timeframe, n_trades)
    tp_arr = np.array(trade_pnls) if trade_pnls else np.array([0.0])
    m["win_rate"] = float((tp_arr > 0).mean())
    m["expectancy"] = float(tp_arr.mean())
    pf_w, pf_l = tp_arr[tp_arr > 0].sum(), -tp_arr[tp_arr < 0].sum()
    m["profit_factor"] = float(pf_w / pf_l) if pf_l > 0 else float("inf")
    return BacktestResult(equity, strat_ret, held, n_trades, m, bh_equity)


def run_backtest(
    df: pd.DataFrame,
    target_position: pd.Series,
    timeframe: str = "5m",
    costs: CostModel | None = None,
    size: pd.Series | None = None,
    min_hold: int = 0,
    stop_atr_mult: float = 0.0,
    tp_atr_mult: float = 0.0,
    risk_per_trade: float = 0.01,
    atr_period: int = 14,
) -> BacktestResult:
    costs = costs or CostModel()
    df = df.copy()

    pos = target_position.reindex(df.index).ffill().fillna(0.0)
    if min_hold > 1:
        pos = apply_min_hold(pos, min_hold)

    # Stop-loss path: intrabar exits need an event loop, so branch off here.
    if stop_atr_mult > 0:
        return _run_with_stops(df, pos, timeframe, costs, stop_atr_mult,
                               tp_atr_mult, risk_per_trade, atr_period)

    bar_ret = df["close"].pct_change().fillna(0.0)
    if size is not None:
        pos = pos * size.reindex(df.index).ffill().fillna(1.0).clip(0, 1)
    held = pos.shift(1).fillna(0.0)

    # Cost charged whenever the held position changes (in fraction of equity).
    turn = held.diff().abs().fillna(held.abs())
    cost = turn * costs.per_turn

    strat_ret = held * bar_ret - cost
    equity = (1 + strat_ret).cumprod()
    bh_equity = (1 + bar_ret).cumprod()

    # Trade accounting: a trade is a round trip (entry then exit/flip).
    changes = held.diff().fillna(held).ne(0)
    n_trades = int(changes.sum())

    metrics = _metrics(strat_ret, equity, bh_equity, held, cost, timeframe, n_trades)
    return BacktestResult(equity, strat_ret, held, n_trades, metrics, bh_equity)


def _metrics(ret, equity, bh_equity, held, cost, timeframe, n_trades) -> dict:
    bpy = BARS_PER_YEAR.get(timeframe, 8_760)
    n = len(ret)
    total_return = equity.iloc[-1] - 1 if n else 0.0
    bh_return = bh_equity.iloc[-1] - 1 if n else 0.0
    ann_return = (equity.iloc[-1]) ** (bpy / max(n, 1)) - 1 if n else 0.0

    vol = ret.std()
    ann_vol = vol * np.sqrt(bpy)
    sharpe = (ret.mean() / vol * np.sqrt(bpy)) if vol > 0 else 0.0
    downside = ret[ret < 0].std()
    sortino = (ret.mean() / downside * np.sqrt(bpy)) if downside > 0 else 0.0

    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1
    max_dd = drawdown.min()
    calmar = (ann_return / abs(max_dd)) if max_dd < 0 else 0.0

    # Per-trade pnl: segment returns between position changes.
    seg = held.diff().fillna(held).ne(0).cumsum()
    trade_pnl = ret.groupby(seg).sum()
    trade_pnl = trade_pnl[held.groupby(seg).first().ne(0)]  # only when in a position
    wins = trade_pnl[trade_pnl > 0]
    losses = trade_pnl[trade_pnl < 0]
    win_rate = len(wins) / len(trade_pnl) if len(trade_pnl) else 0.0
    profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    expectancy = trade_pnl.mean() if len(trade_pnl) else 0.0  # avg P&L per trade

    return {
        "total_return": total_return,
        "buy_hold_return": bh_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "trades": n_trades,
        "exposure": held.ne(0).mean(),
        "total_cost": cost.sum(),
    }
