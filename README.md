# Crypto Strategy Validation Framework

A Python framework for **honestly** backtesting and validating crypto trading
strategies — to find out whether a strategy has a real edge, or just looks good on
one lucky window. Most retail backtests lie (they tune on all history, report the
in-sample number, and ignore fees). This one is built to catch that, and to surface
the rare strategies that actually hold up.

> **Available for freelance work** — strategy backtesting, validation, and custom
> trading tools in Python. If you have a strategy you want tested *honestly* before
> risking money, that's exactly what this is for.

---

## What it found

### ✅ An edge that holds up: funding-rate carry

The delta-neutral spot-perp basis trade (long spot, short perpetual, collect
funding) tested across 3.5 years of BTC data:

| Metric | Result |
|---|---|
| Annualised net yield | **+5.9% / year** (after fees) |
| Collected funding | 85% of intervals |
| Max drawdown | **−0.41%** |

Small, boring, survivable — what a *real* edge looks like. The framework confirmed
it isn't lookahead or leverage: it's a genuine structural premium.

### ❌ Two viral "AI quant" strategies that don't

**A Hidden Markov Model "regime" strategy** (marketed as trading "like a hedge
fund"), tested causally — HMM fit on past data only, no lookahead — across 5 coins
and 2 years. It beat buy-and-hold on **0 of 5 coins.** Example: on XRP it returned
−77.6% while simply holding returned **+119.2%.** Once lookahead bias and leverage
were removed, the advertised edge vanished.

**A "Gaussian Channel" strategy** (advertised "7,492% since 2018"). The big numbers
were real but misleading — it *underperformed simply holding the coins on 4 of 5
assets.* That "profit" was bull-market beta, not skill.

**The discipline this enforces:** the honest benchmark isn't zero, it's
buy-and-hold. A strategy showing +1500% can still be a failure if holding made
+2600%. This toolkit always shows that comparison — which is how it separates a real
edge (the carry above) from an expensive illusion (the two below).

---

## What it does

- **Realistic backtesting** — fees + slippage on every trade, no lookahead (signals
  act next bar), full metrics always vs buy-and-hold.
- **Walk-forward validation** — out-of-sample testing with a blunt verdict:
  ROBUST / MODERATE / WEAK / OVERFITTED.
- **Regime stress-testing** — attributes P&L to trend-vs-chop and high-vs-low
  volatility, so you see *where* an edge lives and where it dies.
- **Cross-asset generalisation** — runs a strategy across many coins and reports
  GENERALISES / MIXED / DOES NOT GENERALISE.
- **Risk management** — ATR position sizing, stop-loss/take-profit engine,
  daily-loss limit, kill switch, expectancy.
- **Causal HMM regime detection** — fit on past data only, state inferred by
  forward-filtering (no lookahead, unlike most HMM demos).
- **Funding-rate carry** — backtest + live scanner + paper simulator for the
  delta-neutral basis trade.
- **Paper trading** — testnet execution loop with risk controls wired in.

## Quick start

```bash
pip install -r requirements.txt

# offline self-test (no network/keys):
python run_backtest.py --strategy donchian --synthetic --regimes

# real data + the generalisation test that matters:
python download_data.py --coins BTC/USDT ETH/USDT SOL/USDT --timeframes 1h --limit 17520
python compare.py --strategy rsi_meanrev --timeframe 1h

# the edge that held up:
python funding_backtest.py --symbol BTC/USDT:USDT --exchange binanceusdm --limit 4000
```

Read metrics in this order: **walk-forward VERDICT → out-of-sample Sharpe → net vs
buy-and-hold → max drawdown.** Win rate is the last thing to look at, not the first.

## Layout

| File | Role |
|---|---|
| `data.py` / `download_data.py` | OHLCV via ccxt, caching, multi-coin bulk download |
| `indicators.py` | Causal indicators (EMA, RSI, ATR, Donchian) |
| `strategies.py` / `gaussian_channel.py` | Pluggable strategies + Gaussian Channel |
| `backtest.py` | Backtester: fees, slippage, stops, sizing, metrics |
| `walkforward.py` | Out-of-sample validation + overfitting verdict |
| `regime.py` | Regime classification + per-regime P&L attribution |
| `compare.py` | Cross-coin generalisation test |
| `hmm_regime.py` / `hmm_backtest.py` | Causal HMM regime detection |
| `funding*.py` / `carry_paper.py` | Funding-carry backtest, scanner, paper sim |
| `paper_trader.py` | Testnet paper-trading loop |

## Design principle

Every feature exists to make it *harder* to fool yourself. A strategy is only worth
paper-trading if it survives walk-forward, holds across multiple coins, and isn't
concentrated in one regime — after realistic costs. That discipline is the
difference between finding a real edge and funding an expensive mistake.

## Tech

Python · pandas · numpy · ccxt · hmmlearn · scikit-learn

## Disclaimer

Research and educational tooling. Not financial advice. Trading carries substantial
risk of loss.
