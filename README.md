# cryptobot — intraday crypto strategy lab (paper-first)

A small, honest framework for building and **validating** intraday crypto
strategies before any real money is involved. Crypto (Binance/Bybit) · intraday ·
paper now, live later.

## The one rule that matters

**The walk-forward verdict is the truth; the single-pass backtest is marketing.**
Anyone can produce a green equity curve by tuning a strategy on past data. This
repo is built to *catch you doing that to yourself*. If a strategy can't keep a
positive Sharpe on out-of-sample data after realistic fees and slippage, it does
not go near capital — paper or real.

This is not a money printer and there is no "winning" bot. The realistic win is
a disciplined process: validate honestly → paper-trade for weeks → risk-manage
hard → only ever risk money you can afford to lose.

## Layout

| File | Role |
|---|---|
| `data.py` | OHLCV via ccxt with caching; `synthetic_ohlcv()` for offline testing |
| `indicators.py` | Causal indicators (EMA, RSI, ATR, Donchian) — no lookahead |
| `strategies.py` | Pluggable strategies (`ema_cross`, `rsi_meanrev`, `donchian`) |
| `backtest.py` | Backtester with fees + slippage, signal shifted +1 bar, honest metrics |
| `walkforward.py` | Rolling out-of-sample validation + overfitting verdict |
| `risk.py` | ATR position sizing, daily-loss limit, kill switch |
| `run_backtest.py` | CLI: backtest + walk-forward |
| `paper_trader.py` | Phase 2: live loop on exchange **testnet** (deterministic) |

## Quick start

```bash
pip install -r requirements.txt

# Works offline, anywhere (no network/exchange needed):
python run_backtest.py --strategy donchian --synthetic

# Real data (run on your own machine — exchanges must be reachable):
python run_backtest.py --strategy donchian --symbol BTC/USDT --timeframe 5m --limit 6000
python run_backtest.py --strategy ema_cross --risk-sizing --taker-bps 6 --slippage-bps 3
```

Read the metrics in this order: **walk-forward VERDICT → out-of-sample Sharpe →
net return vs buy & hold → max drawdown.** Ignore win rate until the rest holds.

## Paper trading (Phase 2)

```bash
cp .env.example .env          # add TESTNET keys only — never real ones
python paper_trader.py --strategy donchian --symbol BTC/USDT --timeframe 5m   # dry-run
python paper_trader.py --strategy donchian --live-paper                       # places orders on TESTNET
```

`paper_trader.py` forces sandbox/testnet mode and defaults to dry-run. The
execution path is plain Python; no language model places orders.

## Where the LLM / MCP fits

Keep the model **out of the live order loop**. Good uses: researching new
signal ideas, reviewing strategy code, sanity-checking a walk-forward report,
summarising market context. The `tradingview-mcp` server is excellent for that
research layer inside Claude Desktop (install it there separately; it's
read-only analysis and never executes trades). Wire it to *inform* you, not to
auto-trade.

## Stress-testing across market regimes

The advice that matters most: a strategy isn't robust because it survives *time*,
it's robust because it survives *regimes*. Attribute P&L to trend-vs-chop and
high-vs-low-vol:

```bash
python run_backtest.py --strategy donchian --synthetic --regimes
```

If the verdict says "edge is TREND-only and loses in CHOP", that strategy will
quietly die the moment the market stops trending — exactly the overfitting trap
to avoid. Curb fee-bleed from overtrading with a minimum hold period:

```bash
python run_backtest.py --strategy ema_cross --synthetic --min-hold 24
```

## The generalisation test (most important workflow)

One coin over one window proves nothing. Download breadth, then test whether an
edge holds across coins:

```bash
# 1. pull ~2 years of hourly history for several coins (once; then it's cached)
python download_data.py --coins BTC/USDT ETH/USDT SOL/USDT BNB/USDT XRP/USDT --timeframes 1h --limit 17520

# 2. run one strategy across all of them and read the verdict at the bottom
python compare.py --strategy rsi_meanrev --timeframe 1h
python compare.py --strategy rsi_meanrev --timeframe 1h --chop-filter   # A/B the filter
```

`compare.py` ends with GENERALISES / MIXED / DOES NOT GENERALISE. Only the first
is worth paper-trading — and even then, paper-trade first.

## Risk management: stops + position sizing

Van Tharp's claim — the exit and bet size matter more than the entry — is testable
here. The stop defines risk; position size is set so every trade risks the same
fixed % of equity (`--risk`), capped at 1x (no leverage).

```bash
# breakout (trend) with a 2-ATR stop — the natural pairing:
python compare.py --strategy donchian --timeframe 1h --stop-atr 2.0 --risk 0.01
# mean-reversion with a stop — expect it to HURT (stops fight mean-reversion):
python compare.py --strategy rsi_meanrev --timeframe 1h --stop-atr 2.0
```

Reality check baked into the design: position sizing cannot flip a negative-
expectancy signal to positive — it only rescales risk. A stop can change
expectancy by truncating the loss tail, but only helps strategies whose losers
genuinely run (trend-following), not mean-reversion.

## Funding-rate carry (the one structural edge)

Long spot + short perpetual of equal size = delta-neutral. You don't predict
price; you collect the funding longs pay shorts. Real, mechanical — but it's a
**bull-market premium**: fat when everyone's leveraged long, negative in fear.

```bash
# what's the carry right now, ranked across coins:
python funding_scan.py --coins BTC ETH SOL BNB XRP DOGE

# does the HISTORY actually pay after fees? (the test that matters)
python funding_backtest.py --symbol BTC/USDT:USDT --exchange binanceusdm --limit 4000
```

The backtest reports annualised NET yield on capital, the % of intervals you
*paid* (negative funding), and the worst 30-day window. Honest bars: under ~4%/yr
net it's worse than stablecoin lending for more risk; and it carries real
liquidation + exchange counterparty risk the backtest can't capture.

## HMM regime detection (done honestly)

Hidden Markov Models label hidden market states (bull/crash/chop) and you trade
only favourable ones. The technique is real — but most demos fit the HMM on the
WHOLE history then backtest on those labels, which is lookahead and inflates
results massively. Here the HMM is fit on past data only and the live state is
inferred by causal forward-filtering. Walk-forward, no leverage:

```bash
python hmm_backtest.py --coins BTC/USDT ETH/USDT SOL/USDT --timeframe 1h --states 4 --min-hold 6
```

Bottom line tells you GENERALISES or "mirage". If an edge only appears with
lookahead or leverage, it isn't real.

## Honest roadmap

- [x] Causal indicators + backtester with realistic costs
- [x] Walk-forward validation with overfitting verdict
- [x] Regime stress-testing (trend/chop, high/low vol) with attribution
- [x] Smoothed regime filters (no flicker) + overtrading control
- [x] Multi-coin / multi-year downloader + cross-coin generalisation test
- [x] Stop-loss / take-profit engine + risk-based position sizing + expectancy
- [ ] Regime-aware walk-forward (require positive OOS Sharpe in every regime)
- [ ] Funding-rate / fee modelling for perp futures
- [ ] Portfolio of pairs + correlation-aware sizing
- [ ] Logging + paper-trading P&L dashboard
- [ ] Only after weeks of green paper results: a deliberate, separately-reviewed
      switch to real capital with hard per-trade and per-day caps

## Disclaimer

Educational tooling, not financial advice. Trading carries substantial risk of
loss. You are responsible for your own decisions and for complying with the laws
and exchange rules that apply to you.
