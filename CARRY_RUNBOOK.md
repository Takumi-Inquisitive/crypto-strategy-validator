# Funding Carry — Runbook

The one edge that survived honest testing. This is a **yield tool**, not an income
machine: realistic net is ~5-9%/yr in normal conditions, lower or negative in bear
markets. It only matters with real capital — on $100 it earns cents. Build the
habit now; deploy size later.

## The trade, plainly

Hold spot long + short the perpetual future of equal size = delta-neutral (price
moves cancel). You collect the funding longs pay shorts. When funding is positive
you get paid; when negative you pay — so you exit to cash when it turns.

## Tools

```bash
# 1. Where's the carry right now?
python funding_scan.py --coins BTC ETH SOL BNB XRP DOGE

# 2. Does the history actually pay after fees?
python funding_backtest.py --symbol BTC/USDT:USDT --exchange binanceusdm --limit 4000

# 3. PAPER-trade it live (zero risk, no real orders) — watch it for weeks:
python carry_paper.py --capital 1000 --coins BTC ETH SOL BNB
```

`carry_paper.py` logs every poll to `carry_paper_log.csv` so you can chart the
equity drip over time. Let it run for weeks. You'll see the truth firsthand: slow,
boring, positive in calm markets — and how it behaves when funding flips.

## The risks the backtest CANNOT show you

These are what actually bankrupt carry traders. Respect them:

1. **Liquidation of the short leg.** A violent price spike can liquidate the perp
   short before the spot gain is realised, especially with isolated margin. Defence:
   low leverage (<=3x), fat margin buffer, cross-margin, and active rebalancing.
2. **Exchange counterparty risk.** An exchange freezing withdrawals has wiped people
   out (FTX, Celsius). Defence: don't keep everything on one venue; withdraw profits.
3. **Funding regime flips.** In bears, funding goes negative for long stretches —
   the carry stops paying or costs you. Defence: the funding floor (auto-unwind).
4. **Basis/execution slippage.** Spot and perp can decouple in a crash, so entry/exit
   isn't free. Defence: size conservatively; don't trade illiquid coins.

## Go-live checklist (only after weeks of green paper results)

- [ ] Paper-traded for at least 4 weeks; understand every unwind it made
- [ ] Real capital you can afford to lock up AND lose
- [ ] Leverage <=3x with a margin buffer; cross-margin on the perp leg
- [ ] Start tiny (a fraction of intended size) with real money
- [ ] Funds split across 2 venues if possible; withdraw profits regularly
- [ ] A hard kill switch and a number that makes you stop, written down in advance

## Honest expectation

This will not make you rich and it is not a salary. It is a better-than-savings
yield for more-than-savings risk and effort. The math: ~6%/yr is $6 on $100, $600
on $10k, $6k on $100k. The strategy doesn't change with size — only the dollars do.
That's why building capital (the freelance path) comes first.
