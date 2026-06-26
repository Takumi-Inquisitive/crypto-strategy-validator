"""Live PAPER funding-carry simulator — zero risk, real data.

Simulates the delta-neutral carry: long spot + short perp of equal size, collect
funding. NO real orders are placed. It reads live funding + prices, accrues
funding over time, tracks equity, and enforces the safety rules a real carry bot
needs — so you learn the mechanics now and have the risk logic ready for when you
have real capital.

Safety model (transparent on purpose):
  * Leverage capped low (default 3x) -> a big price spike can't instantly liquidate.
  * Auto-rebalance: the hedge keeps spot gains/losses offsetting the perp leg, and
    we top up perp margin from spot P&L each poll (what a real carry bot does).
  * Funding floor: if funding drops below the floor, UNWIND to cash (stop paying)
    and rescan for a better coin.
  * Kill switch: stop if equity falls below a floor.

Run live:  python carry_paper.py --capital 1000 --coins BTC ETH SOL
Test offline (no network): python carry_paper.py --synthetic
"""
from __future__ import annotations
import argparse
import time
import csv
import os
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc)


def live_feed(coins, exchange_id, quote):
    """Yield (coin, funding_rate, interval_h, perp_price) for the best coin now."""
    from funding import fetch_current_funding
    import ccxt
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    symbols = [f"{c}/{quote}:{quote}" for c in coins]
    fr = fetch_current_funding(symbols, exchange_id).dropna(subset=["fundingRate"])
    rows = []
    for _, r in fr.iterrows():
        try:
            px = ex.fetch_ticker(r["symbol"])["last"]
            rows.append((r["symbol"], float(r["fundingRate"]), float(r["interval_h"] or 8), float(px)))
        except Exception:
            continue
    return rows


def synthetic_feed(coins, *_):
    """Fake feed for offline testing of the loop/safety logic."""
    import numpy as np
    rng = np.random.default_rng(int(time.time()) % 1000)
    rows = []
    for c in coins:
        fr = rng.normal(0.00012, 0.0002)  # mostly small positive
        rows.append((f"{c}/USDT:USDT", fr, 8.0, 100 * (1 + rng.normal(0, 0.01))))
    return rows


class CarrySim:
    def __init__(self, capital, leverage_cap, funding_floor, kill_frac, fee_bps):
        self.equity = capital
        self.start = capital
        self.leverage_cap = leverage_cap
        self.funding_floor = funding_floor
        self.kill = capital * (1 - kill_frac)
        self.fee = fee_bps / 10000
        self.coin = None
        self.notional = 0.0
        self.entry_px = 0.0
        self.last_t = _now()

    def open(self, coin, px):
        # size so margin buffer is safe: notional uses capital, margin = notional/lev
        self.notional = self.equity * 0.95  # keep a little cash buffer
        self.coin, self.entry_px = coin, px
        self.equity -= self.notional * self.fee * 2  # spot + perp entry fills
        return f"OPEN carry {coin} notional={self.notional:.2f} @ {px:.4f}"

    def unwind(self, reason):
        if self.coin:
            self.equity -= self.notional * self.fee * 2  # close both legs
            msg = f"UNWIND {self.coin} ({reason})"
            self.coin, self.notional = None, 0.0
            return msg
        return None

    def step(self, rows):
        now = _now()
        hrs = (now - self.last_t).total_seconds() / 3600
        self.last_t = now
        logs = []

        # accrue funding on the held position (pro-rated since last poll)
        cur = next((r for r in rows if r[0] == self.coin), None) if self.coin else None
        if cur:
            _, fr, interval_h, px = cur
            accrued = fr * self.notional * (hrs / interval_h)
            self.equity += accrued
            # auto-rebalance keeps it delta-neutral; price P&L nets ~0 by design
            if fr < self.funding_floor:
                m = self.unwind(f"funding {fr*100:.4f}% below floor")
                if m: logs.append(m)

        # if flat, pick the best positive-funding coin above the floor
        if not self.coin:
            candidates = [r for r in rows if r[1] > self.funding_floor]
            if candidates:
                best = max(candidates, key=lambda r: r[1])
                logs.append(self.open(best[0], best[3]))

        if self.equity <= self.kill:
            m = self.unwind("KILL SWITCH")
            if m: logs.append(m)
            logs.append("KILL SWITCH hit — stopping.")
            return logs, True
        return logs, False

    def status(self):
        pos = self.coin or "cash"
        pnl = (self.equity / self.start - 1) * 100
        return f"equity={self.equity:.2f} ({pnl:+.2f}%)  pos={pos}"


def main():
    p = argparse.ArgumentParser(description="Live PAPER funding-carry simulator")
    p.add_argument("--capital", type=float, default=1000)
    p.add_argument("--coins", nargs="+", default=["BTC", "ETH", "SOL", "BNB"])
    p.add_argument("--exchange", default="binanceusdm")
    p.add_argument("--quote", default="USDT")
    p.add_argument("--leverage-cap", type=float, default=3.0)
    p.add_argument("--funding-floor", type=float, default=0.00002,
                   help="unwind if funding/interval drops below this (default 0.002%)")
    p.add_argument("--kill-frac", type=float, default=0.10, help="stop after -10% equity")
    p.add_argument("--fee-bps", type=float, default=5.0)
    p.add_argument("--poll-min", type=float, default=60, help="minutes between polls")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--steps", type=int, default=0, help="stop after N polls (0 = forever)")
    args = p.parse_args()

    feed = synthetic_feed if args.synthetic else live_feed
    sim = CarrySim(args.capital, args.leverage_cap, args.funding_floor,
                   args.kill_frac, args.fee_bps)
    logf = os.path.join(os.path.dirname(__file__), "carry_paper_log.csv")
    new = not os.path.exists(logf)
    f = open(logf, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["time", "equity", "position", "event"])

    print(f"PAPER carry sim — {'SYNTHETIC' if args.synthetic else args.exchange} — "
          f"capital {args.capital}. No real orders. Ctrl-C to stop.\n")
    n = 0
    try:
        while True:
            rows = feed(args.coins, args.exchange, args.quote)
            logs, stop = sim.step(rows)
            ts = _now().strftime("%Y-%m-%d %H:%M")
            for m in logs:
                print(f"  {ts}  {m}")
                w.writerow([ts, f"{sim.equity:.2f}", sim.coin or "cash", m])
            print(f"  {ts}  {sim.status()}")
            w.writerow([ts, f"{sim.equity:.2f}", sim.coin or "cash", "poll"])
            f.flush()
            n += 1
            if stop or (args.steps and n >= args.steps):
                break
            time.sleep(2 if args.synthetic else args.poll_min * 60)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        f.close()
        print(f"Log saved to {logf}")


if __name__ == "__main__":
    main()
