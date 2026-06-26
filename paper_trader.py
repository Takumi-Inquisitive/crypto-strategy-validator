"""Paper-trading loop — Phase 2.

Runs your chosen strategy live against an exchange TESTNET (fake money, real
prices). The execution path is plain deterministic Python — no LLM decides
orders. The LLM/agent layer is for research and review only.

Safety rails:
  * Defaults to --dry-run: prints intended orders, places nothing.
  * Real order placement requires BOTH --live-paper AND testnet API keys.
  * RiskGuard halts on daily-loss / kill-switch breaches.
  * It will refuse to run against a mainnet (real-money) endpoint here; flipping
    to real capital is a deliberate, separate change you make with eyes open.

Set keys in .env (see .env.example). Get free testnet keys at:
  Binance: https://testnet.binance.vision
  Bybit:   https://testnet.bybit.com
"""
from __future__ import annotations
import argparse
import os
import time

from strategies import REGISTRY
from risk import RiskGuard, atr_position_size


def load_env():
    path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def make_exchange(exchange_id: str, dry_run: bool):
    import ccxt
    key = os.environ.get(f"{exchange_id.upper()}_TESTNET_KEY", "")
    secret = os.environ.get(f"{exchange_id.upper()}_TESTNET_SECRET", "")
    ex = getattr(ccxt, exchange_id)({
        "apiKey": key, "secret": secret, "enableRateLimit": True,
    })
    ex.set_sandbox_mode(True)  # <-- force TESTNET. Refuses real money.
    return ex


def run(args):
    load_env()
    strat = REGISTRY[args.strategy]()
    ex = make_exchange(args.exchange, args.dry_run)

    # Establish starting equity from testnet balance (fallback to nominal).
    try:
        bal = ex.fetch_balance()
        equity = float(bal["total"].get("USDT", args.equity)) or args.equity
    except Exception:
        equity = args.equity

    guard = RiskGuard(starting_equity=equity, max_daily_loss=args.max_daily_loss)
    position_open = False
    print(f"Paper trader: {strat.name} on {args.symbol} {args.timeframe} "
          f"({'DRY-RUN' if args.dry_run else 'LIVE-PAPER on testnet'})")
    print(f"Starting equity ~{equity:.2f} USDT. Ctrl-C to stop.\n")

    while True:
        try:
            ohlcv = ex.fetch_ohlcv(args.symbol, args.timeframe, limit=300)
            import pandas as pd
            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df = df.set_index("ts").astype(float)

            target = strat.positions(df).iloc[-1]
            ts = df.index[-1]

            # mark-to-market for the guard
            try:
                bal = ex.fetch_balance()
                equity = float(bal["total"].get("USDT", equity)) or equity
            except Exception:
                pass
            can_trade, reason = guard.update(equity, ts)

            want_long = target > 0
            if want_long and not position_open and can_trade:
                size = atr_position_size(df).iloc[-1] * equity / df["close"].iloc[-1]
                _act("BUY", args.symbol, size, df["close"].iloc[-1], ex, args.dry_run)
                position_open = True
            elif not want_long and position_open:
                _act("SELL", args.symbol, None, df["close"].iloc[-1], ex, args.dry_run)
                position_open = False
            else:
                state = "in position" if position_open else "flat"
                gate = "" if can_trade else f" [{reason}]"
                print(f"{ts}  signal={target:+.0f}  {state}{gate}")

            time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"  ! error: {e}; retrying in {args.poll_seconds}s")
            time.sleep(args.poll_seconds)


def _act(side, symbol, amount, price, ex, dry_run):
    if dry_run or amount is None and side == "BUY":
        print(f"  >> {side} {symbol} ~{amount} @ {price:.2f}  (dry-run, no order sent)"
              if dry_run else f"  >> {side} {symbol} @ {price:.2f}")
        if dry_run:
            return
    try:
        if side == "BUY":
            ex.create_market_buy_order(symbol, amount)
        else:
            base = symbol.split("/")[0]
            held = ex.fetch_balance()["free"].get(base, 0)
            if held:
                ex.create_market_sell_order(symbol, held)
        print(f"  >> {side} {symbol} order placed @ ~{price:.2f}")
    except Exception as e:
        print(f"  !! order failed: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Crypto paper trader (testnet only)")
    p.add_argument("--strategy", choices=REGISTRY, default="donchian")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--exchange", default="binance")
    p.add_argument("--equity", type=float, default=10000.0)
    p.add_argument("--max-daily-loss", type=float, default=0.03)
    p.add_argument("--poll-seconds", type=int, default=60)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", default=True)
    group.add_argument("--live-paper", dest="dry_run", action="store_false",
                       help="actually place orders on the TESTNET")
    run(p.parse_args())
