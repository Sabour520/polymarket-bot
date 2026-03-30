"""
deploy/run_bot.py
-----------------
Live trading entry point.

Usage:
    python deploy/run_bot.py --strategy macd --token-id 0xABC... --interval 60

Environment:
    Copy .env.example to .env and fill in your credentials.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from bot.risk_manager import PortfolioState, RiskConfig
from bot.trader import Trader
from strategies import CVDStrategy, MACDStrategy, RSIStrategy

STRATEGY_MAP = {
    "macd": MACDStrategy,
    "rsi": RSIStrategy,
    "cvd": CVDStrategy,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_bot")


def fetch_candles(client, token_id: str, n: int = 100) -> pd.DataFrame:
    """
    Fetch recent candle/tick data from the Polymarket CLOB.
    The CLOB provides trade history; we resample into 1-min OHLCV bars here.
    """
    trades = client.get_last_trades_for_token(token_id=token_id)
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["price"] = df["price"].astype(float)
    df["size"] = df["size"].astype(float)
    df = df.sort_values("timestamp")

    ohlcv = (
        df.set_index("timestamp")
        .resample("1min")
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("size", "sum"),
        )
        .dropna()
        .reset_index()
        .tail(n)
    )
    return ohlcv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket limit-order trading bot")
    parser.add_argument("--strategy", choices=STRATEGY_MAP.keys(), default="macd")
    parser.add_argument("--token-id", required=True, help="Polymarket outcome token ID")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds")
    parser.add_argument("--capital", type=float, default=500.0, help="Starting cash (USD)")
    parser.add_argument("--max-position", type=float, default=100.0, help="Max position per token (USD)")
    parser.add_argument("--order-size", type=float, default=10.0, help="Default order size (USD)")
    parser.add_argument("--order-ttl", type=int, default=300, help="Stale order TTL in seconds (failsafe cancel, in addition to cancel-before-place)")
    parser.add_argument("--min-confidence", type=float, default=0.1, help="Minimum signal confidence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    risk_config = RiskConfig(
        max_position_usd=args.max_position,
        max_total_exposure_usd=args.capital * 0.8,
        max_order_size_usd=args.order_size * 2,
    )

    trader = Trader.from_env(risk_config=risk_config)
    strategy_cls = STRATEGY_MAP[args.strategy]
    strategy = strategy_cls(
        token_id=args.token_id,
        params={"order_size": args.order_size},
    )
    state = PortfolioState(cash=args.capital)

    logger.info(
        "Bot started | strategy=%s  token=%s  capital=%.2f  interval=%ds",
        args.strategy,
        args.token_id,
        args.capital,
        args.interval,
    )

    while True:
        try:
            # Sync fills first to keep state accurate
            fills = trader.sync_fills(state)
            if fills:
                logger.info("Fills: %s", fills)

            # Cancel orders that have been open too long
            stale = trader.cancel_stale_orders(state)
            if stale:
                logger.info("Cancelled %d stale orders", stale)

            # Fetch latest candle data
            data = fetch_candles(trader.client, args.token_id)
            if data.empty:
                logger.warning("No candle data returned — skipping cycle")
            else:
                signal = strategy.generate_signal(data)
                logger.info(
                    "Signal: %s  price=%.4f  size=%.2f  confidence=%.3f",
                    signal.signal_type.value,
                    signal.price,
                    signal.size,
                    signal.confidence,
                )

                if signal.confidence >= args.min_confidence:
                    order_id = trader.execute(signal, state)
                    if order_id:
                        logger.info("Order submitted: %s", order_id)

            logger.info(
                "State | cash=%.2f  exposure=%.2f  open_orders=%d",
                state.cash,
                state.total_exposure,
                sum(state.open_orders.values()),
            )

        except KeyboardInterrupt:
            logger.info("Shutting down — cancelling all open orders...")
            trader.cancel_all()
            break
        except Exception as exc:
            logger.error("Unexpected error: %s", exc, exc_info=True)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
