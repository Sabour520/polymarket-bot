"""
deploy/run_backtest.py
----------------------
Backtest entry point.

Usage:
    # Run all three strategies on synthetic data
    python deploy/run_backtest.py

    # Run on a CSV file
    python deploy/run_backtest.py --data path/to/data.csv --strategy rsi

    # Tune parameters
    python deploy/run_backtest.py --strategy macd --fast 8 --slow 21 --signal 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")   # headless-safe backend
import matplotlib.pyplot as plt

from backtesting.data_loader import DataLoader
from backtesting.engine import BacktestEngine
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
logger = logging.getLogger("run_backtest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket strategy backtester")
    parser.add_argument("--strategy", choices=list(STRATEGY_MAP.keys()) + ["all"], default="all")
    parser.add_argument("--data", default=None, help="Path to OHLCV CSV file (optional)")
    parser.add_argument("--token-id", default="BACKTEST_TOKEN", help="Token ID label for backtest")
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--order-size", type=float, default=20.0)
    parser.add_argument("--max-position", type=float, default=200.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--plot", action="store_true", help="Save equity curve plots")
    parser.add_argument("--min-trades", type=int, default=100, help="Minimum trades required for benchmark pass")
    # MACD params
    parser.add_argument("--fast", type=int, default=3)
    parser.add_argument("--slow", type=int, default=15)
    parser.add_argument("--signal", type=int, default=3)
    parser.add_argument("--stop-loss-pct", type=float, default=0.05)
    parser.add_argument("--take-profit-pct", type=float, default=0.10)
    # RSI params
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--oversold", type=float, default=30.0)
    parser.add_argument("--exit-rsi", type=float, default=50.0)
    # CVD params
    parser.add_argument("--cvd-lookback", type=int, default=10)
    parser.add_argument("--cvd-price-min-move", type=float, default=0.005)
    parser.add_argument("--cvd-min-move", type=float, default=0.005)
    return parser.parse_args()


def strategy_params(args: argparse.Namespace) -> dict:
    return {
        "macd": {
            "fast_period": args.fast,
            "slow_period": args.slow,
            "signal_period": args.signal,
            "stop_loss_pct": args.stop_loss_pct,
            "take_profit_pct": args.take_profit_pct,
            "order_size": args.order_size,
        },
        "rsi": {
            "period": args.rsi_period,
            "oversold": args.oversold,
            "exit_rsi": args.exit_rsi,
            "order_size": args.order_size,
        },
        "cvd": {
            "lookback": args.cvd_lookback,
            "price_min_move": args.cvd_price_min_move,
            "cvd_min_move": args.cvd_min_move,
            "order_size": args.order_size,
        },
    }


BENCHMARKS = {
    "win_rate":      ("Win Rate > 55%",        lambda r, a: r.win_rate > 0.55),
    "profit_factor": ("Profit Factor > 1.5",   lambda r, a: r.profit_factor > 1.5),
    "max_drawdown":  ("Max Drawdown < 20%",    lambda r, a: r.max_drawdown > -0.20),
    "min_trades":    ("Trades >= {n}",         lambda r, a: r.total_trades >= a.min_trades),
}


def benchmark_report(result, args: argparse.Namespace) -> bool:
    """Print a pass/fail table against the profitability benchmarks. Returns True if all pass."""
    print("\nBenchmark Check")
    print("-" * 36)
    all_pass = True
    for key, (label_tpl, test) in BENCHMARKS.items():
        label = label_tpl.format(n=args.min_trades)
        passed = test(result, args)
        mark = "PASS" if passed else "FAIL"
        all_pass = all_pass and passed

        # Show the actual value next to each check
        if key == "win_rate":
            actual = f"{result.win_rate:.1%}"
        elif key == "profit_factor":
            pf = result.profit_factor
            actual = f"{pf:.3f}" if pf != float('inf') else "inf"
        elif key == "max_drawdown":
            actual = f"{result.max_drawdown:.1%}"
        else:
            actual = str(result.total_trades)

        print(f"  [{mark}]  {label:<28}  actual: {actual}")

    print("-" * 36)
    verdict = "DEPLOYABLE" if all_pass else "NOT READY — tune parameters or gather more data"
    print(f"  Verdict: {verdict}")
    return all_pass


def run_single(name: str, strategy_cls, params: dict, data, args: argparse.Namespace):
    engine = BacktestEngine(
        strategy_cls=strategy_cls,
        token_id=args.token_id,
        strategy_params=params,
        initial_capital=args.capital,
        max_position_usd=args.max_position,
        min_confidence=args.min_confidence,
    )
    result = engine.run(data)

    print(f"\n{'='*44}")
    print(f"  Strategy: {name.upper()}")
    print(f"{'='*44}")
    print(result.summary())
    benchmark_report(result, args)

    if args.plot and len(result.equity_curve) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]})

        result.equity_curve.plot(ax=axes[0], color="steelblue", label="Portfolio value")
        axes[0].axhline(args.capital, linestyle="--", color="grey", label="Starting capital")
        axes[0].set_title(f"{name.upper()} — Equity Curve")
        axes[0].set_ylabel("Portfolio Value (USD)")
        axes[0].legend()

        # Drawdown subplot
        dd = (result.equity_curve - result.equity_curve.cummax()) / result.equity_curve.cummax()
        dd.plot(ax=axes[1], color="firebrick", label="Drawdown")
        axes[1].axhline(-0.20, linestyle="--", color="orange", linewidth=0.8, label="−20% threshold")
        axes[1].set_ylabel("Drawdown")
        axes[1].legend(fontsize=8)
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

        fig.tight_layout()
        out = Path(f"backtest_{name}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nEquity curve saved to {out}")

    return result


def main() -> None:
    args = parse_args()
    params = strategy_params(args)

    if args.data:
        data = DataLoader.from_csv(args.data)
        logger.info("Loaded data from %s (%d rows)", args.data, len(data))
    else:
        data = DataLoader.synthetic(n=500)
        logger.info("Using synthetic data (%d rows)", len(data))

    strategies_to_run = (
        list(STRATEGY_MAP.items())
        if args.strategy == "all"
        else [(args.strategy, STRATEGY_MAP[args.strategy])]
    )

    for name, cls in strategies_to_run:
        run_single(name, cls, params[name], data, args)

    print("\nBacktest complete.")


if __name__ == "__main__":
    main()
