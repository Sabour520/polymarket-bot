"""
Backtest Engine
---------------
Event-driven backtest engine that simulates limit-order execution on
historical OHLCV data.

Limit order fill logic:
  - BUY limit at price P fills when the candle's LOW <= P (passive fill)
  - SELL limit at price P fills when the candle's HIGH >= P
  - Partial fills are not modelled; each signal triggers one full order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Type

import pandas as pd

from strategies.base_strategy import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    timestamp: pd.Timestamp
    signal_type: SignalType
    limit_price: float
    fill_price: float
    size: float          # USD notional
    shares: float        # size / fill_price
    pnl: float = 0.0     # populated on close


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float   # gross_profit / gross_loss  (inf if no losses)
    avg_win: float         # average P&L of winning trades
    avg_loss: float        # average P&L of losing trades (negative value)
    total_trades: int

    def summary(self) -> str:
        pf = f"{self.profit_factor:.3f}" if self.profit_factor != float("inf") else "∞"
        return (
            f"Trades       : {self.total_trades}\n"
            f"Total Return : {self.total_return:.2%}\n"
            f"Sharpe Ratio : {self.sharpe_ratio:.3f}\n"
            f"Max Drawdown : {self.max_drawdown:.2%}\n"
            f"Win Rate     : {self.win_rate:.2%}\n"
            f"Profit Factor: {pf}\n"
            f"Avg Win      : ${self.avg_win:.4f}\n"
            f"Avg Loss     : ${self.avg_loss:.4f}"
        )


class BacktestEngine:
    """
    Walk-forward backtest engine.

    Usage:
        engine = BacktestEngine(strategy_cls=MACDStrategy, token_id="0x...",
                                initial_capital=1000.0)
        result = engine.run(data)
        print(result.summary())
    """

    def __init__(
        self,
        strategy_cls: Type[BaseStrategy],
        token_id: str,
        strategy_params: dict | None = None,
        initial_capital: float = 1000.0,
        max_position_usd: float = 100.0,
        fee_rate: float = 0.0,          # Polymarket charges no maker fees
        min_confidence: float = 0.0,
    ):
        self.strategy_cls = strategy_cls
        self.token_id = token_id
        self.strategy_params = strategy_params or {}
        self.initial_capital = initial_capital
        self.max_position_usd = max_position_usd
        self.fee_rate = fee_rate
        self.min_confidence = min_confidence

    def run(self, data: pd.DataFrame) -> BacktestResult:
        strategy = self.strategy_cls(self.token_id, self.strategy_params)
        required = strategy._required_rows()

        cash = self.initial_capital
        position_shares = 0.0
        position_cost = 0.0
        equity_values: list[float] = []
        trades: list[Trade] = []

        for i in range(required, len(data)):
            window = data.iloc[: i + 1]
            current = data.iloc[i]

            signal: Signal = strategy.generate_signal(window)

            if signal.is_actionable() and signal.confidence >= self.min_confidence:
                if signal.signal_type == SignalType.BUY and position_shares == 0:
                    order_size = min(signal.size, cash, self.max_position_usd)
                    if order_size > 0 and current["low"] <= signal.price:
                        fill = signal.price
                        fee = order_size * self.fee_rate
                        shares_bought = (order_size - fee) / fill
                        cash -= order_size
                        position_shares += shares_bought
                        position_cost = fill
                        t = Trade(
                            timestamp=current.get("timestamp", pd.Timestamp(i)),
                            signal_type=SignalType.BUY,
                            limit_price=signal.price,
                            fill_price=fill,
                            size=order_size,
                            shares=shares_bought,
                        )
                        trades.append(t)
                        logger.debug("BUY  %.4f @ %.4f  cash=%.2f", shares_bought, fill, cash)

                elif signal.signal_type == SignalType.SELL and position_shares > 0:
                    if current["high"] >= signal.price:
                        fill = signal.price
                        proceeds = position_shares * fill
                        fee = proceeds * self.fee_rate
                        pnl = proceeds - fee - (position_shares * position_cost)
                        cash += proceeds - fee

                        t = Trade(
                            timestamp=current.get("timestamp", pd.Timestamp(i)),
                            signal_type=SignalType.SELL,
                            limit_price=signal.price,
                            fill_price=fill,
                            size=proceeds,
                            shares=position_shares,
                            pnl=pnl,
                        )
                        trades.append(t)
                        logger.debug("SELL %.4f @ %.4f  pnl=%.4f  cash=%.2f", position_shares, fill, pnl, cash)
                        position_shares = 0.0
                        position_cost = 0.0

            mark = current["close"]
            equity_values.append(cash + position_shares * mark)

        equity = pd.Series(equity_values)
        return self._compute_metrics(trades, equity)

    def _compute_metrics(self, trades: list[Trade], equity: pd.Series) -> BacktestResult:
        total_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital if len(equity) else 0.0

        returns = equity.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0.0

        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_dd = drawdown.min() if len(drawdown) else 0.0

        sell_trades = [t for t in trades if t.signal_type == SignalType.SELL]
        winning = [t.pnl for t in sell_trades if t.pnl > 0]
        losing  = [t.pnl for t in sell_trades if t.pnl <= 0]

        win_rate = len(winning) / len(sell_trades) if sell_trades else 0.0

        gross_profit = sum(winning)
        gross_loss   = abs(sum(losing))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        avg_win  = (sum(winning) / len(winning)) if winning else 0.0
        avg_loss = (sum(losing)  / len(losing))  if losing  else 0.0

        return BacktestResult(
            trades=trades,
            equity_curve=equity,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=len(sell_trades),
        )
