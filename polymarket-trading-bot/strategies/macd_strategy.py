"""
MACD Histogram Strategy  (fast=3, slow=15, signal=3)
-----------------------------------------------------
Tuned for short 5-minute windows on Polymarket binary markets.

Entry  — MACD histogram crosses zero (MACD line crosses signal line):
           histogram[-2] <= 0 and histogram[-1] > 0  →  BUY
           histogram[-2] >= 0 and histogram[-1] < 0  →  SELL (exit / short)

Exit   — reverse crossover (opposite histogram zero-cross)
         OR stop-loss / take-profit levels reached (embedded in signal metadata
         so the trader or run_bot loop can act on them)

Limit price is placed passively inside the spread by `spread_offset`.
"""

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType


class MACDStrategy(BaseStrategy):
    """
    Params (all optional, defaults shown):
        fast_period   : int   = 3
        slow_period   : int   = 15
        signal_period : int   = 3
        order_size    : float = 10.0
        spread_offset : float = 0.002
        stop_loss_pct : float = 0.05   (5% below entry price)
        take_profit_pct: float = 0.10  (10% above entry price)
    """

    DEFAULT_PARAMS = {
        "fast_period": 3,
        "slow_period": 15,
        "signal_period": 3,
        "order_size": 10.0,
        "spread_offset": 0.002,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.10,
    }

    def __init__(self, token_id: str, params: dict | None = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(token_id, merged)
        self._entry_price: float | None = None

    def _required_rows(self) -> int:
        return self.params["slow_period"] + self.params["signal_period"]

    def _compute_macd(self, close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = close.ewm(span=self.params["fast_period"], adjust=False).mean()
        ema_slow = close.ewm(span=self.params["slow_period"], adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.params["signal_period"], adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        if len(data) < self._required_rows():
            return self._hold()

        close = data["close"]
        mid = close.iloc[-1]
        offset = self.params["spread_offset"]
        size = self.params["order_size"]

        _, _, histogram = self._compute_macd(close)
        prev_hist = histogram.iloc[-2]
        curr_hist = histogram.iloc[-1]

        # ── Exit checks (SL / TP) — evaluated before entry logic ──────────
        if self._entry_price is not None:
            sl = self._entry_price * (1 - self.params["stop_loss_pct"])
            tp = self._entry_price * (1 + self.params["take_profit_pct"])
            if mid <= sl or mid >= tp:
                reason = "stop_loss" if mid <= sl else "take_profit"
                self._entry_price = None
                return Signal(
                    signal_type=SignalType.SELL,
                    price=round(mid * (1 + offset), 4),
                    size=size,
                    confidence=1.0,
                    token_id=self.token_id,
                    metadata={"reason": reason, "exit_price": mid},
                )

        # ── Bullish histogram zero-cross → BUY entry ───────────────────────
        if prev_hist <= 0 and curr_hist > 0:
            limit_price = round(mid * (1 - offset), 4)
            confidence = min(abs(curr_hist) / (mid + 1e-9) * 200, 1.0)
            self._entry_price = mid
            return Signal(
                signal_type=SignalType.BUY,
                price=limit_price,
                size=size,
                confidence=confidence,
                token_id=self.token_id,
                metadata={
                    "histogram": round(curr_hist, 6),
                    "stop_loss": round(mid * (1 - self.params["stop_loss_pct"]), 4),
                    "take_profit": round(mid * (1 + self.params["take_profit_pct"]), 4),
                },
            )

        # ── Bearish histogram zero-cross → SELL / exit ────────────────────
        if prev_hist >= 0 and curr_hist < 0:
            limit_price = round(mid * (1 + offset), 4)
            confidence = min(abs(curr_hist) / (mid + 1e-9) * 200, 1.0)
            self._entry_price = None
            return Signal(
                signal_type=SignalType.SELL,
                price=limit_price,
                size=size,
                confidence=confidence,
                token_id=self.token_id,
                metadata={
                    "histogram": round(curr_hist, 6),
                    "reason": "reverse_crossover",
                },
            )

        return self._hold()
