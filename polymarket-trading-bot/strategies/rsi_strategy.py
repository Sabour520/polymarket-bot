"""
RSI Mean-Reversion Strategy  (RSI 14)
--------------------------------------
Suited for pullbacks after sharp moves in Polymarket binary markets.

Entry  — RSI < 30 (oversold) → limit BUY

Exit   — EITHER condition triggers a limit SELL:
           1. RSI rises back above 50 (momentum recovered)
           2. Current price crosses above VWAP (price returns to fair value)

VWAP is computed over the full window passed to generate_signal.
Requires 'high', 'low', 'volume' columns in addition to 'close'.
"""

import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy, Signal, SignalType


class RSIStrategy(BaseStrategy):
    """
    Params (all optional, defaults shown):
        period        : int   = 14
        oversold      : float = 30.0
        exit_rsi      : float = 50.0   (RSI level that triggers exit)
        order_size    : float = 10.0
        spread_offset : float = 0.002
    """

    DEFAULT_PARAMS = {
        "period": 14,
        "oversold": 30.0,
        "exit_rsi": 50.0,
        "order_size": 10.0,
        "spread_offset": 0.002,
    }

    def __init__(self, token_id: str, params: dict | None = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(token_id, merged)
        self._in_position: bool = False

    def _required_rows(self) -> int:
        return self.params["period"] + 1

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_vwap(data: pd.DataFrame) -> float:
        """Session VWAP over the entire window."""
        if "high" not in data.columns or "low" not in data.columns:
            return float("nan")
        typical = (data["high"] + data["low"] + data["close"]) / 3
        vol = data["volume"].replace(0, np.nan)
        vwap = (typical * vol).cumsum() / vol.cumsum()
        return float(vwap.iloc[-1])

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        if len(data) < self._required_rows():
            return self._hold()

        close = data["close"]
        mid = close.iloc[-1]
        offset = self.params["spread_offset"]
        size = self.params["order_size"]

        rsi = self._compute_rsi(close, self.params["period"])
        current_rsi = float(rsi.iloc[-1])
        vwap = self._compute_vwap(data)

        # ── Exit: RSI recovered above 50, OR price crossed back to VWAP ───
        if self._in_position:
            rsi_exit = current_rsi >= self.params["exit_rsi"]
            vwap_exit = (not np.isnan(vwap)) and (mid >= vwap)

            if rsi_exit or vwap_exit:
                self._in_position = False
                reason = "rsi_recovery" if rsi_exit else "vwap_cross"
                # If both triggered simultaneously, note both
                if rsi_exit and vwap_exit:
                    reason = "rsi_recovery+vwap_cross"
                return Signal(
                    signal_type=SignalType.SELL,
                    price=round(mid * (1 + offset), 4),
                    size=size,
                    confidence=1.0,
                    token_id=self.token_id,
                    metadata={"reason": reason, "rsi": round(current_rsi, 2), "vwap": round(vwap, 4)},
                )
            return self._hold()

        # ── Entry: RSI oversold ────────────────────────────────────────────
        if current_rsi < self.params["oversold"]:
            limit_price = round(mid * (1 - offset), 4)
            # Confidence scales with depth below oversold threshold
            confidence = round((self.params["oversold"] - current_rsi) / self.params["oversold"], 4)
            self._in_position = True
            return Signal(
                signal_type=SignalType.BUY,
                price=limit_price,
                size=size,
                confidence=min(confidence, 1.0),
                token_id=self.token_id,
                metadata={
                    "rsi": round(current_rsi, 2),
                    "vwap": round(vwap, 4) if not np.isnan(vwap) else None,
                    "exit_targets": {
                        "rsi_exit": self.params["exit_rsi"],
                        "vwap": round(vwap, 4) if not np.isnan(vwap) else None,
                    },
                },
            )

        return self._hold()
