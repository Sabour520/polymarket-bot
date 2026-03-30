"""
CVD Divergence Strategy (Cumulative Volume Delta)
--------------------------------------------------
Detects price-volume divergence to identify hidden order flow and
likely reversal points in Polymarket binary markets.

CVD per candle:
  close >= open  →  +volume  (aggressive buyers)
  close <  open  →  -volume  (aggressive sellers)

Divergence signals:
  Bullish  — price falls but CVD rises over the lookback window
             → hidden buying pressure beneath the surface → BUY
  Bearish  — price rises but CVD falls over the lookback window
             → hidden selling pressure disguised as a rally → SELL

Both price change and CVD change must exceed minimum thresholds
to avoid triggering on noise.

Requires 'open', 'high', 'low', 'close', 'volume' columns.
"""

import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy, Signal, SignalType


class CVDStrategy(BaseStrategy):
    """
    Params (all optional, defaults shown):
        lookback          : int   = 10   candles over which divergence is measured
        price_min_move    : float = 0.005  minimum price change to qualify (0.5%)
        cvd_min_move      : float = 0.005  minimum CVD change to qualify (0.5% of CVD range)
        order_size        : float = 10.0
        spread_offset     : float = 0.002
    """

    DEFAULT_PARAMS = {
        "lookback": 10,
        "price_min_move": 0.005,
        "cvd_min_move": 0.005,
        "order_size": 10.0,
        "spread_offset": 0.002,
    }

    def __init__(self, token_id: str, params: dict | None = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(token_id, merged)

    def _required_rows(self) -> int:
        return self.params["lookback"] + 1

    @staticmethod
    def _compute_cvd(data: pd.DataFrame) -> pd.Series:
        """Cumulative Volume Delta: +vol for bullish candles, -vol for bearish."""
        delta = data["volume"].copy().astype(float)
        bearish = data["close"] < data["open"]
        delta[bearish] *= -1
        return delta.cumsum()

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        if len(data) < self._required_rows():
            return self._hold()

        required_cols = {"open", "close", "volume"}
        missing = required_cols - set(data.columns)
        if missing:
            raise ValueError(f"CVDStrategy requires columns: {missing}")

        lookback = self.params["lookback"]
        window = data.iloc[-lookback - 1:]

        cvd = self._compute_cvd(window)
        close = window["close"]

        price_start = float(close.iloc[0])
        price_end = float(close.iloc[-1])
        cvd_start = float(cvd.iloc[0])
        cvd_end = float(cvd.iloc[-1])

        # Normalised changes
        price_change = (price_end - price_start) / (price_start + 1e-9)
        cvd_range = float(cvd.abs().max()) + 1e-9
        cvd_change = (cvd_end - cvd_start) / cvd_range

        min_price = self.params["price_min_move"]
        min_cvd = self.params["cvd_min_move"]

        mid = price_end
        offset = self.params["spread_offset"]
        size = self.params["order_size"]

        # ── Bullish divergence: price ↓ but CVD ↑ ─────────────────────────
        if price_change < -min_price and cvd_change > min_cvd:
            divergence_strength = min(abs(price_change) + abs(cvd_change), 1.0)
            return Signal(
                signal_type=SignalType.BUY,
                price=round(mid * (1 - offset), 4),
                size=size,
                confidence=divergence_strength,
                token_id=self.token_id,
                metadata={
                    "divergence": "bullish",
                    "price_change_pct": round(price_change * 100, 3),
                    "cvd_change_pct": round(cvd_change * 100, 3),
                    "cvd_end": round(cvd_end, 2),
                },
            )

        # ── Bearish divergence: price ↑ but CVD ↓ ─────────────────────────
        if price_change > min_price and cvd_change < -min_cvd:
            divergence_strength = min(abs(price_change) + abs(cvd_change), 1.0)
            return Signal(
                signal_type=SignalType.SELL,
                price=round(mid * (1 + offset), 4),
                size=size,
                confidence=divergence_strength,
                token_id=self.token_id,
                metadata={
                    "divergence": "bearish",
                    "price_change_pct": round(price_change * 100, 3),
                    "cvd_change_pct": round(cvd_change * 100, 3),
                    "cvd_end": round(cvd_end, 2),
                },
            )

        return self._hold()
