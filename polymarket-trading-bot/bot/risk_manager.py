"""
Risk Manager
------------
Enforces position limits, exposure caps, and per-trade sizing rules
before any order reaches the Polymarket CLOB.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from strategies.base_strategy import Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    max_position_usd: float = 100.0      # max notional per token
    max_total_exposure_usd: float = 500.0 # max total open exposure
    max_order_size_usd: float = 50.0      # hard cap on a single order
    min_order_size_usd: float = 1.0       # Polymarket minimum (~$1)
    max_open_orders: int = 10             # concurrent open orders
    min_price: float = 0.02              # don't buy at < 2 cents
    max_price: float = 0.98              # don't buy at > 98 cents
    Kelly_fraction: float = 0.25         # fraction of Kelly criterion to apply


@dataclass
class PortfolioState:
    cash: float = 0.0
    positions: dict[str, float] = field(default_factory=dict)   # token_id -> USD notional held
    open_orders: dict[str, int] = field(default_factory=dict)   # token_id -> count

    @property
    def total_exposure(self) -> float:
        return sum(self.positions.values())

    def position_for(self, token_id: str) -> float:
        return self.positions.get(token_id, 0.0)

    def open_order_count(self, token_id: str) -> int:
        return self.open_orders.get(token_id, 0)


class RiskManager:
    """
    Validates and adjusts signals before order submission.

    Usage:
        rm = RiskManager(config=RiskConfig(max_position_usd=50))
        approved, reason = rm.approve(signal, state)
        if approved:
            adjusted_signal = rm.size_order(signal, state)
    """

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()

    def approve(self, signal: Signal, state: PortfolioState) -> tuple[bool, str]:
        """Return (approved, reason_string)."""
        cfg = self.config

        if not signal.is_actionable():
            return False, "Signal is not actionable (HOLD or zero size)"

        if signal.signal_type == SignalType.BUY:
            if signal.price < cfg.min_price:
                return False, f"Price {signal.price} below min {cfg.min_price}"
            if signal.price > cfg.max_price:
                return False, f"Price {signal.price} above max {cfg.max_price}"

            current_pos = state.position_for(signal.token_id)
            if current_pos >= cfg.max_position_usd:
                return False, f"Position {current_pos:.2f} at max {cfg.max_position_usd}"

            if state.total_exposure >= cfg.max_total_exposure_usd:
                return False, f"Total exposure {state.total_exposure:.2f} at cap {cfg.max_total_exposure_usd}"

            total_open = sum(state.open_orders.values())
            if total_open >= cfg.max_open_orders:
                return False, f"Open orders {total_open} at max {cfg.max_open_orders}"

        if signal.signal_type == SignalType.SELL:
            current_pos = state.position_for(signal.token_id)
            if current_pos <= 0:
                return False, "No position to sell"

        return True, "OK"

    def size_order(self, signal: Signal, state: PortfolioState) -> Signal:
        """Return a copy of the signal with adjusted size, capped by risk rules."""
        cfg = self.config
        size = signal.size

        if signal.signal_type == SignalType.BUY:
            # Kelly-fraction sizing: confidence acts as edge proxy
            kelly_size = state.cash * signal.confidence * cfg.Kelly_fraction
            size = min(size, kelly_size) if kelly_size > 0 else size

            # Hard caps
            size = min(size, cfg.max_order_size_usd)
            size = min(size, cfg.max_position_usd - state.position_for(signal.token_id))
            size = min(size, cfg.max_total_exposure_usd - state.total_exposure)
            size = min(size, state.cash)

        if signal.signal_type == SignalType.SELL:
            size = min(size, state.position_for(signal.token_id))

        if size < cfg.min_order_size_usd:
            logger.debug("Sized order below minimum (%.4f), setting to 0", size)
            size = 0.0

        return Signal(
            signal_type=signal.signal_type,
            price=signal.price,
            size=round(size, 4),
            confidence=signal.confidence,
            token_id=signal.token_id,
            metadata=signal.metadata,
        )
