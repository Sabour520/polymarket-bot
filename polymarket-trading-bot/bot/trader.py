"""
Trader
------
Manages the lifecycle of limit orders on Polymarket via py-clob-client.
Responsibilities:
  - Submit new limit orders (BUY / SELL)
  - Track open orders and cancel stale ones
  - Update PortfolioState after fills
  - Poll for fills and report P&L
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

from strategies.base_strategy import Signal, SignalType
from bot.risk_manager import PortfolioState, RiskConfig, RiskManager

logger = logging.getLogger(__name__)


class Trader:
    """
    High-level order manager for Polymarket limit orders.

    Usage:
        trader = Trader.from_env()
        state  = PortfolioState(cash=500.0)
        trader.execute(signal, state)
    """

    def __init__(
        self,
        client: ClobClient,
        risk_config: RiskConfig | None = None,
        order_ttl_seconds: int = 300,
    ):
        self.client = client
        self.risk = RiskManager(risk_config)
        self.order_ttl = order_ttl_seconds
        # order_id -> (signal, submitted_at)
        self._open_orders: dict[str, tuple[Signal, float]] = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, risk_config: RiskConfig | None = None) -> "Trader":
        """
        Build a Trader from environment variables (loaded from .env).

        Required env vars:
            POLYMARKET_PRIVATE_KEY   — EOA private key (0x...)
            POLYMARKET_FUNDER_ADDRESS — wallet address (0x...)
        Optional:
            POLYMARKET_HOST          — default https://clob.polymarket.com
            POLYMARKET_CHAIN_ID      — default 137 (Polygon mainnet)
        """
        host = os.environ.get("POLYMARKET_HOST", "https://clob.polymarket.com")
        key = os.environ["POLYMARKET_PRIVATE_KEY"]
        chain_id = int(os.environ.get("POLYMARKET_CHAIN_ID", "137"))
        funder = os.environ["POLYMARKET_FUNDER_ADDRESS"]

        client = ClobClient(
            host=host,
            key=key,
            chain_id=chain_id,
            funder=funder,
            signature_type=2,  # EIP-1271
        )
        # Derive API credentials from the wallet — no manual key management needed
        client.set_api_creds(client.create_or_derive_api_creds())
        return cls(client, risk_config)

    # ------------------------------------------------------------------
    # Core order management
    # ------------------------------------------------------------------

    def execute(self, signal: Signal, state: PortfolioState) -> Optional[str]:
        """
        Validate, size, and submit a limit order.
        Returns the order_id on success, None if rejected or skipped.
        """
        approved, reason = self.risk.approve(signal, state)
        if not approved:
            logger.info("Signal rejected by risk manager: %s", reason)
            return None

        sized = self.risk.size_order(signal, state)
        if not sized.is_actionable():
            logger.info("Signal sized to zero — skipping.")
            return None

        side = BUY if sized.signal_type == SignalType.BUY else SELL

        # Cancel any existing open orders for this token before placing a new one
        self.cancel_all(token_id=sized.token_id)

        order_args = OrderArgs(
            price=sized.price,
            size=sized.size,
            side=side,
            token_id=sized.token_id,
        )

        try:
            signed_order = self.client.create_order(order_args)
            resp = self.client.post_order(signed_order, OrderType.GTC)
            order_id = resp.get("orderID") or resp.get("id", "")
            logger.info(
                "Submitted %s limit order id=%s  price=%.4f  size=%.4f",
                side,
                order_id,
                sized.price,
                sized.size,
            )
            self._open_orders[order_id] = (sized, time.time())
            self._update_state_on_submit(sized, state)
            return order_id
        except Exception as exc:
            logger.error("Order submission failed: %s", exc)
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific open order."""
        try:
            self.client.cancel(order_id=order_id)
            self._open_orders.pop(order_id, None)
            logger.info("Cancelled order %s", order_id)
            return True
        except Exception as exc:
            logger.error("Cancel failed for %s: %s", order_id, exc)
            return False

    def cancel_all(self, token_id: str | None = None) -> int:
        """Cancel all open orders, optionally filtered by token_id."""
        ids = [
            oid
            for oid, (sig, _) in list(self._open_orders.items())
            if token_id is None or sig.token_id == token_id
        ]
        cancelled = sum(1 for oid in ids if self.cancel_order(oid))
        return cancelled

    def cancel_stale_orders(self, state: PortfolioState) -> int:
        """Cancel orders that have been open longer than order_ttl_seconds."""
        now = time.time()
        stale = [
            oid
            for oid, (sig, ts) in list(self._open_orders.items())
            if now - ts > self.order_ttl
        ]
        cancelled = 0
        for oid in stale:
            if self.cancel_order(oid):
                sig, _ = self._open_orders.get(oid, (None, None))
                if sig:
                    self._revert_state_on_cancel(sig, state)
                cancelled += 1
        return cancelled

    def sync_fills(self, state: PortfolioState) -> list[dict]:
        """
        Poll the CLOB for fill events and update portfolio state accordingly.
        Returns a list of fill dicts.
        """
        fills = []
        for order_id in list(self._open_orders):
            try:
                order = self.client.get_order(order_id)
                status = order.get("status", "")
                if status in ("MATCHED", "FILLED"):
                    sig, _ = self._open_orders.pop(order_id)
                    fill_price = float(order.get("avgPrice", sig.price))
                    fill_size = float(order.get("sizeMatched", sig.size))
                    self._update_state_on_fill(sig, fill_price, fill_size, state)
                    fills.append({"order_id": order_id, "fill_price": fill_price, "fill_size": fill_size})
                    logger.info("Filled order %s @ %.4f  size=%.4f", order_id, fill_price, fill_size)
            except Exception as exc:
                logger.warning("Could not fetch order %s: %s", order_id, exc)
        return fills

    # ------------------------------------------------------------------
    # Internal state helpers
    # ------------------------------------------------------------------

    def _update_state_on_submit(self, signal: Signal, state: PortfolioState) -> None:
        if signal.signal_type == SignalType.BUY:
            state.cash -= signal.size
            state.open_orders[signal.token_id] = state.open_orders.get(signal.token_id, 0) + 1

    def _revert_state_on_cancel(self, signal: Signal, state: PortfolioState) -> None:
        if signal.signal_type == SignalType.BUY:
            state.cash += signal.size
            count = state.open_orders.get(signal.token_id, 0)
            state.open_orders[signal.token_id] = max(0, count - 1)

    def _update_state_on_fill(
        self, signal: Signal, fill_price: float, fill_size: float, state: PortfolioState
    ) -> None:
        token = signal.token_id
        if signal.signal_type == SignalType.BUY:
            state.positions[token] = state.positions.get(token, 0.0) + fill_size
            count = state.open_orders.get(token, 0)
            state.open_orders[token] = max(0, count - 1)
        else:
            current = state.positions.get(token, 0.0)
            state.positions[token] = max(0.0, current - fill_size)
            state.cash += fill_price * fill_size
