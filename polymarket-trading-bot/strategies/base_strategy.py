from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    signal_type: SignalType
    price: float          # Limit price to place the order at
    size: float           # Dollar size of the order
    confidence: float     # 0.0 – 1.0
    token_id: str         # Polymarket outcome token ID
    metadata: dict = field(default_factory=dict)

    def is_actionable(self) -> bool:
        return self.signal_type != SignalType.HOLD and self.size > 0


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Every strategy receives a DataFrame of OHLCV+book data and returns a Signal.
    Columns expected in `data`:
        timestamp, open, high, low, close, volume,
        best_bid, best_ask  (optional but used by some strategies)
    """

    def __init__(self, token_id: str, params: Optional[dict] = None):
        self.token_id = token_id
        self.params = params or {}

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Analyse `data` and return a Signal."""

    def _hold(self) -> Signal:
        return Signal(
            signal_type=SignalType.HOLD,
            price=0.0,
            size=0.0,
            confidence=0.0,
            token_id=self.token_id,
        )

    def _required_rows(self) -> int:
        """Minimum number of rows needed to produce a valid signal."""
        return 1
