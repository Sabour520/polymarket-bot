"""
DataLoader
----------
Loads historical price/volume data for backtesting.
Supports loading from CSV files or generating synthetic data for strategy testing.

Expected CSV columns: timestamp, open, high, low, close, volume
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess OHLCV data for backtesting."""

    @staticmethod
    def from_csv(filepath: str | Path, datetime_col: str = "timestamp") -> pd.DataFrame:
        """
        Load OHLCV data from a CSV file.

        The CSV must contain at minimum: timestamp, close
        Recommended: timestamp, open, high, low, close, volume
        """
        df = pd.read_csv(filepath)
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.sort_values(datetime_col).reset_index(drop=True)
        df = DataLoader._validate_and_fill(df)
        logger.info("Loaded %d rows from %s", len(df), filepath)
        return df

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalise an existing DataFrame."""
        df = df.copy().reset_index(drop=True)
        return DataLoader._validate_and_fill(df)

    @staticmethod
    def synthetic(
        n: int = 500,
        start_price: float = 0.50,
        volatility: float = 0.02,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate a synthetic OHLCV DataFrame for strategy development and testing.
        Prices are constrained to [0.01, 0.99] to match Polymarket probability bounds.
        """
        rng = np.random.default_rng(seed)
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1h")

        returns = rng.normal(0, volatility, n)
        close = np.cumprod(1 + returns) * start_price
        close = np.clip(close, 0.01, 0.99)

        noise = rng.uniform(0.001, 0.005, n)
        high = np.minimum(close + noise, 0.99)
        low = np.maximum(close - noise, 0.01)
        open_ = np.roll(close, 1)
        open_[0] = start_price
        volume = rng.uniform(100, 5000, n)

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": np.round(open_, 4),
                "high": np.round(high, 4),
                "low": np.round(low, 4),
                "close": np.round(close, 4),
                "volume": np.round(volume, 2),
            }
        )

    @staticmethod
    def _validate_and_fill(df: pd.DataFrame) -> pd.DataFrame:
        required = {"close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        if "open" not in df.columns:
            df["open"] = df["close"].shift(1).fillna(df["close"])
        if "high" not in df.columns:
            df["high"] = df[["open", "close"]].max(axis=1)
        if "low" not in df.columns:
            df["low"] = df[["open", "close"]].min(axis=1)
        if "volume" not in df.columns:
            df["volume"] = 1.0

        return df
