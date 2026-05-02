"""Configuration objects for factor research and backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class FactorConfig:
    """Settings for cross-sectional factor processing."""

    winsor_quantile: float = 0.025
    neutralize_industry: bool = True
    neutralize_size: bool = True
    factor_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "momentum_60": 0.25,
            "reversal_20": 0.15,
            "volatility_20": -0.15,
            "liquidity_20": 0.15,
            "value_pb": 0.15,
            "quality_roe": 0.15,
        }
    )


@dataclass(frozen=True)
class BacktestConfig:
    """Portfolio simulation settings."""

    initial_cash: float = 1_000_000.0
    rebalance: str = "ME"
    top_n: int | None = 30
    top_quantile: float | None = None
    max_weight: float = 0.08
    commission: float = 0.0003
    slippage: float = 0.0005
    price_field: str = "adj_close"
    annualization: int = 252

    def trading_cost(self) -> float:
        return self.commission + self.slippage
