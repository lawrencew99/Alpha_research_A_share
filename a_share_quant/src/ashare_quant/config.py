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
    # Price-volume defaults avoid momentum vs reversal conflict; liquidity often behaves as a crowding
    # proxy on large-cap indices—negative prefers lower near-term turnover until IC proves otherwise.
    factor_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "momentum_60": 0.45,
            "reversal_20": 0.0,
            "volatility_20": -0.35,
            "liquidity_20": -0.20,
            "value_pb": 0.0,
            "quality_roe": 0.0,
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
    weighting: str = "equal"
    execution_delay: int = 1
    commission: float = 0.0003
    sell_commission: float | None = None
    stamp_tax: float = 0.001
    slippage: float = 0.0005
    min_commission: float = 0.0
    price_field: str = "adj_close"
    benchmark: str | None = None
    annualization: int = 252
    min_amount: float | None = None
    max_industry_weight: float | None = None
    industry_field: str = "industry"
    volatility_field: str = "volatility_20"

    def buy_cost_rate(self) -> float:
        return self.commission + self.slippage

    def sell_cost_rate(self) -> float:
        commission = self.commission if self.sell_commission is None else self.sell_commission
        return commission + self.stamp_tax + self.slippage

    def trading_cost(self) -> float:
        """Backward-compatible one-way average cost estimate."""

        return 0.5 * (self.buy_cost_rate() + self.sell_cost_rate())
