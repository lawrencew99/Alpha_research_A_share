"""Long-only multi-factor portfolio backtesting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ashare_quant.config import BacktestConfig
from ashare_quant.portfolio import equal_weight


@dataclass(frozen=True)
class BacktestResult:
    """Backtest outputs used by reports and notebooks."""

    equity_curve: pd.Series
    daily_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    metrics: pd.Series
    monthly_returns: pd.Series


def run_backtest(
    market: pd.DataFrame,
    scores: pd.Series,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a close-to-close A-share long-only factor backtest."""

    config = config or BacktestConfig()
    prices = market[config.price_field].unstack("ticker").sort_index()
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0)
    score_panel = scores.unstack("ticker").reindex(prices.index)
    rebalance_dates = _get_rebalance_dates(prices.index, config.rebalance)

    current_weights = pd.Series(0.0, index=prices.columns)
    equity = config.initial_cash
    equity_records: list[tuple[pd.Timestamp, float]] = []
    return_records: list[tuple[pd.Timestamp, float]] = []
    turnover_records: list[tuple[pd.Timestamp, float]] = []
    weight_records: list[pd.Series] = []

    for date in prices.index:
        portfolio_return = float((current_weights * returns.loc[date]).sum())
        equity *= 1 + portfolio_return

        if date in rebalance_dates:
            target = _select_and_weight(score_panel.loc[date], config)
            target = target.reindex(prices.columns).fillna(0)
            one_way_turnover = 0.5 * (target - current_weights).abs().sum()
            equity *= 1 - one_way_turnover * config.trading_cost()
            current_weights = target
            turnover_records.append((date, float(one_way_turnover)))
            weight_records.append(target.rename(date))

        equity_records.append((date, equity))
        return_records.append((date, portfolio_return))

    equity_curve = pd.Series(dict(equity_records), name="equity").sort_index()
    daily_returns = equity_curve.pct_change().fillna(0).rename("daily_return")
    weights = pd.DataFrame(weight_records).sort_index() if weight_records else pd.DataFrame()
    turnover = pd.Series(dict(turnover_records), name="turnover").sort_index()
    monthly_returns = equity_curve.resample("ME").last().pct_change().dropna()
    monthly_returns.name = "monthly_return"
    metrics = calculate_metrics(equity_curve, daily_returns, turnover, config.annualization)

    return BacktestResult(
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        weights=weights,
        turnover=turnover,
        metrics=metrics,
        monthly_returns=monthly_returns,
    )


def calculate_metrics(
    equity_curve: pd.Series,
    daily_returns: pd.Series,
    turnover: pd.Series,
    annualization: int = 252,
) -> pd.Series:
    """Calculate common performance and risk metrics."""

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    years = max(len(daily_returns) / annualization, 1 / annualization)
    annual_return = (1 + total_return) ** (1 / years) - 1
    annual_vol = daily_returns.std(ddof=0) * np.sqrt(annualization)
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan
    drawdown = equity_curve / equity_curve.cummax() - 1
    max_drawdown = drawdown.min()
    win_rate = (daily_returns > 0).mean()

    return pd.Series(
        {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "avg_turnover": turnover.mean() if not turnover.empty else 0.0,
        },
        name="metrics",
    )


def _get_rebalance_dates(index: pd.DatetimeIndex, rule: str) -> set[pd.Timestamp]:
    rule = _normalize_resample_rule(rule)
    date_series = index.to_series(index=index)
    return set(date_series.resample(rule).last().dropna().tolist())


def _normalize_resample_rule(rule: str) -> str:
    aliases = {"M": "ME", "Q": "QE", "Y": "YE", "A": "YE"}
    return aliases.get(rule.upper(), rule)


def _select_and_weight(scores: pd.Series, config: BacktestConfig) -> pd.Series:
    valid_scores = scores.dropna().sort_values(ascending=False)
    if valid_scores.empty:
        return pd.Series(dtype=float)

    if config.top_quantile is not None:
        n_selected = max(1, int(len(valid_scores) * config.top_quantile))
    elif config.top_n is not None:
        n_selected = min(config.top_n, len(valid_scores))
    else:
        n_selected = max(1, int(len(valid_scores) * 0.2))

    selected = valid_scores.head(n_selected)
    return equal_weight(selected, max_weight=config.max_weight)
