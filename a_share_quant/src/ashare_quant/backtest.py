"""Long-only multi-factor portfolio backtesting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ashare_quant.config import BacktestConfig
from ashare_quant.portfolio import apply_industry_cap, equal_weight, inverse_volatility_weight, score_weight


@dataclass(frozen=True)
class BacktestResult:
    """Backtest outputs used by reports and notebooks."""

    equity_curve: pd.Series
    daily_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    metrics: pd.Series
    monthly_returns: pd.Series
    benchmark_curve: pd.Series | None = None
    benchmark_returns: pd.Series | None = None
    excess_returns: pd.Series | None = None
    trade_log: pd.DataFrame | None = None


def run_backtest(
    market: pd.DataFrame,
    scores: pd.Series,
    config: BacktestConfig | None = None,
    benchmark_returns: pd.Series | None = None,
    context: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run a long-only factor backtest with delayed execution and realistic costs."""

    config = config or BacktestConfig()
    prices = market[config.price_field].unstack("ticker").sort_index()
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0)
    score_panel = scores.unstack("ticker").reindex(prices.index)
    rebalance_dates = _get_rebalance_dates(prices.index, config.rebalance)
    context = context if context is not None else market
    volatility_panel = _context_panel(context, config.volatility_field, prices.index, prices.columns)
    if volatility_panel is None:
        volatility_panel = returns.rolling(20).std()
    industry_panel = _context_panel(context, config.industry_field, prices.index, prices.columns)
    amount_panel = _context_panel(market, "amount", prices.index, prices.columns)
    benchmark_returns = _resolve_benchmark_returns(returns, config, benchmark_returns)

    current_weights = pd.Series(0.0, index=prices.columns)
    equity = config.initial_cash
    equity_records: list[tuple[pd.Timestamp, float]] = []
    turnover_records: list[tuple[pd.Timestamp, float]] = []
    weight_records: list[pd.Series] = []
    trade_records: list[dict[str, float | pd.Timestamp]] = []
    pending_trades: dict[pd.Timestamp, pd.Series] = {}

    for position, date in enumerate(prices.index):
        portfolio_return = float((current_weights * returns.loc[date]).sum())
        equity *= 1 + portfolio_return

        if date in rebalance_dates:
            tradable = _tradable_mask(amount_panel.loc[date] if amount_panel is not None else None, config)
            target = _select_and_weight(
                score_panel.loc[date],
                config,
                volatility=volatility_panel.loc[date],
                industry=industry_panel.loc[date] if industry_panel is not None else None,
                tradable=tradable,
            )
            target = target.reindex(prices.columns).fillna(0)
            execution_position = min(position + max(config.execution_delay, 0), len(prices.index) - 1)
            pending_trades[prices.index[execution_position]] = target

        if date in pending_trades:
            target = pending_trades.pop(date)
            delta = target - current_weights
            buy_turnover = float(delta.clip(lower=0).sum())
            sell_turnover = float((-delta.clip(upper=0)).sum())
            one_way_turnover = 0.5 * float(delta.abs().sum())
            cost = _calculate_trade_cost(delta, equity, config)
            equity *= 1 - cost
            current_weights = target
            turnover_records.append((date, float(one_way_turnover)))
            weight_records.append(target.rename(date))
            trade_records.append(
                {
                    "date": date,
                    "buy_turnover": buy_turnover,
                    "sell_turnover": sell_turnover,
                    "one_way_turnover": one_way_turnover,
                    "cost": cost,
                }
            )

        equity_records.append((date, equity))

    equity_curve = pd.Series(dict(equity_records), name="equity").sort_index()
    daily_returns = equity_curve.pct_change().fillna(0).rename("daily_return")
    weights = pd.DataFrame(weight_records).sort_index() if weight_records else pd.DataFrame()
    turnover = pd.Series(dict(turnover_records), name="turnover").sort_index()
    monthly_returns = equity_curve.resample("ME").last().pct_change().dropna()
    monthly_returns.name = "monthly_return"
    benchmark_curve = None
    excess_returns = None
    if benchmark_returns is not None:
        benchmark_returns = benchmark_returns.reindex(daily_returns.index).fillna(0).rename("benchmark_return")
        benchmark_curve = (1 + benchmark_returns).cumprod() * config.initial_cash
        benchmark_curve.name = "benchmark"
        excess_returns = (daily_returns - benchmark_returns).rename("excess_return")
    metrics = calculate_metrics(equity_curve, daily_returns, turnover, config.annualization, benchmark_returns)
    trade_log = pd.DataFrame(trade_records).set_index("date") if trade_records else pd.DataFrame()

    return BacktestResult(
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        weights=weights,
        turnover=turnover,
        metrics=metrics,
        monthly_returns=monthly_returns,
        benchmark_curve=benchmark_curve,
        benchmark_returns=benchmark_returns,
        excess_returns=excess_returns,
        trade_log=trade_log,
    )


def calculate_metrics(
    equity_curve: pd.Series,
    daily_returns: pd.Series,
    turnover: pd.Series,
    annualization: int = 252,
    benchmark_returns: pd.Series | None = None,
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

    metrics = {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "avg_turnover": turnover.mean() if not turnover.empty else 0.0,
    }
    if benchmark_returns is not None:
        aligned = pd.concat([daily_returns.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1).dropna()
        if not aligned.empty:
            excess = aligned["strategy"] - aligned["benchmark"]
            benchmark_total = (1 + aligned["benchmark"]).prod() - 1
            benchmark_years = max(len(aligned) / annualization, 1 / annualization)
            benchmark_annual = (1 + benchmark_total) ** (1 / benchmark_years) - 1
            tracking_error = excess.std(ddof=0) * np.sqrt(annualization)
            annual_excess = annual_return - benchmark_annual
            benchmark_var = aligned["benchmark"].var(ddof=0)
            beta = aligned["strategy"].cov(aligned["benchmark"]) / benchmark_var if benchmark_var > 0 else np.nan
            alpha = (aligned["strategy"].mean() - beta * aligned["benchmark"].mean()) * annualization
            metrics.update(
                {
                    "benchmark_total_return": benchmark_total,
                    "benchmark_annual_return": benchmark_annual,
                    "excess_annual_return": annual_excess,
                    "tracking_error": tracking_error,
                    "information_ratio": annual_excess / tracking_error if tracking_error > 0 else np.nan,
                    "beta": beta,
                    "alpha": alpha,
                }
            )

    return pd.Series(
        metrics,
        name="metrics",
    )


def _get_rebalance_dates(index: pd.DatetimeIndex, rule: str) -> set[pd.Timestamp]:
    date_series = index.to_series(index=index)
    return set(date_series.resample(_normalize_resample_rule(rule)).last().dropna().tolist())


def _normalize_resample_rule(rule: str) -> str:
    aliases = {"M": "ME", "Q": "QE", "Y": "YE", "A": "YE"}
    return aliases.get(rule.upper(), rule)


def _select_and_weight(
    scores: pd.Series,
    config: BacktestConfig,
    volatility: pd.Series,
    industry: pd.Series | None,
    tradable: pd.Series | None,
) -> pd.Series:
    valid_scores = scores.dropna().sort_values(ascending=False)
    if tradable is not None:
        valid_scores = valid_scores.loc[valid_scores.index.intersection(tradable[tradable].index)]
    if valid_scores.empty:
        return pd.Series(dtype=float)

    if config.top_quantile is not None:
        n_selected = max(1, int(len(valid_scores) * config.top_quantile))
    elif config.top_n is not None:
        n_selected = min(config.top_n, len(valid_scores))
    else:
        n_selected = max(1, int(len(valid_scores) * 0.2))

    selected = valid_scores.head(n_selected)
    if config.weighting == "equal":
        weights = equal_weight(selected, max_weight=config.max_weight)
    elif config.weighting == "score":
        weights = score_weight(selected, max_weight=config.max_weight)
    elif config.weighting == "inverse_volatility":
        weights = inverse_volatility_weight(selected, volatility, max_weight=config.max_weight)
    else:
        raise ValueError(f"unknown weighting method: {config.weighting!r}")
    return apply_industry_cap(weights, industry, config.max_industry_weight, max_weight=config.max_weight)


def _calculate_trade_cost(delta: pd.Series, equity: float, config: BacktestConfig) -> float:
    if equity <= 0:
        return 0.0

    buy = delta[delta > 0]
    sell = -delta[delta < 0]
    buy_cost = _side_cost(buy, equity, config.commission, config.slippage, config.min_commission)
    sell_commission = config.commission if config.sell_commission is None else config.sell_commission
    sell_cost = _side_cost(sell, equity, sell_commission, config.slippage + config.stamp_tax, config.min_commission)
    return buy_cost + sell_cost


def _side_cost(turnover: pd.Series, equity: float, commission: float, extra_rate: float, min_commission: float) -> float:
    if turnover.empty:
        return 0.0
    notional = turnover * equity
    commission_cost = (notional * commission).clip(lower=min_commission if min_commission > 0 else 0)
    extra_cost = notional * extra_rate
    return float((commission_cost + extra_cost).sum() / equity)


def _context_panel(
    frame: pd.DataFrame,
    column: str,
    index: pd.DatetimeIndex,
    columns: pd.Index,
) -> pd.DataFrame | None:
    if column not in frame.columns:
        return None
    return frame[column].unstack("ticker").reindex(index=index, columns=columns)


def _tradable_mask(amount: pd.Series | None, config: BacktestConfig) -> pd.Series | None:
    if amount is None or config.min_amount is None:
        return None
    return amount.fillna(0) >= config.min_amount


def _resolve_benchmark_returns(
    returns: pd.DataFrame,
    config: BacktestConfig,
    benchmark_returns: pd.Series | None,
) -> pd.Series | None:
    if benchmark_returns is not None:
        return benchmark_returns
    if config.benchmark is None or config.benchmark.lower() in {"none", ""}:
        return None
    benchmark = config.benchmark.lower()
    if benchmark in {"equal_weight", "universe_equal"}:
        return returns.mean(axis=1).rename("benchmark_return")
    if config.benchmark in returns.columns:
        return returns[config.benchmark].rename("benchmark_return")
    raise ValueError(f"benchmark {config.benchmark!r} was not found in market returns")
