from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return float(drawdown.min())


def annual_return(daily_returns: pd.Series, periods: int = 252) -> float:
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        return 0.0
    return float((1 + daily_returns).prod() ** (periods / len(daily_returns)) - 1)


def annual_volatility(daily_returns: pd.Series, periods: int = 252) -> float:
    return float(daily_returns.dropna().std() * np.sqrt(periods))


def sharpe_ratio(daily_returns: pd.Series, periods: int = 252) -> float:
    vol = annual_volatility(daily_returns, periods)
    if vol == 0:
        return 0.0
    return annual_return(daily_returns, periods) / vol


def monthly_returns(equity: pd.DataFrame | pd.Series) -> pd.Series:
    series = equity["equity"] if isinstance(equity, pd.DataFrame) else equity
    month_end = series.resample("ME").last()
    return month_end.pct_change().dropna()


def performance_summary(equity: pd.DataFrame) -> dict[str, float]:
    returns = equity["portfolio_return"].fillna(0)
    ann_ret = annual_return(returns)
    mdd = max_drawdown(equity["equity"])
    return {
        "annual_return": ann_ret,
        "annual_volatility": annual_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": mdd,
        "calmar": ann_ret / abs(mdd) if mdd else 0.0,
        "win_rate": float((returns > 0).mean()),
        "total_return": float(equity["equity"].iloc[-1] - 1),
    }
