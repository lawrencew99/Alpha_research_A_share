from __future__ import annotations

import pandas as pd


def rebalance_dates(dates: pd.Series, frequency: str = "M") -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(dates).unique()).sort_values()
    return index.to_series().groupby(index.to_period(frequency)).last().pipe(pd.DatetimeIndex)


def select_top_portfolio(
    scores: pd.DataFrame,
    tradeable: pd.DataFrame,
    date: pd.Timestamp,
    top_n: int,
    max_weight: float,
) -> pd.Series:
    day_scores = scores.loc[scores["date"].eq(date), ["symbol", "score"]]
    day_tradeable = tradeable.loc[tradeable["date"].eq(date), ["symbol", "is_tradeable"]]
    candidates = day_scores.merge(day_tradeable, on="symbol", how="left")
    candidates = candidates.loc[candidates["is_tradeable"].fillna(False)]
    candidates = candidates.dropna(subset=["score"]).sort_values("score", ascending=False).head(top_n)
    if candidates.empty:
        return pd.Series(dtype=float)
    raw_weight = min(1 / len(candidates), max_weight)
    weights = pd.Series(raw_weight, index=candidates["symbol"])
    residual = 1 - weights.sum()
    if residual > 0 and len(weights) > 0:
        weights += residual / len(weights)
        weights = weights.clip(upper=max_weight)
        weights /= weights.sum()
    return weights


def portfolio_turnover(previous: pd.Series, current: pd.Series) -> float:
    symbols = previous.index.union(current.index)
    prev = previous.reindex(symbols).fillna(0)
    curr = current.reindex(symbols).fillna(0)
    return float((curr - prev).abs().sum() / 2)
