"""Portfolio construction helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight(selected_scores: pd.Series, max_weight: float = 0.08) -> pd.Series:
    """Create capped equal weights for selected stocks."""

    if selected_scores.empty:
        return pd.Series(dtype=float)
    weights = pd.Series(1.0 / len(selected_scores), index=selected_scores.index)
    return cap_and_renormalize(weights, max_weight=max_weight)


def score_weight(selected_scores: pd.Series, max_weight: float = 0.08) -> pd.Series:
    """Create long-only weights proportional to positive factor scores."""

    positive = selected_scores.clip(lower=0)
    if positive.sum() <= 0:
        return equal_weight(selected_scores, max_weight=max_weight)
    weights = positive / positive.sum()
    return cap_and_renormalize(weights, max_weight=max_weight)


def inverse_volatility_weight(
    selected_scores: pd.Series,
    volatility: pd.Series,
    max_weight: float = 0.08,
) -> pd.Series:
    """Create risk-balanced weights using inverse realized volatility."""

    aligned_vol = volatility.reindex(selected_scores.index).replace(0, np.nan)
    inv_vol = 1 / aligned_vol
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).dropna()
    if inv_vol.empty:
        return equal_weight(selected_scores, max_weight=max_weight)
    return cap_and_renormalize(inv_vol / inv_vol.sum(), max_weight=max_weight)


def apply_industry_cap(
    weights: pd.Series,
    industry: pd.Series | None,
    max_industry_weight: float | None,
    max_weight: float = 0.08,
) -> pd.Series:
    """Limit aggregate industry exposure and redistribute remaining capital."""

    if industry is None or max_industry_weight is None or weights.empty:
        return weights
    aligned_industry = industry.reindex(weights.index).fillna("UNKNOWN")
    capped = weights.copy()
    for _ in range(20):
        industry_weight = capped.groupby(aligned_industry).transform("sum")
        over = industry_weight > max_industry_weight
        if not over.any():
            break
        over_industries = aligned_industry.loc[over].drop_duplicates()
        for industry_name in over_industries:
            members = aligned_industry[aligned_industry == industry_name].index
            current = capped.loc[members].sum()
            if current > 0:
                capped.loc[members] *= max_industry_weight / current
        leftover = 1.0 - capped.sum()
        under = ~aligned_industry.isin(over_industries)
        if leftover <= 0 or not under.any() or capped.loc[under].sum() <= 0:
            break
        capped.loc[under] += leftover * capped.loc[under] / capped.loc[under].sum()
        capped = cap_and_renormalize(capped, max_weight=max_weight)
    total = capped.sum()
    return capped / total if total > 0 else capped


def cap_and_renormalize(weights: pd.Series, max_weight: float = 0.08) -> pd.Series:
    """Apply a single-name cap and redistribute leftover capital."""

    weights = weights.astype(float).clip(lower=0)
    if weights.sum() <= 0:
        return weights
    weights = weights / weights.sum()

    capped = weights.copy()
    for _ in range(20):
        over = capped > max_weight
        if not over.any():
            break
        excess = capped.loc[over].sum() - max_weight * over.sum()
        capped.loc[over] = max_weight
        under = ~over
        if not under.any() or capped.loc[under].sum() <= 0:
            break
        capped.loc[under] += excess * capped.loc[under] / capped.loc[under].sum()

    total = capped.sum()
    return capped / total if total > 0 else capped
