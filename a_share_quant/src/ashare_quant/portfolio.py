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
