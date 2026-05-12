"""Multiple-testing adjustments for reported Sharpe and information ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Bailey, D. H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier"
# — variance of the Sharpe estimator under non-Gaussian returns (Eq. relating skew/kurtosis).
# Bailey, D., & López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection
# Bias, Backtest Overfitting and Non-Normality" — PSR uses that variance with a benchmark Sharpe.


def variance_of_sharpe_estimator(
    sharpe_estimate: float,
    n_observations: int,
    skewness: float,
    excess_kurtosis: float,
) -> float:
    """Moment-based asymptotic variance of the Sharpe ratio estimator (Bailey & López de Prado, 2012)."""

    if n_observations < 2:
        return np.nan
    e = sharpe_estimate
    return (1.0 - skewness * e + (excess_kurtosis / 4.0) * (e**2)) / max(n_observations - 1, 1)


def probability_sharpe_ratio(
    sharpe_estimate: float,
    benchmark_sharpe: float,
    n_observations: int,
    skewness: float,
    excess_kurtosis: float,
) -> float:
    """PSR: Pr(SR_estimate > benchmark) under the SR variance approximation (Bailey & López de Prado, 2014)."""

    v = variance_of_sharpe_estimator(benchmark_sharpe, n_observations, skewness, excess_kurtosis)
    if not np.isfinite(v) or v <= 0:
        return np.nan
    z = (sharpe_estimate - benchmark_sharpe) / np.sqrt(v)
    return float(stats.norm.cdf(z))


def bonferroni_critical_sharpe(
    n_trials: int,
    n_observations: int,
    skewness: float,
    excess_kurtosis: float,
    alpha: float = 0.05,
    benchmark_sharpe: float = 0.0,
) -> float:
    """
    Bonferroni-adjusted one-sided critical value for the Sharpe statistic.

    Uses SR variance at H0 (benchmark_sharpe) and Sidak/Bonferroni family-wise alpha split: alpha/N.
    """

    if n_trials < 1:
        n_trials = 1
    adj_alpha = alpha / n_trials
    v = variance_of_sharpe_estimator(benchmark_sharpe, n_observations, skewness, excess_kurtosis)
    if not np.isfinite(v) or v <= 0:
        return np.nan
    z = stats.norm.ppf(1.0 - adj_alpha)
    return float(benchmark_sharpe + z * np.sqrt(v))


def bonferroni_deflated_sharpe(
    sharpe: float,
    n_trials: int,
    daily_returns: pd.Series,
    alpha: float = 0.05,
) -> float:
    """Conservative 'haircut': Sharpe minus Bonferroni critical SR at H0 (same moments, N trials)."""

    r = daily_returns.dropna().astype(float)
    n = len(r)
    if n < 3:
        return np.nan
    skew = float(r.skew())
    kurt = float(r.kurt())  # pandas Fisher definition -> excess kurtosis
    crit = bonferroni_critical_sharpe(n_trials, n, skew, kurt, alpha=alpha, benchmark_sharpe=0.0)
    if not np.isfinite(crit):
        return np.nan
    return float(sharpe - crit)


def bonferroni_deflated_information_ratio(
    information_ratio: float,
    n_trials: int,
    excess_daily: pd.Series,
    alpha: float = 0.05,
) -> float:
    """Apply the same Bonferroni SR adjustment to excess returns (treat as IR on daily alphas)."""

    return bonferroni_deflated_sharpe(information_ratio, n_trials, excess_daily, alpha=alpha)


def deflated_summary_from_returns(
    daily_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    *,
    n_trials: int,
    annualization: int = 252,
    alpha: float = 0.05,
) -> pd.Series:
    """Attach PSR vs 0, Bonferroni-deflated Sharpe/IR when benchmark present."""

    r = daily_returns.dropna().astype(float)
    n = len(r)
    ann_vol = float(r.std(ddof=0) * np.sqrt(annualization)) if n > 1 else np.nan
    ann_mean = float(r.mean() * annualization) if n > 0 else np.nan
    sharpe = ann_mean / ann_vol if ann_vol and ann_vol > 0 else np.nan
    skew = float(r.skew()) if n > 2 else 0.0
    exc_kurt = float(r.kurt()) if n > 3 else 0.0
    psr = probability_sharpe_ratio(sharpe, 0.0, n, skew, exc_kurt)
    d_sharpe = bonferroni_deflated_sharpe(sharpe, n_trials, r, alpha=alpha)
    out = {
        "n_observations": n,
        "n_trials_adjustment": n_trials,
        "sharpe": sharpe,
        "probability_sharpe_ratio_vs_0": psr,
        "bonferroni_deflated_sharpe": d_sharpe,
    }
    if benchmark_returns is not None:
        aligned = pd.concat([r.rename("s"), benchmark_returns.reindex(r.index).fillna(0).rename("b")], axis=1).dropna()
        if len(aligned) > 3:
            xs = aligned["s"] - aligned["b"]
            te = float(xs.std(ddof=0) * np.sqrt(annualization))
            ir = float((aligned["s"].mean() - aligned["b"].mean()) * annualization) / te if te > 0 else np.nan
            out["information_ratio"] = ir
            out["bonferroni_deflated_information_ratio"] = bonferroni_deflated_information_ratio(ir, n_trials, xs, alpha=alpha)
    return pd.Series(out)
