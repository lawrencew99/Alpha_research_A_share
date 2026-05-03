"""Factor diagnostics for A-share multi-factor research."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(market: pd.DataFrame, periods: int = 1, price_field: str = "adj_close") -> pd.Series:
    """Calculate future returns aligned to the signal date."""

    prices = market[price_field].unstack("ticker").sort_index()
    future = prices.pct_change(periods=periods, fill_method=None).shift(-periods)
    name = f"forward_return_{periods}"
    result = (
        future.rename_axis(index="date", columns="ticker")
        .reset_index()
        .melt(id_vars="date", var_name="ticker", value_name=name)
        .set_index(["date", "ticker"])[name]
        .sort_index()
    )
    return result.replace([np.inf, -np.inf], np.nan)


def factor_ic(
    factors: pd.DataFrame,
    future_returns: pd.Series,
    method: str = "spearman",
    min_periods: int = 5,
) -> pd.DataFrame:
    """Compute cross-sectional IC or Rank IC by date for each numeric factor."""

    numeric_factors = _numeric_factor_frame(factors)
    aligned = numeric_factors.join(future_returns.rename("future_return"), how="inner")
    records: list[pd.Series] = []
    for date, cross_section in aligned.groupby(level="date"):
        cross_section = cross_section.droplevel("date").dropna(subset=["future_return"])
        values: dict[str, float] = {}
        for column in numeric_factors.columns:
            sample = cross_section[[column, "future_return"]].dropna()
            values[column] = (
                sample[column].corr(sample["future_return"], method=method) if len(sample) >= min_periods else np.nan
            )
        records.append(pd.Series(values, name=date))
    result = pd.DataFrame(records).sort_index()
    result.index.name = "date"
    return result


def summarize_ic(ic: pd.DataFrame, annualization: int = 252) -> pd.DataFrame:
    """Summarize IC level, stability and hit rate."""

    summary = pd.DataFrame(index=ic.columns)
    summary["mean"] = ic.mean()
    summary["std"] = ic.std(ddof=0)
    summary["icir"] = summary["mean"] / summary["std"].replace(0, np.nan) * np.sqrt(annualization)
    summary["positive_rate"] = (ic > 0).mean()
    summary["t_stat"] = summary["mean"] / (summary["std"].replace(0, np.nan) / np.sqrt(ic.count()))
    return summary


def quantile_returns(
    factor: pd.Series,
    future_returns: pd.Series,
    quantiles: int = 5,
    min_periods: int = 20,
) -> pd.DataFrame:
    """Calculate equal-weight future returns for each factor quantile by date."""

    aligned = pd.concat([factor.rename("factor"), future_returns.rename("future_return")], axis=1).dropna()
    rows: list[pd.Series] = []
    for date, cross_section in aligned.groupby(level="date"):
        cross_section = cross_section.droplevel("date")
        if len(cross_section) < max(min_periods, quantiles):
            continue
        ranks = cross_section["factor"].rank(method="first")
        buckets = pd.qcut(ranks, quantiles, labels=[f"Q{i}" for i in range(1, quantiles + 1)])
        row = cross_section.groupby(buckets, observed=False)["future_return"].mean()
        row["long_short"] = row.iloc[-1] - row.iloc[0]
        row.name = date
        rows.append(row)
    result = pd.DataFrame(rows).sort_index()
    result.index.name = "date"
    return result


def factor_correlation(factors: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    """Return pooled numeric factor correlations."""

    return _numeric_factor_frame(factors).corr(method=method)


def exposure_summary(
    weights: pd.DataFrame,
    market: pd.DataFrame,
    industry_field: str = "industry",
    size_field: str = "market_cap",
) -> dict[str, pd.DataFrame | pd.Series]:
    """Summarize concentration and available industry/size exposures."""

    result: dict[str, pd.DataFrame | pd.Series] = {}
    if weights.empty:
        return result

    result["concentration"] = pd.DataFrame(
        {
            "holding_count": weights.gt(0).sum(axis=1),
            "top1_weight": weights.max(axis=1),
            "top5_weight": weights.apply(lambda row: row.nlargest(5).sum(), axis=1),
            "herfindahl": weights.pow(2).sum(axis=1),
        }
    )

    if industry_field in market.columns:
        industry_panel = market[industry_field].unstack("ticker").reindex(index=weights.index, columns=weights.columns)
        rows = []
        for date, row in weights.iterrows():
            industry = industry_panel.loc[date].fillna("UNKNOWN")
            exposure = row.groupby(industry).sum()
            exposure.name = date
            rows.append(exposure)
        result["industry_exposure"] = pd.DataFrame(rows).fillna(0).sort_index()

    if size_field in market.columns:
        size_panel = market[size_field].unstack("ticker").reindex(index=weights.index, columns=weights.columns)
        weighted_size = (weights * np.log(size_panel.replace(0, np.nan))).sum(axis=1, min_count=1)
        weighted_size.name = "weighted_log_size"
        result["size_exposure"] = weighted_size

    return result


def _numeric_factor_frame(factors: pd.DataFrame) -> pd.DataFrame:
    numeric = factors.select_dtypes(include=[np.number]).copy()
    return numeric.drop(columns=[column for column in ["market_cap"] if column in numeric.columns], errors="ignore")
