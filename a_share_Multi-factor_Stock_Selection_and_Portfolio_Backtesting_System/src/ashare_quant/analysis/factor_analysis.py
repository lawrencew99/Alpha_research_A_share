from __future__ import annotations

import pandas as pd

from ashare_quant.data.cleaning import forward_returns
from ashare_quant.factors.preprocess import factor_columns


def information_coefficient(frame: pd.DataFrame, factor: str) -> pd.DataFrame:
    rows = []
    for date, group in frame.groupby("date"):
        valid = group[[factor, "forward_return"]].dropna()
        if len(valid) < 3:
            continue
        rows.append(
            {
                "date": date,
                "factor": factor,
                "ic": valid[factor].corr(valid["forward_return"], method="pearson"),
                "rank_ic": valid[factor].corr(valid["forward_return"], method="spearman"),
            }
        )
    return pd.DataFrame(rows)


def quantile_returns(frame: pd.DataFrame, factor: str, quantiles: int) -> pd.DataFrame:
    rows = []
    for date, group in frame.groupby("date"):
        valid = group[[factor, "forward_return"]].dropna()
        if valid[factor].nunique() < quantiles:
            continue
        buckets = pd.qcut(valid[factor], quantiles, labels=False, duplicates="drop") + 1
        values = valid.assign(quantile=buckets).groupby("quantile")["forward_return"].mean()
        for quantile, value in values.items():
            rows.append({"date": date, "factor": factor, "quantile": int(quantile), "return": value})
    return pd.DataFrame(rows)


def factor_turnover(factors: pd.DataFrame, factor: str, top_quantile: float = 0.2) -> pd.DataFrame:
    rows = []
    previous: set[str] | None = None
    for date, group in factors.groupby("date"):
        cutoff = group[factor].quantile(1 - top_quantile)
        current = set(group.loc[group[factor] >= cutoff, "symbol"])
        if previous is not None and current:
            rows.append({"date": date, "factor": factor, "turnover": 1 - len(current & previous) / len(current)})
        previous = current
    return pd.DataFrame(rows)


def analyze_factors(
    factors: pd.DataFrame,
    panel: pd.DataFrame,
    quantiles: int = 5,
    forward_days: int = 20,
) -> dict[str, pd.DataFrame]:
    frame = factors.merge(panel[["date", "symbol", "close"]], on=["date", "symbol"], how="left")
    frame = frame.sort_values(["symbol", "date"])
    frame["forward_return"] = forward_returns(frame, forward_days)
    columns = factor_columns(factors)
    ic = pd.concat([information_coefficient(frame, factor) for factor in columns], ignore_index=True)
    qret = pd.concat([quantile_returns(frame, factor, quantiles) for factor in columns], ignore_index=True)
    turnover = pd.concat([factor_turnover(factors, factor) for factor in columns], ignore_index=True)
    summary = (
        ic.groupby("factor")
        .agg(ic_mean=("ic", "mean"), rank_ic_mean=("rank_ic", "mean"), ic_std=("ic", "std"))
        .assign(ir=lambda x: x["ic_mean"] / x["ic_std"].replace(0, pd.NA))
        .reset_index()
    )
    corr = factors[columns].corr()
    return {"ic": ic, "quantile_returns": qret, "turnover": turnover, "summary": summary, "corr": corr}
