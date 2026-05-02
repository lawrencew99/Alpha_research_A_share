from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ashare_quant.data.cleaning import winsorize_by_date, zscore_by_date


def factor_columns(data: pd.DataFrame) -> list[str]:
    ignore = {
        "date",
        "symbol",
        "industry",
        "score",
        "close",
        "return_1d",
        "is_tradeable",
        "market_cap",
    }
    return [column for column in data.columns if column not in ignore and pd.api.types.is_numeric_dtype(data[column])]


def neutralize_cross_section(
    factors: pd.DataFrame,
    panel: pd.DataFrame,
    columns: list[str],
    industry: bool = True,
    market_cap: bool = True,
) -> pd.DataFrame:
    frame = factors.merge(panel[["date", "symbol", "industry", "market_cap"]], on=["date", "symbol"], how="left")
    output = factors.copy()
    for date, group in frame.groupby("date"):
        exposures = pd.DataFrame(index=group.index)
        if market_cap:
            exposures["log_market_cap"] = np.log(group["market_cap"].replace(0, np.nan))
        if industry:
            dummies = pd.get_dummies(group["industry"], prefix="industry", dtype=float)
            exposures = pd.concat([exposures, dummies], axis=1)
        exposures = sm.add_constant(exposures.fillna(exposures.median(numeric_only=True)), has_constant="add")
        for column in columns:
            valid = group[column].notna() & exposures.notna().all(axis=1)
            if valid.sum() <= exposures.shape[1]:
                continue
            model = sm.OLS(group.loc[valid, column], exposures.loc[valid]).fit()
            output.loc[group.loc[valid].index, column] = model.resid
    return output


def preprocess_factors(
    factors: pd.DataFrame,
    panel: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
    neutralize: bool = True,
) -> pd.DataFrame:
    columns = factor_columns(factors)
    frame = winsorize_by_date(factors, columns, lower, upper)
    if neutralize:
        frame = neutralize_cross_section(frame, panel, columns)
    return zscore_by_date(frame, columns)
