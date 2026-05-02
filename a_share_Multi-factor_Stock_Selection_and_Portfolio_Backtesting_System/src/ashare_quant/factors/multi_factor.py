from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_quant.data.cleaning import forward_returns
from ashare_quant.factors.preprocess import factor_columns


def estimate_ic_weights(
    factors: pd.DataFrame,
    panel: pd.DataFrame,
    forward_days: int = 20,
    use_ir: bool = False,
) -> pd.Series:
    frame = factors.merge(panel[["date", "symbol", "close"]], on=["date", "symbol"], how="left")
    frame["forward_return"] = forward_returns(frame.sort_values(["symbol", "date"]), forward_days)
    weights = {}
    for column in factor_columns(factors):
        ic = frame.groupby("date").apply(
            lambda group: group[column].corr(group["forward_return"], method="spearman"),
            include_groups=False,
        )
        if use_ir:
            value = ic.mean() / ic.std() if ic.std() else 0
        else:
            value = ic.mean()
        weights[column] = 0 if pd.isna(value) else value
    series = pd.Series(weights, dtype=float)
    if series.abs().sum() == 0:
        series[:] = 1
    return series / series.abs().sum()


def risk_parity_weights(factors: pd.DataFrame) -> pd.Series:
    columns = factor_columns(factors)
    vol = factors[columns].std().replace(0, np.nan)
    inverse = 1 / vol
    inverse = inverse.replace([np.inf, -np.inf], np.nan).fillna(0)
    if inverse.sum() == 0:
        inverse[:] = 1
    return inverse / inverse.sum()


def build_multi_factor_score(
    factors: pd.DataFrame,
    panel: pd.DataFrame,
    method: str = "ic_weighted",
    forward_days: int = 20,
) -> pd.DataFrame:
    frame = factors.copy()
    columns = factor_columns(frame)
    if method == "equal":
        weights = pd.Series(1 / len(columns), index=columns)
    elif method == "ir_weighted":
        weights = estimate_ic_weights(frame, panel, forward_days, use_ir=True)
    elif method == "risk_parity":
        weights = risk_parity_weights(frame)
    else:
        weights = estimate_ic_weights(frame, panel, forward_days, use_ir=False)

    aligned = frame[columns].fillna(0)
    frame["score"] = aligned.mul(weights.reindex(columns).fillna(0), axis=1).sum(axis=1)
    frame.attrs["factor_weights"] = weights.to_dict()
    return frame
