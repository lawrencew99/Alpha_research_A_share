from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_quant.factors.preprocess import preprocess_factors


def _rolling(group: pd.Series, window: int, func: str) -> pd.Series:
    return group.rolling(window, min_periods=max(3, window // 2)).agg(func)


def compute_factor_panel(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.sort_values(["symbol", "date"]).copy()
    grouped = data.groupby("symbol", group_keys=False)
    factors = data[["date", "symbol"]].copy()

    factors["momentum_20"] = grouped["close"].pct_change(20)
    factors["momentum_60"] = grouped["close"].pct_change(60)
    factors["reversal_5"] = -grouped["close"].pct_change(5)
    factors["volatility_20"] = -grouped["return_1d"].apply(lambda x: _rolling(x, 20, "std"))
    factors["liquidity_20"] = -grouped["amount"].apply(lambda x: np.log(x.replace(0, np.nan)).rolling(20, min_periods=10).mean())
    factors["turnover_20"] = -grouped["turnover"].apply(lambda x: _rolling(x, 20, "mean"))
    factors["value_ep"] = 1 / pd.to_numeric(data.get("pe_ttm", np.nan), errors="coerce").replace(0, np.nan)
    factors["value_bp"] = 1 / pd.to_numeric(data.get("pb", np.nan), errors="coerce").replace(0, np.nan)
    factors["quality_roe"] = pd.to_numeric(data.get("roe", np.nan), errors="coerce")
    factors["quality_low_leverage"] = -pd.to_numeric(data.get("debt_to_asset", np.nan), errors="coerce")
    factors["growth_sales_proxy"] = grouped["amount"].pct_change(60)

    factor_cols = [column for column in factors.columns if column not in {"date", "symbol"}]
    factors[factor_cols] = factors.groupby("symbol")[factor_cols].ffill()
    return preprocess_factors(factors, panel)
