from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_by_date(
    data: pd.DataFrame,
    columns: list[str],
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    frame = data.copy()
    for column in columns:
        bounds = frame.groupby("date")[column].quantile([lower, upper]).unstack()
        frame = frame.join(bounds.rename(columns={lower: "_lo", upper: "_hi"}), on="date")
        frame[column] = frame[column].clip(frame["_lo"], frame["_hi"])
        frame = frame.drop(columns=["_lo", "_hi"])
    return frame


def zscore_by_date(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = data.copy()
    grouped = frame.groupby("date")
    for column in columns:
        mean = grouped[column].transform("mean")
        std = grouped[column].transform("std").replace(0, np.nan)
        frame[column] = (frame[column] - mean) / std
    return frame


def filter_tradeable(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    previous_close = frame.groupby("symbol")["close"].shift(1)
    limit_up = frame.get("limit_up", previous_close * 1.1)
    limit_down = frame.get("limit_down", previous_close * 0.9)
    at_limit = frame["close"].ge(limit_up * 0.999) | frame["close"].le(limit_down * 1.001)
    suspended = frame.get("is_suspended", False)
    frame["is_tradeable"] = ~(pd.Series(suspended, index=frame.index).fillna(False) | at_limit.fillna(False))
    return frame


def prepare_panel(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    industries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["symbol", "date"])
    numeric_cols = ["open", "high", "low", "close", "volume", "amount", "turnover"]
    for column in numeric_cols:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[numeric_cols] = frame.groupby("symbol")[numeric_cols].ffill()
    frame = filter_tradeable(frame)
    frame["return_1d"] = frame.groupby("symbol")["close"].pct_change()
    frame["forward_return_1d"] = frame.groupby("symbol")["close"].pct_change().shift(-1)

    if fundamentals is not None and not fundamentals.empty:
        fundamentals = fundamentals.copy()
        fundamentals["date"] = pd.to_datetime(fundamentals["date"])
        frame = frame.merge(fundamentals, on=["date", "symbol"], how="left")
    if industries is not None and not industries.empty:
        frame = frame.merge(industries, on="symbol", how="left")
    frame["industry"] = frame.get("industry", "未知").fillna("未知")
    frame["market_cap"] = pd.to_numeric(frame.get("market_cap", np.nan), errors="coerce")
    frame["market_cap"] = frame.groupby("symbol")["market_cap"].ffill().bfill()
    return frame.dropna(subset=["close"])


def forward_returns(panel: pd.DataFrame, days: int) -> pd.Series:
    future = panel.groupby("symbol")["close"].shift(-days)
    return future / panel["close"] - 1
