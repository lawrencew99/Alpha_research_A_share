"""Data loading, adjustment and A-share universe cleaning utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_PRICE_COLUMNS = {"date", "ticker", "close"}


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load local A-share OHLCV data and return a date/ticker indexed panel."""

    frame = pd.read_csv(path, parse_dates=["date"])
    missing = REQUIRED_PRICE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    frame = frame.sort_values(["date", "ticker"]).set_index(["date", "ticker"])
    return add_adjusted_prices(frame)


def add_adjusted_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Create adjusted OHLC prices when an adjustment factor is available."""

    result = frame.copy()
    if "adj_factor" not in result.columns:
        result["adj_factor"] = 1.0

    for column in ["open", "high", "low", "close"]:
        if column in result.columns:
            result[f"adj_{column}"] = result[column] * result["adj_factor"]

    if "adj_close" not in result.columns and "close" in result.columns:
        result["adj_close"] = result["close"] * result["adj_factor"]
    return result


def clean_universe(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter observations unsuitable for a cross-sectional A-share backtest."""

    result = frame.copy()
    tradable = result["adj_close"].notna()

    for flag in ["is_suspended", "is_limit_up", "is_limit_down"]:
        if flag in result.columns:
            tradable &= ~result[flag].fillna(False).astype(bool)

    for column in ["amount", "volume"]:
        if column in result.columns:
            tradable &= result[column].fillna(0) > 0

    return result.loc[tradable].copy()


def make_synthetic_ashare_data(
    n_stocks: int = 120,
    start: str = "2021-01-01",
    end: str = "2024-12-31",
    seed: int = 7,
) -> pd.DataFrame:
    """Generate reproducible A-share-like panel data for demos and tests."""

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    tickers = [f"{i:06d}.SZ" if i % 2 else f"{i:06d}.SH" for i in range(1, n_stocks + 1)]
    industries = ["电子", "计算机", "医药", "机械", "消费", "金融", "新能源", "周期"]

    records: list[pd.DataFrame] = []
    for i, ticker in enumerate(tickers):
        industry = industries[i % len(industries)]
        market_beta = rng.normal(0.00025, 0.00015)
        idio_vol = rng.uniform(0.012, 0.035)
        returns = rng.normal(market_beta, idio_vol, len(dates))
        close = 20 * np.exp(np.cumsum(returns))
        volume = rng.lognormal(mean=14.5, sigma=0.5, size=len(dates))
        turnover_noise = rng.uniform(0.8, 1.2, len(dates))
        amount = close * volume * turnover_noise
        market_cap = close * rng.uniform(2e8, 2e9)
        roe = rng.normal(0.09, 0.04, len(dates)).clip(-0.1, 0.35)
        pb = rng.lognormal(mean=0.75, sigma=0.45, size=len(dates))
        pe = (pb / np.maximum(roe, 0.01)).clip(5, 120)

        records.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": close * (1 + rng.normal(0, 0.004, len(dates))),
                    "high": close * (1 + rng.uniform(0, 0.025, len(dates))),
                    "low": close * (1 - rng.uniform(0, 0.025, len(dates))),
                    "close": close,
                    "volume": volume,
                    "amount": amount,
                    "adj_factor": 1.0,
                    "industry": industry,
                    "market_cap": market_cap,
                    "pe": pe,
                    "pb": pb,
                    "roe": roe,
                    "is_suspended": rng.random(len(dates)) < 0.002,
                    "is_limit_up": returns > 0.095,
                    "is_limit_down": returns < -0.095,
                }
            )
        )

    frame = pd.concat(records, ignore_index=True).sort_values(["date", "ticker"])
    frame = frame.set_index(["date", "ticker"])
    return add_adjusted_prices(frame)
