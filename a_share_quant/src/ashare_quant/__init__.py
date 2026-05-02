"""A-share multi-factor research and backtesting toolkit."""

from ashare_quant.backtest import BacktestResult, run_backtest
from ashare_quant.config import BacktestConfig, FactorConfig
from ashare_quant.data import clean_universe, load_ohlcv_csv, make_synthetic_ashare_data
from ashare_quant.factors import build_factor_panel, combine_factors
from ashare_quant.report import write_report
from ashare_quant.universe import (
    fetch_hs300_constituents,
    load_hs300_constituents,
    load_universe_csv,
    normalize_ashare_ticker,
    write_hs300_constituents,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "FactorConfig",
    "build_factor_panel",
    "clean_universe",
    "combine_factors",
    "fetch_hs300_constituents",
    "load_hs300_constituents",
    "load_ohlcv_csv",
    "load_universe_csv",
    "make_synthetic_ashare_data",
    "normalize_ashare_ticker",
    "run_backtest",
    "write_hs300_constituents",
    "write_report",
]
