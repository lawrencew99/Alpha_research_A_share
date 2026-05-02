"""A-share multi-factor research and backtesting toolkit."""

from ashare_quant.backtest import BacktestResult, run_backtest
from ashare_quant.config import BacktestConfig, FactorConfig
from ashare_quant.data import clean_universe, load_ohlcv_csv, make_synthetic_ashare_data
from ashare_quant.factors import build_factor_panel, combine_factors
from ashare_quant.report import write_report

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "FactorConfig",
    "build_factor_panel",
    "clean_universe",
    "combine_factors",
    "load_ohlcv_csv",
    "make_synthetic_ashare_data",
    "run_backtest",
    "write_report",
]
