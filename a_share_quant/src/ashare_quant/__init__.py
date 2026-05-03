"""A-share multi-factor research and backtesting toolkit."""

from ashare_quant.analysis import (
    exposure_summary,
    factor_correlation,
    factor_ic,
    forward_returns,
    quantile_returns,
    summarize_ic,
)
from ashare_quant.backtest import BacktestResult, run_backtest
from ashare_quant.config import BacktestConfig, FactorConfig
from ashare_quant.data import (
    clean_universe,
    limit_pct_for_ticker,
    load_akshare_ashare_history,
    load_akshare_ashare_history_with_skips,
    load_ohlcv_csv,
    refresh_trade_flags,
)
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
    "exposure_summary",
    "factor_correlation",
    "factor_ic",
    "fetch_hs300_constituents",
    "forward_returns",
    "limit_pct_for_ticker",
    "load_akshare_ashare_history",
    "load_akshare_ashare_history_with_skips",
    "load_hs300_constituents",
    "load_ohlcv_csv",
    "load_universe_csv",
    "normalize_ashare_ticker",
    "quantile_returns",
    "refresh_trade_flags",
    "run_backtest",
    "summarize_ic",
    "write_hs300_constituents",
    "write_report",
]
