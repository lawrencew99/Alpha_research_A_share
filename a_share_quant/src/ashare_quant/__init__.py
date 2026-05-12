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
    load_akshare_index_history,
    load_ohlcv_csv,
    merge_benchmark_if_needed,
    normalize_benchmark_ticker,
    refresh_trade_flags,
)
from ashare_quant.deflated_metrics import (
    bonferroni_deflated_information_ratio,
    bonferroni_deflated_sharpe,
    deflated_summary_from_returns,
    probability_sharpe_ratio,
)
from ashare_quant.factors import build_factor_panel, combine_factors
from ashare_quant.ic_weights import estimate_factor_weights_ic
from ashare_quant.report import write_report
from ashare_quant.research_pipeline import (
    WindowedBacktestSpec,
    bench_cfg_from_cli,
    build_folds_from_strings,
    expand_grid,
    metrics_from_stitched_returns,
    prepare_market_for_backtest,
    run_backtest_on_date_range,
    run_nested_walk_forward_oos,
    run_walk_forward_oos,
    score_for_selection,
)
from ashare_quant.sample_split import (
    SampleSplitConfig,
    slice_for_eval_window,
    slice_panel_by_date,
    union_split_bounds,
)
from ashare_quant.universe import (
    fetch_hs300_constituents,
    load_hs300_constituents,
    load_universe_csv,
    normalize_ashare_ticker,
    write_hs300_constituents,
)
from ashare_quant.walk_forward import WalkForwardFold, walk_forward_folds

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "FactorConfig",
    "SampleSplitConfig",
    "WindowedBacktestSpec",
    "WalkForwardFold",
    "bench_cfg_from_cli",
    "bonferroni_deflated_information_ratio",
    "bonferroni_deflated_sharpe",
    "build_factor_panel",
    "build_folds_from_strings",
    "clean_universe",
    "combine_factors",
    "deflated_summary_from_returns",
    "estimate_factor_weights_ic",
    "expand_grid",
    "exposure_summary",
    "factor_correlation",
    "factor_ic",
    "fetch_hs300_constituents",
    "forward_returns",
    "limit_pct_for_ticker",
    "load_akshare_ashare_history",
    "load_akshare_ashare_history_with_skips",
    "load_akshare_index_history",
    "load_hs300_constituents",
    "merge_benchmark_if_needed",
    "load_ohlcv_csv",
    "load_universe_csv",
    "metrics_from_stitched_returns",
    "normalize_ashare_ticker",
    "normalize_benchmark_ticker",
    "prepare_market_for_backtest",
    "probability_sharpe_ratio",
    "quantile_returns",
    "refresh_trade_flags",
    "run_backtest",
    "run_backtest_on_date_range",
    "run_nested_walk_forward_oos",
    "run_walk_forward_oos",
    "score_for_selection",
    "slice_for_eval_window",
    "slice_panel_by_date",
    "summarize_ic",
    "union_split_bounds",
    "walk_forward_folds",
    "write_hs300_constituents",
    "write_report",
]
