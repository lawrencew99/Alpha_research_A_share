from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_quant.backtest import run_backtest
from ashare_quant.config import BacktestConfig, FactorConfig
from ashare_quant.factors import combine_factors
from ashare_quant.research_pipeline import WindowedBacktestSpec, run_backtest_on_date_range
from ashare_quant.sample_split import slice_panel_by_date


def _minimal_market() -> pd.DataFrame:
    dates = pd.date_range("2022-01-03", periods=40, freq="B")
    tickers = ["000001.SZ", "000002.SZ", "600000.SH"]
    rows = []
    rng = np.random.default_rng(42)
    for d in dates:
        for j, t in enumerate(tickers):
            px = 10 + 0.01 * rng.standard_normal() + 0.001 * j
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "open": px,
                    "high": px,
                    "low": px,
                    "close": px,
                    "adj_close": px,
                    "adj_factor": 1.0,
                    "volume": 1e6,
                    "amount": 1e8,
                    "is_suspended": False,
                    "is_limit_up": False,
                    "is_limit_down": False,
                    "can_buy": True,
                    "can_sell": True,
                }
            )
    df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    return df


def test_test_window_metrics_match_direct_run() -> None:
    market = _minimal_market()
    factors = pd.DataFrame(index=market.index)
    rng = np.random.default_rng(1)
    factors["momentum_60"] = rng.standard_normal(len(factors))
    factors["reversal_20"] = rng.standard_normal(len(factors))
    factors["volatility_20"] = rng.standard_normal(len(factors))
    factors["liquidity_20"] = rng.standard_normal(len(factors))
    fc = FactorConfig(
        neutralize_industry=False,
        neutralize_size=False,
        weights_source="manual",
        factor_weights={
            "momentum_60": 0.2,
            "reversal_20": 0.3,
            "volatility_20": -0.25,
            "liquidity_20": -0.25,
        },
    )
    bt = BacktestConfig(
        rebalance="W-FRI",
        top_n=2,
        benchmark="equal_weight",
        weighting="equal",
        execution_delay=0,
        commission=0.0,
        stamp_tax=0.0,
        slippage=0.0,
    )
    spec = WindowedBacktestSpec(factor_config=fc, backtest_config=bt, forward_period=1)
    d0 = pd.Timestamp("2022-02-01")
    d1 = pd.Timestamp("2022-03-15")
    r_pipe = run_backtest_on_date_range(market, start=d0, end=d1, spec=spec, factors_full=factors)
    m_slice = slice_panel_by_date(market, d0, d1)
    f_slice = slice_panel_by_date(factors, d0, d1)
    scores = combine_factors(f_slice, fc)
    r_direct = run_backtest(m_slice, scores, bt, context=f_slice)
    pd.testing.assert_series_equal(r_pipe.metrics, r_direct.metrics, rtol=1e-9, atol=1e-9)
