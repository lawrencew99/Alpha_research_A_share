from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_quant.config import BacktestConfig, FactorConfig
from ashare_quant.deflated_metrics import bonferroni_deflated_sharpe, deflated_summary_from_returns
from ashare_quant.research_pipeline import (
    WindowedBacktestSpec,
    expand_grid,
    run_nested_walk_forward_oos,
)
from ashare_quant.walk_forward import walk_forward_folds


def _synth_market_and_factors(
    start: str = "2021-01-04",
    n_days: int = 130,
    tickers: tuple[str, ...] = ("000001.SZ", "000002.SZ", "600000.SH", "600519.SH"),
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a small panel with random walk prices and noise factors."""

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="B")
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    n = len(idx)

    # geometric random walk per ticker
    rets = rng.normal(loc=0.0005, scale=0.012, size=n)
    rets_df = pd.Series(rets, index=idx).unstack("ticker")
    prices = (1 + rets_df).cumprod() * 10.0

    market_rows = []
    for d in dates:
        for t in tickers:
            px = float(prices.loc[d, t])
            market_rows.append(
                {
                    "date": d, "ticker": t,
                    "open": px, "high": px, "low": px, "close": px,
                    "adj_close": px, "adj_factor": 1.0,
                    "volume": 1e6, "amount": 1e8,
                    "is_suspended": False, "is_limit_up": False, "is_limit_down": False,
                    "can_buy": True, "can_sell": True,
                }
            )
    market = pd.DataFrame(market_rows).set_index(["date", "ticker"]).sort_index()

    factors = pd.DataFrame(
        {
            "f_alpha": rng.standard_normal(n),
            "f_beta": rng.standard_normal(n),
        },
        index=idx,
    )
    return market, factors


def _make_spec() -> WindowedBacktestSpec:
    fc = FactorConfig(
        neutralize_industry=False,
        neutralize_size=False,
        weights_source="manual",
        factor_weights={"f_alpha": 0.6, "f_beta": 0.4},
    )
    bt = BacktestConfig(
        rebalance="W-FRI",
        top_n=2,
        benchmark="equal_weight",
        weighting="equal",
        execution_delay=0,
        commission=0.0,
        sell_commission=0.0,
        stamp_tax=0.0,
        slippage=0.0,
    )
    return WindowedBacktestSpec(factor_config=fc, backtest_config=bt, forward_period=1)


def _run_nested_smoke() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, dict, list]:
    market, factors = _synth_market_and_factors()
    folds = walk_forward_folds("2021-01-04", "2021-07-02", train_months=2, test_months=1, step_months=1)
    spec = _make_spec()
    stitched, fold_df, chosen_df, diag = run_nested_walk_forward_oos(
        market,
        factors,
        spec,
        folds,
        grid={"top_n": [2, 3]},
        selection_metric="sharpe",
    )
    return stitched, fold_df, chosen_df, diag, folds


def test_expand_grid_cartesian_product() -> None:
    out = expand_grid({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(out) == 6
    assert {tuple(sorted(d.items())) for d in out} == {
        (("a", 1), ("b", "x")),
        (("a", 1), ("b", "y")),
        (("a", 1), ("b", "z")),
        (("a", 2), ("b", "x")),
        (("a", 2), ("b", "y")),
        (("a", 2), ("b", "z")),
    }
    assert expand_grid({}) == [{}]


def test_nested_no_train_test_leakage() -> None:
    """Each stitched date must come from some fold's test window AND lie at or after
    that fold's own ``train_end_exclusive`` (i.e. not used to train its own predictor).

    Note: adjacent folds can have *training* windows that span an *earlier* fold's
    test window — that is correct (later fold's IC fit doesn't predict the earlier
    fold's test). The relevant invariant is therefore per-fold, not global.
    """

    stitched, _, _, _, folds = _run_nested_smoke()
    assert len(stitched) > 0

    for ts in stitched.index:
        owning = [f for f in folds if f.test_start <= ts < f.test_end_exclusive]
        assert owning, f"stitched date {ts} not inside any test window"
        # half-open invariant in walk_forward_folds: test_start == train_end_exclusive
        for f in owning:
            assert ts >= f.train_end_exclusive, (
                f"stitched date {ts} leaked into its own fold's train window"
            )


def test_chosen_df_shape_and_columns() -> None:
    _, fold_df, chosen_df, diag, folds = _run_nested_smoke()
    assert len(chosen_df) <= len(folds)
    assert len(chosen_df) == len(fold_df)
    assert "top_n" in chosen_df.columns
    assert "train_metric" in chosen_df.columns
    # all picks should be one of the grid values
    assert set(chosen_df["top_n"].unique()).issubset({2, 3})
    assert diag["grid_size"] == 2
    assert diag["selection_metric"] == "sharpe"


def test_deflated_sharpe_le_raw() -> None:
    stitched, _, _, _, _ = _run_nested_smoke()
    summary = deflated_summary_from_returns(stitched, benchmark_returns=None, n_trials=2)
    raw = float(summary["sharpe"])
    deflated = float(summary["bonferroni_deflated_sharpe"])
    assert np.isfinite(deflated)
    # Bonferroni critical value is strictly positive for any n_trials >= 1, alpha < 0.5
    assert deflated < raw

    # explicit equivalence: single-call helper should match the summary's deflated value
    direct = bonferroni_deflated_sharpe(raw, 2, stitched.dropna())
    assert np.isclose(direct, deflated)


def test_grid_size_used_in_deflation() -> None:
    stitched, _, _, _, _ = _run_nested_smoke()
    s1 = deflated_summary_from_returns(stitched, benchmark_returns=None, n_trials=1)
    s10 = deflated_summary_from_returns(stitched, benchmark_returns=None, n_trials=10)
    # more trials => stricter Bonferroni threshold => smaller deflated value
    assert float(s10["bonferroni_deflated_sharpe"]) <= float(s1["bonferroni_deflated_sharpe"])
