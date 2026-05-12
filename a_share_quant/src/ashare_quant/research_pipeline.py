"""High-level experiment orchestration: splits, IC weights, stitched OOS metrics."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import pandas as pd

from ashare_quant.analysis import forward_returns
from ashare_quant.backtest import BacktestResult, calculate_metrics, run_backtest
from ashare_quant.config import BacktestConfig, FactorConfig
from ashare_quant.data import merge_benchmark_if_needed, normalize_benchmark_ticker
from ashare_quant.factors import build_factor_panel, combine_factors
from ashare_quant.sample_split import slice_panel_by_date
from ashare_quant.walk_forward import WalkForwardFold, fold_test_mask, fold_train_mask, walk_forward_folds

SelectionMetric = Literal["information_ratio", "sharpe", "annual_return"]

_FACTOR_FIELDS = frozenset(
    {"winsor_quantile", "neutralize_industry", "neutralize_size", "weights_source", "factor_weights"}
)
_BACKTEST_FIELDS = frozenset(
    {
        "initial_cash", "rebalance", "top_n", "top_quantile", "max_weight", "weighting",
        "execution_delay", "commission", "sell_commission", "stamp_tax", "slippage",
        "min_commission", "price_field", "benchmark", "annualization", "min_amount",
        "max_industry_weight", "industry_field", "volatility_field",
    }
)


@dataclass(frozen=True)
class WindowedBacktestSpec:
    factor_config: FactorConfig
    backtest_config: BacktestConfig
    forward_period: int = 1


def bench_cfg_from_cli(benchmark_cli: str) -> str | None:
    if benchmark_cli.lower() == "none":
        return None
    if benchmark_cli.lower() in {"equal_weight", "universe_equal"}:
        return benchmark_cli.lower()
    return normalize_benchmark_ticker(benchmark_cli)


def prepare_market_for_backtest(
    market_stocks: pd.DataFrame,
    benchmark_cli: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    return merge_benchmark_if_needed(market_stocks, benchmark_cli, start, end)


def run_backtest_on_date_range(
    market_bt: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    spec: WindowedBacktestSpec,
    factors_full: pd.DataFrame | None = None,
    ic_train_slice: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> BacktestResult:
    """Slice market to [start,end], rebuild scores on the slice, run ``run_backtest``."""

    m = slice_panel_by_date(market_bt, start, end)
    if m.empty:
        raise ValueError("empty market slice")
    if spec.factor_config.weights_source == "ic_train":
        if factors_full is None:
            raise ValueError("ic_train requires factors_full panel")
        if ic_train_slice is None:
            raise ValueError("ic_train requires ic_train_slice (train_start, train_end)")
        tr0, tr1 = ic_train_slice
        f_tr = slice_panel_by_date(factors_full, tr0, tr1)
        m_tr = slice_panel_by_date(market_bt, tr0, tr1)
        fut_tr = forward_returns(m_tr, periods=spec.forward_period)
        factors_slice = slice_panel_by_date(factors_full, start, end)
        scores = combine_factors(
            factors_slice,
            spec.factor_config,
            ic_calibration_factors=f_tr,
            ic_calibration_future=fut_tr,
        )
    else:
        if factors_full is not None:
            factors_slice = slice_panel_by_date(factors_full, start, end)
        else:
            factors_slice = build_factor_panel(m)
        scores = combine_factors(factors_slice, spec.factor_config)

    return run_backtest(m, scores, spec.backtest_config, context=factors_slice)


def run_walk_forward_oos(
    market_bt: pd.DataFrame,
    factors_full: pd.DataFrame,
    spec: WindowedBacktestSpec,
    folds: list[WalkForwardFold],
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    """Fit IC weights on each train window, backtest OOS on test; stitch daily returns."""

    dates = market_bt.index.get_level_values("date").unique().sort_values()
    pieces: list[pd.Series] = []
    rows: list[dict[str, object]] = []
    for i, fold in enumerate(folds):
        tr_mask = fold_train_mask(dates, fold)
        te_mask = fold_test_mask(dates, fold)
        train_dates = dates[tr_mask]
        test_dates = dates[te_mask]
        if len(train_dates) < 5 or len(test_dates) < 5:
            continue
        tr0, tr1 = train_dates.min(), train_dates.max()
        te0, te1 = test_dates.min(), test_dates.max()
        res = run_backtest_on_date_range(
            market_bt,
            start=te0,
            end=te1,
            spec=spec,
            factors_full=factors_full,
            ic_train_slice=(tr0, tr1),
        )
        pieces.append(res.daily_returns.copy())
        row = res.metrics.to_dict()
        row.update(
            {
                "fold_index": i,
                "train_start": fold.train_start,
                "train_end_exclusive": fold.train_end_exclusive,
                "test_start": fold.test_start,
                "test_end_exclusive": fold.test_end_exclusive,
            }
        )
        rows.append(row)
    if not pieces:
        raise ValueError("walk-forward produced no OOS segments")
    stitched = pd.concat(pieces).sort_index()
    stitched = stitched.groupby(stitched.index).first()
    stitched.name = "oos_daily_return"
    fold_df = pd.DataFrame(rows).set_index("fold_index")
    ann = spec.backtest_config.annualization
    te = stitched.std(ddof=0) * np.sqrt(ann) if len(stitched) > 1 else np.nan
    years = max(len(stitched) / ann, 1 / ann)
    total_r = float((1 + stitched.fillna(0)).prod() - 1)
    ar = (1 + total_r) ** (1 / years) - 1 if years > 0 else np.nan
    diag = {
        "stitched_sharpe": float(ar / te) if te and np.isfinite(te) and te > 0 else np.nan,
        "stitched_annual_return": float(ar),
    }
    return stitched, fold_df, diag


def build_folds_from_strings(
    global_start: str,
    global_end: str,
    train_months: int,
    test_months: int,
    step_months: int,
) -> list[WalkForwardFold]:
    return walk_forward_folds(global_start, global_end, train_months=train_months, test_months=test_months, step_months=step_months)


def metrics_from_stitched_returns(
    stitched: pd.Series,
    market_bt: pd.DataFrame,
    config: BacktestConfig,
) -> pd.Series:
    equity = (1 + stitched.fillna(0)).cumprod()
    turnover = pd.Series(0.0, index=stitched.index)
    bench_ret = None
    if config.benchmark in {"equal_weight", "universe_equal"}:
        prices = market_bt[config.price_field].unstack("ticker").sort_index()
        rets = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0)
        bench_ret = rets.mean(axis=1).reindex(stitched.index).fillna(0)
    elif config.benchmark is not None:
        prices = market_bt[config.price_field].unstack("ticker").sort_index()
        if config.benchmark in prices.columns:
            bench_ret = prices[config.benchmark].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0).reindex(stitched.index).fillna(0)
    return calculate_metrics(equity, stitched.rename("daily_return"), turnover, config.annualization, bench_ret)


def expand_grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of a parameter grid. Empty grid yields a single empty dict."""

    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    if any(len(v) == 0 for v in values):
        return []
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def score_for_selection(metrics: pd.Series, metric: SelectionMetric) -> float:
    """Pull a single selection score from a BacktestResult.metrics Series; -inf when missing or non-finite."""

    if metric not in metrics.index:
        return float("-inf")
    val = metrics[metric]
    try:
        f = float(val)
    except (TypeError, ValueError):
        return float("-inf")
    if not np.isfinite(f):
        return float("-inf")
    return f


def _spec_with_overrides(base_spec: WindowedBacktestSpec, cfg: Mapping[str, Any]) -> WindowedBacktestSpec:
    """Apply a flat override dict to FactorConfig / BacktestConfig and rebuild the spec."""

    fc_overrides = {k: v for k, v in cfg.items() if k in _FACTOR_FIELDS}
    bt_overrides = {k: v for k, v in cfg.items() if k in _BACKTEST_FIELDS}
    unknown = set(cfg) - _FACTOR_FIELDS - _BACKTEST_FIELDS
    if unknown:
        raise ValueError(f"unknown override fields for nested walk-forward grid: {sorted(unknown)}")
    fc = replace(base_spec.factor_config, **fc_overrides) if fc_overrides else base_spec.factor_config
    bt = replace(base_spec.backtest_config, **bt_overrides) if bt_overrides else base_spec.backtest_config
    return WindowedBacktestSpec(factor_config=fc, backtest_config=bt, forward_period=base_spec.forward_period)


def run_nested_walk_forward_oos(
    market_bt: pd.DataFrame,
    factors_full: pd.DataFrame,
    base_spec: WindowedBacktestSpec,
    folds: list[WalkForwardFold],
    grid: Mapping[str, Sequence[Any]],
    *,
    selection_metric: SelectionMetric = "information_ratio",
    min_dates_per_window: int = 5,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Per-fold nested grid search: pick a champion on each train window, record only its
    test-window performance, then stitch all test-window daily returns into one OOS series.

    The grid keys must be field names of ``FactorConfig`` or ``BacktestConfig``; ``factor_weights``
    may be passed as a Mapping value. ``weights_source='ic_train'`` is honoured per-candidate and
    the train slice is the **fold's** train window (no peeking).
    """

    candidates = expand_grid(grid)
    if not candidates:
        raise ValueError("grid produced no candidates")

    dates = market_bt.index.get_level_values("date").unique().sort_values()
    pieces: list[pd.Series] = []
    fold_rows: list[dict[str, Any]] = []
    chosen_rows: list[dict[str, Any]] = []

    for i, fold in enumerate(folds):
        assert fold.train_end_exclusive <= fold.test_start, "leakage guard: train_end_exclusive > test_start"

        tr_mask = fold_train_mask(dates, fold)
        te_mask = fold_test_mask(dates, fold)
        train_dates = dates[tr_mask]
        test_dates = dates[te_mask]
        if len(train_dates) < min_dates_per_window or len(test_dates) < min_dates_per_window:
            continue

        tr0, tr1 = train_dates.min(), train_dates.max()
        te0, te1 = test_dates.min(), test_dates.max()
        assert tr1 < te0, "leakage guard: last train date >= first test date"

        scored: list[tuple[float, dict[str, Any]]] = []
        for cfg in candidates:
            spec_i = _spec_with_overrides(base_spec, cfg)
            ic_slice = (tr0, tr1) if spec_i.factor_config.weights_source == "ic_train" else None
            try:
                res_tr = run_backtest_on_date_range(
                    market_bt,
                    start=tr0,
                    end=tr1,
                    spec=spec_i,
                    factors_full=factors_full,
                    ic_train_slice=ic_slice,
                )
                s = score_for_selection(res_tr.metrics, selection_metric)
            except Exception:
                s = float("-inf")
            scored.append((s, cfg))

        scored.sort(key=lambda kv: kv[0], reverse=True)
        best_score, best_cfg = scored[0]
        if best_score == float("-inf"):
            continue

        spec_best = _spec_with_overrides(base_spec, best_cfg)
        ic_slice_te = (tr0, tr1) if spec_best.factor_config.weights_source == "ic_train" else None
        res_te = run_backtest_on_date_range(
            market_bt,
            start=te0,
            end=te1,
            spec=spec_best,
            factors_full=factors_full,
            ic_train_slice=ic_slice_te,
        )

        pieces.append(res_te.daily_returns.copy())

        fold_row: dict[str, Any] = {
            "fold_index": i,
            "train_start": fold.train_start,
            "train_end_exclusive": fold.train_end_exclusive,
            "test_start": fold.test_start,
            "test_end_exclusive": fold.test_end_exclusive,
            "train_selection_score": best_score,
        }
        fold_row.update(res_te.metrics.to_dict())
        fold_row.update({f"cfg_{k}": _stringify_cfg_value(v) for k, v in best_cfg.items()})
        fold_rows.append(fold_row)

        chosen_row: dict[str, Any] = {"fold_index": i, "train_metric": best_score}
        chosen_row.update({k: _stringify_cfg_value(v) for k, v in best_cfg.items()})
        chosen_rows.append(chosen_row)

    if not pieces:
        raise ValueError("nested walk-forward produced no OOS segments")

    stitched = pd.concat(pieces).sort_index()
    stitched = stitched.groupby(stitched.index).first()
    stitched.name = "oos_daily_return"

    fold_df = pd.DataFrame(fold_rows).set_index("fold_index")
    chosen_df = pd.DataFrame(chosen_rows).set_index("fold_index")

    ann = base_spec.backtest_config.annualization
    te = stitched.std(ddof=0) * np.sqrt(ann) if len(stitched) > 1 else np.nan
    years = max(len(stitched) / ann, 1 / ann)
    total_r = float((1 + stitched.fillna(0)).prod() - 1)
    ar = (1 + total_r) ** (1 / years) - 1 if years > 0 else np.nan
    diag = {
        "n_folds_evaluated": len(fold_rows),
        "n_folds_input": len(folds),
        "grid_size": len(candidates),
        "selection_metric": selection_metric,
        "stitched_sharpe": float(ar / te) if te and np.isfinite(te) and te > 0 else np.nan,
        "stitched_annual_return": float(ar),
    }
    return stitched, fold_df, chosen_df, diag


def _stringify_cfg_value(value: Any) -> Any:
    """Make grid values CSV-friendly: keep scalars, render mappings as JSON-ish strings."""

    if isinstance(value, Mapping):
        items = ",".join(f"{k}={float(v):g}" for k, v in value.items())
        return "{" + items + "}"
    return value
