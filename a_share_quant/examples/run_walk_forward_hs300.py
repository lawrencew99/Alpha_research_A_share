"""Rolling walk-forward OOS backtest (primary research entry-point).

Modes:

- ``--grid-mode none`` (default, legacy): IC weights are refit each train window for
  a single fixed (top_n, rebalance, weighting) combination. Equivalent to the prior
  behaviour of this script.
- ``--grid-mode default`` (recommended): nested per-fold grid search. On each train
  window all candidates are scored by ``--selection-metric``; the champion is
  applied **only** to the matching test window. Test-window daily returns are
  stitched into a single OOS series and Bonferroni-deflated against ``|grid|``.
- ``--grid-mode json:PATH``: load the grid from a JSON object ``{field: [values]}``
  where each field is a member of ``FactorConfig`` / ``BacktestConfig``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ashare_quant import (
    BacktestConfig,
    FactorConfig,
    build_factor_panel,
    clean_universe,
    load_akshare_ashare_history_with_skips,
    load_hs300_constituents,
)
from ashare_quant.backtest import BacktestResult
from ashare_quant.deflated_metrics import deflated_summary_from_returns
from ashare_quant.report import write_report
from ashare_quant.research_pipeline import (
    WindowedBacktestSpec,
    bench_cfg_from_cli,
    build_folds_from_strings,
    expand_grid,
    metrics_from_stitched_returns,
    prepare_market_for_backtest,
    run_nested_walk_forward_oos,
    run_walk_forward_oos,
)

DEFAULT_GRID: dict[str, list[object]] = {
    "top_n": [20, 30, 40],
    "weighting": ["equal", "score"],
}


def _resolve_grid(arg: str, script_path: Path) -> dict[str, list[object]]:
    if arg == "none":
        return {}
    if arg == "default":
        return {k: list(v) for k, v in DEFAULT_GRID.items()}
    if arg.startswith("json:"):
        rel = arg[len("json:") :]
        path = Path(rel)
        if not path.is_absolute():
            path = script_path.resolve().parents[1] / path
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or not data:
            raise SystemExit("grid JSON must be a non-empty object {field: [values]}")
        return {str(k): list(v) for k, v in data.items()}
    raise SystemExit(f"unknown --grid-mode: {arg!r} (expected none|default|json:PATH)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward OOS backtest with optional nested grid.")
    parser.add_argument("--global-start", default="2021-01-01")
    parser.add_argument("--global-end", default="2024-12-31")
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--step-months", type=int, default=6)
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--top-n", type=int, default=30, help="Used when --grid-mode none.")
    parser.add_argument("--rebalance", default="ME", help="Used when --grid-mode none.")
    parser.add_argument("--weighting", default="equal", help="Used when --grid-mode none.")
    parser.add_argument("--forward-period", type=int, default=1)
    parser.add_argument("--adjust", default="qfq")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs_walk_forward_hs300")
    parser.add_argument(
        "--grid-mode",
        default="none",
        help="none | default | json:PATH (see module docstring).",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["information_ratio", "sharpe", "annual_return"],
        default="information_ratio",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=1,
        help="Override n_trials for deflated metrics. When --grid-mode != none, defaults to |grid|.",
    )
    args = parser.parse_args()

    grid = _resolve_grid(args.grid_mode, Path(__file__))

    tickers = load_hs300_constituents()
    if args.limit is not None:
        tickers = tickers[: args.limit]
    market_raw, skipped = load_akshare_ashare_history_with_skips(
        tickers=tickers,
        start=args.global_start,
        end=args.global_end,
        adjust=args.adjust,
        max_workers=args.max_workers,
        show_progress=True,
    )
    market_stocks = clean_universe(market_raw)
    bench = bench_cfg_from_cli(args.benchmark)
    market_bt = prepare_market_for_backtest(market_stocks, args.benchmark, args.global_start, args.global_end)
    factors_full = build_factor_panel(market_stocks)

    folds = build_folds_from_strings(
        args.global_start,
        args.global_end,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
    )
    if not folds:
        raise SystemExit("no walk-forward folds for the given range and window sizes")

    fc = FactorConfig(neutralize_industry=False, neutralize_size=False, weights_source="ic_train")
    bt = BacktestConfig(rebalance=args.rebalance, top_n=args.top_n, benchmark=bench, weighting=args.weighting)
    base_spec = WindowedBacktestSpec(factor_config=fc, backtest_config=bt, forward_period=args.forward_period)

    chosen_df: pd.DataFrame | None = None
    diag_extra: dict[str, object] = {}
    if grid:
        stitched, fold_df, chosen_df, diag_extra = run_nested_walk_forward_oos(
            market_bt,
            factors_full,
            base_spec,
            folds,
            grid,
            selection_metric=args.selection_metric,
        )
        n_trials_for_deflated = (
            args.n_trials if args.n_trials != 1 else max(len(expand_grid(grid)), 1)
        )
    else:
        stitched, fold_df, _diag = run_walk_forward_oos(market_bt, factors_full, base_spec, folds)
        n_trials_for_deflated = args.n_trials

    metrics_stitched = metrics_from_stitched_returns(stitched, market_bt, bt)
    prices = market_bt[bt.price_field].unstack("ticker").sort_index()
    rets = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0)
    bench_for_deflated = rets.mean(axis=1).reindex(stitched.index).fillna(0)
    deflated = deflated_summary_from_returns(stitched, bench_for_deflated, n_trials=n_trials_for_deflated)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stitched.to_csv(out / "oos_stitched_daily_returns.csv", encoding="utf-8-sig")
    fold_df.to_csv(out / "walk_forward_fold_metrics.csv", encoding="utf-8-sig")
    metrics_stitched.to_frame("value").to_csv(out / "oos_stitched_metrics.csv", encoding="utf-8-sig")
    deflated.to_frame("value").to_csv(out / "oos_deflated_metrics.csv", encoding="utf-8-sig")

    if chosen_df is not None:
        chosen_df.to_csv(out / "walk_forward_chosen_configs.csv", encoding="utf-8-sig")
        spec_payload = {
            "grid_mode": args.grid_mode,
            "grid": grid,
            "selection_metric": args.selection_metric,
            "train_months": args.train_months,
            "test_months": args.test_months,
            "step_months": args.step_months,
            "n_trials_for_deflated": n_trials_for_deflated,
            "diag": {k: (str(v) if not isinstance(v, (int, float, str, bool)) else v) for k, v in diag_extra.items()},
        }
        with (out / "walk_forward_grid_spec.json").open("w", encoding="utf-8") as fh:
            json.dump(spec_payload, fh, ensure_ascii=False, indent=2, default=str)

    pseudo = BacktestResult(
        equity_curve=(1 + stitched.fillna(0)).cumprod(),
        daily_returns=stitched,
        weights=pd.DataFrame(),
        turnover=pd.Series(0.0, index=stitched.index),
        metrics=metrics_stitched,
        monthly_returns=(1 + stitched.fillna(0)).resample("ME").prod() - 1,
        benchmark_curve=None,
        benchmark_returns=None,
        excess_returns=None,
        trade_log=None,
    )
    write_report(pseudo, out)
    if not skipped.empty:
        skipped.to_csv(out / "skipped_tickers.csv", index=False, encoding="utf-8-sig")

    print("=== Walk-forward stitched OOS ===")
    print(f"Grid mode: {args.grid_mode} | folds: {len(fold_df)} | n_trials (deflated): {n_trials_for_deflated}")
    print(metrics_stitched.round(4).to_string())
    if chosen_df is not None:
        print("\nChosen config per fold:")
        print(chosen_df.round(4).to_string())
    print("\nFold metrics:")
    print(fold_df.round(4).to_string())
    print(f"\nWrote {out.resolve()}")


if __name__ == "__main__":
    main()
