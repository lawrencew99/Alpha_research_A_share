"""Run a CSI 300 backtest on real AKShare daily history.

Legacy / 对照基线：主推荐流程为嵌套滚动 walk-forward
(`examples/run_walk_forward_hs300.py --grid-mode default`)。本脚本保留两种用途：

1. 不带 `--sample-split` 的旧式整段回测，作为最简化基线；
2. `--sample-split research` 的 train/val/test 三段切分，作为 walk-forward 的"一次切片"对照。

两者均不应作为最终样本外结论。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ashare_quant import (
    BacktestConfig,
    FactorConfig,
    build_factor_panel,
    clean_universe,
    combine_factors,
    exposure_summary,
    factor_correlation,
    factor_ic,
    forward_returns,
    load_akshare_ashare_history_with_skips,
    load_hs300_constituents,
    quantile_returns,
    run_backtest,
    summarize_ic,
    write_report,
)
from ashare_quant.data import merge_benchmark_if_needed, normalize_benchmark_ticker
from ashare_quant.research_pipeline import WindowedBacktestSpec, run_backtest_on_date_range
from ashare_quant.sample_split import SampleSplitConfig, slice_for_eval_window, slice_panel_by_date, union_split_bounds


def _apply_split_overrides(cfg: SampleSplitConfig, args: argparse.Namespace) -> SampleSplitConfig:
    return SampleSplitConfig(
        train_start=args.train_start or cfg.train_start,
        train_end=args.train_end or cfg.train_end,
        val_start=args.val_start or cfg.val_start,
        val_end=args.val_end or cfg.val_end,
        test_start=args.test_start or cfg.test_start,
        test_end=args.test_end or cfg.test_end,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real-data CSI 300 multi-factor backtest.")
    parser.add_argument("--start", default="2021-01-01", help="Backtest start date, e.g. 2021-01-01.")
    parser.add_argument("--end", default="2024-12-31", help="Backtest end date, e.g. 2024-12-31.")
    parser.add_argument("--sample-split", choices=["none", "research"], default="none", help="Use train/val/test calendar research defaults (see README).")
    parser.add_argument(
        "--eval-window",
        choices=["train", "val", "test", "train_val", "all"],
        default=None,
        help="When --sample-split research: which segment to backtest (default: test). Ignored when sample-split none.",
    )
    parser.add_argument("--weights-source", choices=["manual", "ic_train"], default="manual")
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--val-start", default=None)
    parser.add_argument("--val-end", default=None)
    parser.add_argument("--test-start", default=None)
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--adjust", default="qfq", help="AKShare adjustment mode: qfq, hfq or empty string.")
    parser.add_argument("--limit", type=int, default=None, help="Optional ticker limit for smoke tests.")
    parser.add_argument("--max-workers", type=int, default=1, help="AKShare download workers; keep 1 for stability.")
    parser.add_argument("--output-dir", default="outputs_hs300_real", help="Directory for report files.")
    parser.add_argument(
        "--rebalance",
        default="ME",
        help="Pandas resample rule for rebalance dates, e.g. ME (month-end), W-FRI (weekly Friday).",
    )
    parser.add_argument("--top-n", type=int, default=40, help="Number of names held at each rebalance.")
    parser.add_argument("--max-weight", type=float, default=0.08, help="Single-name max portfolio weight.")
    parser.add_argument(
        "--weighting",
        choices=["equal", "score", "inverse_volatility"],
        default="equal",
        help="Portfolio weighting method.",
    )
    parser.add_argument("--execution-delay", type=int, default=1, help="Trading bars between signal and execution.")
    parser.add_argument(
        "--benchmark",
        default="equal_weight",
        help=(
            "Benchmark: equal_weight (universe equal-weight), none, "
            "or index ticker merged via AKShare e.g. 000300 / 000300.SH for CSI 300."
        ),
    )
    parser.add_argument("--commission", type=float, default=0.0003, help="Buy-side commission rate.")
    parser.add_argument("--sell-commission", type=float, default=None, help="Sell-side commission rate.")
    parser.add_argument("--stamp-tax", type=float, default=0.001, help="A-share sell-side stamp tax rate.")
    parser.add_argument("--slippage", type=float, default=0.0005, help="One-way slippage rate.")
    parser.add_argument("--min-amount", type=float, default=None, help="Minimum daily amount for candidate stocks.")
    parser.add_argument("--neutralize-industry", action="store_true", help="Neutralize factor industry exposure.")
    parser.add_argument("--neutralize-size", action="store_true", help="Neutralize factor size exposure.")
    parser.add_argument("--forward-period", type=int, default=1, help="Forward return horizon for factor IC.")
    parser.add_argument("--quantiles", type=int, default=5, help="Factor quantile count for layered returns.")
    args = parser.parse_args()

    if args.sample_split == "none" and args.eval_window is not None:
        parser.error("--eval-window is only valid with --sample-split research")

    split_cfg: SampleSplitConfig | None = None
    eval_window = args.eval_window or "all"
    if args.sample_split == "research":
        split_cfg = _apply_split_overrides(SampleSplitConfig.research_defaults_hs300(), args)
        split_cfg.validate()
        if args.eval_window is None:
            eval_window = "test"

    if split_cfg is not None:
        lo, hi = union_split_bounds(split_cfg)
        data_start = min(pd.Timestamp(args.start), lo).strftime("%Y-%m-%d")
        data_end = max(pd.Timestamp(args.end), hi).strftime("%Y-%m-%d")
    else:
        data_start, data_end = args.start, args.end

    tickers = load_hs300_constituents()
    if args.limit is not None:
        tickers = tickers[: args.limit]

    market_raw, skipped = load_akshare_ashare_history_with_skips(
        tickers=tickers,
        start=data_start,
        end=data_end,
        adjust=args.adjust,
        max_workers=args.max_workers,
        show_progress=True,
    )
    market_stocks = clean_universe(market_raw)
    market_bt = merge_benchmark_if_needed(market_stocks, args.benchmark, data_start, data_end)

    if args.benchmark.lower() == "none":
        bench_cfg: str | None = None
    elif args.benchmark.lower() in {"equal_weight", "universe_equal"}:
        bench_cfg = args.benchmark.lower()
    else:
        bench_cfg = normalize_benchmark_ticker(args.benchmark)

    factors = build_factor_panel(market_stocks)
    factor_config = FactorConfig(
        neutralize_industry=args.neutralize_industry,
        neutralize_size=args.neutralize_size,
        weights_source=args.weights_source,
    )
    bt_cfg = BacktestConfig(
        rebalance=args.rebalance,
        top_n=args.top_n,
        max_weight=args.max_weight,
        weighting=args.weighting,
        execution_delay=args.execution_delay,
        benchmark=bench_cfg,
        commission=args.commission,
        sell_commission=args.sell_commission,
        stamp_tax=args.stamp_tax,
        slippage=args.slippage,
        min_amount=args.min_amount,
    )

    if split_cfg is None:
        scores = combine_factors(factors, factor_config)
        result = run_backtest(market_bt, scores, bt_cfg, context=factors)
    else:
        spec = WindowedBacktestSpec(factor_config=factor_config, backtest_config=bt_cfg, forward_period=args.forward_period)
        m_slice = slice_for_eval_window(market_bt, split_cfg, eval_window)
        d0 = m_slice.index.get_level_values("date").min()
        d1 = m_slice.index.get_level_values("date").max()
        ic_train = (
            (pd.Timestamp(split_cfg.train_start), pd.Timestamp(split_cfg.train_end))
            if args.weights_source == "ic_train"
            else None
        )
        result = run_backtest_on_date_range(
            market_bt,
            start=d0,
            end=d1,
            spec=spec,
            factors_full=factors,
            ic_train_slice=ic_train,
        )

    output_dir = write_report(result, Path(args.output_dir))
    if not skipped.empty:
        skipped.to_csv(output_dir / "skipped_tickers.csv", index=False, encoding="utf-8-sig")

    if split_cfg is None:
        future = forward_returns(market_stocks, periods=args.forward_period)
        ic = factor_ic(factors, future)
        scores_for_q = scores
    else:
        m_tr = slice_panel_by_date(market_stocks, split_cfg.train_start, split_cfg.train_end)
        f_tr = slice_panel_by_date(factors, split_cfg.train_start, split_cfg.train_end)
        future_tr = forward_returns(m_tr, periods=args.forward_period)
        ic = factor_ic(f_tr, future_tr)
        m_ev = slice_panel_by_date(market_stocks, d0, d1)
        f_ev = slice_panel_by_date(factors, d0, d1)
        future_ev = forward_returns(m_ev, periods=args.forward_period)
        if args.weights_source == "ic_train":
            assert ic_train is not None
            tr0, tr1 = ic_train
            m_ic = slice_panel_by_date(market_stocks, tr0, tr1)
            f_ic = slice_panel_by_date(factors, tr0, tr1)
            fut_ic = forward_returns(m_ic, periods=args.forward_period)
            scores_for_q = combine_factors(
                f_ev,
                factor_config,
                ic_calibration_factors=f_ic,
                ic_calibration_future=fut_ic,
            )
        else:
            scores_for_q = combine_factors(f_ev, factor_config)

    if not ic.empty:
        ic.to_csv(output_dir / "factor_ic.csv", encoding="utf-8-sig")
        summarize_ic(ic).to_csv(output_dir / "factor_ic_summary.csv", encoding="utf-8-sig")
    factor_correlation(factors).to_csv(output_dir / "factor_correlation.csv", encoding="utf-8-sig")
    layered = quantile_returns(
        scores_for_q,
        future_ev if split_cfg is not None else future,
        quantiles=args.quantiles,
    )
    if not layered.empty:
        layered.to_csv(output_dir / "score_quantile_returns.csv", encoding="utf-8-sig")
    exposures = exposure_summary(result.weights, market_stocks)
    for name, frame in exposures.items():
        frame.to_csv(output_dir / f"{name}.csv", encoding="utf-8-sig")

    print("=== CSI 300 real-data backtest ===")
    print("Data source: AKShare stock_zh_a_daily / stock_zh_a_hist")
    print(f"Universe size: {len(tickers)}")
    print(f"Tickers with data: {market_stocks.index.get_level_values('ticker').nunique()}")
    print(f"Skipped tickers: {len(skipped)}")
    print(f"Weighting: {args.weighting}")
    print(f"Execution delay: T+{args.execution_delay}")
    print(f"Benchmark: {args.benchmark}")
    if split_cfg is not None:
        print(f"Sample split: research | eval-window={eval_window} | weights-source={args.weights_source}")
    print(result.metrics.round(4).to_string())
    print(f"\nReport files written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
