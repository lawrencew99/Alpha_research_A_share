"""Grid search on train+validation only; writes ``outputs_sensitivity_summary.csv`` with trial metadata.

Legacy / 对照基线：主推荐流程为嵌套滚动 walk-forward
(`examples/run_walk_forward_hs300.py --grid-mode default`)，每折独立选参，避免「一次切片」
带来的 regime 偏差。本脚本只在 train+val 上选参、再交给 `eval_selected_on_test.py` 在 test 上
一次性记账，仅作为对照保留。
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from ashare_quant import (
    BacktestConfig,
    FactorConfig,
    build_factor_panel,
    clean_universe,
    load_akshare_ashare_history_with_skips,
    load_hs300_constituents,
)
from ashare_quant.data import merge_benchmark_if_needed
from ashare_quant.research_pipeline import WindowedBacktestSpec, bench_cfg_from_cli, prepare_market_for_backtest, run_backtest_on_date_range
from ashare_quant.sample_split import SampleSplitConfig, union_split_bounds


def _grid() -> itertools.product:
    top_ns = (20, 25, 30, 40)
    rebalances = ("ME",)
    weightings = ("equal", "score")
    return itertools.product(top_ns, rebalances, weightings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyper-parameter grid on train+val only (no test leakage).")
    parser.add_argument("--adjust", default="qfq")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--val-start", default=None)
    parser.add_argument("--val-end", default=None)
    parser.add_argument("--test-start", default=None)
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--forward-period", type=int, default=1)
    parser.add_argument("--output-csv", default="outputs_sensitivity_summary.csv")
    args = parser.parse_args()

    scfg = SampleSplitConfig.research_defaults_hs300()
    scfg = SampleSplitConfig(
        train_start=args.train_start or scfg.train_start,
        train_end=args.train_end or scfg.train_end,
        val_start=args.val_start or scfg.val_start,
        val_end=args.val_end or scfg.val_end,
        test_start=args.test_start or scfg.test_start,
        test_end=args.test_end or scfg.test_end,
    )
    scfg.validate()
    lo, hi = union_split_bounds(scfg)
    start_s, end_s = str(lo.date()), str(hi.date())

    tickers = load_hs300_constituents()
    if args.limit is not None:
        tickers = tickers[: args.limit]
    market_raw, _ = load_akshare_ashare_history_with_skips(
        tickers=tickers,
        start=start_s,
        end=end_s,
        adjust=args.adjust,
        max_workers=args.max_workers,
        show_progress=True,
    )
    market_stocks = clean_universe(market_raw)
    bench = bench_cfg_from_cli(args.benchmark)
    market_bt = prepare_market_for_backtest(market_stocks, args.benchmark, start_s, end_s)
    factors_full = build_factor_panel(market_stocks)

    tv0 = pd.Timestamp(scfg.train_start)
    tv1 = pd.Timestamp(scfg.val_end) if scfg.val_end else pd.Timestamp(scfg.train_end)
    rows: list[dict[str, object]] = []
    for top_n, rebalance, weighting in _grid():
        label = f"top{top_n}_{rebalance}_{weighting}"
        bt = BacktestConfig(
            rebalance=rebalance,
            top_n=top_n,
            benchmark=bench,
            weighting=weighting,
        )
        fc = FactorConfig(neutralize_industry=False, neutralize_size=False)
        spec = WindowedBacktestSpec(factor_config=fc, backtest_config=bt, forward_period=args.forward_period)
        res = run_backtest_on_date_range(market_bt, start=tv0, end=tv1, spec=spec, factors_full=factors_full)
        row = {"run_label": label, "top_n": top_n, "rebalance": rebalance, "weighting": weighting}
        row.update(res.metrics.to_dict())
        rows.append(row)

    df = pd.DataFrame(rows)
    n = len(df)
    df["window"] = "train_val"
    df["n_configs_tried"] = n
    rank_key = "information_ratio" if df["information_ratio"].notna().any() else "sharpe"
    df["rank_in_search"] = df[rank_key].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("rank_in_search")

    root = Path(__file__).resolve().parents[1]
    out = root / args.output_csv
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.round(4).to_string(index=False))
    print(f"\nWrote {out.resolve()}")


if __name__ == "__main__":
    main()
