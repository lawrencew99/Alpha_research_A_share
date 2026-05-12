"""Run a single hold-out evaluation on the test window (after train+val selection).

Legacy / 对照基线：与 `examples/run_sensitivity_train_val.py` 配套使用，构成"三段切分一次性 hold-out"
对照流程。主推荐为嵌套滚动 walk-forward
(`examples/run_walk_forward_hs300.py --grid-mode default`)，因其覆盖多段 regime、避免单次抽签。
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
    load_akshare_ashare_history_with_skips,
    load_hs300_constituents,
)
from ashare_quant.deflated_metrics import deflated_summary_from_returns
from ashare_quant.research_pipeline import WindowedBacktestSpec, bench_cfg_from_cli, prepare_market_for_backtest, run_backtest_on_date_range
from ashare_quant.sample_split import SampleSplitConfig, union_split_bounds
from ashare_quant.report import write_report


def _load_champion_from_summary(path: Path, rank: int) -> dict[str, object]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "rank_in_search" not in df.columns:
        raise ValueError("summary CSV must contain rank_in_search (use run_sensitivity_train_val.py)")
    row = df.loc[df["rank_in_search"] == rank].iloc[0]
    return row.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot test-window backtest for a selected configuration.")
    parser.add_argument("--from-summary", type=Path, default=None, help="outputs_sensitivity_summary.csv from train+val grid.")
    parser.add_argument("--rank", type=int, default=1, help="rank_in_search row to take when --from-summary is set.")
    parser.add_argument("--n-trials", type=int, default=None, help="For deflated metrics; default = row count of summary.")
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--rebalance", default=None)
    parser.add_argument("--weighting", default=None)
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--weights-source", choices=["manual", "ic_train"], default="manual")
    parser.add_argument("--forward-period", type=int, default=1)
    parser.add_argument("--adjust", default="qfq")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs_test_holdout")
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--val-start", default=None)
    parser.add_argument("--val-end", default=None)
    parser.add_argument("--test-start", default=None)
    parser.add_argument("--test-end", default=None)
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

    n_trials = args.n_trials
    if args.from_summary is not None:
        summary_path = Path(args.from_summary)
        if not summary_path.is_absolute():
            summary_path = Path(__file__).resolve().parents[1] / summary_path
        champ = _load_champion_from_summary(summary_path, args.rank)
        top_n = int(champ["top_n"])
        rebalance = str(champ["rebalance"])
        weighting = str(champ["weighting"])
        if n_trials is None:
            n_trials = int(pd.read_csv(summary_path, encoding="utf-8-sig").shape[0])
    else:
        if args.top_n is None or args.rebalance is None or args.weighting is None:
            raise SystemExit("Provide --from-summary or explicit --top-n --rebalance --weighting")
        top_n = args.top_n
        rebalance = args.rebalance
        weighting = args.weighting
        if n_trials is None:
            n_trials = 1

    lo, hi = union_split_bounds(scfg)
    start_s, end_s = str(lo.date()), str(hi.date())
    tickers = load_hs300_constituents()
    if args.limit is not None:
        tickers = tickers[: args.limit]
    market_raw, skipped = load_akshare_ashare_history_with_skips(
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

    fc = FactorConfig(neutralize_industry=False, neutralize_size=False, weights_source=args.weights_source)
    bt = BacktestConfig(rebalance=rebalance, top_n=top_n, benchmark=bench, weighting=weighting)
    spec = WindowedBacktestSpec(factor_config=fc, backtest_config=bt, forward_period=args.forward_period)
    t0, t1 = scfg.test_start, scfg.test_end
    assert t0 is not None and t1 is not None
    ic_train = (pd.Timestamp(scfg.train_start), pd.Timestamp(scfg.train_end)) if args.weights_source == "ic_train" else None
    result = run_backtest_on_date_range(
        market_bt,
        start=pd.Timestamp(t0),
        end=pd.Timestamp(t1),
        spec=spec,
        factors_full=factors_full,
        ic_train_slice=ic_train,
    )
    out = write_report(result, Path(args.output_dir))
    if not skipped.empty:
        skipped.to_csv(out / "skipped_tickers.csv", index=False, encoding="utf-8-sig")

    deflated = deflated_summary_from_returns(
        result.daily_returns,
        result.benchmark_returns,
        n_trials=n_trials,
    )
    deflated.to_frame("value").to_csv(out / "deflated_metrics.csv", encoding="utf-8-sig")

    print("=== Test-window hold-out ===")
    print(result.metrics.round(4).to_string())
    print("\nDeflated / multiple-testing diagnostics:")
    print(deflated.round(4).to_string())
    print(f"\nWrote report to {out.resolve()}")


if __name__ == "__main__":
    main()
