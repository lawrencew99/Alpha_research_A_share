"""Run a CSI 300 backtest on real AKShare daily history."""

from __future__ import annotations

import argparse
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real-data CSI 300 multi-factor backtest.")
    parser.add_argument("--start", default="2021-01-01", help="Backtest start date, e.g. 2021-01-01.")
    parser.add_argument("--end", default="2024-12-31", help="Backtest end date, e.g. 2024-12-31.")
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
        help="Benchmark source: equal_weight, none, or a ticker present in the market panel.",
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

    tickers = load_hs300_constituents()
    if args.limit is not None:
        tickers = tickers[: args.limit]

    market, skipped = load_akshare_ashare_history_with_skips(
        tickers=tickers,
        start=args.start,
        end=args.end,
        adjust=args.adjust,
        max_workers=args.max_workers,
        show_progress=True,
    )
    market = clean_universe(market)

    factors = build_factor_panel(market)
    factor_config = FactorConfig(neutralize_industry=args.neutralize_industry, neutralize_size=args.neutralize_size)
    scores = combine_factors(factors, factor_config)
    result = run_backtest(
        market,
        scores,
        BacktestConfig(
            rebalance=args.rebalance,
            top_n=args.top_n,
            max_weight=args.max_weight,
            weighting=args.weighting,
            execution_delay=args.execution_delay,
            benchmark=None if args.benchmark.lower() == "none" else args.benchmark,
            commission=args.commission,
            sell_commission=args.sell_commission,
            stamp_tax=args.stamp_tax,
            slippage=args.slippage,
            min_amount=args.min_amount,
        ),
        context=factors,
    )
    output_dir = write_report(result, Path(args.output_dir))
    if not skipped.empty:
        skipped.to_csv(output_dir / "skipped_tickers.csv", index=False, encoding="utf-8-sig")

    future = forward_returns(market, periods=args.forward_period)
    ic = factor_ic(factors, future)
    if not ic.empty:
        ic.to_csv(output_dir / "factor_ic.csv", encoding="utf-8-sig")
        summarize_ic(ic).to_csv(output_dir / "factor_ic_summary.csv", encoding="utf-8-sig")
    factor_correlation(factors).to_csv(output_dir / "factor_correlation.csv", encoding="utf-8-sig")
    layered = quantile_returns(scores, future, quantiles=args.quantiles)
    if not layered.empty:
        layered.to_csv(output_dir / "score_quantile_returns.csv", encoding="utf-8-sig")
    exposures = exposure_summary(result.weights, market)
    for name, frame in exposures.items():
        frame.to_csv(output_dir / f"{name}.csv", encoding="utf-8-sig")

    print("=== CSI 300 real-data backtest ===")
    print("Data source: AKShare stock_zh_a_daily / stock_zh_a_hist")
    print(f"Universe size: {len(tickers)}")
    print(f"Tickers with data: {market.index.get_level_values('ticker').nunique()}")
    print(f"Skipped tickers: {len(skipped)}")
    print(f"Weighting: {args.weighting}")
    print(f"Execution delay: T+{args.execution_delay}")
    print(f"Benchmark: {args.benchmark}")
    print(result.metrics.round(4).to_string())
    print(f"\nReport files written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
