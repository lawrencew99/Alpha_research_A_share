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
    load_akshare_ashare_history_with_skips,
    load_hs300_constituents,
    run_backtest,
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
    scores = combine_factors(factors, FactorConfig(neutralize_industry=False, neutralize_size=False))
    result = run_backtest(market, scores, BacktestConfig(top_n=30, max_weight=0.08))
    output_dir = write_report(result, Path(args.output_dir))
    if not skipped.empty:
        skipped.to_csv(output_dir / "skipped_tickers.csv", index=False, encoding="utf-8-sig")

    print("=== CSI 300 real-data backtest ===")
    print("Data source: AKShare stock_zh_a_daily / stock_zh_a_hist")
    print(f"Universe size: {len(tickers)}")
    print(f"Tickers with data: {market.index.get_level_values('ticker').nunique()}")
    print(f"Skipped tickers: {len(skipped)}")
    print(result.metrics.round(4).to_string())
    print(f"\nReport files written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
