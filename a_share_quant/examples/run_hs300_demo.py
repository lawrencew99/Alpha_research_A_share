"""Run the demo on the CSI 300 constituent ticker universe."""

from __future__ import annotations

from pathlib import Path

from ashare_quant import (
    BacktestConfig,
    FactorConfig,
    build_factor_panel,
    clean_universe,
    combine_factors,
    load_hs300_constituents,
    make_synthetic_ashare_data,
    run_backtest,
    write_report,
)


def main() -> None:
    tickers = load_hs300_constituents()
    market = make_synthetic_ashare_data(
        tickers=tickers,
        start="2021-01-01",
        end="2024-12-31",
    )
    market = clean_universe(market)

    factors = build_factor_panel(market)
    scores = combine_factors(factors, FactorConfig())
    result = run_backtest(market, scores, BacktestConfig(top_n=30, max_weight=0.08))
    output_dir = write_report(result, Path("outputs_hs300"))

    print("=== CSI 300 universe demo ===")
    print(f"Universe size: {len(tickers)}")
    print(result.metrics.round(4).to_string())
    print(f"\nReport files written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
