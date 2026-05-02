"""Run the A-share multi-factor research pipeline on synthetic data."""

from __future__ import annotations

from pathlib import Path

from ashare_quant import (
    BacktestConfig,
    FactorConfig,
    build_factor_panel,
    clean_universe,
    combine_factors,
    make_synthetic_ashare_data,
    run_backtest,
    write_report,
)


def main() -> None:
    market = make_synthetic_ashare_data(n_stocks=120, start="2021-01-01", end="2024-12-31")
    market = clean_universe(market)

    factors = build_factor_panel(market)
    scores = combine_factors(
        factors,
        FactorConfig(
            winsor_quantile=0.025,
            neutralize_industry=True,
            neutralize_size=True,
        ),
    )

    result = run_backtest(
        market,
        scores,
        BacktestConfig(
            rebalance="ME",
            top_n=30,
            max_weight=0.08,
            commission=0.0003,
            slippage=0.0005,
        ),
    )
    output_dir = write_report(result, Path("outputs"))

    print("=== A-share multi-factor demo ===")
    print(result.metrics.round(4).to_string())
    print(f"\nReport files written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
