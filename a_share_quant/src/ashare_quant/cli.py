"""Command line entry points."""

from __future__ import annotations

from pathlib import Path

from ashare_quant.backtest import run_backtest
from ashare_quant.config import BacktestConfig, FactorConfig
from ashare_quant.data import clean_universe, make_synthetic_ashare_data
from ashare_quant.factors import build_factor_panel, combine_factors
from ashare_quant.report import write_report


def main() -> None:
    """Run a full synthetic-data research pipeline."""

    market = clean_universe(make_synthetic_ashare_data())
    factors = build_factor_panel(market)
    scores = combine_factors(factors, FactorConfig())
    result = run_backtest(market, scores, BacktestConfig(top_n=30, max_weight=0.08))
    output_dir = write_report(result, Path("outputs"))

    print("Backtest finished.")
    print(f"Output directory: {output_dir.resolve()}")
    print(result.metrics.round(4).to_string())


if __name__ == "__main__":
    main()
