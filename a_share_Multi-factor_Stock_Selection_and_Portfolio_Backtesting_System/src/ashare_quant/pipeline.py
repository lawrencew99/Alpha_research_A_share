from __future__ import annotations

from pathlib import Path

import pandas as pd

from ashare_quant.analysis.factor_analysis import analyze_factors
from ashare_quant.backtest.engine import BacktestConfig, run_backtest
from ashare_quant.config import Config
from ashare_quant.data.akshare_client import AKShareClient
from ashare_quant.data.cleaning import prepare_panel
from ashare_quant.factors.library import compute_factor_panel
from ashare_quant.factors.multi_factor import build_multi_factor_score
from ashare_quant.reporting.html import build_html_report
from ashare_quant.visualization.charts import save_all_figures


def ensure_dirs(config: Config) -> None:
    for path in [
        config.data_dir / "raw",
        config.data_dir / "processed",
        config.output_dir / "figures",
        config.output_dir / "reports",
        config.output_dir / "backtests",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def fetch_data(config: Config, demo: bool = False) -> dict[str, pd.DataFrame]:
    ensure_dirs(config)
    client = AKShareClient(config.data_dir / "raw")
    symbols = config.get("universe.symbols", [])
    start = config.get("universe.start_date")
    end = config.get("universe.end_date")
    adjust = config.get("data.adjust", "qfq")
    fallback = bool(config.get("data.demo_if_akshare_fails", True))
    return client.fetch_universe(
        symbols=symbols,
        start_date=start,
        end_date=end,
        adjust=adjust,
        demo=demo,
        fallback_demo=fallback,
    )


def run_factor_pipeline(config: Config, demo: bool = False) -> dict[str, pd.DataFrame]:
    raw = fetch_data(config, demo=demo)
    panel = prepare_panel(raw["prices"], raw.get("fundamentals"), raw.get("industries"))
    factors = compute_factor_panel(panel)
    factors = build_multi_factor_score(
        factors=factors,
        panel=panel,
        method=config.get("portfolio.method", "ic_weighted"),
        forward_days=int(config.get("analysis.forward_return_days", 20)),
    )
    processed_dir = config.data_dir / "processed"
    panel.to_parquet(processed_dir / "panel.parquet")
    factors.to_parquet(processed_dir / "factors.parquet")
    return {"panel": panel, "factors": factors}


def run_research(config: Config, demo: bool = False) -> dict[str, pd.DataFrame | dict]:
    ensure_dirs(config)
    data = run_factor_pipeline(config, demo=demo)
    panel = data["panel"]
    factors = data["factors"]
    analysis = analyze_factors(
        factors=factors,
        panel=panel,
        quantiles=int(config.get("analysis.quantiles", 5)),
        forward_days=int(config.get("analysis.forward_return_days", 20)),
    )
    backtest = run_backtest(
        panel=panel,
        scores=factors[["date", "symbol", "score"]],
        config=BacktestConfig(
            rebalance=config.get("portfolio.rebalance", "M"),
            top_n=int(config.get("portfolio.top_n", 20)),
            max_weight=float(config.get("portfolio.max_weight", 0.1)),
            fee_rate=float(config.get("portfolio.fee_rate", 0.0003)),
            slippage=float(config.get("portfolio.slippage", 0.0005)),
        ),
    )
    out = config.output_dir / "backtests"
    backtest["equity"].to_parquet(out / "equity.parquet")
    backtest["positions"].to_parquet(out / "positions.parquet")
    pd.Series(backtest["metrics"]).to_frame("value").to_csv(out / "metrics.csv")
    save_all_figures(config.output_dir / "figures", analysis, backtest)
    build_html_report(config.output_dir / "reports" / "research_report.html", analysis, backtest)
    return {"panel": panel, "factors": factors, "analysis": analysis, "backtest": backtest}


def load_or_run(config: Config, demo: bool = False) -> dict[str, pd.DataFrame | dict]:
    panel_path = config.data_dir / "processed" / "panel.parquet"
    factor_path = config.data_dir / "processed" / "factors.parquet"
    if panel_path.exists() and factor_path.exists():
        panel = pd.read_parquet(panel_path)
        factors = pd.read_parquet(factor_path)
        return {"panel": panel, "factors": factors}
    return run_factor_pipeline(config, demo=demo)


def project_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
