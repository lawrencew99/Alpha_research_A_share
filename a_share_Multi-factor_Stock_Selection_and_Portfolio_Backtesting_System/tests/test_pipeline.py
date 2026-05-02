from ashare_quant.backtest.engine import BacktestConfig, run_backtest
from ashare_quant.config import Config
from ashare_quant.data.cleaning import prepare_panel
from ashare_quant.data.demo import make_demo_data
from ashare_quant.factors.library import compute_factor_panel
from ashare_quant.factors.multi_factor import build_multi_factor_score
from ashare_quant.pipeline import run_research


def test_demo_pipeline_runs_end_to_end() -> None:
    raw = make_demo_data(["000001", "000002", "600000", "600519", "000858"], "2022-01-01", "2022-06-30")
    panel = prepare_panel(raw["prices"], raw["fundamentals"], raw["industries"])
    factors = compute_factor_panel(panel)
    scores = build_multi_factor_score(factors, panel, method="equal", forward_days=5)
    result = run_backtest(panel, scores[["date", "symbol", "score"]], BacktestConfig(top_n=3))

    assert not panel.empty
    assert "score" in scores
    assert not result["equity"].empty
    assert result["metrics"]["total_return"] > -1


def test_run_research_with_demo_config(tmp_path) -> None:
    cfg = Config(
        values={
            "project": {"data_dir": str(tmp_path / "data"), "output_dir": str(tmp_path / "outputs")},
            "universe": {
                "symbols": ["000001", "000002", "600000", "600519", "000858"],
                "start_date": "2022-01-01",
                "end_date": "2022-06-30",
            },
            "data": {"adjust": "qfq", "demo_if_akshare_fails": True},
            "portfolio": {"method": "equal", "rebalance": "M", "top_n": 3, "max_weight": 0.4},
            "analysis": {"quantiles": 3, "forward_return_days": 5},
        },
        path=tmp_path / "config.yaml",
    )
    result = run_research(cfg, demo=True)
    assert not result["backtest"]["equity"].empty
    assert (tmp_path / "outputs" / "reports" / "research_report.html").exists()
