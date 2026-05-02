import pandas as pd

from ashare_quant.analysis.metrics import max_drawdown, performance_summary


def test_max_drawdown() -> None:
    equity = pd.Series([1.0, 1.2, 0.9, 1.1])
    assert round(max_drawdown(equity), 4) == -0.25


def test_performance_summary_has_core_metrics() -> None:
    equity = pd.DataFrame(
        {
            "portfolio_return": [0.01, -0.02, 0.03],
            "equity": [1.01, 0.9898, 1.019494],
        }
    )
    summary = performance_summary(equity)
    assert {"annual_return", "annual_volatility", "sharpe", "max_drawdown", "win_rate"} <= set(summary)
