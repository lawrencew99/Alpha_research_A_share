from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ashare_quant.analysis.metrics import performance_summary
from ashare_quant.backtest.portfolio import portfolio_turnover, rebalance_dates, select_top_portfolio


@dataclass(frozen=True)
class BacktestConfig:
    rebalance: str = "M"
    top_n: int = 20
    max_weight: float = 0.1
    fee_rate: float = 0.0003
    slippage: float = 0.0005


def run_backtest(
    panel: pd.DataFrame,
    scores: pd.DataFrame,
    config: BacktestConfig,
) -> dict[str, pd.DataFrame | dict[str, float]]:
    data = panel.sort_values(["date", "symbol"]).copy()
    data["next_return"] = data.groupby("symbol")["close"].pct_change().shift(-1)
    daily_returns = data.pivot(index="date", columns="symbol", values="next_return").fillna(0)
    all_dates = pd.DatetimeIndex(daily_returns.index).sort_values()
    rebalance_index = set(rebalance_dates(pd.Series(all_dates), config.rebalance))

    current = pd.Series(dtype=float)
    positions = []
    equity_rows = []
    equity = 1.0
    cost_rate = config.fee_rate + config.slippage

    for date in all_dates[:-1]:
        turnover = 0.0
        if date in rebalance_index or current.empty:
            new_weights = select_top_portfolio(scores, data, date, config.top_n, config.max_weight)
            turnover = portfolio_turnover(current, new_weights)
            current = new_weights
            for symbol, weight in current.items():
                positions.append({"date": date, "symbol": symbol, "weight": weight})

        ret = float(daily_returns.loc[date, current.index].mul(current, fill_value=0).sum()) if not current.empty else 0.0
        cost = turnover * cost_rate
        net_ret = ret - cost
        equity *= 1 + net_ret
        equity_rows.append(
            {
                "date": date,
                "portfolio_return": net_ret,
                "gross_return": ret,
                "turnover": turnover,
                "cost": cost,
                "equity": equity,
            }
        )

    equity_frame = pd.DataFrame(equity_rows).set_index("date")
    positions_frame = pd.DataFrame(positions)
    metrics = performance_summary(equity_frame)
    metrics["avg_turnover"] = float(equity_frame["turnover"].mean())
    metrics["total_cost"] = float(equity_frame["cost"].sum())
    return {"equity": equity_frame, "positions": positions_frame, "metrics": metrics}
