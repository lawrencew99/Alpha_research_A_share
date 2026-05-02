from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ashare_quant.analysis.metrics import monthly_returns


def equity_curve(equity: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity["equity"], name="组合净值"))
    fig.update_layout(title="组合净值曲线", xaxis_title="日期", yaxis_title="净值")
    return fig


def drawdown_curve(equity: pd.DataFrame) -> go.Figure:
    drawdown = equity["equity"] / equity["equity"].cummax() - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, fill="tozeroy", name="回撤"))
    fig.update_layout(title="最大回撤曲线", xaxis_title="日期", yaxis_title="回撤")
    return fig


def monthly_heatmap(equity: pd.DataFrame) -> go.Figure:
    monthly = monthly_returns(equity).to_frame("return")
    monthly["year"] = monthly.index.year
    monthly["month"] = monthly.index.month
    pivot = monthly.pivot(index="year", columns="month", values="return")
    fig = px.imshow(pivot, text_auto=".2%", aspect="auto", title="月度收益热力图")
    fig.update_xaxes(title="月份")
    fig.update_yaxes(title="年份")
    return fig


def ic_series(ic: pd.DataFrame) -> go.Figure:
    if ic.empty:
        return go.Figure().update_layout(title="IC 序列")
    fig = px.line(ic, x="date", y="rank_ic", color="factor", title="RankIC 序列")
    return fig


def quantile_bar(quantile_returns: pd.DataFrame) -> go.Figure:
    if quantile_returns.empty:
        return go.Figure().update_layout(title="分层收益")
    agg = quantile_returns.groupby(["factor", "quantile"], as_index=False)["return"].mean()
    return px.bar(agg, x="quantile", y="return", color="factor", barmode="group", title="单因子分层收益")


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    return px.imshow(corr, text_auto=".2f", aspect="auto", title="因子相关性矩阵")


def save_all_figures(output_dir: str | Path, analysis: dict, backtest: dict) -> dict[str, Path]:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    figures = {
        "equity": equity_curve(backtest["equity"]),
        "drawdown": drawdown_curve(backtest["equity"]),
        "monthly_heatmap": monthly_heatmap(backtest["equity"]),
        "ic": ic_series(analysis["ic"]),
        "quantile_returns": quantile_bar(analysis["quantile_returns"]),
        "factor_corr": correlation_heatmap(analysis["corr"]),
    }
    saved = {}
    for name, fig in figures.items():
        file_path = path / f"{name}.html"
        fig.write_html(file_path)
        saved[name] = file_path
    return saved
