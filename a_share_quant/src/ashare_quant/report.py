"""Report writers for backtest results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ashare_quant.backtest import BacktestResult


def write_report(result: BacktestResult, output_dir: str | Path = "outputs") -> Path:
    """Write metrics and figures to disk."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result.metrics.to_frame("value").to_csv(output_path / "metrics.csv", encoding="utf-8-sig")
    result.equity_curve.to_csv(output_path / "equity_curve.csv", encoding="utf-8-sig")
    result.turnover.to_csv(output_path / "turnover.csv", encoding="utf-8-sig")
    if result.trade_log is not None and not result.trade_log.empty:
        result.trade_log.to_csv(output_path / "trade_log.csv", encoding="utf-8-sig")
    if not result.weights.empty:
        result.weights.to_csv(output_path / "rebalance_weights.csv", encoding="utf-8-sig")
    if result.benchmark_curve is not None:
        result.benchmark_curve.to_csv(output_path / "benchmark_curve.csv", encoding="utf-8-sig")
    if result.excess_returns is not None:
        result.excess_returns.to_csv(output_path / "excess_returns.csv", encoding="utf-8-sig")

    _plot_equity_curve(result.equity_curve, output_path / "equity_curve.png", result.benchmark_curve)
    _plot_drawdown(result.equity_curve, output_path / "drawdown.png", result.benchmark_curve)
    if result.excess_returns is not None:
        _plot_excess_curve(result.excess_returns, output_path / "excess_curve.png")
    _plot_monthly_heatmap(result.monthly_returns, output_path / "monthly_returns_heatmap.png")
    return output_path


def _plot_equity_curve(equity_curve: pd.Series, path: Path, benchmark_curve: pd.Series | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    net_value = equity_curve / equity_curve.iloc[0]
    net_value.plot(ax=ax, color="#1f77b4", linewidth=1.8, label="Strategy")
    if benchmark_curve is not None:
        benchmark_net_value = benchmark_curve.reindex(equity_curve.index) / benchmark_curve.dropna().iloc[0]
        benchmark_net_value.plot(ax=ax, color="#ff7f0e", linewidth=1.5, label="Benchmark")
    ax.set_title("Strategy Net Value")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_drawdown(equity_curve: pd.Series, path: Path, benchmark_curve: pd.Series | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    drawdown = equity_curve / equity_curve.cummax() - 1
    drawdown.plot(ax=ax, color="#1f77b4", linewidth=1.5, label="Strategy")
    if benchmark_curve is not None:
        benchmark = benchmark_curve.reindex(equity_curve.index).dropna()
        benchmark_drawdown = benchmark / benchmark.cummax() - 1
        benchmark_drawdown.plot(ax=ax, color="#ff7f0e", linewidth=1.3, label="Benchmark")
    ax.set_title("Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_excess_curve(excess_returns: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    excess_net_value = (1 + excess_returns.fillna(0)).cumprod()
    excess_net_value.plot(ax=ax, color="#2ca02c", linewidth=1.5)
    ax.set_title("Excess Net Value")
    ax.set_xlabel("Date")
    ax.set_ylabel("Relative Net Value")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_monthly_heatmap(monthly_returns: pd.Series, path: Path) -> None:
    if monthly_returns.empty:
        return

    heatmap = monthly_returns.to_frame("return")
    heatmap["year"] = heatmap.index.year
    heatmap["month"] = heatmap.index.month
    matrix = heatmap.pivot(index="year", columns="month", values="return").reindex(columns=range(1, 13))

    fig, ax = plt.subplots(figsize=(11, 4))
    image = ax.imshow(matrix.fillna(0), cmap="RdYlGn", aspect="auto", vmin=-0.12, vmax=0.12)
    ax.set_title("Monthly Returns")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_xticks(range(12), labels=[str(i) for i in range(1, 13)])
    ax.set_yticks(range(len(matrix.index)), labels=[str(i) for i in matrix.index])

    for y_pos, year in enumerate(matrix.index):
        for x_pos, month in enumerate(matrix.columns):
            value = matrix.loc[year, month]
            if pd.notna(value):
                ax.text(x_pos, y_pos, f"{value:.1%}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
