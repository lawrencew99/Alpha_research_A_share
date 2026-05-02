from __future__ import annotations

import argparse

import streamlit as st

from ashare_quant.config import load_config
from ashare_quant.pipeline import run_research
from ashare_quant.visualization.charts import (
    correlation_heatmap,
    drawdown_curve,
    equity_curve,
    ic_series,
    monthly_heatmap,
    quantile_bar,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    st.set_page_config(page_title="A 股多因子看板", layout="wide")
    st.title("A 股多因子选股与组合回测看板")
    st.caption("数据源：AKShare；网络不可用时可切换模拟数据。")

    demo = st.sidebar.checkbox("使用模拟数据", value=True)
    run = st.sidebar.button("运行研究流程")
    if run or "result" not in st.session_state:
        with st.spinner("正在计算因子、回测并生成图表..."):
            st.session_state["result"] = run_research(cfg, demo=demo)

    result = st.session_state["result"]
    analysis = result["analysis"]
    backtest = result["backtest"]
    metrics = backtest["metrics"]

    cols = st.columns(4)
    for col, key in zip(cols, ["annual_return", "annual_volatility", "sharpe", "max_drawdown"]):
        col.metric(key, f"{metrics[key]:.2%}" if key != "sharpe" else f"{metrics[key]:.2f}")

    st.plotly_chart(equity_curve(backtest["equity"]), use_container_width=True)
    st.plotly_chart(drawdown_curve(backtest["equity"]), use_container_width=True)
    st.plotly_chart(monthly_heatmap(backtest["equity"]), use_container_width=True)

    st.subheader("因子评价")
    st.dataframe(analysis["summary"], use_container_width=True)
    st.plotly_chart(ic_series(analysis["ic"]), use_container_width=True)
    st.plotly_chart(quantile_bar(analysis["quantile_returns"]), use_container_width=True)
    st.plotly_chart(correlation_heatmap(analysis["corr"]), use_container_width=True)


if __name__ == "__main__":
    main()
