from __future__ import annotations

from pathlib import Path

from jinja2 import Template

from ashare_quant.visualization.charts import (
    correlation_heatmap,
    drawdown_curve,
    equity_curve,
    ic_series,
    monthly_heatmap,
    quantile_bar,
)


REPORT_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>A 股多因子研究报告</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; color: #222; }
    h1, h2 { color: #16324f; }
    table { border-collapse: collapse; margin: 16px 0; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
    th:first-child, td:first-child { text-align: left; }
    .metric { display: inline-block; margin: 8px 20px 8px 0; }
  </style>
</head>
<body>
  <h1>A 股多因子选股与组合回测研究报告</h1>
  <h2>组合绩效</h2>
  {% for key, value in metrics.items() %}
    <div class="metric"><b>{{ key }}</b>: {{ "%.4f"|format(value) }}</div>
  {% endfor %}

  <h2>净值与风险</h2>
  {{ equity_fig | safe }}
  {{ drawdown_fig | safe }}
  {{ monthly_fig | safe }}

  <h2>因子有效性</h2>
  {{ summary | safe }}
  {{ ic_fig | safe }}
  {{ quantile_fig | safe }}
  {{ corr_fig | safe }}
</body>
</html>
"""


def build_html_report(path: str | Path, analysis: dict, backtest: dict) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    template = Template(REPORT_TEMPLATE)
    html = template.render(
        metrics=backtest["metrics"],
        summary=analysis["summary"].to_html(index=False, float_format=lambda x: f"{x:.4f}"),
        equity_fig=equity_curve(backtest["equity"]).to_html(full_html=False, include_plotlyjs="cdn"),
        drawdown_fig=drawdown_curve(backtest["equity"]).to_html(full_html=False, include_plotlyjs=False),
        monthly_fig=monthly_heatmap(backtest["equity"]).to_html(full_html=False, include_plotlyjs=False),
        ic_fig=ic_series(analysis["ic"]).to_html(full_html=False, include_plotlyjs=False),
        quantile_fig=quantile_bar(analysis["quantile_returns"]).to_html(full_html=False, include_plotlyjs=False),
        corr_fig=correlation_heatmap(analysis["corr"]).to_html(full_html=False, include_plotlyjs=False),
    )
    output.write_text(html, encoding="utf-8")
    return output
