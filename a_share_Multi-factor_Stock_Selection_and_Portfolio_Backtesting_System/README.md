# A Share Quant

基于 A 股场景的多因子选股与组合回测 Python 工程包，覆盖简历描述中的行情读取、复权处理、因子计算、截面标准化、中性化、多因子合成、交易成本回测和报告输出。

## 快速开始

```powershell
cd "C:\Users\w4727\Desktop\2027应届量化交易方向\a_share_quant"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python examples\run_demo.py
```

运行后会在 `outputs/` 下生成回测净值曲线、月度收益热力图和指标 CSV。

## 工程结构

```text
a_share_quant/
  pyproject.toml
  README.md
  examples/run_demo.py
  src/ashare_quant/
    __init__.py
    backtest.py
    cli.py
    config.py
    data.py
    factors.py
    portfolio.py
    report.py
```

## 输入数据格式

真实行情数据可用 CSV 读入，推荐字段：

- `date`: 交易日期
- `ticker`: 股票代码
- `open`, `high`, `low`, `close`: 未复权价格
- `volume`, `amount`: 成交量和成交额
- `adj_factor`: 复权因子
- `industry`: 行业分类
- `market_cap`: 总市值或流通市值
- `pe`, `pb`, `roe`: 估值与质量指标
- `is_suspended`, `is_limit_up`, `is_limit_down`: 停牌和涨跌停标记

## 核心流程

1. 用 `load_ohlcv_csv()` 或 `make_synthetic_ashare_data()` 得到 MultiIndex 行情面板。
2. 用 `clean_universe()` 过滤停牌、涨跌停和缺失样本。
3. 用 `build_factor_panel()` 生成动量、反转、波动率、流动性、估值和质量因子。
4. 用 `combine_factors()` 做 winsorize、z-score、行业/市值中性化和多因子打分。
5. 用 `run_backtest()` 加入调仓频率、持仓上限、手续费和滑点约束，输出净值、持仓和绩效指标。
6. 用 `write_report()` 输出指标表和图表。

该包默认提供合成数据 demo，便于无外部数据时验证完整研究链路；接入 Tushare、聚宽、米筐或本地行情库时，只需把数据整理成上述字段即可。
