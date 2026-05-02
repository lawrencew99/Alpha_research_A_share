# A Share Multi-Factor Alpha Research and Portfolio Backtesting System

基于 A 股场景的多因子 Alpha 研究与组合回测 Python 工程包，覆盖真实行情读取、复权处理、因子计算、截面标准化、中性化、多因子合成、交易成本回测和报告输出。

当前主流程使用沪深 300 成分股和 AKShare 真实历史日线行情。

## 快速开始

```powershell
cd "C:\Users\w4727\Desktop\2027应届量化交易方向\a_share_quant"
pip install -e ".[data]"
python examples\run_hs300_demo.py --start 2021-01-01 --end 2024-12-31
```

运行后会在 `outputs_hs300_real/` 下生成回测净值曲线、月度收益热力图、调仓权重和指标 CSV。

## 工程结构

```text
a_share_quant/
  pyproject.toml
  README.md
  data/
    hs300_constituents.csv
    raw/akshare_daily/
  examples/
    fetch_hs300_universe.py
    run_hs300_demo.py
  src/ashare_quant/
    __init__.py
    backtest.py
    config.py
    data.py
    factors.py
    portfolio.py
    report.py
    universe.py
```

## 数据来源

- 成分股与权重：AKShare 中证指数成分权重接口，缓存为 `data/hs300_constituents.csv`
- 历史行情：AKShare `stock_zh_a_daily` / `stock_zh_a_hist` 个股日线接口
- 复权方式：默认前复权 `qfq`
- 行情缓存：`data/raw/akshare_daily/`
- 回测输出：`outputs_hs300_real/`
- 跳过股票记录：`skipped_tickers.csv`，用于记录回测区间内无可用日线的股票及原因

当前 `data/raw/akshare_daily/` 已缓存 2021-2024 年沪深 300 全部 300 只成分股的真实日线 CSV。再次运行同一区间会优先读取缓存，不会重复逐只下载。

如果成分股在回测区间内尚未上市或无可用行情，程序会跳过该股票并记录到输出目录的 `skipped_tickers.csv`，不会用模拟价格或填充数据污染回测。

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

1. 用 `load_hs300_constituents()` 读取沪深 300 股票池。
2. 用 `load_akshare_ashare_history()` 拉取或读取缓存中的真实日线行情。
3. 用 `clean_universe()` 过滤停牌、涨跌停和缺失样本。
4. 用 `build_factor_panel()` 生成动量、反转、波动率、流动性、估值和质量因子。
5. 用 `combine_factors()` 做 winsorize、z-score 和多因子打分。
6. 用 `run_backtest()` 加入调仓频率、持仓上限、手续费和滑点约束，输出净值、持仓和绩效指标。
7. 用 `write_report()` 输出指标表、净值曲线和月度收益热力图。

## 沪深 300 股票池

安装数据接口依赖后可拉取最新沪深 300 成分股：

```powershell
pip install -e ".[data]"
python examples\fetch_hs300_universe.py
```

成分股会保存到 `data/hs300_constituents.csv`，字段为 `ticker,name,weight,as_of_date`，代码格式统一为 `600519.SH` / `000001.SZ`。

运行真实沪深 300 回测：

```powershell
python examples\run_hs300_demo.py --start 2021-01-01 --end 2024-12-31
```

可先用小样本验证：

```powershell
python examples\run_hs300_demo.py --limit 10 --start 2023-01-01 --end 2023-12-31 --output-dir outputs_hs300_real_sample
```

当前环境下 AKShare 依赖不适合多线程并发，建议保持默认单线程：

```powershell
python examples\run_hs300_demo.py --max-workers 1
```

如果当前虚拟环境缺少 AKShare，请在项目根目录运行：

```powershell
pip install -e ".[data]"
```
