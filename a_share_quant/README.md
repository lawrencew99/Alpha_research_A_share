# A Share Multi-Factor Alpha Research and Portfolio Backtesting System

基于 A 股场景的多因子 Alpha 研究与组合回测 Python 工程包，覆盖真实行情读取、复权处理、因子计算、截面标准化、中性化、多因子合成、交易成本回测和报告输出。

当前主流程使用沪深 300 成分股和 AKShare 真实历史日线行情。

## 快速开始

```powershell
cd "C:\Users\w4727\Desktop\2027应届量化交易方向\a_share_quant"
pip install -e ".[data]"
python examples\run_hs300_demo.py --start 2021-01-01 --end 2024-12-31
```

运行后会在 `outputs_hs300_real/` 下生成回测净值曲线、基准对比、回撤图、月度收益热力图、调仓权重、交易日志、因子 IC、分层收益和指标 CSV。

## 工程结构

```text
a_share_quant/
  pyproject.toml
  README.md
  data/
    hs300_constituents.csv
    raw/akshare_daily/
  examples/
    aggregate_sensitivity.py
    eval_selected_on_test.py
    fetch_hs300_universe.py
    run_hs300_demo.py
    run_sensitivity_train_val.py
    run_walk_forward_hs300.py
  src/ashare_quant/
    __init__.py
    analysis.py
    backtest.py
    config.py
    data.py
    deflated_metrics.py
    factors.py
    ic_weights.py
    portfolio.py
    report.py
    research_pipeline.py
    sample_split.py
    universe.py
    walk_forward.py
  tests/
    ...
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
- `can_buy`, `can_sell`: 是否允许买入/卖出（涨停不可买、跌停不可卖；停牌均不可交易）；可由 `refresh_trade_flags()` 从行情重算

## 核心流程

1. 用 `load_hs300_constituents()` 读取沪深 300 股票池。
2. 用 `load_akshare_ashare_history()` 拉取或读取缓存中的真实日线行情。
3. 用 `clean_universe()` 过滤停牌与无成交样本；**保留**涨跌停日行情以免收益被截断，执行层用 `can_buy` / `can_sell`（由 `limit_pct_for_ticker` 按板块判断涨跌停）约束买卖。
4. 用 `build_factor_panel()` 生成动量、反转、波动率、流动性、估值和质量因子。
5. 用 `combine_factors()` 做 winsorize、z-score 和多因子打分。
6. 用 `run_backtest()` 加入调仓频率、T+1 执行、持仓上限、买卖非对称费用、滑点和基准约束，输出净值、持仓和绩效指标。
7. 用 `factor_ic()`、`quantile_returns()` 和 `factor_correlation()` 检查因子 IC、Rank IC、分层收益和因子冗余。
8. 用 `write_report()` 输出指标表、净值曲线、基准对比、回撤图和月度收益热力图。

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

常用研究参数：

```powershell
python examples\run_hs300_demo.py `
  --start 2021-01-01 `
  --end 2024-12-31 `
  --top-n 40 `
  --rebalance ME `
  --weighting equal `
  --execution-delay 1 `
  --benchmark equal_weight `
  --commission 0.0003 `
  --stamp-tax 0.001 `
  --slippage 0.0005
```

可选权重方式：

- `equal`: 入选股票等权，并应用单票权重上限
- `score`: 按正向因子分数加权
- `inverse_volatility`: 按过去波动率倒数加权

可选研究诊断输出：

- `factor_ic.csv`: 单因子 Rank IC 时间序列
- `factor_ic_summary.csv`: IC 均值、标准差、ICIR、胜率和 t 统计量
- `score_quantile_returns.csv`: 合成分数分层收益和 Top-Bottom 收益
- `factor_correlation.csv`: 因子相关性矩阵
- `concentration.csv`: 持仓数量、Top 权重和 Herfindahl 集中度

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

## 研究假设和边界

### 数据字段契约

AKShare 日线主流程稳定提供的是价量字段：

- `date`, `ticker`
- `open`, `high`, `low`, `close`
- `volume`, `amount`
- `adj_factor`, `adj_open`, `adj_high`, `adj_low`, `adj_close`
- `is_suspended`, `is_limit_up`, `is_limit_down`

`pb`、`roe`、`industry`、`market_cap` 属于可选增强字段。只有输入数据中存在这些字段时，估值、质量、行业中性和市值中性才会实际生效。默认 AKShare 日线回测主要是价量因子研究，不应被解读为已经完整覆盖基本面多因子。

### 成交假设

默认 `--execution-delay 1`，含义是用 T 日收盘后可得的信息生成调仓目标，并在下一根交易 bar 执行。回测仍是日频近似模型，不模拟逐笔成交、盘口冲击和真实挂单排队。

涨跌停与撮合：`run_backtest()` 在调仓日对不可买入标的不上调权重（加仓需满足 `can_buy`）；不可卖出标的暂不减仓（保留持仓权重直至可卖）。本地 CSV 缓存会用 `refresh_trade_flags()` 按代码前缀重算涨跌停（主板 ±10%、科创/创业板 ±20%、北交所 ±30%），避免旧缓存里「一律 ±10%」对创业板、科创板等造成的误判和收益丢失。

交易成本默认包括：

- 买入佣金：`--commission`
- 卖出佣金：`--sell-commission`，未设置时等于买入佣金
- 卖出印花税：`--stamp-tax`
- 双边滑点：`--slippage`

### 股票池偏差

当前 `data/hs300_constituents.csv` 是沪深 300 成分股快照。用于历史回测时可能存在幸存者偏差，严格研究应接入历史调样日成分或 point-in-time 指数权重。当前结果适合策略原型验证和研究流程展示，不应直接作为实盘收益承诺。

### 基准说明

默认 `--benchmark equal_weight` 使用回测股票池内所有可用股票的等权收益作为基准。它不是官方沪深 300 全收益指数。若要做正式研究，应接入沪深 300 全收益指数或其他 point-in-time 基准序列。

### 默认因子权重（价量）

未接入 `pb`/`roe` 时，合成 Score 仅使用价量因子。默认 `FactorConfig` 将 **反转权重设为 0**（与中期动量冲突时关闭）、**流动性为负**（偏拥挤因子时常反向）、波动率为负（偏好低波）。请仍以输出目录中的 `factor_ic_summary.csv` 为准微调权重。

### 过拟合与样本外评估

**主推荐流程：嵌套滚动 walk-forward（nested rolling walk-forward）。** 把历史按固定长度的滚动训练 / 测试窗切成多折，每个训练窗上**单独**跑一次超参网格，挑出 champion 后**只**应用于紧随其后的测试窗记账；所有测试窗日收益按时间拼接成一条样本外净值曲线，整体指标按 `|grid|` 做 Bonferroni 折扣。这样**因子 IC 权重每折重估、超参也每折重选**，没有任何一段测试数据曾被用来选参，结构性地避免了「同一段历史既调参又报告」的 selection bias 与多重检验问题。

- **核心 API**：`run_nested_walk_forward_oos(market_bt, factors_full, base_spec, folds, grid, selection_metric="information_ratio")`。返回 `(stitched_oos_returns, fold_metrics_df, chosen_configs_df, diag)`。`folds` 由 `walk_forward_folds(global_start, global_end, train_months, test_months, step_months)` 构造，使用半开区间且当 `step == test` 时各测试窗互不重叠。
- **CLI 入口**：`examples/run_walk_forward_hs300.py`。
  - `--grid-mode none`（默认，向后兼容）：固定一组超参 + IC 权重每折重估，等价于旧脚本。
  - `--grid-mode default`（**推荐**）：内置 6 组合网格 `top_n ∈ {20, 30, 40}`、`weighting ∈ {equal, score}`、`rebalance = ME`、`weights_source = ic_train`。
  - `--grid-mode json:PATH`：从 JSON 对象 `{field: [values]}` 读取自定义网格，键必须是 `FactorConfig` / `BacktestConfig` 的字段。
  - `--selection-metric {information_ratio, sharpe, annual_return}`：训练窗上选 champion 的口径。
- **输出**：`outputs_walk_forward_hs300/` 下
  - `oos_stitched_daily_returns.csv`、`oos_stitched_metrics.csv`：拼接 OOS 日收益与整体指标
  - `oos_deflated_metrics.csv`：基于 Bailey & López de Prado（2012, 2014）的 PSR 与按试验次数 N 的 Bonferroni 折扣 Sharpe / IR（详见 `deflated_metrics.py`）；`N` 自动取 `|grid|`
  - `walk_forward_fold_metrics.csv`：每折测试窗 metrics + `cfg_*` 列
  - `walk_forward_chosen_configs.csv`：每折选了什么；**多折之间不稳定** 即提示策略对超参敏感、谨慎对待
  - `walk_forward_grid_spec.json`：本次跑的网格与切窗参数，便于复现
- **IC 驱动权重**：每折训练窗内调用 `estimate_factor_weights_ic` 估出 ICIR 权重，应用到该折测试窗的截面打分；不需要也不允许跨折共享 IC 信息。

**对照基线（legacy）**：若需要传统的「train → val → test」三段切分对照（仅作展示，不再作主结论），可用 `examples/run_sensitivity_train_val.py` + `examples/eval_selected_on_test.py` 与 `run_hs300_demo.py --sample-split research`。

**仍未覆盖的偏差**：成分股非 point-in-time（README 上文「股票池偏差」）、基本面字段前视 / 修订滞后、逐笔成交与冲击成本等。

**PowerShell 示例（先 `--limit 5` 冒烟，再跑完整）：**

```powershell
cd "C:\Users\w4727\Desktop\2027应届量化交易方向\a_share_quant"
pip install -e ".[data,dev]"
python -m pytest tests -q

python examples\run_walk_forward_hs300.py --limit 5 --grid-mode default --benchmark 000300.SH --output-dir outputs_walk_forward_smoke
python examples\run_walk_forward_hs300.py --grid-mode default --benchmark 000300.SH --output-dir outputs_walk_forward_nested
```

自定义网格示例（保存为 `my_grid.json`）：

```json
{
  "top_n": [20, 30],
  "weighting": ["equal", "score"],
  "rebalance": ["ME", "W-FRI"]
}
```

```powershell
python examples\run_walk_forward_hs300.py --grid-mode json:my_grid.json --benchmark 000300.SH --output-dir outputs_walk_forward_custom
```
