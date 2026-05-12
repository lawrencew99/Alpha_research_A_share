# 推荐参数与样本外流程（2026 更新 · nested rolling walk-forward）

> **最新一轮结果**：见 [`outputs_walk_forward_nested/RESULTS_REPORT.md`](outputs_walk_forward_nested/RESULTS_REPORT.md)（stitched OOS 2023-01..2024-12，IR vs 000300.SH = 1.66、bonferroni_deflated_sharpe = 0.97、weighting=equal 4/4 折胜出、top_n 不稳定）。

## 状态说明

- 旧版「在整段 2021–2024 上按 `information_ratio` 挑最优」**已废弃**：存在 selection bias / 多重检验，不再作为结论。
- 旧版「train / val / test 三段切分（2021–2022 / 2023 / 2024）」**降级为对照基线**：仅一次随机切片，不能反映跨 regime 表现。
- **当前主推荐**：**嵌套滚动 walk-forward**——每折训练窗单独跑超参网格，champion 只在紧邻测试窗记账，全部测试窗日收益拼接成 OOS 净值后再做多重检验折扣。

## 推荐流程

1. **构造滚动折**：`walk_forward_folds(global_start, global_end, train_months=24, test_months=6, step_months=6)`。半开区间且 `step==test` 时各测试窗互不重叠，可直接拼接。
2. **每折训练窗** → 在网格内逐组回测、按 `selection_metric`（默认 `information_ratio`）挑 champion；IC 权重每折用该折训练窗重估。
3. **每折测试窗** → 只用 champion 做一次记账；测试日收益加入 stitched 序列。
4. **整体评估** → `oos_stitched_metrics.csv` 给原始指标，`oos_deflated_metrics.csv` 给按 `|grid|` 折扣后的 Sharpe / IR 与 PSR（Bailey & López de Prado 2012/2014）。
5. **稳定性核对** → 读 `walk_forward_chosen_configs.csv`：若同一组配置在多数折胜出，则结论较稳健；若 champion 在折间漂移剧烈，意味着策略对超参敏感、不能直接固化。

## 推荐命令

```powershell
cd "C:\Users\w4727\Desktop\2027应届量化交易方向\a_share_quant"
pip install -e ".[data,dev]"

# 冒烟（少量股票，几十秒）
python examples\run_walk_forward_hs300.py --limit 5 --grid-mode default --benchmark 000300.SH --output-dir outputs_walk_forward_smoke

# 完整（约 7 折 × 6 组合 = 42 次回测，几分钟级）
python examples\run_walk_forward_hs300.py --grid-mode default --benchmark 000300.SH --output-dir outputs_walk_forward_nested
```

## 默认网格

由 `examples/run_walk_forward_hs300.py` 的 `DEFAULT_GRID` 定义：

```yaml
grid:
  top_n: [20, 30, 40]
  weighting: ["equal", "score"]
# rebalance 固定 ME；weights_source 固定 ic_train
selection_metric: information_ratio
windows:
  train_months: 24
  test_months: 6
  step_months: 6
```

如需更广的搜索，可用 `--grid-mode json:my_grid.json`，键必须是 `FactorConfig` / `BacktestConfig` 的字段。**注意**：网格越大，`|grid|` 越大，Bonferroni 折扣越严，`oos_deflated_metrics.csv` 给出的指标越保守。

## 配置固化（用 champion 进入实盘）

固化前**必须**满足三件事：

1. `walk_forward_chosen_configs.csv` 中该 config 在大多数折胜出；
2. `oos_stitched_metrics.csv` 的 IR / Sharpe 显著为正；
3. `oos_deflated_metrics.csv` 的 `bonferroni_deflated_sharpe`（按 `|grid|` 折扣后）仍为正。

```yaml
# 由 walk_forward_chosen_configs.csv 多折胜出项填入；以下为模板
backtest_config:
  top_n: <从 chosen 表统计众数>
  rebalance: "ME"
  weighting: "<equal|score>"
  benchmark: "000300.SH"
  max_weight: 0.08
  execution_delay: 1

factor_config:
  weights_source: "ic_train"   # 实盘按"每月再校准 IC 权重"重跑训练窗
  neutralize_industry: false
  neutralize_size: false
```

## 仍须自行警惕的偏差

- 沪深 300 **成分股快照非 point-in-time**（README 已说明）。
- 基本面字段、行业分类等若存在 **前视 / 修订滞后未建模**，walk-forward 不会自动消除。
- 逐笔成交、流动性冲击、停牌可卖假设等微观结构误差，仍属模型简化。
- `deflated_metrics` 仅是简化统计近似，无法替代经济意义检验或贝叶斯模型比较。

## 旧版对照基线（保留以便复现）

```powershell
python examples\run_sensitivity_train_val.py --benchmark 000300.SH
python examples\eval_selected_on_test.py --from-summary outputs_sensitivity_summary.csv --rank 1 --benchmark 000300.SH --output-dir outputs_test_holdout_champion1
python examples\run_hs300_demo.py --sample-split research --eval-window test --weights-source ic_train --benchmark 000300.SH --output-dir outputs_research_test
```

这些脚本的输出**不再**作为主结论，仅用于与 `outputs_walk_forward_nested/` 对比、说明「一次切片」与「滚动 OOS」差异。
