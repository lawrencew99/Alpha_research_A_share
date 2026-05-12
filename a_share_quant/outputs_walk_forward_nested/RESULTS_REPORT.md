# Nested rolling walk-forward — 结果报告

本报告由 `examples/run_walk_forward_hs300.py --grid-mode default --benchmark 000300.SH` 的 4 折嵌套滚动 walk-forward 产出，全部数字直接来自本目录下 4 个核心 CSV 与 `walk_forward_grid_spec.json`，未经手动修饰。

## 1. 运行参数

| 项 | 值 |
| --- | --- |
| global window | 2021-01-01 ~ 2024-12-31 |
| 测试窗联合区间（stitched） | 2023-01-03 ~ 2024-12-31 (484 个交易日) |
| train / test / step (月) | 24 / 6 / 6 |
| 折数 | 4 |
| grid_mode | default（6 个组合） |
| grid | top_n ∈ {20, 30, 40}，weighting ∈ {equal, score} |
| rebalance | ME（月末） |
| weights_source | ic_train（每折训练窗内估 ICIR 权重） |
| selection_metric | information_ratio |
| benchmark | 000300.SH（沪深 300 指数收盘价） |
| n_trials（用于 Bonferroni） | 6 |

> 注：plan 里曾估算 5 折，但 `walk_forward_folds(2021-01-01, 2024-12-31, 24, 6, 6)` 实际产出 4 折 —— 算法严格只能在 train_start + 24M + test ≤ global_end 之前开折，48 月 ÷ 6 = 4 折是正确结果。

## 2. OOS 整体指标（stitched）

**对照基准为 000300.SH 指数**（见 `oos_stitched_metrics.csv`）：

| 指标 | 值 |
| --- | --- |
| total_return | 36.51% |
| annual_return | 17.59% |
| annual_volatility | 16.23% |
| sharpe | 1.084 |
| max_drawdown | -14.43% |
| benchmark_total_return | 1.63% |
| benchmark_annual_return | 0.85% |
| excess_annual_return | 16.74% |
| tracking_error | 10.10% |
| **information_ratio** | **1.66** |
| beta | 0.754 |
| alpha | 15.68% |
| avg_turnover (stitched) | 0.0（**伪零** — 见 §6 limitations） |

**deflated（按 `n_trials=6` 做 Bonferroni 校正，见 `oos_deflated_metrics.csv`）**：

| 指标 | 值 | 解释 |
| --- | --- | --- |
| n_observations | 484 | 484 个交易日参与统计 |
| sharpe (deflated 视角) | 1.079 | 与上面 1.084 同义，重算一遍 |
| probability_sharpe_ratio_vs_0 | **1.000** | PSR：在估计精度下"真 Sharpe > 0"几乎是确定的 |
| **bonferroni_deflated_sharpe** | **0.970** | Sharpe − Bonferroni 临界值（α/6）。仍**强正** |
| information_ratio (vs universe equal-weight) | 0.060 | 严苛口径：用回测股票池等权日收益做基准 |
| bonferroni_deflated_information_ratio (vs universe equal-weight) | **-0.049** | 经多重检验折扣后**接近零、略负** |

> **重要**：`oos_deflated_metrics.csv` 里的 IR 不是 vs 000300.SH，而是 vs 「回测股票池等权」（`run_walk_forward_hs300.py:158`）。这是一个**比 000300.SH 更严**的虚拟基准 —— 它隐含「你能瞬时等权买入整个池子」。因此 IR=1.66（vs HS300 指数）和 IR=0.06（vs universe-equal）同时为真，并不矛盾。

## 3. 每折表现

| fold | 训练窗 | 测试窗 | chosen | train_score | test annual | test sharpe | test MDD | test IR | benchmark_ann | excess | turnover |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2021-01..2023-01 | 2023-01..2023-07 | top_n=30, equal | 2.04 | +18.72% | 1.64 | -5.48% | **2.14** | -2.48% | +21.20% | 0.44 |
| 1 | 2021-07..2023-07 | 2023-07..2024-01 | top_n=40, equal | 2.52 | -16.36% | -1.74 | -11.39% | 0.72 | -22.63% | +6.28% | 0.47 |
| 2 | 2022-01..2024-01 | 2024-01..2024-07 | top_n=20, equal | 2.35 | +20.97% | 1.58 | -6.27% | **1.69** | +4.85% | +16.12% | 0.52 |
| 3 | 2022-07..2024-07 | 2024-07..2025-01 | top_n=20, equal | 3.41 | +59.12% | 2.35 | -9.75% | **2.68** | +28.24% | +30.88% | 0.62 |

观察：

- **fold 1 是关键警示**：策略本身亏 -16.4%，但因为 2023H2 沪深 300 跌得更狠 (-22.6%)，仍跑出正 IR=0.72。这说明 stitched IR=1.66 里有一部分是"在熊市里少亏"带来的相对优势，不是绝对正收益。
- **fold 3 的 train_score=3.41 异常高**，可能与该窗（含 2024Q3-Q4 反弹）样本质量有关，需要在更长 history 上重核。
- 全部 4 折 IR 都为正，4 折中 3 折总收益为正（仅 fold 1 为负）。

## 4. Chosen config 稳定性

| 字段 | 各折取值 | 众数 | 稳定度 |
| --- | --- | --- | --- |
| weighting | equal, equal, equal, equal | **equal (4/4)** | 完全稳定 |
| top_n | 30, 40, 20, 20 | 20 (2/4) | **不稳定 / 漂移** |
| 组合 (top_n, weighting) | (30,e),(40,e),(20,e),(20,e) | (20, equal) 2/4 = 50% | 不稳定 |

按 [`outputs_recommended_config.md`](../outputs_recommended_config.md) 的稳定度量度（≥3/折数 稳定；2/折数 不稳定；1/折数 高敏感），此处 `top_n` 处于"不稳定"档。结论：**weighting=equal 可作为硬约束固化**，**top_n 不应该硬编码到一个数字，建议实盘按训练窗每月再选**（这正是 nested walk-forward 本身在做的事）。

## 5. 固化前三件事判定

| # | 检查 | 结果 | 备注 |
| --- | --- | --- | --- |
| 1 | chosen config 在多数折胜出 | **部分通过** | weighting=equal 100%；top_n 仅 50% |
| 2 | stitched IR / Sharpe 显著为正 | **通过（vs 000300.SH）** | IR=1.66、Sharpe=1.08 |
| 3 | deflated Sharpe 仍为正 | **通过** | bonferroni_deflated_sharpe=0.97（vs 0） |

附加严苛口径：

- vs **universe equal-weight** 的 deflated IR = -0.05 ≈ 0：意味着若把基准换成"在 HS300 池里随机等权买"，本策略在 6 试验的多重检验折扣后**没有显著 alpha**。
- 这说明 2023-2024 stitched 净值 36.5% 里，大约 **30%（universe equal-weight 涨幅相当）来自于 HS300 池在 2024 后半段的反弹**，约 **6.5%（vs HS300 指数的超额）来自策略择股**。

**结论**：
- 可以固化为「**weighting=equal、rebalance=ME、weights_source=ic_train**」的研究流水线模板。
- **不建议**把 `top_n` 写死成 20。在实盘中应继续用 nested walk-forward 每月（或每季度）训练窗里重选 top_n。
- 在「真有 alpha」这件事上，**对 000300.SH 而言证据较强**（IR=1.66、deflated Sharpe=0.97、PSR=1.0），**对 universe equal-weight 而言证据接近零**（deflated IR≈0）。务必在简历 / 路演里同时披露这两个数字，不要只挑前者。

## 6. 与 legacy 基线对照

数据来自 [`outputs_sensitivity_summary.csv`](../outputs_sensitivity_summary.csv) 与 [`outputs_top20/metrics.csv`](../outputs_top20/metrics.csv)（全样本 2021-2024，**单次 in-sample 选优**）：

| 视角 | legacy（in-sample，2021-2024 全期） | 新流水线 stitched OOS（2023-2025） |
| --- | --- | --- |
| best top_n（按 IR） | 20 | 20（chosen 众数） |
| best weighting | equal | equal (4/4) |
| **IR vs 000300.SH** | 1.98 | **1.66** |
| **IR vs equal_weight 基准** | 0.80 | 0.06（**经 6 试验 Bonferroni 折扣 → -0.05**） |
| annual_return | 18.29% | 17.59% |
| max_drawdown | -14.95% | -14.43% |

观察：

- 在 **HS300 指数基准** 下，OOS 与 IS 数字非常接近（IR 1.66 vs 1.98，annual_return 17.6% vs 18.3%），说明**当前实现的 selection bias 并不严重 / 不是吹起来的**。
- 在 **equal_weight 基准** 下，IS 的 0.80 → OOS 0.06 → deflated -0.05：**IS 与 OOS 的差异巨大**。这才是真正暴露 selection bias 的口径。

综合看，原来的 "outputs_recommended_config.md 旧版选 top_n=20 + equal" 这个结论本身**不算错**，但**它的 IR 数字（特别是 0.80）需要重新解释为「IS 估计、未折扣」**。

## 7. 局限性 / 仍未消除的偏差

1. **股票池快照非 point-in-time**：当前 `data/hs300_constituents.csv` 是单一时点快照，会带幸存者偏差，walk-forward 不能修正这一点。
2. **基本面字段未启用**：默认 grid 只对价量因子起作用（动量、反转、波动率、流动性），pb / roe 等若未接入则不参与；策略本质上是「**HS300 价量因子组合**」，不是完整多因子。
3. **`avg_turnover` 在 stitched 文件里是伪零**：源码 `metrics_from_stitched_returns` 把 turnover 设为 0；真实换手率请看 `walk_forward_fold_metrics.csv` 的 `avg_turnover` 列（4 折均值约 **0.51**）。
4. **deflated 仅按 `|grid|=6` 折扣**：实际隐含搜索空间远大于 6（因子权重网格、benchmark 选择等都已被尝试过几轮）。报告里的 deflated Sharpe / IR 应理解为"保守下界中的乐观下界"。
5. **fold 1 表现不一致**：负绝对收益 + 正 IR，说明策略在 2023H2 那种系统性下跌行情里没有真正回避风险，仅是 beta < 1 起了缓冲作用。需要在"做空 / 现金仓位"机制上做进一步研究才能正面应对类似 regime。
6. **样本量只有 4 折**：t 检验 / 跨 regime 推断都不太可靠。如果要进一步严谨：把 global window 扩到 2018 起，或用 `test_months=3 / step_months=3` 拿到约 8 折。

## 8. 下一步可做（不在本次报告范围内）

- 把 grid 拓宽到含 `rebalance ∈ {ME, W-FRI}`、`max_weight ∈ {0.06, 0.08, 0.10}`、`neutralize_industry ∈ {True, False}` —— 注意 `|grid|` 会从 6 增到 ~36，Bonferroni 折扣会更严，**不一定让结论变好看**。
- 接 PIT HS300 成分股 + akshare 财务数据，启用 pb / roe / size 中性化，再跑一遍。
- 把 base case 也加 anchored / expanding window 跑一次，与 rolling 比对。
- 在 fold 1 这种 down regime 做 sub-period attribution，识别"为何在该窗里相对赢，但绝对亏"的因子来源。
