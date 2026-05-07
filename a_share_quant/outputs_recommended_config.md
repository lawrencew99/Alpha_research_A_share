# 推荐参数结论（基于 outputs_sensitivity_summary.csv）

数据来源：`outputs_sensitivity_summary.csv`  
筛选规则：优先 `information_ratio`，并约束 `max_drawdown >= -0.20`。

## 主推荐（实盘优先）

- 结论：采用 **top_n=20 + 月频调仓 + equal 加权 + 000300.SH 基准**
- 选择理由：
  - `top_n_20` 在等权基准下 `information_ratio=0.8031`，为参数敏感性测试中最优；
  - `max_drawdown=-0.1495`，满足回撤约束；
  - 周频(`rebalance_wfri`)与 score 加权(`weighting_score`)均未超过该方案的 IR。

### 配置清单

```yaml
factor_config:
  momentum_60: 0.00
  reversal_20: 0.30
  volatility_20: -0.40
  liquidity_20: -0.30
  value_pb: 0.00
  quality_roe: 0.00

backtest_config:
  top_n: 20
  rebalance: "ME"
  weighting: "equal"
  benchmark: "000300.SH"
  max_weight: 0.08
  execution_delay: 1
```

### 推荐命令

```powershell
python examples\run_hs300_demo.py --top-n 20 --rebalance ME --weighting equal --benchmark 000300.SH --output-dir outputs_recommended_top20
```

## 备选方案 A（更稳健分散）

- 结论：**top_n=40 + 月频 + score 加权 + 000300.SH 基准**
- 适用场景：希望在保持 40 只持仓分散度前提下，较基线提升 IR。
- 关键指标（对比 top_n_40 等权）：
  - `information_ratio: 0.5848 -> 0.6377`
  - `max_drawdown: -0.1710 -> -0.1662`

### 配置清单

```yaml
factor_config:
  momentum_60: 0.00
  reversal_20: 0.30
  volatility_20: -0.40
  liquidity_20: -0.30
  value_pb: 0.00
  quality_roe: 0.00

backtest_config:
  top_n: 40
  rebalance: "ME"
  weighting: "score"
  benchmark: "000300.SH"
  max_weight: 0.08
  execution_delay: 1
```

### 推荐命令

```powershell
python examples\run_hs300_demo.py --top-n 40 --rebalance ME --weighting score --benchmark 000300.SH --output-dir outputs_recommended_score
```

## 不推荐切换项

- `rebalance=W-FRI`：`information_ratio=0.4230`，低于月频，不建议切周频。

