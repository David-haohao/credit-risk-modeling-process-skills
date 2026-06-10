# 阶段 4：模型训练

> **可执行脚本**: `scripts/04_model_training.py`

## 4.0 路由与阶段信息

- 所有模型必须读取本文件。
- 仅当 `model_type` 为 XGB/LGB 时，额外读取 `references/4.1-xgb-lgb-tuning.md`。
- **输入**: 阶段 3 最终模型数据、`03_final_features.txt` 和 `00_modeling_config.yaml`
- **输出**: 最终模型、最终变量、原始预测、训练过程审计和人工决策

阶段 4 仅训练阶段 0 已确认的模型。OOT 只用于最终泛化评估，禁止参与变量筛选、调参、Early Stopping 或模型选择。

模型预测统一命名为 `y_pred_raw`。是否能够解释为真实坏账率，由阶段 5 校准评估决定。

---

## 4.1 输入契约

### 共同必需输入

| 输入 | 用途 |
|------|------|
| `00_modeling_config.yaml` | 模型类型、字段、权重、随机种子和用户确认参数 |
| `03_final_features.txt` | 阶段 4 唯一允许使用的模型特征列表 |

禁止通过排除目标字段的方式推断模型特征。`sample_id`、目标字段、时间字段和 `sample_weight` 不得进入模型。

### 路径输入

| 模型路径 | 必需输入 |
|----------|----------|
| LR | `03_train_woe.csv`、`03_oot_woe.csv` |
| XGB/LGB | `03_train_final.csv`、`03_test_final.csv`、`03_oot_final.csv` |

LR 可参考 `03_iv_table.csv`、`03_feature_adjustments.csv` 和 `03_binning_detail.csv` 判断异常变量是否返回阶段 3。

### 输入验收

1. 各数据集均包含 `sample_id`、目标字段和最终特征。
2. `sample_id` 非空、唯一且不进入模型。
3. 各适用数据集的特征列顺序和类型一致。
4. 阶段 3 不存在待确认项。
5. 存在 `sample_weight` 时，训练和调参必须使用该权重。

---

## 4.2 LR 路径

```text
Train WOE
→ VIF 诊断
→ 可审计逐步回归
→ 入模变量软上限检查
→ 系数方向与显著性诊断
→ 用户确认异常项
→ k-fold CV + 网格搜索
→ 最终训练
→ Train/OOT 预测
```

### 4.2.1 入模变量软上限

- 默认建议值为 15，但不是自动剔除规则。
- 阶段开始前询问用户采用 15、其他数量或不限制。
- 超过软上限时生成 `decision=pending` 并暂停，禁止直接按 IV 自动截断。

### 4.2.2 可审计逐步回归

支持 `forward`、`backward` 和 `both`，默认 `both`；选择标准支持 AIC/BIC，默认 AIC。

每轮记录：迭代轮次、动作、变化变量、变化前后变量列表、AIC/BIC 变化、改善量和停止原因。

必须输出：

- `04_lr_stepwise_history.csv`
- `04_lr_training_process.xlsx`
- `04_lr_criterion_curve.png`

### 4.2.3 诊断与人工决策

VIF、负系数和 `p_value > 0.05` 均属于诊断项，不自动剔除。用户可确认：

- `keep`
- `drop`
- `return_stage3`

存在 `pending` 或 `return_stage3` 时不得完成阶段 4。

### 4.2.4 最终模型参数

必须区分：

1. statsmodels 诊断模型：系数、标准误、z 值和 p 值。
2. 最终 sklearn 正则化 LR：实际权重和截距。

`04_lr_coefficients.csv` 必须包含 `__INTERCEPT__`；同时输出 `04_model_formula.txt`。

---

## 4.3 XGB/LGB 路径

详细调优规则、默认搜索空间和高级选项见 `references/4.1-xgb-lgb-tuning.md`。

默认核心流程：

```text
确认搜索空间与 Trial 数
→ Train 训练、Test 调参和 Early Stopping
→ 记录每个 Trial 的效果、过拟合差值、参数和耗时
→ 固化最佳参数与最佳迭代轮数
→ 使用完整 Train 训练最终模型
→ 生成 Train/Test/OOT 预测
```

核心约束：

- Test 是调参集，不属于独立最终验证集。
- OOT 禁止参与搜索空间调整、参数选择和最佳 Trial 选择。
- 数据采样仅是用户确认的性能选项。
- XGB 与 LGB 必须调用各自真实实现；缺少依赖时明确报错。
- 每次执行前确认使用默认搜索空间还是自定义搜索空间。

默认必须输出：

- 每个 Trial 的参数、状态、耗时、Train/Test AUC、AUC Gap 和最佳迭代轮数
- 优化历史、参数重要性、最佳参数和最终选择理由
- 可续跑 Study 或明确记录当前仅保存辅助快照

Pruner、多目标优化、粗细两阶段搜索和扩展可视化属于按需能力，不作为每次执行的默认强制项。

---

## 4.4 人工决策表

`04_model_decisions.csv` 至少包含：

| 字段 | 说明 |
|------|------|
| `feature` | 特征或决策对象 |
| `issue_type` | `soft_feature_limit`、`high_vif`、`negative_coef`、`non_significant` 或模型选择问题 |
| `metric_value` | 对应诊断值 |
| `decision` | `pending`、`keep`、`drop`、`return_stage3` |
| `decision_reason` / `confirmed_by` / `confirmed_at` | 确认记录 |

---

## 4.5 阶段 5 回退重训入口

当阶段 5 决策为 `return_stage4` 时，阶段 4 可作为回退重训入口。除原阶段 4 输入外，还必须读取：

- `05_evaluation_decisions.csv`：确认问题类型、证据和建议动作。
- `05_rework_history.csv`：确认历史模型版本、已采取动作及 OOT 使用状态。

回退重训要求：

1. 根据诊断原因调整模型，不得无依据地全面扩大搜索范围。
2. 过拟合候选：LR 可加强 L1/L2、减少变量；XGB/LGB 可降低模型复杂度、加强正则、采样或早停。
3. 欠拟合候选只有在阶段 5 已确认特征有效时才直接回退阶段 4；否则先回退阶段 3。
4. 每次重训生成新的模型版本，并在重新进入阶段 5 后更新 `05_rework_history.csv`。
5. 若模型调整已使用 OOT 结果作为选择依据，该 OOT 不再视为独立最终验证集。

---

## 4.6 核心输出

### 共同输出

| 文件 | 说明 |
|------|------|
| `04_model.pkl` | 最终模型 |
| `04_model_metadata.json` | 模型类型、最终特征、最佳参数和运行配置 |
| `04_final_features.txt` | 最终模型特征 |
| `04_pred_train.csv` | Train 原始预测 |
| `04_pred_oot.csv` | OOT 原始预测 |
| `04_model_decisions.csv` | 人工确认记录 |
| `04-output-list.xlsx` | 阶段汇总 |

XGB/LGB 额外输出 `04_pred_test.csv` 和调优子 reference 规定的核心审计文件。

预测文件统一包含：

```text
sample_id, time_col, y_true, y_pred_raw, sample_weight（如存在）
```

`time_col` 使用阶段 0 确认的实际字段名，仅用于阶段 5 周期稳定性评估，不作为模型特征。

### LR 专属输出

- `04_lr_diagnostics.csv`
- `04_lr_coefficients.csv`
- `04_lr_grid_search.csv`
- `04_lr_stepwise_history.csv`
- `04_lr_training_process.xlsx`
- `04_model_formula.txt`

---

## 4.7 输出验收

1. 模型特征与 `04_final_features.txt` 一致，不包含保留字段。
2. LR 输出 Train/OOT；XGB/LGB 输出 Train/Test/OOT。
3. 预测文件的 `sample_id` 与阶段 3 输入可追溯。
4. OOT 未参与筛选、调参、Early Stopping 或模型选择。
5. 不存在 `pending` 或未处理的 `return_stage3`。
6. LR 权重表包含截距项，逐步回归过程可追溯。
7. XGB/LGB Trial 参数、核心指标和最终选择依据可追溯。

阶段 4 输出确认后进入阶段 5。
