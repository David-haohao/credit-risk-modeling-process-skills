# 阶段 5：模型评估

> **可执行脚本**：`scripts/05_model_evaluation.py`。阶段 5 reference 完成确认后，再同步脚本实现。

## 5.0 阶段目的与输入契约

阶段 5 只负责诊断阶段 4 已冻结模型的原始预测，不在阶段 5 内直接重新训练、调参或执行概率校准。评估通过后方可进入阶段 6；评估不通过时，必须根据问题原因回退至阶段 1、3 或 4 处理并重新执行后续评估。

### 核心输入

| 文件 | 必需性 | 用途 |
|------|--------|------|
| `04_model_metadata.json` | 必需 | 模型类型、运行配置与最终模型信息 |
| `04_pred_train.csv` | 必需 | Train 评估与基准分箱 |
| `04_pred_test.csv` | XGB/LGB 必需 | 调参集表现评估，不视为独立泛化验证 |
| `04_pred_oot.csv` | 必需 | OOT 泛化、稳定性与校准评估 |
| `04_final_features.txt` | 必需 | 记录最终模型特征，不直接用于核心指标计算 |

预测文件统一字段：

```text
sample_id, time_col, y_true, y_pred_raw, sample_weight（如存在）
```

- `sample_id` 只用于追溯与结果关联。
- `time_col` 使用阶段 0 确认的实际字段名，用于周期 PSI。
- 有 `sample_weight` 时，核心统计指标与分箱统计应同时支持加权口径，并在输出中标注口径。
- LR 评估 Train 与 OOT；XGB/LGB 评估 Train、Test 与 OOT。

---

## 5.1 区分度评估

对各可用数据集计算：

- KS、AUC、Gini。
- ROC 曲线与 KS 曲线。
- Train-OOT 的 KS/AUC 变化。
- XGB/LGB 额外展示 Train-Test、Test-OOT 的变化，并明确 Test 是调参集。

输出：

- `05_evaluation_summary.csv`
- `05_ks_curve_<dataset>.png`
- `05_roc_curve_<dataset>.png`

---

## 5.2 排序性与 Lift 评估

排序性评估同时提供两种分箱口径：

1. **各数据集内部等频分箱**：用于查看该数据集自身排序能力。
2. **Train 固定分箱边界**：在 Train 拟合边界后应用至 Test/OOT，用于跨数据集直接比较。

每个分箱至少输出：样本数、样本占比、好/坏样本数、坏账率、分箱 Lift、累计样本占比、累计坏样本占比、累计 Lift、累计 KS。

### Lift 曲线

- **分箱 Lift 曲线**：展示每个风险分箱的 `bucket_bad_rate / overall_bad_rate`。
- **累计 Lift 曲线**：按预测风险从高到低累计，展示前若干比例样本捕获坏样本的能力。
- 同时输出累计 Gains 曲线，辅助解释累计 Lift。
- Train、Test（如有）、OOT 分别绘制，并提供 Train 固定分箱口径下的跨数据集对比图。

输出：

- `05_bucket_internal.csv`
- `05_bucket_fixed.csv`
- `05_bucket_edges.json`
- `05_lift_bucket_<dataset>.png`
- `05_lift_cumulative_<dataset>.png`
- `05_gains_<dataset>.png`
- `05_lift_fixed_comparison.png`

---

## 5.3 稳定性评估

### 5.3.1 整体分数 PSI

以 Train 原始预测为基准拟合固定分箱边界，计算 Train 与 OOT 的模型分数 PSI。必须输出每个分箱的 Train/OOT 占比、占比差异、单箱 PSI 贡献和总 PSI。

PSI 参考区间可展示为：

| PSI | 参考解释 |
|-----|----------|
| `< 0.10` | 分布较稳定 |
| `0.10 - 0.25` | 存在一定漂移，需要结合业务检查 |
| `> 0.25` | 漂移明显，需要重点检查 |

参考区间用于提示，不直接替代人工评估决策。

### 5.3.2 周期 PSI

阶段 0 必须询问并确认 `psi_period_granularity`：按周、按月或按季。

- 使用 Train 固定分箱边界。
- 将 OOT 按确认粒度切分为连续周期。
- 每个 OOT 周期均与整体 Train 基准分布比较，不能以前一周期替代 Train 基准。
- 输出每周期的样本数、坏样本数、坏账率、PSI 总值和各分箱 PSI 贡献。
- 不对周期样本数设置固定硬阈值；样本数或坏样本数不足时，在结果中标注，由用户判断是否合并周期或调整观察粒度。

输出：

- `05_score_psi_detail.csv`
- `05_score_psi_summary.csv`
- `05_period_psi_detail.csv`
- `05_period_psi_summary.csv`
- `05_period_psi_trend.png`

### 5.3.3 特征 PSI（可选增强项）

特征 PSI 不是阶段 5 核心验收项。仅当用户确认需要时，通过上阶段传递的特征数据与转换规则计算，避免阶段 5 默认回读阶段 3 文件。

---

## 5.4 校准度评估

阶段 5 只判断是否需要校准，不执行校准。对各可用数据集输出：

- Brier Score。
- Log Loss。
- Reliability Curve。
- 校准截距与校准斜率。
- 各概率分箱的平均预测概率、实际坏账率、样本数与差异。

输出：

- `05_calibration_summary.csv`
- `05_calibration_detail.csv`
- `05_reliability_<dataset>.png`

---

## 5.5 模型评估决策

模型评估决策统一写入 `05_evaluation_decisions.csv`。每个核心问题单独一行，不能只在报告正文中给结论。

| 字段 | 内容 |
|------|------|
| `decision_id` | 决策记录唯一编号 |
| `issue_type` | `overfitting` / `underfitting` / `data_shift` / `data_or_split_issue` / `ranking_issue` / `poor_calibration` / `other` |
| `metric_scope` | 涉及的数据集、周期或指标范围 |
| `observed_value` | 实际观察值或结果摘要 |
| `reference_value` | 参考值、用户确认标准或对比基准 |
| `assessment` | 对问题的业务与统计解释 |
| `recommended_action` | 建议动作 |
| `return_stage` | 无需回退时为空；否则填写 `stage1` / `stage3` / `stage4` |
| `decision` | `pending` / `accept` / `return_stage1` / `return_stage3` / `return_stage4` / `calibrate_stage6` / `reject_model` |
| `confirmed_by` | 确认人 |
| `confirmed_at` | 确认时间 |
| `reason` | 最终决策原因 |

### 5.5.1 Train / Test / OOT 差异诊断

不能仅根据 Train、Test、OOT 的 KS/AUC 差异直接认定过拟合。应结合各数据集绝对表现、稳定性、样本量和数据口径完成诊断：

| 典型表现 | 初步诊断 | 主要检查 | 建议回退与处理 |
|----------|----------|----------|----------------|
| Train 明显优于 Test/OOT | 过拟合候选 | Train-Test/OOT Gap、模型复杂度、变量稳定性、是否存在泄漏 | 回退阶段 4：LR 可加强 L1/L2、减少变量或复杂交互；XGB/LGB 可降低树复杂度、加强正则、采样与早停 |
| Train、Test、OOT 均较差且差距不大 | 欠拟合候选 | Train 绝对 KS/AUC、特征有效性、模型表达能力 | 优先回退阶段 3 检查特征与分箱；确认特征有效后再回退阶段 4 调整模型 |
| Train/Test 尚可，但 OOT 明显下降且 PSI 较高 | 数据漂移候选 | 周期 PSI、样本口径、变量 PSI、客群变化 | 回退阶段 1 检查样本和切分口径，或阶段 3 处理不稳定特征；不能只通过加大正则化处理 |
| Test 明显弱于 Train 和 OOT，或结果关系异常 | 数据或切分问题候选 | Test 样本量、坏样本量、随机性、分层与时间边界 | 回退阶段 1 检查切分；必要时重新生成 Test 后重跑阶段 3-5 |
| 区分度可接受，但固定分箱排序异常 | 排序问题 | 分箱坏账率、Lift、局部分数段样本量 | 回退阶段 3 调整特征/分箱，或阶段 4 调整模型 |
| 区分度与排序可接受，但概率偏差明显 | 校准问题 | Reliability、Brier、Log Loss、校准截距/斜率 | 进入阶段 6 执行校准，不要求回退阶段 4 |

L1/L2 是处理过拟合的候选手段，不是看到数据集差异后的默认动作。若主要原因是数据漂移、切分异常或特征失效，应回退至对应上游阶段处理。

### 5.5.2 回退与重新验证规则

1. 评估结论为过拟合、欠拟合、数据漂移、切分异常或排序问题时，不得进入阶段 6。
2. 根据诊断结果回退至对应阶段处理；完成修改后，必须重新执行该阶段至阶段 5 的全部受影响流程。
3. 每次回退必须记录原模型版本、问题证据、回退阶段、采取动作、新模型版本和复评结果。
4. OOT 原则上只用于最终泛化评估。若根据 OOT 结果反复修改特征、样本或模型，原 OOT 已参与模型选择，不再视为独立 OOT；应使用新的后续时间窗口再次验证，或明确披露其已被用于迭代。

回退与复评过程统一写入 `05_rework_history.csv`：

```text
iteration_id, source_model_version, issue_type, evidence, return_stage,
action_taken, new_model_version, oot_usage_status, reevaluation_result,
confirmed_by, confirmed_at
```

建议的核心决策行：

| 核心检查内容 | 可能的 `issue_type` | 常见后续动作 |
|--------------|----------------------|--------------|
| Train/Test/OOT 的 KS/AUC 绝对表现与差异 | `overfitting` / `underfitting` / `data_or_split_issue` | 接受、返回阶段 1/3/4、拒绝模型 |
| 固定分箱坏账率排序、分箱 Lift 与累计 Lift | `ranking_issue` | 接受、返回阶段 3/4 |
| Train-OOT 整体及周期 PSI | `data_shift` | 接受、调整观察粒度、返回阶段 1/3、拒绝模型 |
| Brier、Log Loss、Reliability、校准截距/斜率 | `poor_calibration` | 接受原始概率、阶段 6 校准、拒绝模型 |

任一核心决策仍为 `pending` 时，不得进入阶段 6。

---

## 5.6 核心输出与验收

核心输出：

| 文件 | 说明 |
|------|------|
| `05_evaluation_summary.csv` | 区分度及跨数据集变化汇总 |
| `05_bucket_internal.csv` | 各数据集内部等频分箱结果 |
| `05_bucket_fixed.csv` | Train 固定边界的跨数据集分箱结果 |
| `05_bucket_edges.json` | Train 分数分箱边界 |
| `05_score_psi_detail.csv` | 整体分数 PSI 分箱贡献 |
| `05_score_psi_summary.csv` | 整体分数 PSI 汇总 |
| `05_period_psi_detail.csv` | 周期 PSI 分箱贡献 |
| `05_period_psi_summary.csv` | 周期 PSI 汇总与样本情况 |
| `05_calibration_detail.csv` | 校准分箱明细 |
| `05_calibration_summary.csv` | 校准指标汇总 |
| `05_evaluation_decisions.csv` | 模型评估决策表 |
| `05_rework_history.csv` | 回退处理、模型版本与重新评估记录；无回退时可为空表 |
| `05-output-list.xlsx` | 阶段 5 输出索引与用户审阅汇总 |

验收规则：

1. 所有评估均基于阶段 4 的 `y_pred_raw`，未在阶段 5 重新训练、调参或校准。
2. 分箱边界只由 Train 拟合；固定分箱、整体 PSI 与周期 PSI 对 Test/OOT 只做应用。
3. 已按阶段 0 确认的周/月/季粒度输出周期 PSI。
4. Lift 同时包含分箱 Lift、累计 Lift 和 Gains。
5. 核心决策均已写入表格且不存在 `pending`。
6. 所有核心输出均登记在 `05-output-list.xlsx`。
7. 如存在回退决策，阶段 5 状态必须标记为未通过；仅在回退处理并重新评估通过后，才允许进入阶段 6。

阶段 5 验收通过后，进入阶段 6。
