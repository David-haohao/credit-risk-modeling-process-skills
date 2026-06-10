# 阶段 6：概率校准、分数尺度变换与风险等级

> **可执行脚本**：`scripts/06_scorecard.py`。阶段 6 reference 完成确认后，再同步脚本实现。

## 6.0 阶段目的与入口

阶段 6 包含三个不同任务：

1. **概率校准**：使预测坏账概率与实际坏账率尽量一致。
2. **分数尺度变换**：将最终坏账概率单调映射为便于业务使用的整数分。
3. **风险等级切分**：使用 Train 确定等级边界，再固定应用至其他样本。

概率校准与分数尺度变换不是同一件事。尺度变换只改变展示尺度，不改善概率准确性，也不应改变模型排序性。

阶段 6 必须先读取 `05_evaluation_decisions.csv`：

- 阶段 5 存在回退、拒绝或 `pending` 决策时，禁止开始阶段 6。
- LR 根据阶段 5 校准决策决定是否校准。
- XGB/LGB 必须执行概率校准，并输出校准前后效果对比。

---

## 6.1 输入契约

### 核心输入

| 输入 | 用途 |
|------|------|
| `05_evaluation_decisions.csv` | 阶段 6 入口检查与 LR 校准决策 |
| `05_calibration_summary.csv`、`05_calibration_detail.csv` | 原始概率校准表现基准 |
| `04_model_metadata.json` | 模型类型、模型版本和特征信息 |
| `04_pred_train.csv`、`04_pred_oot.csv` | Train/OOT 原始预测与评分 |
| `04_pred_test.csv` | XGB/LGB Test 原始预测与评分 |
| `00_modeling_config.yaml` | 评分参数和字段配置 |

预测文件统一使用：

```text
sample_id, time_col, y_true, y_pred_raw, sample_weight（如存在）
```

### 条件输入

| 输入 | 条件 |
|------|------|
| `04_model.pkl`、`04_final_features.txt`、最终转换链 | 需要对灰样本或新增样本预测时 |
| `01_grey_samples.csv` | 存在灰样本并需要评分时 |
| `03_combiner.pkl`、`03_woe_transformer.pkl`、`03_binning_detail.csv` | LR 必需，用于生成标准评分卡逐变量逐分箱分值 |
| `04_lr_coefficients.csv`、`04_model_formula.txt` | LR 必需，用于还原最终模型系数、截距和评分公式 |

阶段 5 是阶段 6 的正式入口和验收门槛，但阶段 6 仍需读取模型、预测和灰样本等必要上游产物。应在输出清单中记录实际使用的文件路径与版本，避免隐式依赖。

---

## 6.2 概率校准

### 6.2.1 路径规则

| 模型 | 概率校准要求 |
|------|--------------|
| LR | 阶段 5 评估后由用户确认是否执行 |
| XGB/LGB | 必须执行，并比较校准前后效果 |

XGB/LGB 默认使用 Platt Scaling。为统一口径，Platt 默认使用裁剪后的 `logit(y_pred_raw)` 作为唯一输入：

```text
raw_logit = ln(y_pred_raw / (1 - y_pred_raw))
logit(y_pred_calibrated) = platt_intercept + platt_coef × raw_logit
```

当 `platt_coef > 0` 时，该映射保持单调，通常不会改变排序性。若拟合得到 `platt_coef <= 0`，应视为校准异常并暂停确认。

Isotonic Regression 可作为用户确认的对比方案，但样本量不足时容易过拟合，且无法还原为 LR 标准逐箱加分卡，因此不作为默认方法；LR 路径如需保留标准评分卡结构，不使用 Isotonic Regression。

### 6.2.2 校准拟合数据

本项目确认使用 Train 拟合校准器。必须在输出中说明：

- 使用 Train 的训练内预测拟合校准器，可能得到偏乐观的校准结果，因为这些预测来自模型已经见过的样本。
- 校准后的 OOT 结果仍可用于观察外推表现，但不得使用 OOT 拟合或选择校准器。

校准器只在 Train 拟合一次，然后固定应用至 Test、OOT、灰样本和后续新增样本。禁止对每个数据集分别重新拟合校准器。

LR 如执行 Platt 校准，标准评分卡中的模型系数和截距必须同步调整：

```text
原始 LR logit = intercept + Σ(coef_j × WOE_j)
Platt 后 logit = platt_intercept + platt_coef × 原始 LR logit

effective_intercept = platt_intercept + platt_coef × intercept
effective_coef_j = platt_coef × coef_j
```

逐箱分值必须基于 `effective_intercept` 和 `effective_coef_j` 计算，确保各变量分箱分值加总后与最终校准概率映射出的总分一致。

### 6.2.3 校准结果复评

校准后必须比较校准前后的：

- Brier Score。
- Log Loss。
- Reliability Curve。
- 校准截距和校准斜率。
- AUC、KS，用于验证排序性未发生异常变化。

输出：

- `06_calibrator.pkl`：执行概率校准时生成。
- `06_calibration_comparison.csv`
- `06_calibration_detail.csv`
- `06_reliability_before_after_<dataset>.png`

---

## 6.3 分数尺度变换

### 6.3.1 风险方向与 Odds 定义

统一定义：

- `p_bad`：最终坏账概率。
- `bad_odds = p_bad / (1 - p_bad)`：坏好比。
- 坏好比越高，风险越高，分数越低。
- **分数越高，信用风险越低。**

基准参数：

| 参数 | 含义 |
|------|------|
| `base_score` | 基准坏好比对应的分数 |
| `base_odds` | 基准坏好比，必须明确为 bad/good |
| `pdo` | 坏好比翻倍时减少的分数 |
| `rate` | Odds 变化倍数，默认 2 |

默认建议值可展示，但执行前必须由用户确认，不得只依赖脚本默认值。

### 6.3.2 评分公式

```text
B = PDO / ln(rate)
A = base_score + B × ln(base_odds)
Score = A - B × ln(bad_odds)
bad_odds = p_bad / (1 - p_bad)
```

当坏好比等于 `base_odds` 时，分数等于 `base_score`；坏好比翻倍时，分数减少一个 PDO。

评分前必须对概率进行数值裁剪，避免 `p_bad=0` 或 `p_bad=1` 导致无穷值。概率裁剪属于计算处理，不代表对最终分数进行业务截断。

### 6.3.3 分数取整

分数由小数转换为整数时，统一使用十进制 `ROUND_HALF_UP` 四舍五入：

```text
234.499 → 234
234.500 → 235
234.513 → 235
```

不使用 Python 内置 `round()` 的银行家舍入。XGB/LGB 先计算完整小数总分，再执行一次 `ROUND_HALF_UP`。LR 标准评分卡对基础分和每个分箱分值分别执行 `ROUND_HALF_UP`，最终整数总分以整数基础分与整数分箱分值加总结果为准。

必须同时保留：

- `score_raw`：评分公式计算的小数分。
- `score`：按模型路径确认规则生成的整数分。
- `rounding_method=ROUND_HALF_UP`。

### 6.3.4 概率校准、Odds 修正与尺度变换的边界

- 概率校准：修正预测概率与真实坏账率之间的一致性。
- Odds 修正：当近期客群坏账率与开发样本存在整体截距错配时，修正整体风险基准；使用前必须由用户确认近期样本口径和适用性。
- 分数尺度变换：将最终概率映射为分数，不改变风险排序。

若使用近期客群执行 Odds 修正，应输出修正前后坏账率、Odds、截距变化及适用样本窗口，不能只修改 `base_odds` 而不保留依据。

### 6.3.5 分数截断

分数截断是指对评分公式产生的极端分数设置固定上下限。例如确认分数范围为 `[300, 900]` 后：

- 原始分数为 `265`，最终记为 `300`。
- 原始分数为 `947`，最终记为 `900`。
- 范围内分数保持不变。

默认不执行分数截断。只有业务系统明确要求固定分数范围时，才由用户确认是否设置上下限、上下限取值以及截断后处理方式。分数截断可满足系统展示或接口范围要求，但会使所有低于下限或高于上限的样本得到相同边界分数，并失去边界外原始 Odds 差异。

如执行截断，必须保留：

- `score_raw`：截断前分数。
- `score`：截断后分数。
- 上下限截断样本数和占比。

---

## 6.4 风险等级切分

### 6.4.1 等级方向

统一采用：

| 等级 | 含义 |
|------|------|
| A | 低风险、高分 |
| E | 高风险、低分 |

如果用户确认使用其他等级标签或等级数量，则按用户方案执行。

### 6.4.2 等级边界拟合与应用

- 风险等级边界只使用 Train 确定。
- 将 Train 确定的固定边界应用至 Test、OOT、灰样本和后续新增样本。
- 禁止对 Test、OOT 或灰样本分别按自身分位数重新切分。

Train 可使用以下方式确定等级边界：

1. Train 分位数切分。
2. 用户提供固定分数阈值。
3. 根据 Train 各分数段坏账率、样本占比和业务策略人工确认。

默认可建议五级 `A/B/C/D/E` 和 Train 等频切分，但属于软默认：必须向用户展示建议并确认后执行。

等级统计至少包含：数据集、等级、分数区间、样本数、样本占比、好坏样本数、坏账率和 Lift。应验证 A 至 E 的坏账率整体递增；局部不单调时输出问题供用户确认，不自动重新切分。

输出：

- `06_grade_edges.json`
- `06_grade_stats.csv`
- `06_grade_distribution.png`

---

## 6.5 LR 标准评分卡

LR 路径必须输出标准评分卡，不得仅输出总分或模型系数。

### 6.5.1 输入来源

- 阶段 3 最终分箱规则与分箱边界。
- 阶段 3 平滑 WOE 映射及逐箱统计。
- 阶段 4 最终 LR 系数和截距。
- 阶段 6 已确认的 `base_score`、`base_odds`、PDO 和校准结果。

必须使用最终入模变量和最终分箱版本，不得使用阶段 3 调整前分箱或阶段 4 诊断模型系数。

### 6.5.2 标准评分卡明细

`06_scorecard_detail.csv` 每一行代表一个特征分箱，至少包含：

| 字段 | 内容 |
|------|------|
| `feature` | 最终入模特征 |
| `bin_order` | 分箱顺序 |
| `bin_label` | 可读分箱范围或类别集合 |
| `lower_bound` / `upper_bound` | 连续变量上下界；离散变量可为空 |
| `categories` | 离散变量类别集合；连续变量可为空 |
| `is_missing_bin` | 是否缺失箱 |
| `count` / `good_cnt` / `bad_cnt` / `bad_rate` | Train 分箱统计 |
| `woe` / `iv_contribution` | 最终平滑 WOE 与 IV 贡献 |
| `coefficient` | 最终 LR 有效系数；执行 Platt 时使用合并后的系数 |
| `bin_points_raw` | 分箱原始分值 |
| `bin_points` | 按确认规则取整后的分箱分值 |

评分卡还必须明确输出基础分或截距分：

- `06_scorecard_base_points.csv`：记录模型截距、Platt调整（如有）、基准参数和基础分。
- 基础分与各特征分箱分值相加得到客户总分。

### 6.5.3 分值计算与一致性验证

当使用坏好比和高分低风险方向时，特征分箱分值来源于：

```text
bin_points_raw = -B × effective_coef_j × WOE_bin
```

截距、基准坏好比和基础分的分配方式必须在 `06_scorecard_base_points.csv` 中记录。分箱分值取整可能造成总分误差，因此必须同时保留原始分值和整数分值。

必须在 Train 和 OOT 执行逐样本一致性验证：

```text
score_from_probability = 概率映射总分
score_from_card = 基础分 + Σ(命中分箱分值)
score_difference = score_from_card - score_from_probability
```

输出 `06_scorecard_validation.csv`，至少包含数据集、样本数、最大绝对误差、平均绝对误差、差异分布和是否通过。默认要求平均绝对误差不超过 1 分；最大允许误差由用户结合最终变量数量确认。未通过时不得完成阶段 6。

---

## 6.6 灰样本评分

灰样本不参与模型训练、概率校准器拟合、等级边界拟合或效果评估。

评分流程：

```text
最终模型与转换链生成 y_pred_raw
→ 应用已拟合的校准器（如适用）
→ 应用已确认的分数公式
→ 应用 Train 固定等级边界
```

灰样本输出只展示分数、等级、数量和占比，不计算坏账率、KS、AUC等需要真实标签的指标。

---

## 6.7 决策表

阶段 6 所有人工确认统一写入 `06_scoring_decisions.csv`：

| 字段 | 内容 |
|------|------|
| `decision_id` | 决策编号 |
| `decision_type` | `lr_calibration` / `calibration_method` / `score_parameters` / `odds_adjustment` / `score_clipping` / `grade_scheme` |
| `suggested_value` | 建议值 |
| `confirmed_value` | 用户确认值 |
| `decision` | `pending` / `accept` / `reject` |
| `reason` / `confirmed_by` / `confirmed_at` | 确认记录 |

存在 `pending` 时不得完成阶段 6。

---

## 6.8 核心输出与验收

### 评分结果字段

```text
sample_id, time_col, y_true, y_pred_raw, y_pred_calibrated（如适用）,
score_raw, score, risk_grade, sample_weight（如存在）
```

灰样本没有真实标签时不输出 `y_true`。

### 核心输出

| 文件 | 说明 |
|------|------|
| `06_scoring_parameters.json` | 评分参数、Odds 定义、评分公式、概率裁剪与分数截断配置 |
| `06_calibrator.pkl` | 校准器；执行概率校准时生成 |
| `06_calibration_comparison.csv` | 校准前后指标对比 |
| `06_calibration_detail.csv` | 校准分箱明细 |
| `06_grade_edges.json` | Train 确定的固定等级边界 |
| `06_grade_stats.csv` | 各数据集等级统计 |
| `06_train_scored.csv` | Train 评分结果 |
| `06_test_scored.csv` | XGB/LGB Test 评分结果 |
| `06_oot_scored.csv` | OOT 评分结果 |
| `06_grey_scored.csv` | 灰样本评分结果，如存在 |
| `06_scorecard_detail.csv` | LR 路径逐变量逐分箱分值 |
| `06_scorecard_base_points.csv` | LR 路径基础分、截距及校准调整 |
| `06_scorecard_validation.csv` | LR 概率映射分与分箱加总分一致性验证 |
| `06_scoring_decisions.csv` | 阶段 6 人工决策 |
| `06-output-list.xlsx` | 阶段 6 输出索引与审阅汇总 |

### 验收规则

1. 阶段 5 已通过且不存在回退、拒绝或 `pending` 决策。
2. XGB/LGB 已完成概率校准及校准前后比较；LR 已按用户确认执行。
3. 校准器仅在 Train 拟合一次，Test/OOT/灰样本只应用。
4. 评分公式明确使用坏好比，且高分代表低风险。
5. 等级边界仅由 Train 确定，并固定应用至其他数据集。
6. 灰样本未参与拟合、等级边界确定或效果评估。
7. 分数截断、等级方案和评分参数均已确认，不存在 `pending`。
8. LR 已输出标准评分卡逐箱分值、基础分及分值一致性验证，且验证通过。
9. 所有核心输出均登记在 `06-output-list.xlsx`。

阶段 6 验收通过后，进入阶段 7 最终报告汇总。
