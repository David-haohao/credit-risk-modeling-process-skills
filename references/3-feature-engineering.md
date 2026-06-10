# 阶段 3：特征工程

> **可执行脚本**: `scripts/03_feature_engineering.py` — 运行 `python scripts/03_feature_engineering.py --help` 查看用法。

## 3.0 阶段信息

- **执行必需输入**: `01_train.csv`, `01_oot.csv` + `00_modeling_config.yaml`；XGB/LGB 路径额外需要 `01_test.csv`
- **审计参考输入**: 阶段 2 已生成的 EDA 质量报告与明细文件，不在阶段 3 重复生成
- **输出**: 核心下游数据 + 审计明细 + `03-output-list.xlsx`
- **耗时记录**: 阶段开始和结束时记录时间戳（精确到分钟）

---

## 3.0.1 输入契约

### 执行必需输入

| 输入 | 必需字段/内容 | 用途 |
|------|--------------|------|
| `01_train.csv` | `sample_id`、Y_label、候选特征；抽样时包含 `sample_weight` | 仅使用 Train 拟合筛选规则、分箱和转换器 |
| `01_test.csv` | `sample_id`、Y_label、与 Train 一致的候选特征 | XGB/LGB 路径必需；应用 Train 确定的特征处理规则 |
| `01_oot.csv` | `sample_id`、Y_label、与 Train 一致的候选特征 | 应用 Train 确定的特征处理规则 |
| `00_modeling_config.yaml` | `model_type`、`target_col`、`sample_id_col`、`time_col`、缺失值配置、阶段 3 阈值及用户确认状态 | 提供机器可读取的全局配置，避免仅依赖对话或 Excel |
| `03_feature_adjustments.csv` | LR 路径首次分箱后生成；首次运行可缺失，确认后重跑时必需 | 提供手动分箱、U 型转换、保留或剔除决策 |

`sample_id` 为样本唯一标识，可以是业务主键或脱敏后的稳定标识。`sample_id`、Y_label、时间字段和 `sample_weight` 均属于保留字段，**不得进入模型特征池**。

### 阶段 2 EDA 质量报告输入

阶段 2 已完成 EDA 和数据质量检查。阶段 3 应读取并引用以下结果，不重复执行同类检查：

| 输入 | 阶段 3 使用方式 |
|------|----------------|
| `02-output-list.xlsx` | 获取阶段 2 总体结论及异常项摘要 |
| `02_quality_decisions.csv` | 执行已确认的 `drop` / `transform` / `cap` 等处理，并检查未确认项 |
| `02_toad_detect.csv` | 参考缺失率、唯一值数量和变量类型 |
| `02_numeric_stats.csv` / `02_categorical_stats.csv` | 辅助判断极端分布、异常取值和类别稀疏问题 |
| `02_time_leakage.csv` | 时间泄露候选特征必须在进入特征池前确认处理结论 |
| `02_extreme_values.csv` | 记录极端值特征的保留、截断或剔除结论 |

除 `02_quality_decisions.csv` 外，阶段 2 质量报告均属于阶段 3 的**审计参考输入**。如果质量报告明细缺失，阶段 3 可以执行，但必须在 `03-output-list.xlsx` 中记录缺失项和风险；如果 `02_quality_decisions.csv` 缺失，或存在未确认的时间泄露候选特征，则不得完成阶段 3。

### 输入验收

执行特征工程前必须检查：

1. `sample_id` 在每个数据集内非空且唯一。
2. Train/Test/OOT 的候选特征列和数据类型一致。
3. Y_label 仅包含阶段 0 确认的好坏样本取值。
4. `sample_id`、Y_label、时间字段、`sample_weight` 未进入候选特征列表。
5. `02_quality_decisions.csv` 存在，且阶段 2 标记的时间泄露候选特征均有明确处理结论。
6. 所有筛选阈值均来自 `00_modeling_config.yaml`；阶段内新增确认结果应回写配置文件。

---

## 前置说明

特征工程在**训练集**上进行，所有筛选逻辑在训练集上确定后，再应用到 Test/OOT。

阶段 3 使用唯一的**活动特征池**逐步筛选，后续步骤只能接收上一环节输出，不得重新从原始字段生成候选列表：

```text
原始字段
→ 排除 sample_id、Y_label、时间字段、sample_weight
→ 应用阶段 2 质量决策
→ 应用规则变量确认结果
→ 预筛选
→ LR 初始分箱、平滑 IV 与单调性审阅
→ LR 人工调整并重新分箱；XGB/LGB 快速平滑 IV
→ IV 阈值筛选
→ 相关性去重
→ 最终特征集
```

**本阶段根据阶段 0 确认的模型选择，走不同的筛选路径**：

| 步骤 | LR 路径 | XGB/LGB 路径 |
|------|---------|-------------|
| 3.1 规则变量识别 | 执行 | 执行 |
| 3.2 预筛选 | 可选，需显式确认 | 可选，需显式确认 |
| 3.3 IV 计算与筛选 | 初始分箱后计算平滑 IV；人工调整完成后按最终 IV 筛选 | 快速平滑 IV |
| 3.4 相关性去重 | Pearson | Pearson |
| 3.5 单调性检验与人工调整 | 初始分箱后逐项审阅，调整后重新验证 | 移至阶段 4（见 §4.1.3） |
| 3.6 分箱 + WOE | **必须** | **不需要** |

> **执行（XGB）**: `python scripts/03_feature_engineering.py --train <01_train.csv> --test <01_test.csv> --oot <01_oot.csv> --target-col <Y列> --time-col <时间列> --model-type XGB`
> **执行（LR 首次分箱）**: `python scripts/03_feature_engineering.py --train <01_train.csv> --oot <01_oot.csv> --target-col <Y列> --time-col <时间列> --model-type LR --woe-smooth 0.5`
> **执行（LR 调整后重跑）**: 在上述命令增加 `--feature-adjustments <03_feature_adjustments.csv>`
> 仅在用户确认需要预筛选时增加 `--pre-filter`。

---

## 3.1 规则变量识别（预筛选前置步骤）

在预筛选之前，必须先识别**规则变量**：缺失率极高但非缺失部分完美（或近乎完美）分离好坏的特征。这类变量不是建模变量，而是业务规则。

**识别标准**（可通过 `--rule-missing-threshold` 和 `--rule-iv-threshold` 调整）：
- 缺失率 > 80%（默认）
- 非缺失部分 IV > 1.0（使用加性平滑 IV，smooth=0.5）
- 或非缺失部分仅含单一类别（全好/全坏）

**处理方式**：
1. 首次运行只标记候选规则变量，输出给用户确认，不直接剔除
2. 用户完成确认后重新执行，根据确认结果决定是否进入后续筛选
3. 确认为规则的变量从建模特征池中移除，可在建模完成后作为前置规则独立评估

> 输出文件: `03_rule_candidates.csv`

### 候选规则确认表

`03_rule_candidates.csv` 至少包含：

| 字段 | 说明 |
|------|------|
| `feature` | 特征名称 |
| `missing_rate` | 缺失率 |
| `non_missing_n` | 非缺失样本数 |
| `non_missing_bad_rate` | 非缺失样本坏账率 |
| `non_missing_iv` | 非缺失部分 IV |
| `candidate_reason` | 成为候选规则的原因 |
| `decision` | `pending` / `rule` / `feature` / `drop` |
| `decision_reason` | 用户确认原因 |
| `confirmed_by` | 确认人 |
| `confirmed_at` | 确认时间 |

### 确认流程

1. 首次运行生成候选规则文件，所有候选项默认为 `decision=pending`。
2. 只要存在 `pending`，暂停阶段 3，不执行后续预筛选、IV 筛选和相关性去重。
3. 用户更新确认表后重新执行：
   - `rule`：移出模型特征池，保留为业务规则候选。
   - `feature`：作为普通特征继续参与后续筛选。
   - `drop`：直接剔除，不作为业务规则。
4. 确认结果同步写入 `03_feature_decisions.csv` 和 `03-output-list.xlsx`。

不得提供将全部候选规则自动确认为 `rule` 的批量跳过参数，避免未经业务审核批量剔除强变量。

---

## 3.2 预筛选（可选）

预筛选用于快速剔除明显无效特征。是否执行由阶段 2 数据质量结果、无效特征情况和实际运行成本决定，并由用户确认；特征数量不作为自动触发条件。确认执行时传入 `--pre-filter`。

**筛选维度**：
1. **缺失率**：> 95%（可通过 `--missing-threshold` 调整）→ 剔除
2. **样本量**：非缺失样本 < 100 → 剔除
3. **唯一值**：≤ 1 → 剔除
4. **方差**：< 1e-6 → 剔除

**特征筛选的完整顺序**（XGB/LGB 路径）：
```
活动特征池
  → 规则变量确认（§3.1）→ 按 decision 更新活动特征池
  → 预筛选（§3.2）→ 在剩余活动特征中剔除缺失率 > 95% / 近零方差 / 样本过少
  → 快速平滑 IV 计算（默认 smooth=0.5）
  → IV 筛选（先看分布，用户确认阈值）
  → Pearson 去重（> 0.7 保留 IV 高的）
  → 最终特征集
```

> 输出文件: `03_pre_filter_dropped.csv`

---

## 3.3 IV 计算与阈值确认

### 平滑 WOE/IV 计算

LR 与 XGB/LGB 路径均使用一致的加性平滑公式，防止箱内 `bad_cnt=0` 或 `good_cnt=0` 导致 `ln(0)`。默认 `smooth=0.5`，LR 可通过 `--woe-smooth` 调整。

IV 公式：`IV = Σ (bad_pct - good_pct) × ln(bad_pct / good_pct)`
- bad_pct = (n_bad + smooth) / (n_bad_total + smooth × n_cats)
- good_pct = (n_good + smooth) / (n_good_total + smooth × n_cats)
- 缺失值单独作为一箱参与计算
- LR：先使用 Train 拟合分箱，再基于分箱结果计算平滑 WOE 和 IV；IV 筛选、最终 WOE 编码使用同一套平滑口径
- XGB/LGB：使用快速分组后的平滑 IV 进行筛选，不执行 WOE 编码

> 输出文件: `03_iv_table.csv`

### IV 阈值确认（自适应策略）

**必须先输出 IV 分布，再让用户确认阈值，不允许自行决定。**

使用 `AskUserQuestion` 工具让用户确认 IV 阈值。建议默认 0.02，根据分布情况可调整。也可以通过 `--iv-threshold` 直接指定，跳过交互确认。

LR 路径如执行手动分箱或 `u_shape_abs`，IV 阈值必须应用于调整后的最终 IV，不得使用调整前 IV 直接决定最终保留结果。

---

## 3.4 Pearson 相关性去重

**每次执行时可通过 `--corr-threshold` 指定阈值**，默认 0.7。相关对中保留 IV 更高的特征。

> 输出文件: `03_high_corr_pairs.csv`

---

## 3.5 单调性检验

- **LR 路径**：初始分箱后输出分箱边界及每箱 `count`、`good_cnt`、`bad_cnt`、`bad_rate`、WOE 和 IV contribution，并检查 bad_rate/WOE 单调性。存在非单调特征时，输出 `03_feature_adjustments.csv` 并暂停，等待用户逐项确认。
- **XGB/LGB 路径**：移至阶段 4 特征重要性输出后执行（见 §4.1.3），不在阶段 3 处理

### LR 人工调整动作

| `action` | 处理方式 |
|----------|----------|
| `keep` | 接受当前分箱及非单调形态，不调整 |
| `manual_bin` | 使用 `manual_breaks` 中已确认的 JSON 数组边界覆盖自动分箱 |
| `u_shape_abs` | 生成 `abs(X - Xmid)` 距离特征，原始特征退出模型特征池；必要时可同时提供 `manual_breaks` 微调转换后特征 |
| `drop` | 原始特征退出模型特征池，不生成替代特征 |

U 型转换约束：

1. `Xmid` 只能基于 Train 的 bad_rate 转折位置确定，并由用户确认。
2. 转换后的默认名称为 `<原特征>__abs_mid`，允许在 `output_feature` 中指定其他名称。
3. 缺失值转换后仍保持缺失。
4. Train/Test/OOT 必须使用同一个 `Xmid`；不得在 Test/OOT 重新估计。
5. `abs(X-Xmid)` 不能直接保证单调，转换后必须重新分箱并再次检查单调性。
6. 如果转换后仍非单调，应重新确认 `Xmid`，或在同一条 `u_shape_abs` 记录中补充 `manual_breaks`；确认前不得进入阶段 4。

`03_feature_adjustments.csv` 至少包含：

| 字段 | 说明 |
|------|------|
| `source_feature` / `output_feature` | 原始特征与最终输出特征 |
| `action` | 人工确认动作 |
| `x_mid` / `manual_breaks` | U 型转换中心点或人工分箱边界 |
| `before_iv` / `after_iv` | 调整前后 IV |
| `before_monotonic` / `after_monotonic` | 调整前后单调性 |
| `u_shape_candidate` | 初始分箱是否呈现 U 型候选 |
| `decision_reason` / `confirmed_by` / `confirmed_at` | 决策原因与确认记录 |

系统默认仅将非单调特征写入候选调整表；用户也可以为其他需要业务分箱微调的特征新增记录。

---

## 3.6 分箱与 WOE 编码（LR 路径专属）

LR 路径使用 `toad.Combiner` 在 Train 上拟合分箱规则，再使用一致加性平滑公式计算 WOE 和 IV。是否使用或分批执行 `toad.Combiner` 应根据代表性数据性能测试决定，不设置固定样本量或特征数门槛。

### 关键约束

- **单特征分箱不超过 6 箱**
- **缺失值单独作为一箱**（`empty_separate=True`）
- 分箱方法：`chi`（卡方）、`dt`（决策树）、`quantile`（等频）、`step`（等距）
- 默认使用卡方分箱（`method="chi"`），可根据特征类型调整

### WOE 转换

在 Train 分箱结果上拟合平滑 WOE 映射，应用至 Test/OOT **不重新拟合**。默认 `smooth=0.5`；WOE 与 IV 必须使用相同平滑参数和公式。

---

## 3.7 保存最终特征集

导出最终的 Train/Test/OOT 特征集 + 特征列表文件。

最终数据集必须保留以下非特征字段：

- `sample_id`：用于阶段 4-7 的预测、评分、监控结果回溯
- Y_label：用于训练和评估
- `sample_weight`：仅在阶段 1 执行抽样且生成权重时保留

`sample_id`、Y_label 和 `sample_weight` 不得写入 `03_final_features.txt`，也不得作为模型训练特征。

> 输出文件: `03_train_final.csv`, `03_test_final.csv`, `03_oot_final.csv`, `03_final_features.txt`
> LR 路径输出: `03_train_woe.csv`, `03_oot_woe.csv`

---

## 3.8 阶段 3 输出项

阶段 3 输出分为三类：

### 3.8.1 核心下游输出

阶段 4 必须消费的文件：

1. `03_train_final.csv`, `03_test_final.csv`, `03_oot_final.csv`（XGB/LGB 路径），或 `03_train_woe.csv`, `03_oot_woe.csv`（LR 路径）
2. `03_final_features.txt`：仅包含最终入模特征名，不包含任何保留字段
3. LR 路径额外输出可复用的 `03_combiner.pkl` 和平滑 WOE 映射对象 `03_woe_transformer.pkl`

### 3.8.2 审计输出

用于还原特征筛选过程和人工决策：

1. `03_rule_candidates.csv`：候选规则变量及用户确认结论
2. `03_pre_filter_dropped.csv`：预筛选剔除变量、剔除原因和关键指标
3. `03_iv_table.csv`：全量 IV、是否通过阈值及最终状态
4. `03_high_corr_pairs.csv`：高相关变量对、保留变量、剔除变量和选择依据
5. `03_feature_decisions.csv`：每个原始候选特征的最终状态及完整原因链
6. `03_feature_adjustments.csv`：LR 调整动作、转换参数及调整前后 IV/单调性
7. `03_binning_detail.csv`：LR 调整前后逐箱 count、bad_rate、WOE 和 IV contribution
8. `00_modeling_config.yaml`：回写阶段 3 最终阈值及用户确认状态

### 3.8.3 展示汇总输出

阶段 3 结束时，统一输出 `03-output-list.xlsx`，包含以下内容：

1. 规则变量识别结果（候选规则变量列表 + 用户确认记录）
2. 预筛选报告（XGB/LGB 路径）：各维度剔除的特征数量
3. IV 分布报告：各阈值下的特征数量 + IV Top 20 + 用户确认的 IV 阈值
4. 相关性去重记录：剔除的特征对及保留/剔除原因
5. 分箱及人工调整结果（LR 路径）：调整前后分箱、U 型转换、人工边界、IV 和单调性变化
6. 最终特征集：特征数量、特征列表；转换特征必须标注原始字段和转换方式
7. 阶段开始/结束时间戳

### 3.8.4 输出验收规则

阶段 3 完成前必须检查：

1. 各适用数据集的输出行数分别与阶段 1 对应输入一致。
2. 各适用输出数据集的最终特征列、顺序和数据类型一致。
3. `sample_id` 在输出中非空、唯一，且与阶段 1 输入逐行可追溯。
4. `03_final_features.txt` 与最终数据集中的模型特征完全一致，不包含保留字段。
5. 每个原始候选特征在 `03_feature_decisions.csv` 中恰好有一条最终结论。
6. 所有人工确认项均有确认结果；不得以“待确认”状态进入阶段 4。
7. LR 路径能够使用保存的 Combiner/平滑 WOE 映射对象对 Test/OOT 重现相同转换。
8. 所有 `u_shape_abs` 特征均已移除原始特征，仅保留转换后字段，并在输出表中记录 `Xmid`、调整前后 IV 和单调性。

### 输出文件清单
| 文件 | 说明 |
|------|------|
| `03-output-list.xlsx` | 阶段 3 输出汇总（以上 7 项） |
| `03_rule_candidates.csv` | 候选规则变量、关键指标和用户确认结论 |
| `03_pre_filter_dropped.csv` | 预筛选剔除变量、剔除原因和关键指标 |
| `03_iv_table.csv` | 全量 IV、阈值通过状态和最终状态 |
| `03_high_corr_pairs.csv` | 高相关特征对、保留/剔除变量及原因 |
| `03_feature_decisions.csv` | 原始候选特征的完整筛选决策链 |
| `03_feature_adjustments.csv` | LR 人工分箱/U 型转换确认表及调整前后结果 |
| `03_binning_detail.csv` | LR 调整前后逐箱统计、WOE 和 IV contribution |
| `03_train_final.csv` | 训练集最终特征（XGB/LGB），保留 `sample_id`、Y_label 和可选权重 |
| `03_test_final.csv` | 测试集最终特征（XGB/LGB），保留 `sample_id`、Y_label 和可选权重 |
| `03_oot_final.csv` | OOT 最终特征（XGB/LGB），保留 `sample_id`、Y_label 和可选权重 |
| `03_train_woe.csv` | 训练集 WOE 编码（LR），保留 `sample_id`、Y_label 和可选权重 |
| `03_oot_woe.csv` | OOT WOE 编码（LR），保留 `sample_id`、Y_label 和可选权重 |
| `03_combiner.pkl` | LR 分箱规则对象 |
| `03_woe_transformer.pkl` | LR 平滑 WOE 映射对象，包含平滑参数和各箱映射 |
| `03_final_features.txt` | 最终模型特征列表，不包含保留字段 |

---

## 后续阶段入口

阶段 3 输出确认无误后，进入阶段 4（references/4-model-training.md）。
