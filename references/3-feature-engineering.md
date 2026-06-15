# 阶段 3：特征工程

> **可执行脚本**: `scripts/03_feature_engineering.py` — 运行 `python scripts/03_feature_engineering.py --help` 查看用法。

## 3.0 阶段信息

- **执行必需输入**: `01_train.csv`、`01_oot.csv` + `00_modeling_config.yaml`；XGB/LGB 路径额外需要 `01_test.csv`
- **审计参考输入**: 阶段 2 已生成的 EDA 质量报告与明细文件
- **输出**: 核心下游数据 + 审计明细 + `03-output-list.xlsx`
- **耗时记录**: 阶段开始和结束时记录时间戳（精确到分钟）

---

## 3.0.1 输入契约

正式运行首先读取 `02_stage_manifest.json`，并仅通过该 manifest 获取累计声明的阶段 0-2 产物路径。

| 输入 | 必需字段/内容 | 用途 |
|------|--------------|------|
| `01_train.csv` | `sample_id`、`Y_label`、候选特征；抽样时包含 `sample_weight` | 仅使用 Train 拟合筛选规则、分箱和转换器 |
| `01_test.csv` | `sample_id`、`Y_label`、与 Train 一致的候选特征 | XGB/LGB 路径必需 |
| `01_oot.csv` | `sample_id`、`Y_label`、与 Train 一致的候选特征 | 应用 Train 确定的特征处理规则 |
| `00_modeling_config.yaml` | `model_type`、`target_col`、`sample_id_col`、`time_col`、缺失值配置、阶段 3 阈值及用户确认状态 | 提供机器可读取的全局配置 |
| `03_feature_adjustments.csv` | LR 路径首次分箱后生成；首次运行可缺失，确认后重跑时必需 | 提供手动分箱、U 型转换、保留或剔除决策 |

---

## 3.1 总体流程

阶段 3 使用唯一的活动特征池逐步筛选，后续步骤只能接收上一环节输出，不得重新从原始字段生成候选列表：

```text
原始字段
→ 排除 sample_id、Y_label、时间字段、sample_weight
→ 应用阶段 2 质量决策
→ 识别并确认规则变量
→ 可选预筛选
→ 全模型先计算快速 IV
→ XGB/LGB: 快速 IV 筛选 + 相关性去重
→ LR: 快速 IV 预筛 + 精细分箱 + 精细 IV + 单调性检查 + 人工调整
→ 输出最终特征集
```

### 3.1.1 模型分支要求

| 步骤 | LR 路径 | XGB/LGB 路径 |
|------|---------|-------------|
| 规则变量识别 | 执行 | 执行 |
| 预筛选 | 可选，需显式确认 | 可选，需显式确认 |
| 快速 IV | 必须 | 必须 |
| 精细分箱/精细 IV | 必须 | 不执行 |
| 单调性检查 | 必须 | 不执行 |
| 相关性去重 | 执行 | 执行 |
| WOE 编码 | 必须 | 不执行 |

---

## 3.2 规则变量识别

在预筛选之前，必须先识别规则变量：缺失率极高但非缺失部分近乎完美分离好坏的特征。

首次运行只输出候选，不自动剔除，必须等待用户确认。

输出文件：`03_rule_candidates.csv`

---

## 3.3 可选预筛选

预筛选仅在用户明确要求时执行。常见维度：

1. 缺失率
2. 非缺失样本量
3. 唯一值数量
4. 方差

输出文件：`03_pre_filter_dropped.csv`

---

## 3.4 全模型先计算快速 IV

无论最终模型是 `LR`、`XGB`、`LGB`，都必须先计算快速 IV，并输出：

- `03_fast_iv_table.csv`
- `03_high_iv_alerts.csv`

快速 IV 的作用：

1. 提供统一、低耗时的首轮筛选口径
2. 为相关性去重提供比较基准
3. 为 LR 路径提供“先粗后细”的入口

### 3.4.1 高 IV 告警

IV 极高的特征必须单独提醒，不得默认视为优质特征。告警原因至少包括：

- 可能存在特征穿越
- 某些箱 `bad=0` 或 `good=0`
- 特征确实极强，但需人工复核

---

## 3.5 XGB/LGB 路径

`XGB/LGB` 路径阶段 3 只做两件事：

1. **快速 IV 筛选**
2. **相关性去重**

### 3.5.1 快速 IV 筛选

先输出 IV 分布，再由用户确认阈值。未确认前不应自作主张固定阈值。

### 3.5.2 相关性去重

对两两相关性超过阈值的特征，只保留快速 IV 更高的特征。

> 输出文件: `03_high_corr_pairs.csv`

XGB/LGB 路径在阶段 3 **不做**：

- 精细分箱
- WOE 编码
- 单调性检查

---

## 3.6 LR 路径

LR 路径采用“两段式”筛选：

1. **快速 IV 预筛**
2. **精细分箱 + 精细 IV + 单调性检查 + 人工调整**

### 3.6.1 快速 IV 预筛

先基于快速 IV 进行首轮粗筛，减少后续精细分箱成本。若全部低于阈值，至少保留少量候选供后续复核，不应直接筛空。

### 3.6.2 精细分箱与精细 IV

对快速 IV 通过的候选特征，在 Train 上进行分箱，输出精细 IV 与单调性结果。

### 3.6.3 单调性检查与人工调整

存在非单调特征时，输出 `03_feature_adjustments.csv` 并暂停，等待用户逐项确认。可选动作：

- `keep`
- `manual_bin`
- `u_shape_abs`
- `drop`

### 3.6.4 WOE 编码

在 Train 分箱结果上拟合平滑 WOE 映射，并应用到 Test/OOT，不重新拟合。

---

## 3.7 增量重跑原则

用户确认后的阶段 3 重跑，应优先采用**局部重算**而不是全量重跑。

原则如下：

1. 已确认且未受影响的特征，不重复跑精细分箱和精细 IV。
2. 仅对以下受影响特征增量重算：
   - 新增调整动作的特征
   - 修改 `manual_breaks` / `x_mid` 的特征
   - 受其相关性去重结果影响的特征
3. 重新合并更新后的结果表，再输出新的最终特征集。

如果当前脚本能力尚未完全实现缓存式增量引擎，也必须在设计和后续实现中以该原则为目标，不应把“有一个 pending”解释为“全量特征重跑”。

---

## 3.8 LR 输出口径

LR 相关 WOE/IV 文件分为两层：

### 3.8.1 特征级汇总文件

`03_iv_table.csv` 至少包含：

- 特征名
- 最终 IV
- 调整前后 IV
- 调整前后单调性
- 是否入模
- 决策原因

### 3.8.2 分箱级明细文件

`03_binning_detail.csv` 至少包含每个箱的：

- `bad_count`
- `good_count`
- `total_count`
- `pct` 或等价占比字段
- `lift`
- `woe`
- `bad_rate`
- `iv_component`

必要时可以同时保留兼容字段，如 `count`、`good_cnt`、`bad_cnt`，但上述字段必须明确存在。

---

## 3.9 输出项

### 3.9.1 核心下游输出

1. `03_train_final.csv`、`03_test_final.csv`、`03_oot_final.csv`（XGB/LGB 路径）
2. `03_train_woe.csv`、`03_oot_woe.csv`（LR 路径）
3. `03_final_features.txt`
4. LR 路径额外输出 `03_combiner.pkl` 和 `03_woe_transformer.pkl`

### 3.9.2 审计输出

1. `03_rule_candidates.csv`
2. `03_pre_filter_dropped.csv`
3. `03_fast_iv_table.csv`
4. `03_high_iv_alerts.csv`
5. `03_iv_table.csv`
6. `03_high_corr_pairs.csv`
7. `03_feature_decisions.csv`
8. `03_feature_adjustments.csv`
9. `03_binning_detail.csv`
10. `00_modeling_config.yaml`

### 3.9.3 阶段验收规则

阶段 3 完成前必须检查：

1. 各适用数据集输出行数与阶段 1 对应输入一致。
2. 各适用输出数据集的最终特征列、顺序和数据类型一致。
3. `sample_id` 在输出中非空、唯一，且与阶段 1 输入逐行可追溯。
4. `03_final_features.txt` 与最终数据集中的模型特征完全一致，不包含保留字段。
5. 每个原始候选特征在 `03_feature_decisions.csv` 中恰好有一条最终结论。
6. 所有人工确认项均有确认结果；不得以 `pending` 状态进入阶段 4。
7. LR 路径能够使用保存的 Combiner/WOE 对 Test/OOT 重现相同转换。
8. 高 IV 告警已单独输出并在汇总文件中展示。

---

## 后续阶段入口

阶段 3 输出确认无误后，进入阶段 4（`references/4-model-training.md`）。
