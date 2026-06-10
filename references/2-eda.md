# 阶段 2：EDA（数据探索性分析）

> **可执行脚本**: `scripts/02_eda.py` — 运行 `python scripts/02_eda.py --help` 查看用法。

## 2.0 阶段信息

- **执行必需输入**: `01_train.csv` + `00_modeling_config.yaml`
- **核心下游输出**: `02_quality_decisions.csv`（阶段 3 必须读取）
- **审计输出**: EDA 统计量和数据质量检查明细
- **展示输出**: `02-output-list.xlsx` + EDA 图表
- **耗时记录**: 阶段开始和结束时记录时间戳（精确到分钟）

---

## 2.0.1 输入契约

| 输入 | 必需内容 | 用途 |
|------|----------|------|
| `01_train.csv` | `sample_id`、Y_label、时间字段、候选特征及可选 `sample_weight` | 仅使用 Train 执行 EDA 和质量检查 |
| `00_modeling_config.yaml` | 字段定义、标签定义、缺失值配置、保留字段列表 | 驱动分析并排除非特征字段 |

`sample_id`、Y_label、时间字段、`sample_weight` 属于保留字段。除专门分析 Y_label 和时间分布外，不得参与特征质量统计、相关性分析或候选特征结论。

---

## 前置说明

EDA 在**训练集**上进行（避免数据泄露到验证集/OOT）。所有图表**统一使用 plotly 生成**，导出为 PNG 格式（`fig.write_image()`），兼顾美观与可嵌入 xlsx 的便利性。

> **执行**: `python scripts/02_eda.py --input <01_train.csv> --target-col <Y列> --time-col <时间列> --sample-id-col sample_id --output-dir <输出目录>`

---

## 2.1 样本时间分布

确认样本在时间维度上的分布，识别是否存在时间断层。

**输出**：
- 每月样本量柱状图
- 每月坏账率折线图（含均值参考线）

> 输出文件: `02_time_distribution.png`, `02_monthly_stats.csv`

**关注点**：
- 是否存在某些月份样本量异常少（可能是数据截断或业务停摆）
- 坏账率是否随时间有明显趋势变化（可能是宏观环境变化或客群迁移）
- 若坏账率随时间单调变化，可能需要在阶段 4 的 LR 模型中考虑引入时间相关控制变量

---

## 2.2 Y_label 整体分布

在训练集上统计 Y_label 的类别分布。

**输出**：
- Y_label 类别分布柱状图（Good/Bad/Grey，标注数量和占比）
- 控制台输出：样本总量、坏账率、灰样本占比（如有）

> 输出文件: `02_target_distribution.png`

---

## 2.3 特征概览

对全部候选特征做结构化概览，目的是快速了解特征类型、数据完整性、取值范围和分布，不在本阶段做筛选。

默认类型识别口径：

- 数值类型且非缺失唯一值数量 > 20：连续特征
- 非数值类型，或非缺失唯一值数量 ≤ 20：离散特征

如业务定义与自动识别结果不一致，应以业务确认类型为准。

所有连续和离散特征均需统计：

| 指标 | 说明 |
|------|------|
| `count` | 总样本数 |
| `non_missing_count` | 非缺失样本数 |
| `unique` | 非缺失唯一值数量 |
| `missing` | 缺失样本数 |
| `missing_rate` | 缺失率 |
| `min` | 最小值；离散特征按可排序文本值统计，不适用时留空 |
| `max` | 最大值；离散特征按可排序文本值统计，不适用时留空 |
| `feature_type` | `continuous` 或 `categorical` |

> 输出文件: `02_feature_overview.csv`

### 2.3.1 连续特征统计

连续特征除通用概览指标外，还需输出：

- mean / std
- p01 / p05 / p10 / p25 / p50 / p75 / p90 / p95 / p99

用于了解中心趋势、离散程度、偏态和极端值情况。

> 输出文件: `02_numeric_stats.csv`

### 2.3.2 离散特征统计

离散特征除通用概览指标外，还需按每个离散取值输出：

- `value`
- `count`
- `pct`

缺失值作为单独类别统计，默认输出全部取值。

> 输出文件: `02_categorical_stats.csv`

---

## 2.4 特征质量概览（toad.detector）

使用 `toad.detector.detect()` 输出数据探测报告，包含各特征的缺失率、unique 值数量、分布类型等。

`02_toad_detect.csv` 作为 `2.3 特征概览` 的补充质量诊断，不替代 `02_feature_overview.csv`、连续特征统计或离散特征频数统计。

> 输出文件: `02_toad_detect.csv`

**关注点**：
- 是否有特征 unique 值数量极少（接近常量，建模价值低）
- 跨特征对比缺失率，为阶段 3 特征筛选提供背景

## 2.5 数据质量检查

### 2.5.1 时间泄露风险检查

计算各数值特征与时间列的相关系数，|r| > 0.5 的特征标记为可疑。

> 输出文件: `02_time_leakage.csv`

### 2.5.2 极端值检查

检查数值特征超过 threshold 个标准差（默认 5σ）的极端值。

> 通过 `--extreme-threshold` 调整阈值，输出文件: `02_extreme_values.csv`

---

## 2.5.3 质量问题确认表

将阶段 2 发现的时间泄露、极端值、异常类型、高缺失率和稀疏类别等问题统一写入 `02_quality_decisions.csv`，至少包含：

| 字段 | 说明 |
|------|------|
| `feature` | 特征名称 |
| `issue_type` | `time_leakage` / `extreme_value` / `high_missing` / `sparse_category` / `type_anomaly` |
| `issue_metric` | 触发问题的关键指标 |
| `suggested_action` | 建议处理方式 |
| `decision` | `pending` / `keep` / `drop` / `transform` / `cap` |
| `decision_detail` | 转换、截断或其他处理细节 |
| `confirmed_by` | 确认人 |
| `confirmed_at` | 确认时间 |

阶段 2 可以在存在 `pending` 时完成 EDA，但必须在汇总报告中突出展示。所有 `time_leakage` 项必须在阶段 3 完成前确认，其他问题可由阶段 3 根据确认状态执行或记录风险。

---

## 2.6 阶段 2 输出项

### 2.6.1 核心下游输出

| 文件 | 说明 |
|------|------|
| `02_quality_decisions.csv` | 阶段 2 质量问题及人工确认结论，阶段 3 必须读取 |

### 2.6.2 审计输出

阶段 2 的统计量和数据质量检查结果均属于审计输出，用于解释阶段 3 的特征处理决策，不直接作为模型训练数据。

### 2.6.3 展示汇总输出

阶段 2 结束时，统一输出 `02-output-list.xlsx`，包含以下内容：

1. 样本时间分布图（逐月样本量 + 坏账率趋势）
2. Y_label 分布图（好/坏/灰占比）
3. 特征概览（全部候选特征的 count / unique / max / min / missing）
4. 连续特征分位数统计
5. 离散特征 value count + pct 统计
6. toad.detector 补充质量诊断
7. 时间泄露风险检查结果
8. 极端值检查结果
9. 质量问题确认状态汇总
10. 阶段开始/结束时间戳

### 输出文件清单

| 文件 | 说明 |
|------|------|
| `02-output-list.xlsx` | 阶段 2 输出汇总（以上 10 项） |
| `02_quality_decisions.csv` | 质量问题候选项、建议动作和人工确认结论 |
| `02_monthly_stats.csv` | 逐月样本量与坏账率 |
| `02_time_distribution.png` | 样本时间分布图（plotly） |
| `02_target_distribution.png` | Y_label 分布图（plotly） |
| `02_feature_overview.csv` | 全部候选特征通用概览 |
| `02_toad_detect.csv` | toad 数据探测报告 |
| `02_numeric_stats.csv` | 连续特征描述性统计和分位数 |
| `02_categorical_stats.csv` | 离散特征各取值 count + pct |
| `02_time_leakage.csv` | 时间泄露风险检查结果 |
| `02_extreme_values.csv` | 极端值检查结果 |

### 2.6.4 阶段验收规则

阶段 2 完成前必须检查：

1. EDA 仅使用 Train，未读取 Test/OOT 标签或统计结果。
2. `sample_id`、时间字段和 `sample_weight` 未被当作候选模型特征分析。
3. 所有数据质量检查结果都能关联到 `02_quality_decisions.csv`。
4. 时间泄露候选、极端值等异常项在 `02-output-list.xlsx` 中有明确汇总。
5. 所有输出均记录所使用的配置版本、Train 样本量和生成时间。
6. 阶段 2 未自行删除或修改阶段 1 数据。

---

## 后续阶段入口

阶段 2 输出确认无误后，进入阶段 3（references/3-feature-engineering.md）。
