# 阶段 1：数据准备与样本划分

> **可执行脚本**: `scripts/01_data_prep.py` — 运行 `python scripts/01_data_prep.py --help` 查看用法。

## 1.0 阶段信息

- **执行必需输入**: 原始数据文件 + `00_modeling_config.yaml`
- **核心输出**: Train/OOT 数据集；XGB/LGB 路径额外输出 Test；存在灰样本时额外输出 Grey
- **审计输出**: 样本划分、主键检查、抽样权重和缺失率明细
- **展示输出**: `01-output-list.xlsx`
- **耗时记录**: 阶段开始和结束时记录时间戳（精确到分钟）

---

## 1.0.1 输入契约

| 输入 | 必需内容 | 用途 |
|------|----------|------|
| 原始数据 | `Y_label`、时间字段、候选特征及样本唯一标识来源字段 | 生成建模样本 |
| `00_stage_manifest.json` | 阶段 0 正式交接入口，声明主配置与原始数据路径 | 禁止绕过交接清单猜测输入 |
| `00_modeling_config.yaml` | 通过 manifest 获取；模型类型、字段定义、标签定义、缺失值配置、样本划分配置 | 驱动数据准备流程 |

阶段 1 不允许在代码中重新定义 `Y_label`、时间字段或模型类型。需要变更时，应先更新配置并保留确认记录。

---

## 1.0.2 开始前强校验

阶段 1 启动前必须先检查以下确认项；存在 `pending` 时必须中止正式执行：

1. `model_type` 已由用户确认。
2. `sample_split.oot_method` 已由用户确认。
3. `sample_split.sample_method` 已由用户确认；未确认时不得默认 `stratified`。
4. 缺失值定义已由用户确认。
5. `sample_id` 方案已确认；未确认时不得默认 `content_hash`。

阶段 1 可以先读取表头和少量样本（如前 5-100 行）做字段检查与候选搜索，但不得在上述项目未确认时输出正式 Train/Test/OOT 文件。

---

## 1.1 按时间切出 OOT

无论 LR 还是 XGB/LGB，OOT 的切分逻辑一致：**取时间范围最近的一段作为 OOT**，确保其与当前客群最接近，同时留足表现期。

OOT 切分方式必须由用户确认，不做预设。典型方式包括：

- 指定 OOT 起始日期：`--oot-method date --oot-start-date`
- 指定 OOT 最近 N 个月：`--oot-method months --oot-months N`
- 指定 OOT 比例：`--oot-method proportion --oot-proportion p`

正常建议：

1. 先用 pandas 轻量读取数据，识别时间字段并查看时间分布。
2. 若用户未另行指定，可建议“最近 1-2 个月作为 OOT，剩余进入开发样本”。
3. 开发样本内再按模型路径拆分 Train / Test。

---

## 1.2 建模样本内部切分

### 1.2.1 LR 模式

- 建模样本全部作为 Train
- LR 的参数寻优和验证在 Train 内使用 k-fold 交叉验证完成
- OOT 仅用于最终泛化能力评估
- 默认不生成 `01_test.csv`

### 1.2.2 XGB/LGB 模式

- 在剔除 OOT 后的开发样本内部按 `7:3` 随机切分 Train / Test，除非用户另有明确要求
- **Train**：用于模型训练
- **Test**：用于调参
- **OOT**：用于最终泛化检验

如果用户未单独指定切分比例，默认采用 `7:3`，但该默认也应在阶段 0 或阶段 1 展示给用户。

---

## 1.3 抽样方法

### 1.3.1 必须先询问用户

建模样本量取决于数据规模，需要向用户确认：

1. 是否做抽样：`full` / `stratified`
2. 目标样本量是多少
3. 是否需要样本权重

未确认前：

- `sample_split.sample_method` 必须保持 `pending`
- 不得因为脚本支持分层抽样就直接执行 `stratified`

### 1.3.2 全量方案

样本量可承受时，直接使用全部样本：

> **执行**: `python scripts/01_data_prep.py --sample-method full`

### 1.3.3 分层抽样方案

仅当用户明确要求时，才允许使用分层抽样：

> **执行**: `python scripts/01_data_prep.py --sample-method stratified --target-total 100000`

抽样逻辑：

- 坏样本全量保留
- 好样本按目标总量抽取
- 若生成 `sample_weight`，后续训练和评估应显式传递

---

## 1.4 样本唯一标识生成与固化

阶段 1 必须为每条样本保留稳定的 `sample_id`：

1. 优先使用阶段 0 已确认的单字段业务主键。
2. 若需要组合字段唯一，使用已确认的 `sample_id_source_cols` 生成稳定标识。
3. 若原始数据不存在稳定唯一标识，应先搜索常见候选字段，如 `apply_no`、手机号/身份证脱敏列、`md5`、`hash` 类字段，并向用户展示结果。
4. 只有在以上方案均不可用且用户明确批准时，才允许生成 `content_hash_confirmed`。
5. `sample_id` 不得因样本切分、排序或重新运行而变化。
6. `sample_id` 只用于追溯和数据关联，不得进入候选特征池。

> 输出文件: `01_sample_id_audit.csv`

---

## 1.5 灰样本处理

当 `Y_label` 中存在非 0 非 1 的灰样本时，处理逻辑固定为：

1. 识别灰样本
2. 将灰样本从建模数据中剔除并单独保存
3. 使用纯二分类样本建模
4. 建模完成后对灰样本单独打分

灰样本不参与训练、验证和评估。

---

## 1.6 缺失值初始摸排

阶段 1 不做缺失值剔除，不设硬阈值，只做摸排记录。

前提约束：

- 缺失值口径必须沿用阶段 0 已确认结果
- 未确认前只允许输出检查表，不得把默认 `-9999/-999` 当作正式口径

建议做法：

1. 使用 pandas 快速统计缺失率
2. 可以使用 `.isna()`、`.describe()` 或逐列 `apply`
3. 输出 `01_missing_report.csv`

输出字段：`column`、`missing_count`、`missing_rate`、`missing_rate_pct`

---

## 1.7 输出项

### 1.7.1 核心下游输出

| 文件 | 适用路径 | 说明 |
|------|----------|------|
| `01_train.csv` | 全部模型 | 训练数据，保留 `sample_id`、`Y_label`、时间字段和可选 `sample_weight` |
| `01_test.csv` | XGB/LGB | 调参数据；LR 默认不生成 |
| `01_oot.csv` | 全部模型 | OOT 数据，保留与 Train 一致的字段 |
| `01_grey_samples.csv` | 存在灰样本时 | 灰样本数据，供阶段 6 单独评分 |

### 1.7.2 审计输出

| 文件 | 说明 |
|------|------|
| `01_split_audit.csv` | 每个数据集的样本量、坏账率、时间范围和切分规则 |
| `01_sample_id_audit.csv` | `sample_id` 生成方式、唯一性、空值和重复检查 |
| `01_sampling_audit.csv` | 抽样策略、抽样前后数量和权重计算结果 |
| `01_missing_report.csv` | 缺失率摸排报告 |
| `01_stage_manifest.json` | 阶段 1 正式交接清单，累计声明后续所需产物 |

### 1.7.3 阶段验收规则

阶段 1 完成前必须检查：

1. `sample_id` 在 Train/Test/OOT/Grey 各数据集内非空且唯一，各数据集之间互斥。
2. 各数据集样本量之和与原始数据经灰样本处理后的数量能够核对。
3. Train/Test/OOT 的字段名称、顺序和数据类型一致；LR 路径不要求 Test。
4. `Y_label` 只包含当前数据集允许的标签值；Grey 不进入 Train/Test/OOT。
5. 时间切分符合配置，OOT 时间不得早于建模样本时间范围。
6. 使用抽样时，`sample_weight` 已生成并通过总体数量/坏账率还原检查。
7. 样本划分和确认结果已回写 `00_modeling_config.yaml`。

---

## 后续阶段入口

阶段 1 输出确认无误后，进入阶段 2（`references/2-eda.md`）。
