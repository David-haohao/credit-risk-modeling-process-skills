# 阶段 6：评分卡转换

> **可执行脚本**: `scripts/06_scorecard.py` — 运行 `python scripts/06_scorecard.py --help` 查看用法。

## 6.0 阶段信息

- **输入**: 阶段 4 模型文件 + 阶段 5 预测结果 + 训练集实际坏账率 + `01_grey_samples.csv`（如有）
- **输出**: `06-output-list.xlsx` + 打分结果 CSV + 风险等级分布 + 分数分布图
- **耗时记录**: 阶段开始和结束时记录时间戳（精确到分钟）

---

## 前置说明

评分卡转换的核心步骤：

1. **概率校准**：LR 和 XGB/LGB **均需要**进行概率校准
2. **概率 → 分数映射**：线性变换为整数分（base_score / base_odds / pdo）
3. **风险等级切分**：基于分数分位数切分为 A/B/C/D/E 五级
4. **灰样本打分**（如有）：使用最终评分公式对灰样本打分

**概率校准差异**：
- **LR**：输出概率近天然校准（log-loss 优化），但仍需校验。偏差较大时需调整 base_odds
- **XGB/LGB**：输出概率**不代表**真实逾期率，**必须**通过 Platt Scaling 校准后才能映射为分数

> **执行（XGB）**: `python scripts/06_scorecard.py --pred-train <04_pred_train.csv> --pred-oot <04_pred_oot.csv> --model-type XGB --base-score 600 --base-odds 35 --pdo 60`
> **执行（LR）**: `python scripts/06_scorecard.py --pred-train <04_pred_train.csv> --pred-oot <04_pred_oot.csv> --model-type LR --base-score 600 --base-odds 35 --pdo 60`

---

## 评分卡参数

以下参数通过 CLI 指定，有合理默认值。建议在首次执行时向用户确认。

| 参数 | 含义 | 默认值 | CLI 参数 |
|------|------|--------|----------|
| base_score | 基准分 | 600 | `--base-score` |
| base_odds | 基准 good/bad 比 | 根据训练集坏账率反推 | `--base-odds` |
| pdo | Odds 翻倍时分数变化 | 60 | `--pdo` |
| rate | odds 变化倍数 | 2 | `--rate` |

**base_odds 默认公式**：`base_odds = (1 - bad_rate) / bad_rate`（自动从训练集计算）

---

## 6.1 Platt Scaling 概率校准（XGB/LGB 专属）

XGB/LGB 的预测概率不直接代表真实违约概率。Platt Scaling 通过训练一个无正则化的 LR（sigmoid）将原始预测概率映射为校准后的真实概率。

**原理**：将模型原始预测概率作为唯一特征 X'，用真实标签 Y 训练一个 `LogisticRegression(penalty=None)`。由于 sigmoid 是单调的，校准不会改变模型的排序性。

校准后，XGB/LGB 的评分直接从校准概率通过线性公式计算，**不依赖 Combiner 和 WOETransformer**，因此无法输出逐特征逐箱的评分卡明细表。

---

## 6.2 概率 → 分数映射

### 6.2.1 Odds 偏差校准（客群漂移修正）

当开发样本的坏账率与实际线上客群的坏账率存在显著偏差时（例如开发样本坏账率 10%，实际客群仅 2%），需要对基准 Odds 进行修正：

```
ln(Odds_calibrated) = ln(Odds_actual) - ln(Odds_expected)
adjusted_base_odds = base_odds × exp(ln(Odds_calibrated))
```

> 通过 `--train-bad-rate` 和 `--actual-bad-rate` 启用 Odds 校准。

**注意**：Odds 校准仅调整基准点，不改变特征分箱和各特征的点数分配。如果校准后 base_odds 变化显著（如 > 50%），建议考虑重新建模。

### 6.2.2 LR：评分卡构建（toad.ScoreCard）

使用 `toad.ScoreCard`，依赖阶段 3 拟合的 Combiner 和 WOETransformer。

输出：评分卡明细 DataFrame（逐特征逐箱的分数分配）。

### 6.2.3 XGB/LGB：评分公式

XGB/LGB 经 Platt 校准后，不依赖分箱逻辑，直接通过线性公式计算：

```
Score = A - B × ln(Odds)
  B = pdo / ln(rate)
  A = base_score + B × ln(base_odds)
  Odds = p_calibrated / (1 - p_calibrated)
```

分数截断到 [300, 900] 范围。

---

## 6.3 风险等级切分

等级切分属于风控策略范畴，默认执行。可通过 `--skip-grade-cut` 跳过。

- `--grade-labels`：等级标签，默认 `A,B,C,D,E`（A=低风险, E=高风险）
- `--grade-method`：切分方式，`quantile`（等频，默认）或 `score`（固定阈值）
- `--grade-cuts`：固定阈值切分点（`grade-method=score` 时使用）

**风险等级方向**：高分 = 高逾期概率 = 高风险（E 级）。

等级切分后需验证：**A → E 的 bad_rate 必须单调递增**。

> 输出文件: `06_grade_stats_train.csv`

---

## 6.4 分数分布可视化

Train 和 OOT 分数分布叠加直方图（probability density），用于直观判断分数分布是否漂移。使用 plotly 生成 PNG。

> 输出文件: `06_score_distribution.png`

---

## 6.5 灰样本打分

当阶段 1 识别出灰样本时，在建模完成后对其进行打分。

流程：
1. 使用模型预测原始概率
2. Platt Scaling 校准（XGB/LGB）
3. 概率 → 分数映射
4. 风险等级划分

输出各等级的灰样本数量和占比。

---

## 6.6 阶段 6 输出项

阶段 6 结束时，统一输出 `06-output-list.xlsx`，包含以下内容：

1. Platt 校准记录（XGB/LGB **必须**，LR 校验后决定）
2. base_odds 确认记录（用户确认值 + 建议值 + 训练集坏账率）
3. 评分卡公式（XGB/LGB）或明细表（LR）
4. 各数据集分数分布图（Train + OOT 叠加）
5. 风险等级切分方案（切分方式 + 阈值 + 各等级分数范围）
6. 各等级统计表（A→E：分数区间、人群占比、坏账率），**验证 A→E 坏账率单调递增**
7. 灰样本打分结果（如有）
8. 阶段开始/结束时间戳

### 输出文件清单

| 文件 | 说明 |
|------|------|
| `06-output-list.xlsx` | 阶段 6 输出汇总（以上 8 项） |
| `06_train_scored.csv` | 训练集打分结果（score + risk_grade） |
| `06_test_scored.csv` | 测试集打分结果 |
| `06_oot_scored.csv` | OOT 打分结果 |
| `06_grey_scored.csv` | 灰样本打分结果（如有） |
| `06_grade_stats_train.csv` | 各等级统计表 |
| `06_scorecard_detail.csv` | 评分卡明细表（LR 路径，逐特征逐箱） |
| `06_score_distribution.png` | 分数分布图（plotly） |

---

## 后续阶段入口

阶段 6 输出确认无误后，进入阶段 7（references/7-deployment-monitoring.md）。