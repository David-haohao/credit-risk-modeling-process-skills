# 阶段 7：最终报告汇总与交付

> **可执行脚本**：`scripts/07_final_report.py`

## 7.0 阶段目的

阶段 7 汇总阶段 0-6 已确认的配置、过程、结果、决策和图表，生成最终建模报告。阶段 7 不重新训练模型、不重新计算核心指标、不修改上游决策，也不包含上线与持续监控。

最终报告必须保存到阶段 0 已确认的 `report_dir`。

---

## 7.1 输入契约

### 必需输入

| 输入 | 用途 |
|------|------|
| `00_modeling_config.yaml` | 项目背景、模型配置、路径及用户确认项 |
| `00-output-list.xlsx` 至 `06-output-list.xlsx` | 各阶段输出索引与审阅汇总 |
| `05_evaluation_decisions.csv` | 模型评估结论及回退记录 |
| `06_scoring_decisions.csv` | 校准、评分参数与等级方案确认 |
| `06_scoring_parameters.json` | 最终评分公式和参数 |

### 路径相关输入

- LR：阶段 3 最终分箱、阶段 4 LR 训练过程、阶段 6 标准评分卡明细与一致性验证。
- XGB/LGB：阶段 4 调优过程、阶段 5 评估结果、阶段 6 校准前后比较。
- 存在灰样本时：阶段 1 灰样本说明与阶段 6 灰样本评分汇总。

阶段 7 只读取已生成的上游产物。缺失必需输入时不得静默跳过，必须写入 `07_report_issues.csv` 并暂停最终交付确认。

---

## 7.2 报告内容

最终报告至少包含：

1. 项目背景、Y_label定义、样本时间范围和模型类型。
2. 数据准备、Train/Test/OOT划分和样本统计。
3. EDA核心结论与数据质量决策。
4. 特征筛选、分箱、人工调整和最终特征。
5. 模型训练过程：
   - LR：逐步回归、AIC/BIC变化、诊断、最终系数与截距。
   - XGB/LGB：调优过程、Trial变化、最佳参数和最终选择依据。
6. 模型评估：KS、AUC、Gini、Lift、PSI、周期PSI、校准度及回退处理。
7. 评分结果：概率校准、评分参数、分数分布、风险等级和灰样本结果。
8. LR标准评分卡：逐变量逐分箱分值、基础分和一致性验证。
9. 最终结论、适用范围、已知限制和人工确认记录。
10. 文件交付清单与生成时间。

正文只引用已确认结果。报告中的指标必须标明数据集和口径，不能将Test描述为独立泛化验证集，也不能将已用于迭代的OOT描述为完全独立OOT。

---

## 7.3 图表与附件

- 优先复用阶段 0-6 已生成的PNG图表，不在阶段 7 重新计算图表数据。
- 最终HTML报告可嵌入或链接图表。
- CSV、JSON、XLSX、PKL等过程和交付文件不嵌入正文，统一登记在附件清单。
- 缺失图表可标记为缺失，但缺失核心指标文件时必须暂停交付。

---

## 7.4 报告问题与确认

`07_report_issues.csv` 至少包含：

| 字段 | 内容 |
|------|------|
| `issue_id` | 问题编号 |
| `issue_type` | `missing_required_input` / `missing_optional_artifact` / `inconsistent_value` / `unconfirmed_decision` |
| `source_stage` | 问题来源阶段 |
| `artifact` | 涉及文件或指标 |
| `description` | 问题说明 |
| `decision` | `pending` / `accept_missing` / `return_source_stage` |
| `reason` / `confirmed_by` / `confirmed_at` | 确认记录 |

存在 `missing_required_input`、`inconsistent_value`、`unconfirmed_decision` 或任何 `pending` 时，不得完成最终报告交付。

---

## 7.5 核心输出与验收

| 文件 | 说明 |
|------|------|
| `07_A卡建模报告.html` | 最终建模报告 |
| `07_artifact_inventory.csv` | 阶段 0-6 文件清单、路径、大小和是否纳入报告 |
| `07_report_issues.csv` | 缺失、不一致及确认记录 |
| `07-output-list.xlsx` | 阶段 7 输出索引和最终交付确认 |

验收规则：

1. 报告保存于阶段 0 确认的最终报告路径。
2. 阶段 0-6 均已完成并不存在未处理的 `pending`。
3. 报告指标、模型版本、评分参数与上游最终产物一致。
4. LR与XGB/LGB报告内容按实际模型路径生成，不混入另一模型路径的强制内容。
5. 所有引用文件均登记在 `07_artifact_inventory.csv`。
6. `07_report_issues.csv` 不存在阻断项。
7. 用户确认最终报告后，全流程结束。

---

## 全流程结束

阶段 7 最终报告确认后，A卡建模流程结束。
