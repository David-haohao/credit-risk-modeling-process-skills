---
name: credit-a-card-modeling
description: "Use when building or reviewing an application credit scorecard workflow, including sample preparation, EDA, feature engineering, LR/XGBoost/LightGBM training, evaluation, score conversion, or final reporting."
---

# A 卡建模

## 核心原则

按阶段读取对应 reference 并执行。不要一次加载全部 reference。

每个阶段开始前确认输入契约，结束前执行验收规则。阶段结果经用户确认后再进入下一阶段。

统一使用 `00_modeling_config.yaml` 传递跨阶段配置和用户确认结果。不得仅依赖对话记录或代码常量。

阶段 1-7 正式运行必须传入上一阶段的 `0X_stage_manifest.json`。manifest 是唯一正式交接入口，声明主配置路径、配置快照、累计上游产物路径、pending 决策和下一阶段；下一阶段不得绕过 manifest 猜测或扫描上游文件。直接文件参数仅用于单元测试或显式 `--compatibility-mode`。

## 全局约束

- 首先确认 Y_label 含义、模型类型、时间字段和样本唯一标识。
- 项目开始时必须确认是否执行阶段 6 概率校准与评分转换，并写入 `enable_stage6_scoring`。选择不执行时，阶段 5 验收通过后直接进入阶段 7。
- 每次开始新的建模任务时，必须向用户确认以下路径，并写入 `00_modeling_config.yaml`：
  - 原始数据路径
  - 过程代码存放路径
  - 过程文件存放路径
  - 最终报告存放路径
- 路径未确认前不得开始执行脚本或生成项目文件；后续阶段统一使用已确认路径，不自行创建其他项目目录。
- `sample_id`、Y_label、时间字段和 `sample_weight` 为保留字段，不得作为模型特征。
- 仅使用 Train 拟合筛选规则、分箱、转换器、校准器和模型；对 Test/OOT 只应用已拟合规则。
- LR 使用 Train + OOT，并在 Train 内执行交叉验证。
- XGBoost/LightGBM 使用 Train + Test + OOT，Test 用于调参，OOT 用于最终泛化评估。
- 灰样本不参与训练和评估；仅在启用阶段 6 时单独评分。
- 所有需要人工确认的项目在确认前保持 `pending`；关键项存在 `pending` 时不得进入下一阶段。
- 所有 CSV 使用 UTF-8-SIG，其他文本使用 UTF-8。

## 阶段路由

| 阶段 | 任务 | 必须读取 |
|------|------|----------|
| 0 | 问题定义、字段与配置确认、环境预检 | `references/0-problem-definition.md` |
| 1 | 样本准备、灰样本处理、Train/Test/OOT 划分 | `references/1-data-prep.md` |
| 2 | Train EDA 与数据质量问题识别 | `references/2-eda.md` |
| 3 | 特征工程、筛选、分箱与 WOE | `references/3-feature-engineering.md` |
| 4 | LR / XGBoost / LightGBM 训练 | `references/4-model-training.md`；仅当 `model_type` 为 XGB/LGB 时额外读取 `references/4.1-xgb-lgb-tuning.md` |
| 5 | KS / AUC / LIFT / PSI / 校准度评估 | `references/5-model-evaluation.md` |
| 6 | 概率校准、分数映射与风险等级（仅当 `enable_stage6_scoring=true`） | `references/6-scorecard.md` |
| 7 | 最终报告汇总与交付 | `references/7-final-report.md` |

## 执行方式

1. 新建模任务首先确认原始数据、过程代码、过程文件和最终报告四类路径。
2. 读取目标阶段 reference。
3. 检查该阶段输入和未完成确认项。
4. 优先复用对应 `scripts/0X_*.py`，根据项目字段和已确认配置调整。
5. 生成核心输出、审计输出和展示输出。
6. 执行 reference 中的验收规则并向用户确认结果。
7. 保存本阶段配置快照、`0X-output-list.xlsx` 和 `0X_stage_manifest.json`；存在阻断性 pending 时 manifest 状态不得标记为完成。

## 脚本索引

| 阶段 | 脚本 |
|------|------|
| 0 | `scripts/00_env_setup.py` |
| 1 | `scripts/01_data_prep.py` |
| 2 | `scripts/02_eda.py` |
| 3 | `scripts/03_feature_engineering.py` |
| 4 | `scripts/04_model_training.py` |
| 5 | `scripts/05_model_evaluation.py` |
| 6 | `scripts/06_scorecard.py` |
| 7 | `scripts/07_final_report.py` |

公共阶段契约由 `scripts/stage_contracts.py` 实现。
