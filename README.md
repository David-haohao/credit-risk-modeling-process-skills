# A卡申请评分模型全流程 Skill

这是一个面向申请信用评分模型的分阶段 Skill，覆盖问题定义、样本准备、EDA、特征工程、模型训练、模型评估、可选评分转换和最终报告交付。

本 Skill 支持 LR 评分卡、XGBoost 和 LightGBM 路径，重点保证：

- 每个阶段具有明确的输入、输出和验收条件。
- 业务参数和人工确认结果统一记录，不依赖对话记忆或隐藏默认值。
- Train、Test、OOT 的职责清晰，避免信息泄露。
- 模型筛选、调参、评估、评分和人工决策过程可追溯。
- 未确认事项保持 `pending` 并阻断后续正式阶段。

## 适用范围

适用于需要建立或审阅 A 卡申请评分模型的项目，包括：

- LR 标准评分卡建模、分箱、WOE、IV 和逐步回归。
- XGBoost / LightGBM 模型训练与 Optuna 参数调优。
- KS、AUC、Gini、Lift、Gains、PSI 和概率校准度评估。
- 可选的概率校准、分数映射、风险等级和 LR 逐箱评分卡输出。
- 汇总过程产物、确认记录和结论的最终 HTML 报告。

本 Skill 不包含模型上线、在线服务部署和持续监控。

## 核心运行契约

### 统一配置

所有业务参数、字段定义、路径和用户确认结果统一写入：

```text
00_modeling_config.yaml
```

项目开始前必须确认：

- 原始数据路径
- 过程代码存放路径
- 过程文件存放路径
- 最终报告存放路径
- Y_label 含义、时间字段、模型类型和样本唯一标识
- 是否执行阶段 6 概率校准与评分转换

未确认的业务参数不得由脚本自行推断。

### 阶段交接

阶段 0-6 生成：

```text
0X_modeling_config_snapshot.yaml
0X-output-list.xlsx
0X_stage_manifest.json
```

`0X_stage_manifest.json` 是下一阶段的唯一正式交接入口，包含配置路径、配置快照、累计产物、待确认决策和下一阶段信息。

阶段 7 是最终交付阶段，生成最终报告、问题清单、文件清单和 `07-output-list.xlsx`，不再生成下一阶段交接 manifest。

阶段 1-7 正式运行必须提供上一阶段已完成的 manifest：

```powershell
python scripts/01_data_prep.py --previous-manifest <00_stage_manifest.json>
```

直接传入数据文件仅用于单元测试或显式 `--compatibility-mode`。上一阶段存在 `pending` 决策时，下一阶段会被阻断。

## 阶段说明

| 阶段 | 主要任务 | 核心说明 |
|---|---|---|
| 0 | 问题定义与预检 | 生成配置模板，检查依赖和原始数据，确认项目范围与路径 |
| 1 | 样本准备 | 生成稳定 `sample_id`，处理灰样本，划分 Train/Test/OOT |
| 2 | EDA | 分析连续与离散特征；可选质量检查仅在配置阈值后执行 |
| 3 | 特征工程 | 执行筛选、分箱、WOE、结构化质量动作和人工分箱调整 |
| 4 | 模型训练 | LR 逐步回归，或 XGB/LGB 调优；完整保存优化过程 |
| 5 | 模型评估 | 评估区分度、Lift、稳定性和校准度，生成四类核心决策 |
| 6 | 评分转换，可选 | 执行概率校准、分数映射、风险等级和 LR 标准评分卡 |
| 7 | 最终报告 | 汇总已确认结论和正式产物，生成最终交付报告 |

详细阶段契约位于 [`references/`](references/)；执行阶段 4 的 XGB/LGB 路径时，额外读取 [`references/4.1-xgb-lgb-tuning.md`](references/4.1-xgb-lgb-tuning.md)。

## 模型与数据集约束

- `sample_id`、Y_label、时间字段和 `sample_weight` 是保留字段，不进入模型特征。
- 所有筛选规则、分箱、转换器、模型和校准器只使用 Train 拟合。
- LR 使用 Train + OOT，并在 Train 内执行交叉验证。
- XGBoost / LightGBM 使用 Train + Test + OOT，Test 用于调参，OOT 用于最终泛化评估。
- XGBoost / LightGBM 的 Train/Test 保持普通随机切分，不按 Y 分层。
- 比例 OOT 按时间排序取末尾样本，同一时间点不拆分。
- 无稳定主键时，根据整行内容生成可重复的哈希 `sample_id`。
- 灰样本不参与训练与评估；仅在启用阶段 6 且具有最终模型链预测时单独评分。

## 人工确认与回退

以下类型的事项不会被脚本静默决定：

- 阶段 2 的高缺失、稀疏类别和类型异常阈值。
- 阶段 3 的规则变量、手动分箱调整和结构化质量动作。
- 阶段 4 的 LR 诊断决策、特征软上限和模型参数。
- 阶段 5 的区分度、Lift、PSI 和校准度四类核心评估结论。
- 阶段 6 的评分参数、校准方式、截断方案和风险等级方案。

首次运行遇到待确认事项时，阶段会输出决策表并生成状态为 `pending` 的 manifest。确认并更新配置或决策表后，重新运行该阶段。

阶段 5 发现过拟合、欠拟合、稳定性或校准问题时，应按问题来源回退至阶段 1、3 或 4，重新执行后续阶段，而不是直接接受异常结果。

## 两条正式交付路由

项目开始时通过 `enable_stage6_scoring` 确认是否需要评分转换：

```text
阶段 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 7
阶段 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7
```

- `enable_stage6_scoring=false`：阶段 5 验收通过后直接生成最终报告。
- `enable_stage6_scoring=true`：阶段 5 验收通过后执行阶段 6，再生成最终报告。

## 快速开始

### 1. 生成配置模板

```powershell
python scripts/00_env_setup.py `
  --config-template `
  --output-dir <过程文件目录>
```

编辑生成的 `00_modeling_config.yaml`，完成项目范围、路径和业务参数确认。

### 2. 执行环境与数据预检

```powershell
python scripts/00_env_setup.py `
  --filepath <原始数据文件> `
  --output-dir <过程文件目录> `
  --check-packages
```

阶段 0 只检查并报告缺失依赖，不自动安装。主要依赖包括 pandas、NumPy、scikit-learn、statsmodels、toad、XGBoost、LightGBM、Optuna、PyYAML 和常用绘图库。

### 3. 通过 manifest 运行后续阶段

```powershell
python scripts/01_data_prep.py `
  --previous-manifest <过程文件目录>\00_stage_manifest.json `
  --output-dir <过程文件目录>

python scripts/02_eda.py `
  --previous-manifest <过程文件目录>\01_stage_manifest.json `
  --output-dir <过程文件目录>
```

其余阶段使用相同模式。运行前应先阅读对应 reference，并处理上一阶段输出的人工确认项。查看具体参数：

```powershell
python scripts/05_model_evaluation.py --help
```

### 4. 生成最终报告

跳过阶段 6 时使用阶段 5 manifest；执行阶段 6 时使用阶段 6 manifest：

```powershell
python scripts/07_final_report.py `
  --previous-manifest <05或06_stage_manifest.json> `
  --report-dir <最终报告目录>
```

最终报告的结论位于报告顶部，并且只引用已确认的阶段 5 和阶段 6 决策。

## 目录结构

```text
credit-a-card-modeling/
├── SKILL.md                 # Agent 使用的主路由与全局约束
├── README.md                # 面向使用者的仓库说明
├── references/              # 阶段 0-7 的详细输入、输出与验收规则
├── scripts/                 # 可复用阶段脚本和通用阶段契约模块
└── tests/                   # 契约、阶段逻辑和完整路由测试
```

通用配置、manifest、配置快照和输出清单能力由 [`scripts/stage_contracts.py`](scripts/stage_contracts.py) 实现。

## 测试

运行全部测试：

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

测试覆盖：

- 配置、manifest、配置快照和正式入口约束
- 时间 OOT、比例 OOT、同时间点边界和哈希样本 ID
- 阶段 2 配置驱动质量检查和阶段 3 结构化质量动作
- 阶段 4 特征排除、预测文件和 LightGBM 依赖错误
- 阶段 5 Lift、Gains、周期 PSI、校准指标和 pending 阻断
- 阶段 6 Platt 校准、整数分数、等级边界和 LR 逐箱评分卡
- 跳过阶段 6 与执行阶段 6 的两条完整交付路由

## 使用原则

- 先阅读 `SKILL.md`，再按目标阶段加载对应 reference，不一次性加载全部文档。
- 以配置文件和 manifest 为正式事实来源，不依赖聊天上下文猜测参数。
- 所有人工决策必须留痕；存在阻断性 `pending` 时不得继续。
- 不支持或未确认的转换、校准和评分方案应明确阻断，而不是自动替代。
