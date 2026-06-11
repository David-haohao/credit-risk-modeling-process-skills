#!/usr/bin/env python3
"""阶段 4：可审计模型训练。"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import sklearn
import statsmodels.api as sm
import xgboost as xgb
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from stage_contracts import load_config, load_previous_manifest, require_formal_entry, write_output_list, write_stage_manifest


DECISION_COLUMNS = [
    "feature", "issue_type", "metric_value", "decision", "decision_reason",
    "confirmed_by", "confirmed_at",
]

XGB_SEARCH_SPACE = {
    "n_estimators": {"type": "int", "low": 100, "high": 500},
    "max_depth": {"type": "int", "low": 3, "high": 8},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
    "subsample": {"type": "float", "low": 0.6, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
    "reg_alpha": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
    "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
    "min_child_weight": {"type": "int", "low": 1, "high": 20},
}


def load_feature_list(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def require_lightgbm() -> Any:
    if lgb is None:
        raise ImportError("LightGBM 路径需要安装 lightgbm，禁止伪装为 XGBoost")
    return lgb


def validate_feature_contract(
    data: pd.DataFrame, feature_cols: list[str], target_col: str,
    sample_id_col: str = "sample_id", weight_col: Optional[str] = "sample_weight",
) -> list[str]:
    reserved = {target_col, sample_id_col}
    if weight_col:
        reserved.add(weight_col)
    invalid = [c for c in feature_cols if c in reserved]
    missing = [c for c in feature_cols if c not in data.columns]
    if invalid:
        raise ValueError(f"模型特征列表包含保留字段: {invalid}")
    if missing:
        raise ValueError(f"数据缺少模型特征: {missing}")
    if sample_id_col not in data.columns:
        raise ValueError(f"数据缺少样本唯一标识: {sample_id_col}")
    if data[sample_id_col].isna().any() or not data[sample_id_col].is_unique:
        raise ValueError(f"{sample_id_col} 必须非空且唯一")
    if target_col not in data.columns or not data[target_col].dropna().isin([0, 1]).all():
        raise ValueError(f"{target_col} 必须存在且仅包含 0/1")
    return feature_cols


def fit_logit(data: pd.DataFrame, target_col: str, features: list[str]) -> Any:
    X = sm.add_constant(data[features], has_constant="add")
    return sm.Logit(data[target_col], X).fit(disp=False, maxiter=200)


def model_criterion(model: Any, criterion: str) -> float:
    return float(model.aic if criterion == "aic" else model.bic)


def auditable_stepwise(
    data: pd.DataFrame, target_col: str, feature_cols: list[str],
    direction: str = "both", criterion: str = "aic", min_improvement: float = 1e-6,
    max_iter: Optional[int] = None,
) -> tuple[list[str], pd.DataFrame]:
    """逐轮记录变量变化和 AIC/BIC 变化。"""
    if direction not in {"forward", "backward", "both"}:
        raise ValueError("direction 必须为 forward/backward/both")
    if criterion not in {"aic", "bic"}:
        raise ValueError("criterion 必须为 aic/bic")

    selected = feature_cols.copy() if direction == "backward" else []
    remaining = [c for c in feature_cols if c not in selected]
    current_score = model_criterion(fit_logit(data, target_col, selected), criterion) if selected else np.inf
    history = []
    iteration = 0

    while True:
        iteration += 1
        if max_iter and iteration > max_iter:
            break
        best_action = None
        best_feature = None
        best_score = current_score
        best_p_value = np.nan

        if direction in {"forward", "both"}:
            for feature in remaining:
                try:
                    model = fit_logit(data, target_col, selected + [feature])
                    score = model_criterion(model, criterion)
                    if score < best_score - min_improvement:
                        best_action, best_feature, best_score = "add", feature, score
                        best_p_value = float(model.pvalues.get(feature, np.nan))
                except Exception:
                    continue

        if direction in {"backward", "both"} and len(selected) > 1:
            for feature in selected:
                try:
                    candidate = [c for c in selected if c != feature]
                    model = fit_logit(data, target_col, candidate)
                    score = model_criterion(model, criterion)
                    if score < best_score - min_improvement:
                        best_action, best_feature, best_score = "remove", feature, score
                        best_p_value = np.nan
                except Exception:
                    continue

        before = selected.copy()
        if best_action == "add":
            selected.append(best_feature)
            remaining.remove(best_feature)
        elif best_action == "remove":
            selected.remove(best_feature)
            remaining.append(best_feature)
        else:
            history.append({
                "iteration": iteration, "action": "stop", "feature": "",
                "features_before": json.dumps(before, ensure_ascii=False),
                "features_after": json.dumps(before, ensure_ascii=False),
                "criterion": criterion, "criterion_before": current_score,
                "criterion_after": current_score, "delta_criterion": 0.0,
                "p_value": np.nan, "decision_reason": "no_further_improvement",
            })
            break

        history.append({
            "iteration": iteration, "action": best_action, "feature": best_feature,
            "features_before": json.dumps(before, ensure_ascii=False),
            "features_after": json.dumps(selected, ensure_ascii=False),
            "criterion": criterion, "criterion_before": current_score,
            "criterion_after": best_score,
            "delta_criterion": current_score - best_score if np.isfinite(current_score) else np.nan,
            "p_value": best_p_value, "decision_reason": f"best_{criterion}_improvement",
        })
        current_score = best_score
        if not remaining and direction == "forward":
            break

    return selected, pd.DataFrame(history)


def calculate_vif(data: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = data[feature_cols].dropna()
    return pd.DataFrame({
        "feature": feature_cols,
        "vif": [variance_inflation_factor(X.values, i) for i in range(len(feature_cols))],
    }).sort_values("vif", ascending=False)


def build_soft_limit_decisions(
    selected_cols: list[str], max_features: Optional[int], diagnostic_table: pd.DataFrame,
) -> pd.DataFrame:
    if max_features is None or len(selected_cols) <= max_features:
        return pd.DataFrame(columns=DECISION_COLUMNS)
    metric_map = {}
    if not diagnostic_table.empty and {"feature", "metric_value"}.issubset(diagnostic_table.columns):
        metric_map = diagnostic_table.set_index("feature")["metric_value"].to_dict()
    return pd.DataFrame([{
        "feature": feature, "issue_type": "soft_feature_limit",
        "metric_value": metric_map.get(feature, np.nan), "decision": "pending",
        "decision_reason": "", "confirmed_by": "", "confirmed_at": "",
    } for feature in selected_cols], columns=DECISION_COLUMNS)


def build_lr_diagnostics(data: pd.DataFrame, target_col: str, features: list[str]) -> tuple[pd.DataFrame, Any]:
    model = fit_logit(data, target_col, features)
    result = pd.DataFrame({
        "feature": model.params.index,
        "coefficient": model.params.values,
        "odds_ratio": np.exp(model.params.values),
        "std_err": model.bse.values,
        "z_value": model.tvalues.values,
        "p_value": model.pvalues.values,
    })
    result["feature"] = result["feature"].replace({"const": "__INTERCEPT__"})
    return result, model


def build_diagnostic_decisions(diagnostics: pd.DataFrame, vif: pd.DataFrame, vif_threshold: float) -> pd.DataFrame:
    rows = []
    for row in vif[vif["vif"] > vif_threshold].to_dict("records"):
        rows.append({
            "feature": row["feature"], "issue_type": "high_vif", "metric_value": row["vif"],
            "decision": "pending", "decision_reason": "", "confirmed_by": "", "confirmed_at": "",
        })
    for row in diagnostics[diagnostics["feature"] != "__INTERCEPT__"].to_dict("records"):
        issue = "negative_coef" if row["coefficient"] < 0 else ("non_significant" if row["p_value"] > 0.05 else None)
        if issue:
            rows.append({
                "feature": row["feature"], "issue_type": issue,
                "metric_value": row["coefficient"] if issue == "negative_coef" else row["p_value"],
                "decision": "pending", "decision_reason": "", "confirmed_by": "", "confirmed_at": "",
            })
    return pd.DataFrame(rows, columns=DECISION_COLUMNS)


def apply_model_decisions(features: list[str], decisions: pd.DataFrame) -> list[str]:
    if decisions.empty:
        return features
    if (decisions["decision"] == "pending").any():
        raise ValueError("04_model_decisions.csv 仍存在 pending")
    if (decisions["decision"] == "return_stage3").any():
        raise ValueError("存在 return_stage3 决策，需返回阶段 3 处理")
    drops = set(decisions.loc[decisions["decision"] == "drop", "feature"])
    return [c for c in features if c not in drops]


def lr_grid_search(
    X: pd.DataFrame, y: pd.Series, cv: int, random_state: int,
    sample_weight: Optional[pd.Series] = None,
) -> tuple[LogisticRegression, pd.DataFrame]:
    grid = GridSearchCV(
        LogisticRegression(solver="saga", max_iter=3000, random_state=random_state),
        {"C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0], "penalty": ["l1", "l2"]},
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring="roc_auc", n_jobs=-1,
    )
    fit_params = {"sample_weight": sample_weight} if sample_weight is not None else {}
    grid.fit(X, y, **fit_params)
    return grid.best_estimator_, pd.DataFrame(grid.cv_results_)


def build_final_lr_coefficients(model: LogisticRegression, features: list[str]) -> pd.DataFrame:
    rows = [{"feature": "__INTERCEPT__", "coefficient": float(model.intercept_[0])}]
    rows.extend({"feature": f, "coefficient": float(c)} for f, c in zip(features, model.coef_[0]))
    result = pd.DataFrame(rows)
    result["odds_ratio"] = np.exp(result["coefficient"])
    return result


def prediction_frame(
    model: Any, data: pd.DataFrame, features: list[str], target_col: str,
    sample_id_col: str, weight_col: Optional[str], time_col: Optional[str] = None,
) -> pd.DataFrame:
    result = pd.DataFrame({sample_id_col: data[sample_id_col].values})
    if time_col:
        if time_col not in data.columns:
            raise ValueError(f"预测数据缺少时间字段: {time_col}")
        result[time_col] = data[time_col].values
    result["y_true"] = data[target_col].values
    result["y_pred_raw"] = model.predict_proba(data[features])[:, 1]
    if weight_col and weight_col in data.columns:
        result[weight_col] = data[weight_col].values
    return result


def calculate_ks(y_true: pd.Series, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def train_xgb_optuna(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
    n_trials: int, scale_pos_weight: float, early_stopping_rounds: int, random_state: int,
    optuna_sample_size: Optional[int] = None, sample_weight_train: Optional[pd.Series] = None,
    sample_weight_test: Optional[pd.Series] = None,
) -> tuple[xgb.XGBClassifier, optuna.Study, pd.DataFrame]:
    if optuna_sample_size and optuna_sample_size < len(X_train):
        X_opt = X_train.sample(n=optuna_sample_size, random_state=random_state)
        y_opt = y_train.loc[X_opt.index]
        weight_opt = sample_weight_train.loc[X_opt.index] if sample_weight_train is not None else None
    else:
        X_opt, y_opt = X_train, y_train
        weight_opt = sample_weight_train

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "scale_pos_weight": scale_pos_weight, "eval_metric": "auc",
            "random_state": random_state, "n_jobs": -1,
            "early_stopping_rounds": early_stopping_rounds,
        }
        model = xgb.XGBClassifier(**params)
        fit_params = {}
        if weight_opt is not None:
            fit_params["sample_weight"] = weight_opt
        if sample_weight_test is not None:
            fit_params["sample_weight_eval_set"] = [sample_weight_test]
        model.fit(X_opt, y_opt, eval_set=[(X_test, y_test)], verbose=False, **fit_params)
        train_auc = roc_auc_score(y_opt, model.predict_proba(X_opt)[:, 1])
        test_pred = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_pred)
        trial.set_user_attr("train_auc", train_auc)
        trial.set_user_attr("test_auc", test_auc)
        trial.set_user_attr("test_ks", calculate_ks(y_test, test_pred))
        trial.set_user_attr("auc_gap", train_auc - test_auc)
        trial.set_user_attr("best_iteration", getattr(model, "best_iteration", None))
        return test_auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)
    trials = study.trials_dataframe()
    trials["historical_best_auc"] = trials["value"].cummax()
    trials["improvement_vs_previous_best"] = trials["historical_best_auc"].diff()

    best = study.best_params.copy()
    best.update({
        "scale_pos_weight": scale_pos_weight, "eval_metric": "auc",
        "random_state": random_state, "n_jobs": -1,
    })
    best_iteration = study.best_trial.user_attrs.get("best_iteration")
    if best_iteration is not None:
        best["n_estimators"] = int(best_iteration) + 1
    model = xgb.XGBClassifier(**best)
    final_fit_params = {"sample_weight": sample_weight_train} if sample_weight_train is not None else {}
    model.fit(X_train, y_train, verbose=False, **final_fit_params)
    return model, study, trials


def train_lgb_optuna(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
    n_trials: int, scale_pos_weight: float, early_stopping_rounds: int, random_state: int,
    optuna_sample_size: Optional[int] = None, sample_weight_train: Optional[pd.Series] = None,
    sample_weight_test: Optional[pd.Series] = None,
) -> tuple[Any, optuna.Study, pd.DataFrame]:
    library = require_lightgbm()
    if optuna_sample_size and optuna_sample_size < len(X_train):
        X_opt = X_train.sample(n=optuna_sample_size, random_state=random_state)
        y_opt = y_train.loc[X_opt.index]
        weight_opt = sample_weight_train.loc[X_opt.index] if sample_weight_train is not None else None
    else:
        X_opt, y_opt, weight_opt = X_train, y_train, sample_weight_train

    def objective(trial: optuna.Trial) -> float:
        max_depth = trial.suggest_int("max_depth", 3, 12)
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 8, min(128, 2 ** max_depth)),
            "max_depth": max_depth,
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 100, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 100, log=True),
            "scale_pos_weight": scale_pos_weight, "random_state": random_state,
            "verbosity": -1,
        }
        model = library.LGBMClassifier(**params)
        model.fit(
            X_opt, y_opt, sample_weight=weight_opt,
            eval_set=[(X_test, y_test)], eval_sample_weight=[sample_weight_test],
            callbacks=[library.early_stopping(early_stopping_rounds, verbose=False)],
        )
        train_auc = roc_auc_score(y_opt, model.predict_proba(X_opt)[:, 1])
        test_pred = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_pred)
        trial.set_user_attr("train_auc", train_auc)
        trial.set_user_attr("test_auc", test_auc)
        trial.set_user_attr("test_ks", calculate_ks(y_test, test_pred))
        trial.set_user_attr("auc_gap", train_auc - test_auc)
        trial.set_user_attr("best_iteration", model.best_iteration_)
        return test_auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)
    trials = study.trials_dataframe()
    trials["historical_best_auc"] = trials["value"].cummax()
    trials["improvement_vs_previous_best"] = trials["historical_best_auc"].diff()
    best = study.best_params | {"scale_pos_weight": scale_pos_weight, "random_state": random_state, "verbosity": -1}
    best_iteration = study.best_trial.user_attrs.get("best_iteration")
    if best_iteration:
        best["n_estimators"] = int(best_iteration)
    model = library.LGBMClassifier(**best)
    model.fit(X_train, y_train, sample_weight=sample_weight_train)
    return model, study, trials


def report_feature_importance(model: xgb.XGBClassifier, features: list[str]) -> pd.DataFrame:
    if lgb is not None and isinstance(model, lgb.LGBMClassifier):
        return pd.DataFrame({
            "feature": features,
            "weight": model.booster_.feature_importance(importance_type="split"),
            "gain": model.booster_.feature_importance(importance_type="gain"),
        }).sort_values("gain", ascending=False)
    booster = model.get_booster()
    weight = booster.get_score(importance_type="weight")
    gain = booster.get_score(importance_type="gain")
    return pd.DataFrame({
        "feature": features,
        "weight": [weight.get(f, 0) for f in features],
        "gain": [gain.get(f, 0) for f in features],
    }).sort_values("gain", ascending=False)


def save_process_plot(data: pd.DataFrame, x: str, y: str, path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data[x], data[y], marker="o", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_bar_plot(data: pd.DataFrame, category: str, value: str, path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ordered = data.sort_values(value, ascending=True)
    ax.barh(ordered[category], ordered[value])
    ax.set_title(title)
    ax.set_xlabel(value)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_excel(path: str, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path) as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)


def save_model(model: Any, path: str) -> None:
    with open(path, "wb") as file:
        pickle.dump(model, file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段 4：可审计模型训练")
    parser.add_argument("--model-type", choices=["LR", "XGB", "LGB"])
    parser.add_argument("--previous-manifest")
    parser.add_argument("--compatibility-mode", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--train")
    parser.add_argument("--test")
    parser.add_argument("--oot")
    parser.add_argument("--features-file")
    parser.add_argument("--target-col")
    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument("--weight-col", default="sample_weight")
    parser.add_argument("--time-col", help="保留到预测文件中的时间字段")
    parser.add_argument("--iv-table")
    parser.add_argument("--model-decisions")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--max-features", type=int, default=15,
                        help="LR 入模变量软上限；设为 0 表示不限制")
    parser.add_argument("--vif-threshold", type=float, default=5.0)
    parser.add_argument("--stepwise-direction", choices=["forward", "backward", "both"], default="both")
    parser.add_argument("--stepwise-criterion", choices=["aic", "bic"], default="aic")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--scale-pos-weight", type=float)
    parser.add_argument("--optuna-sample-size", type=int)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sep", default=",")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_formal_entry(args.previous_manifest, args.compatibility_mode, 4)
    manifest = load_previous_manifest(args.previous_manifest, 3) if args.previous_manifest else None
    config_path = Path(args.config or (manifest or {}).get("config_path", ""))
    config = load_config(config_path) if config_path.exists() else {}
    declared = (manifest or {}).get("outputs", {})
    fields = config.get("fields", {})
    training = config.get("training", {})
    args.model_type = config.get("model_type", args.model_type)
    args.target_col = fields.get("target_col", args.target_col)
    args.sample_id_col = fields.get("sample_id_col") or args.sample_id_col
    args.weight_col = fields.get("sample_weight_col", args.weight_col)
    args.time_col = fields.get("time_col", args.time_col)
    args.n_trials = int(training.get("n_trials", args.n_trials))
    if args.model_type == "LR":
        args.train = args.train or declared.get("03_train_woe")
        args.oot = args.oot or declared.get("03_oot_woe")
    else:
        args.train = args.train or declared.get("03_train_final")
        args.test = args.test or declared.get("03_test_final")
        args.oot = args.oot or declared.get("03_oot_final")
    args.features_file = args.features_file or declared.get("03_final_features")
    if not all([args.model_type, args.train, args.oot, args.features_file, args.target_col]):
        raise ValueError("阶段 4 manifest/config 缺少模型类型、数据、特征列表或 target_col")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.model_type == "LGB":
        require_lightgbm()
    if args.model_type in {"XGB", "LGB"} and not args.test:
        raise ValueError("XGB/LGB 路径必须提供 --test")

    train = pd.read_csv(args.train, sep=args.sep, encoding="utf-8-sig")
    oot = pd.read_csv(args.oot, sep=args.sep, encoding="utf-8-sig")
    test = pd.read_csv(args.test, sep=args.sep, encoding="utf-8-sig") if args.test else None
    features = validate_feature_contract(
        train, load_feature_list(args.features_file), args.target_col,
        args.sample_id_col, args.weight_col,
    )
    for name, data in [("OOT", oot), ("Test", test)]:
        if data is not None:
            validate_feature_contract(data, features, args.target_col, args.sample_id_col, args.weight_col)

    metadata = {
        "model_type": args.model_type, "target_col": args.target_col,
        "sample_id_col": args.sample_id_col, "input_features": features,
        "library_versions": {
            "optuna": optuna.__version__, "xgboost": xgb.__version__,
            "sklearn": sklearn.__version__,
        },
    }

    if args.model_type == "LR":
        selected, history = auditable_stepwise(
            train, args.target_col, features, args.stepwise_direction, args.stepwise_criterion,
        )
        vif = calculate_vif(train, selected)
        diagnostics, _ = build_lr_diagnostics(train, args.target_col, selected)
        decisions = pd.concat([
            build_soft_limit_decisions(selected, args.max_features or None, pd.DataFrame()),
            build_diagnostic_decisions(diagnostics, vif, args.vif_threshold),
        ], ignore_index=True).drop_duplicates(["feature", "issue_type"])

        if args.model_decisions:
            supplied = pd.read_csv(args.model_decisions, encoding="utf-8-sig")
            decisions = decisions.drop(columns=["decision", "decision_reason", "confirmed_by", "confirmed_at"]).merge(
                supplied[DECISION_COLUMNS], on=["feature", "issue_type"], how="left",
            )
            decisions["decision"] = decisions["decision"].fillna("pending")
        decisions.to_csv(output / "04_model_decisions.csv", index=False, encoding="utf-8-sig")
        history.to_csv(output / "04_lr_stepwise_history.csv", index=False, encoding="utf-8-sig")
        diagnostics.to_csv(output / "04_lr_diagnostics.csv", index=False, encoding="utf-8-sig")
        vif.to_csv(output / "04_lr_vif.csv", index=False, encoding="utf-8-sig")
        save_excel(str(output / "04_lr_training_process.xlsx"), {
            "stepwise_history": history, "vif": vif,
            "diagnostics": diagnostics, "decisions": decisions,
        })
        save_process_plot(history, "iteration", "criterion_after", str(output / "04_lr_criterion_curve.png"),
                          f"LR Stepwise {args.stepwise_criterion.upper()} History")
        if not decisions.empty and (decisions["decision"] == "pending").any():
            write_output_list(output / "04-output-list.xlsx",
                              pd.DataFrame([{"status": "pending", "reason": "lr_model_decisions"}]),
                              list(output.glob("04_*")), decisions.loc[decisions["decision"].eq("pending")])
            outputs = {path.stem: path for path in output.glob("04_*")}
            outputs["04_output_list"] = output / "04-output-list.xlsx"
            write_stage_manifest(output, 4, "pending", config_path,
                                 {"previous_manifest": args.previous_manifest or ""}, outputs,
                                 decisions.loc[decisions["decision"].eq("pending")].to_dict("records"), 5)
            print("存在待确认 LR 诊断项，已输出 04_model_decisions.csv，确认后重新执行阶段 4。")
            return
        selected = apply_model_decisions(selected, decisions)
        train_weight = train[args.weight_col] if args.weight_col in train.columns else None
        model, grid_results = lr_grid_search(
            train[selected], train[args.target_col], args.cv, args.random_state, train_weight,
        )
        coefficients = build_final_lr_coefficients(model, selected)
        coefficients.to_csv(output / "04_lr_coefficients.csv", index=False, encoding="utf-8-sig")
        grid_results.to_csv(output / "04_lr_grid_search.csv", index=False, encoding="utf-8-sig")
        save_excel(str(output / "04_lr_training_process.xlsx"), {
            "stepwise_history": history, "vif": vif, "diagnostics": diagnostics,
            "decisions": decisions, "grid_search": grid_results, "final_coefficients": coefficients,
        })
        save_model(model, str(output / "04_model.pkl"))
        (output / "04_model_formula.txt").write_text(
            "logit(p) = " + " + ".join(
                [f"{model.intercept_[0]:.12g}"] + [f"({c:.12g} * {f})" for f, c in zip(selected, model.coef_[0])]
            ), encoding="utf-8",
        )
        datasets = {"train": train, "oot": oot}
        metadata["best_params"] = model.get_params()
    else:
        scale = args.scale_pos_weight
        n_bad = int((train[args.target_col] == 1).sum())
        n_good = int((train[args.target_col] == 0).sum())
        imbalance_ratio = n_good / n_bad
        if scale is None:
            scale = 1.0
        trainer = train_lgb_optuna if args.model_type == "LGB" else train_xgb_optuna
        model, study, trials = trainer(
            train[features], train[args.target_col], test[features], test[args.target_col],
            args.n_trials, scale, args.early_stopping_rounds, args.random_state, args.optuna_sample_size,
            train[args.weight_col] if args.weight_col in train.columns else None,
            test[args.weight_col] if args.weight_col in test.columns else None,
        )
        trials.to_csv(output / "04_optuna_trials.csv", index=False, encoding="utf-8-sig")
        save_excel(str(output / "04_optuna_trials.xlsx"), {"trials": trials})
        save_process_plot(trials, "number", "historical_best_auc", str(output / "04_optuna_history.png"),
                          "Optuna Historical Best Test AUC")
        importance = report_feature_importance(model, features)
        importance.to_csv(output / "04_feature_importance.csv", index=False, encoding="utf-8-sig")
        try:
            param_importance = pd.DataFrame(
                optuna.importance.get_param_importances(study).items(),
                columns=["parameter", "importance"],
            )
            param_importance.to_csv(
                output / "04_optuna_param_importance.csv", index=False, encoding="utf-8-sig",
            )
            save_bar_plot(
                param_importance, "parameter", "importance",
                str(output / "04_optuna_param_importance.png"), "Optuna Parameter Importance",
            )
        except Exception as exc:
            param_importance = pd.DataFrame([{"parameter": "unavailable", "importance": np.nan, "reason": str(exc)}])
        save_model(model, str(output / "04_model.pkl"))
        save_model(study, str(output / "04_optuna_study.pkl"))
        (output / "04_best_params.json").write_text(
            json.dumps(study.best_params, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        (output / "04_optuna_search_space.json").write_text(
            json.dumps(XGB_SEARCH_SPACE, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        selected = features
        datasets = {"train": train, "test": test, "oot": oot}
        metadata["best_params"] = study.best_params
        metadata["best_test_auc"] = study.best_value
        metadata["scale_pos_weight"] = scale
        metadata["suggested_imbalance_ratio"] = imbalance_ratio
        metadata["optuna"] = {
            "n_trials": args.n_trials,
            "search_space_source": "default",
            "study_persistence": "pickle_snapshot_only",
            "sampler": "TPESampler",
            "random_state": args.random_state,
        }

    (output / "04_final_features.txt").write_text("\n".join(selected), encoding="utf-8")
    metadata["final_features"] = selected
    (output / "04_model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    for name, data in datasets.items():
        prediction_frame(
            model, data, selected, args.target_col, args.sample_id_col, args.weight_col,
            args.time_col,
        ).to_csv(output / f"04_pred_{name}.csv", index=False, encoding="utf-8-sig")
    inventory = pd.DataFrame({
        "file": sorted(path.name for path in output.iterdir() if path.is_file()),
    })
    summary = pd.DataFrame([{
        "model_type": args.model_type,
        "input_feature_count": len(features),
        "final_feature_count": len(selected),
        "train_rows": len(train),
        "test_rows": len(test) if test is not None else np.nan,
        "oot_rows": len(oot),
    }])
    workbook_sheets = {"summary": summary, "file_inventory": inventory}
    if args.model_type == "LR":
        workbook_sheets.update({
            "stepwise_history": history, "final_coefficients": coefficients,
            "model_decisions": decisions,
        })
    else:
        workbook_sheets.update({
            "optuna_trials": trials, "feature_importance": importance,
            "parameter_importance": param_importance,
        })
    save_excel(str(output / "04-output-list.xlsx"), workbook_sheets)
    outputs = {path.stem: path for path in output.glob("04_*")}
    outputs["04_output_list"] = output / "04-output-list.xlsx"
    write_stage_manifest(output, 4, "completed", config_path,
                         {"previous_manifest": args.previous_manifest or ""}, outputs, [], 5)
    print(f"阶段 4 完成，最终特征数: {len(selected)}")


if __name__ == "__main__":
    main()
