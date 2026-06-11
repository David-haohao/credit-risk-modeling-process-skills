#!/usr/bin/env python3
"""阶段 6：概率校准、分数映射、风险等级与 LR 标准评分卡。"""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
import json
import pickle
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

from stage_contracts import (
    load_config, load_previous_manifest, require_formal_entry, write_output_list,
    write_stage_manifest,
)


def round_half_up(values: Any) -> Any:
    def convert(value: float) -> int:
        return int(Decimal(str(float(value))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    if np.isscalar(values):
        return convert(values)
    return np.asarray([convert(value) for value in np.asarray(values)])


def _raw_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-10, 1 - 1e-10)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit_platt_calibrator(
    y_pred_train: np.ndarray, y_train: pd.Series, sample_weight: Optional[pd.Series] = None,
) -> LogisticRegression:
    calibrator = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    calibrator.fit(_raw_logit(y_pred_train), y_train, sample_weight=sample_weight)
    if calibrator.coef_[0][0] <= 0:
        raise ValueError("Platt 校准系数 <= 0，校准异常，必须暂停确认")
    return calibrator


def apply_calibrator(calibrator: LogisticRegression, probabilities: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(_raw_logit(probabilities))[:, 1]


def probability_to_score(
    p_bad: np.ndarray, base_score: float, base_odds: float, pdo: float, rate: float = 2,
    clip_min: Optional[float] = None, clip_max: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.clip(np.asarray(p_bad, dtype=float), 1e-10, 1 - 1e-10)
    factor = pdo / np.log(rate)
    offset = base_score + factor * np.log(base_odds)
    score_raw = offset - factor * np.log(probability / (1 - probability))
    score = round_half_up(score_raw)
    if clip_min is not None or clip_max is not None:
        score = np.clip(score, clip_min, clip_max).astype(int)
    return score_raw, score


def fit_grade_edges(scores: pd.Series, n_grades: int) -> list[float]:
    _, edges = pd.qcut(scores, q=n_grades, retbins=True, duplicates="drop")
    edges = np.asarray(edges, dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges.tolist()


def apply_grade_edges(scores: pd.Series, edges: list[float], labels_low_to_high: list[str]) -> pd.Series:
    if len(labels_low_to_high) != len(edges) - 1:
        raise ValueError("等级标签数量必须与边界数量一致")
    return pd.cut(scores, bins=edges, labels=labels_low_to_high, include_lowest=True)


def build_lr_scorecard_tables(
    binning_detail: pd.DataFrame, coefficients: pd.DataFrame,
    base_score: float, base_odds: float, pdo: float, rate: float = 2,
    platt_intercept: float = 0.0, platt_coef: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coef = coefficients.set_index("feature")["coefficient"]
    intercept = float(coef.get("__INTERCEPT__", 0.0))
    factor = pdo / np.log(rate)
    detail = binning_detail.copy()
    detail["coefficient"] = detail["feature"].map(coef) * platt_coef
    detail["bin_points_raw"] = -factor * detail["coefficient"] * detail["woe"]
    detail["bin_points"] = round_half_up(detail["bin_points_raw"].to_numpy())
    effective_intercept = platt_intercept + platt_coef * intercept
    base_points_raw = base_score + factor * np.log(base_odds) - factor * effective_intercept
    base = pd.DataFrame([{
        "intercept": intercept, "platt_intercept": platt_intercept, "platt_coef": platt_coef,
        "effective_intercept": effective_intercept, "base_score": base_score,
        "base_odds": base_odds, "pdo": pdo, "rate": rate,
        "base_points_raw": base_points_raw, "base_points": round_half_up(base_points_raw),
    }])
    return detail, base


def validate_lr_card_scores(
    woe_data: pd.DataFrame, scored_data: pd.DataFrame, coefficients: pd.DataFrame,
    base_points: int, factor: float, platt_coef: float, dataset: str,
) -> dict:
    coef = coefficients.loc[coefficients["feature"].ne("__INTERCEPT__")].set_index("feature")["coefficient"]
    missing = [feature for feature in coef.index if feature not in woe_data.columns]
    if missing:
        raise ValueError(f"LR WOE 数据缺少最终模型特征: {missing}")
    integer_points = pd.DataFrame({
        feature: round_half_up(-factor * platt_coef * coefficient * woe_data[feature].to_numpy())
        for feature, coefficient in coef.items()
    })
    card_score = base_points + integer_points.sum(axis=1).to_numpy()
    difference = card_score - scored_data["score"].to_numpy()
    return {
        "dataset": dataset, "sample_count": len(difference),
        "max_abs_error": float(np.max(np.abs(difference))),
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "within_mean_1_point": bool(np.mean(np.abs(difference)) <= 1),
    }


def calibration_comparison(data: pd.DataFrame, calibrated: np.ndarray, dataset: str) -> dict:
    raw = np.clip(data["y_pred_raw"].to_numpy(dtype=float), 1e-10, 1 - 1e-10)
    y = data["y_true"]
    weight = data["sample_weight"] if "sample_weight" in data else None
    fpr_raw, tpr_raw, _ = roc_curve(y, raw, sample_weight=weight)
    fpr_cal, tpr_cal, _ = roc_curve(y, calibrated, sample_weight=weight)
    return {
        "dataset": dataset,
        "brier_before": brier_score_loss(y, raw, sample_weight=weight),
        "brier_after": brier_score_loss(y, calibrated, sample_weight=weight),
        "log_loss_before": log_loss(y, raw, sample_weight=weight),
        "log_loss_after": log_loss(y, calibrated, sample_weight=weight),
        "auc_before": roc_auc_score(y, raw, sample_weight=weight),
        "auc_after": roc_auc_score(y, calibrated, sample_weight=weight),
        "ks_before": float(np.max(tpr_raw - fpr_raw)),
        "ks_after": float(np.max(tpr_cal - fpr_cal)),
    }


def build_scoring_decisions() -> pd.DataFrame:
    return pd.DataFrame([{
        "decision_id": f"S{index:03d}", "decision_type": decision_type,
        "suggested_value": suggested, "confirmed_value": "", "decision": "pending",
        "reason": "", "confirmed_by": "", "confirmed_at": "",
    } for index, (decision_type, suggested) in enumerate([
        ("lr_calibration", "use_stage5_decision"), ("calibration_method", "platt"),
        ("score_parameters", "confirm_base_score_base_odds_pdo_rate"),
        ("odds_adjustment", "none"), ("score_clipping", "none"),
        ("grade_scheme", "A/B/C/D/E using Train quantiles"),
    ], start=1)])


def parse_confirmed_value(decisions: pd.DataFrame, decision_type: str) -> Any:
    row = decisions.loc[decisions["decision_type"].eq(decision_type)]
    if row.empty:
        raise ValueError(f"缺少评分决策: {decision_type}")
    value = row.iloc[0]["confirmed_value"]
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def grade_stats(scored: pd.DataFrame, dataset: str) -> pd.DataFrame:
    frame = scored.copy()
    frame["weight"] = frame["sample_weight"] if "sample_weight" in frame else 1.0
    frame["bad_weight"] = frame["weight"] * frame["y_true"]
    result = frame.groupby("risk_grade", observed=False).agg(
        count=("score", "size"), sample_weight=("weight", "sum"),
        bad_cnt=("bad_weight", "sum"), score_min=("score", "min"), score_max=("score", "max"),
    ).reset_index()
    result["sample_pct"] = result["sample_weight"] / result["sample_weight"].sum()
    result["bad_rate"] = result["bad_cnt"] / result["sample_weight"]
    result["lift"] = result["bad_rate"] / (frame["bad_weight"].sum() / frame["weight"].sum())
    result.insert(0, "dataset", dataset)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段 6：概率校准、评分与风险等级")
    parser.add_argument("--previous-manifest")
    parser.add_argument("--compatibility-mode", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--pred-train", required=True)
    parser.add_argument("--pred-test")
    parser.add_argument("--pred-oot", required=True)
    parser.add_argument("--scoring-decisions")
    parser.add_argument("--binning-detail")
    parser.add_argument("--lr-coefficients")
    parser.add_argument("--lr-woe-train")
    parser.add_argument("--lr-woe-oot")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--sep", default=",")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_formal_entry(args.previous_manifest, args.compatibility_mode, 6)
    manifest = load_previous_manifest(args.previous_manifest, 5) if args.previous_manifest else None
    config_path = Path(args.config or (manifest or {}).get("config_snapshot", ""))
    config = load_config(config_path) if config_path.exists() else {}
    if config.get("enable_stage6_scoring") is not True:
        raise ValueError("阶段 6 仅在 enable_stage6_scoring=true 时执行")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    decisions = pd.read_csv(args.scoring_decisions, encoding="utf-8-sig") if args.scoring_decisions else build_scoring_decisions()
    decisions.to_csv(output / "06_scoring_decisions.csv", index=False, encoding="utf-8-sig")
    pending = decisions.loc[decisions["decision"].eq("pending")]
    if not pending.empty:
        write_output_list(output / "06-output-list.xlsx", pd.DataFrame([{"status": "pending"}]),
                          [output / "06_scoring_decisions.csv"], pending)
        write_stage_manifest(output, 6, "pending", config_path, {"previous_manifest": args.previous_manifest or ""},
                             {"scoring_decisions": output / "06_scoring_decisions.csv",
                              "output_list": output / "06-output-list.xlsx"},
                             pending.to_dict("records"), 7)
        raise SystemExit("阶段 6 评分决策仍为 pending，请确认后重跑。")

    scoring = parse_confirmed_value(decisions, "score_parameters")
    if not isinstance(scoring, dict):
        raise ValueError("score_parameters confirmed_value 必须为 JSON 对象")
    grade_scheme = parse_confirmed_value(decisions, "grade_scheme")
    if not isinstance(grade_scheme, dict):
        raise ValueError("grade_scheme confirmed_value 必须为 JSON 对象")
    clipping = parse_confirmed_value(decisions, "score_clipping")
    clipping = clipping if isinstance(clipping, dict) and clipping.get("enabled") else {}
    model_type = config.get("model_type")
    datasets = {
        "Train": pd.read_csv(args.pred_train, sep=args.sep, encoding="utf-8-sig"),
        "OOT": pd.read_csv(args.pred_oot, sep=args.sep, encoding="utf-8-sig"),
    }
    if args.pred_test:
        datasets["Test"] = pd.read_csv(args.pred_test, sep=args.sep, encoding="utf-8-sig")
    calibration_required = model_type in {"XGB", "LGB"} or str(
        parse_confirmed_value(decisions, "lr_calibration")
    ).lower() in {"true", "calibrate", "platt"}
    calibrator = None
    comparisons = []
    if calibration_required:
        train = datasets["Train"]
        calibrator = fit_platt_calibrator(
            train["y_pred_raw"].to_numpy(), train["y_true"],
            train["sample_weight"] if "sample_weight" in train else None,
        )
        with open(output / "06_calibrator.pkl", "wb") as file:
            pickle.dump(calibrator, file)
    scored = {}
    for name, frame in datasets.items():
        final_prob = apply_calibrator(calibrator, frame["y_pred_raw"].to_numpy()) if calibrator else frame["y_pred_raw"].to_numpy()
        if calibrator:
            comparisons.append(calibration_comparison(frame, final_prob, name))
        score_raw, score = probability_to_score(
            final_prob, scoring["base_score"], scoring["base_odds"], scoring["pdo"],
            scoring.get("rate", 2), clipping.get("min"), clipping.get("max"),
        )
        result = frame.copy()
        result["y_pred_calibrated"] = final_prob if calibrator else np.nan
        result["score_raw"], result["score"] = score_raw, score
        scored[name] = result
    labels_low_to_high = grade_scheme.get("labels_low_to_high", ["E", "D", "C", "B", "A"])
    edges = fit_grade_edges(scored["Train"]["score"], len(labels_low_to_high))
    (output / "06_grade_edges.json").write_text(json.dumps(edges), encoding="utf-8")
    all_grade_stats = []
    for name, result in scored.items():
        result["risk_grade"] = apply_grade_edges(result["score"], edges, labels_low_to_high)
        result.to_csv(output / f"06_{name.lower()}_scored.csv", index=False, encoding="utf-8-sig")
        all_grade_stats.append(grade_stats(result, name))
    pd.concat(all_grade_stats, ignore_index=True).to_csv(output / "06_grade_stats.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparisons).to_csv(output / "06_calibration_comparison.csv", index=False, encoding="utf-8-sig")
    parameters = {
        **scoring, "rounding_method": "ROUND_HALF_UP", "score_clipping": clipping or None,
        "odds_definition": "bad/good", "score_direction": "higher_score_lower_risk",
    }
    (output / "06_scoring_parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
    if model_type == "LR":
        if not all([args.binning_detail, args.lr_coefficients, args.lr_woe_train, args.lr_woe_oot]):
            raise ValueError("LR 标准评分卡必须提供分箱明细、系数及 Train/OOT WOE 数据")
        binning = pd.read_csv(args.binning_detail, encoding="utf-8-sig")
        coefficients = pd.read_csv(args.lr_coefficients, encoding="utf-8-sig")
        platt_intercept = float(calibrator.intercept_[0]) if calibrator else 0.0
        platt_coef = float(calibrator.coef_[0][0]) if calibrator else 1.0
        detail, base = build_lr_scorecard_tables(
            binning, coefficients, scoring["base_score"], scoring["base_odds"], scoring["pdo"],
            scoring.get("rate", 2), platt_intercept, platt_coef,
        )
        detail.to_csv(output / "06_scorecard_detail.csv", index=False, encoding="utf-8-sig")
        base.to_csv(output / "06_scorecard_base_points.csv", index=False, encoding="utf-8-sig")
        factor = scoring["pdo"] / np.log(scoring.get("rate", 2))
        validation = []
        for name, path in [("Train", args.lr_woe_train), ("OOT", args.lr_woe_oot)]:
            validation.append(validate_lr_card_scores(
                pd.read_csv(path, encoding="utf-8-sig"), scored[name], coefficients,
                int(base.iloc[0]["base_points"]), factor, platt_coef, name,
            ))
        validation_frame = pd.DataFrame(validation)
        validation_frame.to_csv(output / "06_scorecard_validation.csv", index=False, encoding="utf-8-sig")
        if not validation_frame["within_mean_1_point"].all():
            raise ValueError("LR 评分卡一致性验证未通过：平均绝对误差超过 1 分")
    artifacts = list(output.glob("06_*"))
    write_output_list(output / "06-output-list.xlsx", pd.DataFrame([{"status": "completed"}]), artifacts,
                      extra_sheets={"scoring_decisions": decisions})
    outputs = {path.stem: path for path in output.glob("06_*")}
    outputs["output_list"] = output / "06-output-list.xlsx"
    if manifest and manifest.get("outputs", {}).get("05_evaluation_decisions"):
        outputs["05_evaluation_decisions"] = manifest["outputs"]["05_evaluation_decisions"]
    write_stage_manifest(output, 6, "completed", config_path, {"previous_manifest": args.previous_manifest or ""},
                         outputs, [], 7)


if __name__ == "__main__":
    main()
