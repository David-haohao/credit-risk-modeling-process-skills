#!/usr/bin/env python3
"""阶段 6：评分卡转换 —— Platt 校准、概率→分数映射、风险等级切分、灰样本打分。"""

from __future__ import annotations

import argparse
import pickle
from typing import Any, Optional

import numpy as np
import pandas as pd
import toad
from sklearn.linear_model import LogisticRegression


# =============================================================================
# Platt Scaling（XGB/LGB 专属）
# =============================================================================

def platt_scaling(
    y_pred_train: np.ndarray, y_train: pd.Series, y_pred_target: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, LogisticRegression]:
    """对 XGB/LGB 预测概率做 Platt 校准。"""
    platt_lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    platt_lr.fit(y_pred_train.reshape(-1, 1), y_train)
    calibrated = platt_lr.predict_proba(y_pred_target.reshape(-1, 1))[:, 1]

    print(f"Platt LR: coef={platt_lr.coef_[0][0]:.4f}, "
          f"intercept={platt_lr.intercept_[0]:.4f}")
    return calibrated, platt_lr


# =============================================================================
# Odds 偏差校准
# =============================================================================

def odds_calibration(
    train_bad_rate: float, actual_bad_rate: float, base_odds: float,
) -> float:
    """根据实际客群坏账率修正 base_odds。"""
    odds_expected = (1 - train_bad_rate) / train_bad_rate
    odds_actual = (1 - actual_bad_rate) / actual_bad_rate
    calibrated_ln = np.log(odds_actual) - np.log(odds_expected)
    adjusted_odds = base_odds * np.exp(calibrated_ln)

    print(f"开发样本坏账率: {train_bad_rate:.4%} → odds_expected = {odds_expected:.2f}")
    print(f"实际客群坏账率: {actual_bad_rate:.4%} → odds_actual = {odds_actual:.2f}")
    print(f"ln(Odds_calibrate) = {calibrated_ln:.4f}")
    print(f"base_odds 修正: {base_odds:.2f} → {adjusted_odds:.2f}")

    return adjusted_odds


# =============================================================================
# LR：评分卡构建（toad.ScoreCard）
# =============================================================================

def build_scorecard(
    combiner: Any, transer: Any, X_train: pd.DataFrame, y_train: pd.Series,
    lr_model: Any, base_score: int = 600, base_odds: float = 35,
    pdo: int = 60, rate: int = 2,
) -> tuple[Any, pd.DataFrame]:
    """使用 toad.ScoreCard 构建评分卡。"""
    card = toad.ScoreCard(
        combiner=combiner, transer=transer,
        base_score=base_score, base_odds=base_odds,
        pdo=pdo, rate=rate,
    )
    card.fit(X_train, y_train)

    card_df = card.export()
    print(f"评分卡构建完成，共 {len(card_df)} 条规则")
    print(f"\n评分卡明细（前 20 条）：")
    print(card_df.head(20).to_string())

    return card, card_df


# =============================================================================
# XGB/LGB：评分公式
# =============================================================================

def xgb_score_formula(
    y_pred_proba_calibrated: np.ndarray, base_score: int = 600,
    base_odds: float = 35, pdo: int = 60, rate: int = 2,
) -> np.ndarray:
    """将 Platt 校准后的概率线性映射为整数分。"""
    B = pdo / np.log(rate)
    A = base_score + B * np.log(base_odds)

    odds = y_pred_proba_calibrated / (1 - y_pred_proba_calibrated)
    scores = A - B * np.log(odds)
    scores = scores.clip(300, 900)

    print(f"评分公式: Score = {A:.2f} - {B:.2f} * ln(Odds)")
    print(f"分数范围: {scores.min():.0f} ~ {scores.max():.0f}")
    return scores.astype(int)


# =============================================================================
# 样本打分
# =============================================================================

def score_samples(card: Any, X_data: pd.DataFrame) -> pd.Series:
    """对数据集打分。"""
    return card.predict(X_data)


# =============================================================================
# 风险等级切分
# =============================================================================

def assign_risk_grades(
    scores: pd.Series, cut_method: str = "quantile",
    custom_cuts: Optional[list[float]] = None,
    labels: Optional[list[str]] = None,
) -> pd.Series:
    """按用户指定方式切分风险等级。A=低风险, E=高风险。"""
    if labels is None:
        labels = ["A", "B", "C", "D", "E"]

    if cut_method == "quantile":
        n = len(labels)
        q_values = [i / n for i in range(n + 1)]
        grades = pd.qcut(scores, q=q_values, labels=labels)
    elif cut_method == "score":
        bins = [-float("inf")] + sorted(custom_cuts or []) + [float("inf")]
        grades = pd.cut(scores, bins=bins, labels=labels)
    else:
        raise ValueError(f"不支持的切分方式: {cut_method}")

    return grades


def verify_grade_badrate(
    scores: pd.Series, y_true: pd.Series, grades: pd.Series,
) -> pd.DataFrame:
    """验证风险等级排序：A→E bad_rate 应单调递增。"""
    df = pd.DataFrame({"score": scores, "y": y_true.values, "grade": grades.values})
    grade_stats = df.groupby("grade", observed=False).agg(
        count=("y", "count"), bad_count=("y", "sum"),
        score_min=("score", "min"), score_max=("score", "max"),
        score_avg=("score", "mean"),
    ).reset_index()
    grade_stats["pct"] = (grade_stats["count"] / len(df) * 100).round(2)
    grade_stats["bad_rate"] = (grade_stats["bad_count"] / grade_stats["count"]).round(4)

    bad_rates = grade_stats["bad_rate"].tolist()
    expected_order = ["A", "B", "C", "D", "E"]
    actual_order = grade_stats["grade"].tolist()

    if actual_order == expected_order and all(
        bad_rates[i] < bad_rates[i + 1] for i in range(len(bad_rates) - 1)
    ):
        print("[OK] Bad_rate A→E 单调递增，风险排序正确")
    else:
        print("[WARN] Bad_rate 不单调或顺序异常，需调整切分")
        print(f"  期望顺序: {expected_order}")
        print(f"  实际顺序: {actual_order}")
        print(f"  Bad rates: {bad_rates}")

    return grade_stats


# =============================================================================
# 分数分布可视化
# =============================================================================

def plot_score_distribution(
    scores_train: pd.Series, scores_oot: Optional[pd.Series] = None,
    output_path: Optional[str] = None,
) -> None:
    """Plotly 交互式分数分布直方图。"""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores_train, name=f"Train (n={len(scores_train)})",
        opacity=0.6, marker_color="steelblue",
        histnorm="probability density",
    ))
    if scores_oot is not None:
        fig.add_trace(go.Histogram(
            x=scores_oot, name=f"OOT (n={len(scores_oot)})",
            opacity=0.6, marker_color="darkred",
            histnorm="probability density",
        ))
    fig.update_layout(
        title="Score Distribution", xaxis_title="Score",
        yaxis_title="Density", barmode="overlay",
    )
    if output_path:
        fig.write_html(output_path)
    fig.show()


# =============================================================================
# 灰样本打分
# =============================================================================

def score_grey_samples(
    grey_samples: pd.DataFrame, model: Any, feature_cols: list[str],
    platt_lr: LogisticRegression, base_score: int = 600,
    base_odds: float = 35, pdo: int = 60,
) -> pd.DataFrame:
    """对灰样本进行打分。"""
    X_grey = grey_samples[feature_cols]
    pred_raw = model.predict_proba(X_grey)[:, 1]
    pred_cal = platt_lr.predict_proba(pred_raw.reshape(-1, 1))[:, 1]

    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    prob = np.clip(pred_cal, 1e-10, 1 - 1e-10)
    scores = offset + factor * np.log(prob / (1 - prob))
    scores = scores.clip(300, 900).astype(int)

    grades = assign_risk_grades(pd.Series(scores))

    result = grey_samples.copy()
    result["pred_prob"] = pred_cal
    result["score"] = scores
    result["risk_grade"] = grades

    print(f"灰样本打分: {len(result)} 条")
    for g in ["A", "B", "C", "D", "E"]:
        n = (result["risk_grade"] == g).sum()
        print(f"  {g}: {n} ({n/len(result):.1%})")

    return result


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="阶段 6：评分卡转换"
    )
    parser.add_argument("--pred-train", required=True, help="训练集预测结果 CSV")
    parser.add_argument("--pred-oot", help="OOT 预测结果 CSV")
    parser.add_argument("--output-dir", default="./output", help="输出目录")
    parser.add_argument("--model-type", choices=["LR", "XGB", "LGB"], default="XGB")
    parser.add_argument("--base-score", type=int, default=600, help="基准分")
    parser.add_argument("--base-odds", type=float, default=35, help="基准 odds")
    parser.add_argument("--pdo", type=int, default=60, help="PDO")
    parser.add_argument("--rate", type=int, default=2, help="odds 变化倍数")
    parser.add_argument("--train-bad-rate", type=float, help="训练集坏账率（用于 odds 校准）")
    parser.add_argument("--actual-bad-rate", type=float, help="实际客群坏账率（用于 odds 校准）")
    parser.add_argument("--sep", default=",", help="CSV 分隔符")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import os as _os
    _os.makedirs(args.output_dir, exist_ok=True)

    pred_train = pd.read_csv(args.pred_train, sep=args.sep, encoding="utf-8-sig")
    y_train, s_train = pred_train["y_true"], pred_train["y_pred"].values

    # Odds 校准
    base_odds = float(args.base_odds)
    if args.train_bad_rate and args.actual_bad_rate:
        base_odds = odds_calibration(args.train_bad_rate, args.actual_bad_rate, base_odds)

    if args.model_type in ("XGB", "LGB"):
        # Platt 校准
        s_cal_train, _platt_lr = platt_scaling(s_train, y_train, s_train)

        # 评分
        scores_train = xgb_score_formula(
            s_cal_train, base_score=args.base_score,
            base_odds=base_odds, pdo=args.pdo, rate=args.rate,
        )
        scores_train = pd.Series(scores_train)

        # OOT
        scores_oot = None
        if args.pred_oot:
            pred_oot = pd.read_csv(args.pred_oot, sep=args.sep, encoding="utf-8-sig")
            s_oot = pred_oot["y_pred"].values
            s_cal_oot, _ = platt_scaling(s_train, y_train, s_oot)
            scores_oot = pd.Series(xgb_score_formula(
                s_cal_oot, base_score=args.base_score,
                base_odds=base_odds, pdo=args.pdo, rate=args.rate,
            ))
    else:
        # LR：直接使用预测概率
        scores_train = pd.Series(xgb_score_formula(
            s_train, base_score=args.base_score,
            base_odds=base_odds, pdo=args.pdo, rate=args.rate,
        ))
        scores_oot = None
        if args.pred_oot:
            pred_oot = pd.read_csv(args.pred_oot, sep=args.sep, encoding="utf-8-sig")
            scores_oot = pd.Series(xgb_score_formula(
                pred_oot["y_pred"].values, base_score=args.base_score,
                base_odds=base_odds, pdo=args.pdo, rate=args.rate,
            ))

    # 风险等级
    grades_train = assign_risk_grades(scores_train)
    grade_stats = verify_grade_badrate(scores_train, y_train, grades_train)
    grade_stats.to_csv(f"{args.output_dir}/06_grade_stats_train.csv", index=False, encoding="utf-8-sig")

    # 分数分布
    plot_score_distribution(
        scores_train, scores_oot,
        output_path=f"{args.output_dir}/06_score_distribution.html",
    )

    # 导出打分结果
    scored_train = pd.DataFrame({"score": scores_train, "risk_grade": grades_train})
    scored_train.to_csv(f"{args.output_dir}/06_scored_train.csv", index=False, encoding="utf-8-sig")

    if scores_oot is not None:
        grades_oot = assign_risk_grades(scores_oot)
        scored_oot = pd.DataFrame({"score": scores_oot, "risk_grade": grades_oot})
        scored_oot.to_csv(f"{args.output_dir}/06_scored_oot.csv", index=False, encoding="utf-8-sig")

    print(f"\n评分卡转换完成。输出已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()