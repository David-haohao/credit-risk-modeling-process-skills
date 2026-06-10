#!/usr/bin/env python3
"""阶段 7：上线与监控 —— 模型导出、Score PSI、Variable PSI、KS 追踪、通过率监控。"""

from __future__ import annotations

import argparse
import pickle
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import toad
from sklearn.metrics import roc_auc_score


# =============================================================================
# 模型导出
# =============================================================================

def export_model_from_card(card: Any, output_path: str) -> None:
    """导出评分卡对象（joblib）。"""
    import joblib
    joblib.dump(card, output_path)
    print(f"评分卡模型已导出: {output_path}")


def export_model_sklearn(model: Any, output_path: str) -> None:
    """导出 sklearn / xgboost / lightgbm 模型（joblib）。"""
    import joblib
    joblib.dump(model, output_path)
    print(f"模型已导出: {output_path}")


# =============================================================================
# Score PSI 监控
# =============================================================================

def monitor_score_psi(
    scores_development: pd.Series, scores_production: pd.Series, bucket: int = 10,
) -> float:
    """监控生产分数与开发样本分数的 PSI。"""
    psi = toad.metrics.PSI(scores_development, scores_production, bucket=bucket)
    print(f"Score PSI: {psi:.4f}")

    if psi < 0.1:
        print("  判定: 稳定")
    elif psi < 0.25:
        print("  判定: 需关注")
    else:
        print("  判定: 显著漂移，建议排查原因")

    return psi


# =============================================================================
# Variable PSI 监控
# =============================================================================

def monitor_variable_psi(
    data_dev: pd.DataFrame, data_prod: pd.DataFrame, features: list[str],
    combiner: Any,
) -> pd.DataFrame:
    """逐特征计算开发样本与生产样本的 PSI。"""
    psi_results = []
    for col in features:
        if col not in data_dev.columns or col not in data_prod.columns:
            continue
        psi_val = toad.metrics.PSI(data_dev[col], data_prod[col], combiner=combiner)
        psi_results.append({"feature": col, "psi": round(psi_val, 4)})

    psi_df = pd.DataFrame(psi_results).sort_values("psi", ascending=False)

    alerts = psi_df[psi_df["psi"] > 0.25]
    if len(alerts) > 0:
        print(f"[ALERT] PSI > 0.25 的特征: {len(alerts)} 个")
        print(alerts.to_string(index=False))

    return psi_df


# =============================================================================
# 区分度监控
# =============================================================================

def monitor_discrimination(
    y_true: pd.Series, y_pred_proba: np.ndarray, period_label: str = "",
) -> dict[str, Any]:
    """按时间段追踪 KS 和 AUC。"""
    ks = toad.metrics.KS(y_pred_proba, y_true)
    auc = roc_auc_score(y_true, y_pred_proba)

    result = {"period": period_label, "KS": round(ks, 4), "AUC": round(auc, 4)}
    print(f"[{period_label}] KS={ks:.4f}, AUC={auc:.4f}")
    return result


# =============================================================================
# 通过率与坏账率监控
# =============================================================================

def monitor_pass_rate(
    df_production: pd.DataFrame, score_col: str,
    cutoff: Optional[float] = None, target_col: Optional[str] = None,
) -> dict[str, Any]:
    """追踪通过率和通过样本的坏账率。"""
    total = len(df_production)
    if cutoff is not None:
        passed = df_production[df_production[score_col] >= cutoff]
    else:
        passed = df_production

    pass_rate = len(passed) / total * 100
    stats: dict[str, Any] = {
        "total": total, "passed": len(passed),
        "pass_rate": round(pass_rate, 2),
    }

    if target_col and target_col in df_production.columns:
        stats["overall_bad_rate"] = round(df_production[target_col].mean() * 100, 4)
        stats["passed_bad_rate"] = round(passed[target_col].mean() * 100, 4)

    print(f"总量: {total}, 通过: {len(passed)} ({pass_rate:.2f}%)")
    if "overall_bad_rate" in stats:
        print(f"整体坏账率: {stats['overall_bad_rate']}%, "
              f"通过坏账率: {stats['passed_bad_rate']}%")

    return stats


# =============================================================================
# 分数分布趋势
# =============================================================================

def plot_score_trend(
    scores_by_period: dict[str, pd.Series],
    development_scores: Optional[pd.Series] = None,
    output_path: Optional[str] = None,
) -> None:
    """按时间段绘制分数分布趋势。"""
    fig, ax = plt.subplots(figsize=(12, 5))

    for label, scores in scores_by_period.items():
        scores.plot(kind="kde", ax=ax, label=label, linewidth=1.2, alpha=0.7)

    if development_scores is not None:
        development_scores.plot(kind="kde", ax=ax, label="Dev (Baseline)",
                                color="black", linewidth=2, linestyle="--")

    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution Trend", fontsize=13)
    ax.legend(fontsize=8)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="阶段 7：上线与监控"
    )
    parser.add_argument("--scores-dev", required=True, help="开发样本分数 CSV（含 score 列）")
    parser.add_argument("--scores-prod", help="生产样本分数 CSV（含 score 列）")
    parser.add_argument("--output-dir", default="./output", help="输出目录")
    parser.add_argument("--score-col", default="score", help="分数列名")
    parser.add_argument("--bucket", type=int, default=10, help="PSI 分桶数")
    parser.add_argument("--sep", default=",", help="CSV 分隔符")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import os as _os
    _os.makedirs(args.output_dir, exist_ok=True)

    dev = pd.read_csv(args.scores_dev, sep=args.sep, encoding="utf-8-sig")
    scores_dev = dev[args.score_col]

    if args.scores_prod:
        prod = pd.read_csv(args.scores_prod, sep=args.sep, encoding="utf-8-sig")
        scores_prod = prod[args.score_col]

        psi = monitor_score_psi(scores_dev, scores_prod, bucket=args.bucket)

        with open(f"{args.output_dir}/07_monitoring_report.txt", "w", encoding="utf-8") as f:
            f.write(f"score_psi: {psi:.4f}\n")

        print(f"\n监控完成。输出已保存至: {args.output_dir}")
    else:
        print("未提供生产样本分数，仅检查开发样本分布。")
        print(f"开发样本分数: mean={scores_dev.mean():.1f}, "
              f"std={scores_dev.std():.1f}, "
              f"range=[{scores_dev.min():.0f}, {scores_dev.max():.0f}]")


if __name__ == "__main__":
    main()