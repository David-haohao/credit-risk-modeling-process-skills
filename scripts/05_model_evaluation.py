#!/usr/bin/env python3
"""阶段 5：模型评估 —— KS/AUC/Gini、KS 曲线、ROC 曲线、KS_bucket、LIFT、PSI、校准度。"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import toad
from sklearn.metrics import roc_auc_score, roc_curve


# =============================================================================
# KS / AUC / Gini
# =============================================================================

def calc_ks_auc(
    y_true: pd.Series, y_pred_proba: np.ndarray, dataset_name: str = "",
) -> dict[str, float]:
    """计算 KS 和 AUC。"""
    ks = toad.metrics.KS(y_pred_proba, y_true)
    auc = toad.metrics.AUC(y_pred_proba, y_true)
    gini = 2 * auc - 1

    print(f"[{dataset_name}] KS={ks:.4f}, AUC={auc:.4f}, Gini={gini:.4f}")
    return {"ks": round(ks, 4), "auc": round(auc, 4), "gini": round(gini, 4)}


# =============================================================================
# KS 曲线
# =============================================================================

def plot_ks_curve(
    y_true: pd.Series, y_pred_proba: np.ndarray, dataset_name: str = "",
    output_path: str | None = None,
) -> None:
    """绘制 KS 曲线。"""
    df = pd.DataFrame({
        "y": y_true.values if hasattr(y_true, "values") else y_true,
        "score": y_pred_proba,
    })
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["cum_good"] = (df["y"] == 0).cumsum() / (df["y"] == 0).sum()
    df["cum_bad"] = (df["y"] == 1).cumsum() / (df["y"] == 1).sum()
    df["ks_curve"] = df["cum_bad"] - df["cum_good"]
    ks_val = df["ks_curve"].max()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["cum_bad"], label="Bad", color="darkred", linewidth=1.5)
    ax.plot(df["cum_good"], label="Good", color="steelblue", linewidth=1.5)
    ax.plot(df["ks_curve"], label=f"KS Curve (max={ks_val:.4f})",
            color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"KS Curve - {dataset_name} (KS={ks_val:.4f})", fontsize=13)
    ax.set_xlabel("Score Rank (descending)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# =============================================================================
# ROC 曲线
# =============================================================================

def plot_roc_curve(
    y_true: pd.Series, y_pred_proba: np.ndarray, dataset_name: str = "",
    output_path: str | None = None,
) -> None:
    """绘制 ROC 曲线。"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkred", linewidth=1.5, label=f"AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {dataset_name}", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# =============================================================================
# KS_bucket 分箱统计
# =============================================================================

def evaluate_ks_bucket(
    y_true: pd.Series, y_pred_proba: np.ndarray, n_buckets: int = 10,
    dataset_name: str = "",
) -> pd.DataFrame:
    """按预测分数分桶，输出 bad_rate、累计好/坏占比、LIFT。"""
    bucket_df = toad.metrics.KS_bucket(y_pred_proba, y_true, bucket=n_buckets)

    print(f"\n=== KS Bucket - {dataset_name} ===")
    print(bucket_df.to_string())

    bad_rates = bucket_df["bad_rate"].values
    is_mono = (
        all(bad_rates[i] <= bad_rates[i + 1] for i in range(len(bad_rates) - 1))
        or all(bad_rates[i] >= bad_rates[i + 1] for i in range(len(bad_rates) - 1))
    )
    print(f"Bad_rate 随桶序单调: {is_mono}")

    return bucket_df


# =============================================================================
# LIFT 曲线
# =============================================================================

def plot_lift_curve(
    y_true: pd.Series, y_pred_proba: np.ndarray, dataset_name: str = "",
    output_path: str | None = None,
) -> pd.DataFrame:
    """绘制 LIFT 曲线。"""
    df = pd.DataFrame({
        "y": y_true.values if hasattr(y_true, "values") else y_true,
        "score": y_pred_proba,
    })
    df["decile"] = pd.qcut(df["score"], q=10, labels=False, duplicates="drop")
    baseline = df["y"].mean()

    lift_df = df.groupby("decile")["y"].agg(["mean", "count"]).reset_index()
    lift_df["lift"] = lift_df["mean"] / baseline
    lift_df = lift_df.sort_values("decile")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(lift_df)), lift_df["lift"], color="darkred", alpha=0.85)
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.8, label="Baseline (LIFT=1)")
    ax.set_xticks(range(len(lift_df)))
    ax.set_xticklabels(lift_df["decile"] + 1)
    ax.set_xlabel("Score Decile (1=Lowest)")
    ax.set_ylabel("LIFT")
    ax.set_title(f"LIFT Curve - {dataset_name}", fontsize=13)
    ax.legend(fontsize=9)

    for i, lift in enumerate(lift_df["lift"]):
        ax.text(i, lift + 0.02, f"{lift:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    return lift_df


# =============================================================================
# 模型分数 PSI
# =============================================================================

def calc_model_psi(
    y_train_pred: np.ndarray, y_oot_pred: np.ndarray, n_buckets: int = 10,
) -> float:
    """计算训练集与 OOT 的模型预测分数 PSI。"""
    psi_val = toad.metrics.PSI(y_train_pred, y_oot_pred, bucket=n_buckets)
    print(f"模型分数 PSI (Train vs OOT): {psi_val:.4f}")

    if psi_val < 0.1:
        print("  判定: 稳定")
    elif psi_val < 0.25:
        print("  判定: 轻微漂移，建议关注")
    else:
        print("  判定: 显著漂移，建议检查客群变化或特征稳定性")

    return psi_val


# =============================================================================
# 特征 PSI
# =============================================================================

def calc_cross_sample_psi(
    data_train: pd.DataFrame, data_oot: pd.DataFrame, final_cols: list[str],
    combiner: Any,
) -> pd.DataFrame:
    """计算最终入模特征在训练集与 OOT 之间的 PSI。"""
    psi_results = []
    for col in final_cols:
        if col not in data_train.columns or col not in data_oot.columns:
            continue
        psi_val = toad.metrics.PSI(data_train[col], data_oot[col], combiner=combiner)
        psi_results.append({"column": col, "psi": round(psi_val, 4)})

    psi_df = pd.DataFrame(psi_results).sort_values("psi", ascending=False)
    print(f"特征 PSI (Train vs OOT) 前 10:")
    print(psi_df.head(10).to_string(index=False))

    high_psi = psi_df[psi_df["psi"] > 0.25]
    if len(high_psi) > 0:
        print(f"\n[WARN] 以下 {len(high_psi)} 个特征 PSI > 0.25:")
        for _, row in high_psi.iterrows():
            print(f"  {row['column']}: PSI={row['psi']:.4f}")

    return psi_df


# =============================================================================
# 校准度曲线
# =============================================================================

def plot_reliability_curve(
    y_true: pd.Series, y_pred_proba: np.ndarray, dataset_name: str = "",
    n_bins: int = 10, output_path: str | None = None,
) -> pd.DataFrame:
    """绘制校准度曲线（reliability curve）。"""
    df = pd.DataFrame({
        "y": y_true.values if hasattr(y_true, "values") else y_true,
        "prob": y_pred_proba,
    })
    df["bin"] = pd.qcut(df["prob"], q=n_bins, labels=False, duplicates="drop")
    cal = df.groupby("bin", observed=False).agg(
        mean_pred=("prob", "mean"),
        actual_rate=("y", "mean"),
        count=("y", "count"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(cal["mean_pred"], cal["actual_rate"], marker="o", color="darkred",
            linewidth=1.5, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8, label="Perfect")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Actual Bad Rate")
    ax.set_title(f"Reliability Curve - {dataset_name}", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    return cal


# =============================================================================
# 综合评估汇总
# =============================================================================

def build_evaluation_summary(metrics_dict: dict[str, dict[str, float]]) -> pd.DataFrame:
    """汇总各数据集评估指标。"""
    rows = []
    for ds_name, metrics in metrics_dict.items():
        rows.append({
            "dataset": ds_name,
            "KS": metrics.get("ks"),
            "AUC": metrics.get("auc"),
            "Gini": metrics.get("gini"),
        })

    summary_df = pd.DataFrame(rows)
    print("\n========== 模型评估汇总 ==========")
    print(summary_df.to_string(index=False))

    if len(summary_df) >= 2:
        train_ks = summary_df[summary_df["dataset"].str.contains("Train|train", na=False)]["KS"].values
        oot_ks = summary_df[summary_df["dataset"].str.contains("OOT|oot", na=False)]["KS"].values
        if len(train_ks) > 0 and len(oot_ks) > 0:
            ks_drop = train_ks[0] - oot_ks[0]
            print(f"\nKS 衰减 (Train → OOT): {ks_drop:.4f}")
            if ks_drop > 0.05:
                print("[WARN] KS 衰减 > 0.05，模型泛化能力需关注")

    return summary_df


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="阶段 5：模型评估"
    )
    parser.add_argument("--pred-train", required=True, help="训练集预测结果 CSV（含 y_true, y_pred）")
    parser.add_argument("--pred-test", help="测试集预测结果 CSV")
    parser.add_argument("--pred-oot", help="OOT 预测结果 CSV")
    parser.add_argument("--output-dir", default="./output", help="输出目录")
    parser.add_argument("--n-buckets", type=int, default=10, help="PSI/KS_bucket 分桶数")
    parser.add_argument("--sep", default=",", help="CSV 分隔符")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import os as _os
    _os.makedirs(args.output_dir, exist_ok=True)

    metrics_dict = {}

    # 训练集
    pred_train = pd.read_csv(args.pred_train, sep=args.sep, encoding="utf-8-sig")
    y_train, s_train = pred_train["y_true"], pred_train["y_pred"].values
    metrics_dict["Train"] = calc_ks_auc(y_train, s_train, "Train")
    plot_ks_curve(y_train, s_train, "Train", f"{args.output_dir}/05_ks_curve_train.png")
    plot_roc_curve(y_train, s_train, "Train", f"{args.output_dir}/05_roc_curve_train.png")
    bucket_train = evaluate_ks_bucket(y_train, s_train, args.n_buckets, "Train")
    bucket_train.to_csv(f"{args.output_dir}/05_ks_bucket_train.csv", index=False, encoding="utf-8-sig")
    plot_lift_curve(y_train, s_train, "Train", f"{args.output_dir}/05_lift_curve_train.png")
    plot_reliability_curve(y_train, s_train, "Train", output_path=f"{args.output_dir}/05_reliability_train.png")

    # 测试集
    if args.pred_test:
        pred_test = pd.read_csv(args.pred_test, sep=args.sep, encoding="utf-8-sig")
        y_test, s_test = pred_test["y_true"], pred_test["y_pred"].values
        metrics_dict["Test"] = calc_ks_auc(y_test, s_test, "Test")
        plot_ks_curve(y_test, s_test, "Test", f"{args.output_dir}/05_ks_curve_test.png")
        plot_roc_curve(y_test, s_test, "Test", f"{args.output_dir}/05_roc_curve_test.png")

    # OOT
    if args.pred_oot:
        pred_oot = pd.read_csv(args.pred_oot, sep=args.sep, encoding="utf-8-sig")
        y_oot, s_oot = pred_oot["y_true"], pred_oot["y_pred"].values
        metrics_dict["OOT"] = calc_ks_auc(y_oot, s_oot, "OOT")
        plot_ks_curve(y_oot, s_oot, "OOT", f"{args.output_dir}/05_ks_curve_oot.png")
        plot_roc_curve(y_oot, s_oot, "OOT", f"{args.output_dir}/05_roc_curve_oot.png")

        psi_val = calc_model_psi(s_train, s_oot, args.n_buckets)
        with open(f"{args.output_dir}/05_psi.txt", "w") as f:
            f.write(f"model_psi: {psi_val:.4f}\n")

    # 综合汇总
    summary = build_evaluation_summary(metrics_dict)
    summary.to_csv(f"{args.output_dir}/05_evaluation_summary.csv", index=False, encoding="utf-8-sig")

    print(f"\n评估完成。输出已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()