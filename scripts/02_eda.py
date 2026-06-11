#!/usr/bin/env python3
"""阶段 2：EDA 数据探索性分析 —— 时间分布、目标分布、特征概览和数据质量检查。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import toad
from stage_contracts import load_config, load_previous_manifest, require_formal_entry, write_output_list, write_stage_manifest


# =============================================================================
# 样本时间分布
# =============================================================================


def configured_quality_candidates(
    data: pd.DataFrame, feature_cols: list[str], rules: dict,
) -> list[dict]:
    """仅执行配置中明确提供阈值或类型规则的质量检查。"""
    rows = []
    if "high_missing_rate" in rules:
        threshold = float(rules["high_missing_rate"])
        for feature in feature_cols:
            rate = data[feature].isna().mean()
            if rate > threshold:
                rows.append({"feature": feature, "issue_type": "high_missing", "issue_metric": rate,
                             "suggested_action": "drop_or_confirm"})
    if "sparse_category_rate" in rules:
        threshold = float(rules["sparse_category_rate"])
        for feature in feature_cols:
            if pd.api.types.is_numeric_dtype(data[feature]) and data[feature].nunique(dropna=True) > 20:
                continue
            sparse = data[feature].astype("string").fillna("__MISSING__").value_counts(normalize=True)
            sparse = sparse[sparse < threshold]
            if not sparse.empty:
                rows.append({"feature": feature, "issue_type": "sparse_category",
                             "issue_metric": json.dumps(sparse.to_dict(), ensure_ascii=False),
                             "suggested_action": "merge_or_confirm"})
    for feature, expected in rules.get("expected_types", {}).items():
        if feature not in data:
            continue
        actual = "numeric" if pd.api.types.is_numeric_dtype(data[feature]) else "categorical"
        if actual != expected:
            rows.append({"feature": feature, "issue_type": "type_anomaly",
                         "issue_metric": f"expected={expected},actual={actual}",
                         "suggested_action": "transform_or_confirm"})
    return rows

def plot_time_distribution(
    data: pd.DataFrame, time_col: str, target_col: str, output_path: Optional[str] = None,
) -> pd.DataFrame:
    """逐月样本量与坏账率趋势。"""
    data = data.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data["_month"] = data[time_col].dt.to_period("M")

    monthly = data.groupby("_month").agg(
        total=("_month", "count"),
        bad_count=(target_col, "sum"),
    ).reset_index()
    monthly["bad_rate"] = monthly["bad_count"] / monthly["total"]
    monthly["_month"] = monthly["_month"].astype(str)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    bars = ax1.bar(range(len(monthly)), monthly["total"], color="steelblue", alpha=0.85)
    ax1.set_xticks(range(len(monthly)))
    ax1.set_xticklabels(monthly["_month"], rotation=45, ha="right", fontsize=8)
    ax1.set_title("Monthly Sample Count", fontsize=13)
    ax1.set_ylabel("Count")
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{int(bar.get_height())}", ha="center", fontsize=7)

    ax2.plot(range(len(monthly)), monthly["bad_rate"], marker="o", color="darkred", linewidth=1.5)
    ax2.set_xticks(range(len(monthly)))
    ax2.set_xticklabels(monthly["_month"], rotation=45, ha="right", fontsize=8)
    ax2.set_title("Monthly Bad Rate", fontsize=13)
    ax2.set_ylabel("Bad Rate")
    for i, rate in enumerate(monthly["bad_rate"]):
        ax2.text(i, rate + 0.001, f"{rate:.3%}", ha="center", fontsize=7)
    ax2.axhline(y=monthly["bad_rate"].mean(), color="gray", linestyle="--",
                label=f"Avg: {monthly['bad_rate'].mean():.3%}")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    return monthly


# =============================================================================
# Y_label 分布
# =============================================================================

def plot_target_distribution(
    data: pd.DataFrame, target_col: str, output_path: Optional[str] = None,
) -> None:
    """Y_label 类别分布（好/坏/灰）。"""
    counts = data[target_col].value_counts().sort_index()
    labels_map = {0: "Good", 1: "Bad", 2: "Grey"}

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["steelblue", "darkred", "gray"]
    bars = ax.bar(
        [labels_map.get(k, str(k)) for k in counts.index],
        counts.values,
        color=[colors[k] if k < 3 else "gray" for k in counts.index],
        alpha=0.85,
    )
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val}\n({val/counts.sum():.2%})", ha="center", fontsize=10)

    ax.set_title("Target Distribution", fontsize=13)
    ax.set_ylabel("Count")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"样本总量: {len(data)}")
    print(f"坏账率: {counts.get(1, 0) / counts.sum():.4%}")
    if 2 in counts.index:
        print(f"灰样本占比: {counts.get(2, 0) / counts.sum():.4%}")


# =============================================================================
# toad 特征质量报告
# =============================================================================

def toad_detect(data: pd.DataFrame) -> pd.DataFrame:
    """使用 toad.detector 输出数据探测报告。"""
    return toad.detector.detect(data)


# =============================================================================
# 描述性统计
# =============================================================================

def feature_overview(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """输出全部候选特征的通用概览。"""
    rows = []
    for col in [c for c in data.columns if c != target_col]:
        series = data[col]
        notna = series.dropna()
        is_continuous = pd.api.types.is_numeric_dtype(series) and notna.nunique() > 20
        rows.append({
            "column": col,
            "feature_type": "continuous" if is_continuous else "categorical",
            "count": len(series),
            "non_missing_count": len(notna),
            "unique": notna.nunique(),
            "missing": series.isna().sum(),
            "missing_rate": round(series.isna().mean(), 6),
            "min": notna.min() if len(notna) > 0 else np.nan,
            "max": notna.max() if len(notna) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def numeric_describe(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """数值特征扩展描述性统计。"""
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [
        c for c in num_cols
        if c != target_col and data[c].nunique(dropna=True) > 20
    ]

    stats = []
    for col in num_cols:
        series = data[col]
        notna = series.dropna()
        stats.append({
            "column": col,
            "count": len(series),
            "distinct": notna.nunique(),
            "missing": series.isna().sum(),
            "missing_rate": round(series.isna().mean() * 100, 2),
            "min": round(notna.min(), 4) if len(notna) > 0 else np.nan,
            "max": round(notna.max(), 4) if len(notna) > 0 else np.nan,
            "mean": round(notna.mean(), 4) if len(notna) > 0 else np.nan,
            "std": round(notna.std(), 4) if len(notna) > 0 else np.nan,
            "p01": round(notna.quantile(0.01), 4) if len(notna) > 0 else np.nan,
            "p05": round(notna.quantile(0.05), 4) if len(notna) > 0 else np.nan,
            "p10": round(notna.quantile(0.10), 4) if len(notna) > 0 else np.nan,
            "p25": round(notna.quantile(0.25), 4) if len(notna) > 0 else np.nan,
            "p50": round(notna.quantile(0.50), 4) if len(notna) > 0 else np.nan,
            "p75": round(notna.quantile(0.75), 4) if len(notna) > 0 else np.nan,
            "p90": round(notna.quantile(0.90), 4) if len(notna) > 0 else np.nan,
            "p95": round(notna.quantile(0.95), 4) if len(notna) > 0 else np.nan,
            "p99": round(notna.quantile(0.99), 4) if len(notna) > 0 else np.nan,
        })

    return pd.DataFrame(stats).set_index("column")


def categorical_describe(
    data: pd.DataFrame, target_col: str,
) -> pd.DataFrame:
    """类别特征频数分布。"""
    cat_cols = [
        c for c in data.columns
        if c != target_col and (
            not pd.api.types.is_numeric_dtype(data[c])
            or data[c].nunique(dropna=True) <= 20
        )
    ]

    results = []
    for col in cat_cols:
        vc = data[col].value_counts(dropna=False)
        total = len(data)
        print(f"\n--- {col} (取值数: {len(vc)}) ---")
        display_df = vc.reset_index()
        display_df.columns = ["value", "count"]
        display_df["pct"] = (display_df["count"] / total).round(6)
        print(display_df.head(20).to_string(index=False))
        display_df.insert(0, "column", col)
        results.append(display_df)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(
        columns=["column", "value", "count", "pct"],
    )


# =============================================================================
# 时间泄露风险检查
# =============================================================================

def check_time_leakage_risk(
    data: pd.DataFrame, time_col: str, target_col: str,
) -> dict[str, float]:
    """检查数值特征是否与时间高度相关。"""
    data = data.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    time_numeric = data[time_col].astype(np.int64) / 1e18

    correlations: dict[str, float] = {}
    for col in data.select_dtypes(include=[np.number]).columns:
        if col == target_col:
            continue
        valid = data[col].notna() & pd.Series(time_numeric, index=data.index).notna()
        if valid.sum() < 10:
            continue
        corr = data.loc[valid, col].corr(pd.Series(time_numeric, index=data.index)[valid])
        if abs(corr) > 0.5:
            correlations[col] = corr

    if correlations:
        print("[WARN] 以下特征与时间相关性较高，可能存在时间泄露风险：")
        for k, v in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {k}: r = {v:.4f}")
    else:
        print("[OK] 未发现与时间高度相关的特征。")

    return correlations


# =============================================================================
# 极端值检查
# =============================================================================

def check_extreme_values(
    data: pd.DataFrame, cols: list[str], threshold: float = 5,
) -> list[dict]:
    """检查数值特征超过 threshold 个标准差的极端值。"""
    extreme_report = []
    for col in cols:
        if col not in data.columns or data[col].dtype not in [np.float64, np.int64]:
            continue
        series = data[col].dropna()
        mean, std = series.mean(), series.std()
        n_extreme = (np.abs(series - mean) > threshold * std).sum()
        if n_extreme > 0:
            extreme_report.append({
                "column": col,
                "mean": round(mean, 2),
                "std": round(std, 2),
                "min": series.min(),
                "max": series.max(),
                "n_extreme": n_extreme,
                "extreme_rate": round(n_extreme / len(series) * 100, 2),
            })

    if extreme_report:
        report_df = pd.DataFrame(extreme_report).sort_values("n_extreme", ascending=False)
        print(f"以下特征存在超过 {threshold}σ 的极端值：")
        print(report_df.to_string(index=False))
    else:
        print(f"[OK] 未发现超过 {threshold}σ 的极端值。")

    return extreme_report


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="阶段 2：EDA 数据探索性分析"
    )
    parser.add_argument("--input", help="兼容模式训练集 CSV 路径")
    parser.add_argument("--previous-manifest")
    parser.add_argument("--compatibility-mode", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--target-col", help="Y_label 列名")
    parser.add_argument("--time-col", help="时间列名")
    parser.add_argument("--sample-id-col", default="sample_id", help="样本唯一标识列")
    parser.add_argument("--sample-weight-col", default="sample_weight", help="样本权重列")
    parser.add_argument("--output-dir", default="./output", help="输出目录")
    parser.add_argument("--extreme-threshold", type=float, default=5,
                        help="极端值标准差倍数阈值")
    parser.add_argument("--sep", default=",", help="CSV 分隔符")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_formal_entry(args.previous_manifest, args.compatibility_mode, 2)
    manifest = load_previous_manifest(args.previous_manifest, 1) if args.previous_manifest else None
    config_path = Path(args.config or (manifest or {}).get("config_path", ""))
    config = load_config(config_path) if config_path.exists() else {}
    args.input = args.input or (manifest or {}).get("outputs", {}).get("train")
    fields = config.get("fields", {})
    args.target_col = fields.get("target_col", args.target_col)
    args.time_col = fields.get("time_col", args.time_col)
    args.sample_id_col = fields.get("sample_id_col") or args.sample_id_col
    if not args.input or not args.target_col or not args.time_col:
        raise ValueError("阶段 2 manifest/config 缺少 train、target_col 或 time_col")

    import os as _os
    _os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input, sep=args.sep, encoding="utf-8-sig")
    print(f"加载训练集: {len(df)} 行, {len(df.columns)} 列")
    reserved_cols = {
        args.target_col, args.time_col, args.sample_id_col, args.sample_weight_col,
    }
    feature_cols = [c for c in df.columns if c not in reserved_cols]
    feature_data = df[feature_cols + [args.target_col]].copy()

    # 时间分布
    monthly = plot_time_distribution(
        df, args.time_col, args.target_col,
        output_path=f"{args.output_dir}/02_time_distribution.png",
    )
    monthly.to_csv(f"{args.output_dir}/02_monthly_stats.csv", index=False, encoding="utf-8-sig")

    # 目标分布
    plot_target_distribution(
        df, args.target_col,
        output_path=f"{args.output_dir}/02_target_distribution.png",
    )

    # 特征概览与描述性统计
    overview = feature_overview(feature_data, args.target_col)
    overview.to_csv(f"{args.output_dir}/02_feature_overview.csv", index=False, encoding="utf-8-sig")

    num_stats = numeric_describe(feature_data, args.target_col)
    num_stats.to_csv(f"{args.output_dir}/02_numeric_stats.csv", encoding="utf-8-sig")
    print(f"\n数值特征描述性统计: {len(num_stats)} 个特征")

    cat_stats = categorical_describe(feature_data, args.target_col)
    cat_stats.to_csv(f"{args.output_dir}/02_categorical_stats.csv", index=False, encoding="utf-8-sig")

    detect_result = toad_detect(feature_data)
    detect_result.to_csv(f"{args.output_dir}/02_toad_detect.csv", encoding="utf-8-sig")

    # 时间泄露检查
    leakage = check_time_leakage_risk(
        df[feature_cols + [args.time_col, args.target_col]], args.time_col, args.target_col,
    )
    leakage_df = pd.DataFrame([
        {"feature": feature, "correlation": corr}
        for feature, corr in leakage.items()
    ])
    leakage_df.to_csv(f"{args.output_dir}/02_time_leakage.csv", index=False, encoding="utf-8-sig")

    # 极端值检查
    num_cols = feature_data.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != args.target_col]
    extreme = check_extreme_values(feature_data, num_cols, threshold=args.extreme_threshold)
    pd.DataFrame(extreme).to_csv(
        f"{args.output_dir}/02_extreme_values.csv", index=False, encoding="utf-8-sig",
    )

    quality_decisions = [
        {
            "feature": feature,
            "issue_type": "time_leakage",
            "issue_metric": corr,
            "suggested_action": "drop_or_confirm",
            "decision": "pending",
            "decision_detail": "",
            "confirmed_by": "",
            "confirmed_at": "",
        }
        for feature, corr in leakage.items()
    ]
    quality_decisions.extend({
        "feature": row["column"],
        "issue_type": "extreme_value",
        "issue_metric": row["extreme_rate"],
        "suggested_action": "cap_or_confirm",
        "decision": "pending",
        "decision_detail": "",
        "confirmed_by": "",
        "confirmed_at": "",
    } for row in extreme)
    quality_decisions.extend({
        **row, "decision": "pending", "decision_detail": "", "confirmed_by": "", "confirmed_at": "",
    } for row in configured_quality_candidates(df, feature_cols, config.get("eda_quality_thresholds", {})))
    pd.DataFrame(quality_decisions, columns=[
        "feature", "issue_type", "issue_metric", "suggested_action",
        "decision", "decision_detail", "confirmed_by", "confirmed_at",
    ]).to_csv(f"{args.output_dir}/02_quality_decisions.csv", index=False, encoding="utf-8-sig")
    output = Path(args.output_dir)
    decisions = pd.DataFrame(quality_decisions)
    summary = pd.DataFrame([{"status": "completed", "train_rows": len(df),
                             "optional_quality_checks_configured": bool(config.get("eda_quality_thresholds"))}])
    artifacts = list(output.glob("02_*"))
    write_output_list(output / "02-output-list.xlsx", summary, artifacts, decisions)
    outputs = {path.stem: path for path in output.glob("02_*")}
    outputs["02_output_list"] = output / "02-output-list.xlsx"
    pending = decisions.loc[
        decisions["decision"].eq("pending") & decisions["issue_type"].eq("time_leakage")
    ].to_dict("records") if not decisions.empty else []
    write_stage_manifest(output, 2, "completed", config_path, {"previous_manifest": args.previous_manifest or ""},
                         outputs, pending, 3)

    print(f"\nEDA 输出已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
