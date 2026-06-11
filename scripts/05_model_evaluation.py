#!/usr/bin/env python3
"""阶段 5：评估冻结模型的 y_pred_raw，并生成审计与人工决策。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

from stage_contracts import (
    load_config, load_previous_manifest, require_formal_entry, write_output_list,
    write_stage_manifest,
)


def validate_prediction_frame(data: pd.DataFrame, time_col: Optional[str] = None) -> None:
    required = {"sample_id", "y_true", "y_pred_raw"}
    if time_col:
        required.add(time_col)
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"预测文件缺少字段: {sorted(missing)}")


def fit_score_edges(scores: pd.Series, n_buckets: int = 10) -> list[float]:
    _, edges = pd.qcut(scores, q=n_buckets, retbins=True, duplicates="drop")
    edges = np.asarray(edges, dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges.tolist()


def _weighted_sum(values: pd.Series, weights: pd.Series) -> float:
    return float(np.sum(values.to_numpy(dtype=float) * weights.to_numpy(dtype=float)))


def build_bucket_table(
    data: pd.DataFrame, edges: list[float], dataset: str, scope: str,
    weight_col: Optional[str] = None,
) -> pd.DataFrame:
    frame = data.copy()
    frame["bucket"] = pd.cut(frame["y_pred_raw"], bins=edges, labels=False, include_lowest=True)
    frame["weight"] = frame[weight_col] if weight_col and weight_col in frame else 1.0
    frame["bad_weight"] = frame["weight"] * frame["y_true"]
    frame["good_weight"] = frame["weight"] * (1 - frame["y_true"])
    grouped = frame.groupby("bucket", observed=False).agg(
        count=("y_true", "size"), sample_weight=("weight", "sum"),
        bad_cnt=("bad_weight", "sum"), good_cnt=("good_weight", "sum"),
        score_min=("y_pred_raw", "min"), score_max=("y_pred_raw", "max"),
    ).reset_index()
    grouped = grouped.sort_values("bucket", ascending=False).reset_index(drop=True)
    total_weight = grouped["sample_weight"].sum()
    total_bad = grouped["bad_cnt"].sum()
    total_good = grouped["good_cnt"].sum()
    overall_bad_rate = total_bad / total_weight
    grouped["sample_pct"] = grouped["sample_weight"] / total_weight
    grouped["bad_rate"] = grouped["bad_cnt"] / grouped["sample_weight"]
    grouped["lift"] = grouped["bad_rate"] / overall_bad_rate
    grouped["cumulative_sample_pct"] = grouped["sample_weight"].cumsum() / total_weight
    grouped["cumulative_bad_pct"] = grouped["bad_cnt"].cumsum() / total_bad
    grouped["cumulative_good_pct"] = grouped["good_cnt"].cumsum() / total_good
    grouped["cumulative_lift"] = grouped["cumulative_bad_pct"] / grouped["cumulative_sample_pct"]
    grouped["cumulative_ks"] = grouped["cumulative_bad_pct"] - grouped["cumulative_good_pct"]
    grouped.insert(0, "weighted", bool(weight_col))
    grouped.insert(0, "scope", scope)
    grouped.insert(0, "dataset", dataset)
    return grouped


def psi_detail(
    expected: pd.DataFrame, actual: pd.DataFrame, edges: list[float],
    expected_name: str, actual_name: str, weight_col: Optional[str] = None,
) -> tuple[pd.DataFrame, float]:
    def proportions(frame: pd.DataFrame) -> pd.Series:
        bucket = pd.cut(frame["y_pred_raw"], edges, labels=False, include_lowest=True)
        if weight_col and weight_col in frame:
            return frame.assign(_bucket=bucket).groupby("_bucket", observed=False)[weight_col].sum() / frame[weight_col].sum()
        return bucket.value_counts(normalize=True)
    exp, act = proportions(expected), proportions(actual)
    buckets = range(len(edges) - 1)
    rows = []
    for bucket in buckets:
        expected_pct = max(float(exp.get(bucket, 0)), 1e-6)
        actual_pct = max(float(act.get(bucket, 0)), 1e-6)
        contribution = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        rows.append({"expected": expected_name, "actual": actual_name, "bucket": bucket,
                     "weighted": bool(weight_col),
                     "expected_pct": expected_pct, "actual_pct": actual_pct,
                     "psi_contribution": contribution})
    detail = pd.DataFrame(rows)
    return detail, float(detail["psi_contribution"].sum())


def period_psi(
    train: pd.DataFrame, oot: pd.DataFrame, edges: list[float], time_col: str, granularity: str,
    weight_col: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    period_codes = {"week": "W", "month": "M", "quarter": "Q"}
    if granularity not in period_codes:
        raise ValueError("psi_period_granularity 必须为 week/month/quarter")
    frame = oot.copy()
    frame[time_col] = pd.to_datetime(frame[time_col])
    frame["period"] = frame[time_col].dt.to_period(period_codes[granularity]).astype(str)
    details, summaries = [], []
    for period, group in frame.groupby("period"):
        detail, psi = psi_detail(train, group, edges, "Train", period, weight_col)
        detail.insert(0, "period", period)
        details.append(detail)
        summaries.append({"period": period, "sample_count": len(group), "weighted": bool(weight_col),
                          "bad_count": int(group["y_true"].sum()),
                          "bad_rate": group["y_true"].mean(), "psi": psi})
    return pd.concat(details, ignore_index=True), pd.DataFrame(summaries)


def discrimination_metrics(data: pd.DataFrame, dataset: str, weight_col: Optional[str] = None) -> dict:
    weight = data[weight_col] if weight_col and weight_col in data else None
    auc = roc_auc_score(data["y_true"], data["y_pred_raw"], sample_weight=weight)
    fpr, tpr, _ = roc_curve(data["y_true"], data["y_pred_raw"], sample_weight=weight)
    return {"dataset": dataset, "auc": auc, "ks": float(np.max(tpr - fpr)), "gini": 2 * auc - 1,
            "weighted": weight is not None}


def calibration_metrics(
    data: pd.DataFrame, dataset: str, n_bins: int = 10, weight_col: Optional[str] = None,
) -> tuple[dict, pd.DataFrame]:
    weight = data[weight_col] if weight_col and weight_col in data else None
    prob = np.clip(data["y_pred_raw"].to_numpy(dtype=float), 1e-10, 1 - 1e-10)
    raw_logit = np.log(prob / (1 - prob)).reshape(-1, 1)
    calibrator = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    calibrator.fit(raw_logit, data["y_true"], sample_weight=weight)
    summary = {
        "dataset": dataset,
        "brier": brier_score_loss(data["y_true"], prob, sample_weight=weight),
        "log_loss": log_loss(data["y_true"], prob, sample_weight=weight),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0][0]),
        "weighted": weight is not None,
    }
    frame = data.copy()
    frame["bin"] = pd.qcut(frame["y_pred_raw"], q=n_bins, labels=False, duplicates="drop")
    frame["weight"] = frame[weight_col] if weight_col and weight_col in frame else 1.0
    frame["weighted_pred"] = frame["weight"] * frame["y_pred_raw"]
    frame["weighted_bad"] = frame["weight"] * frame["y_true"]
    detail = frame.groupby("bin", observed=False).agg(
        count=("y_true", "size"), weight=("weight", "sum"),
        weighted_pred=("weighted_pred", "sum"), weighted_bad=("weighted_bad", "sum"),
    ).reset_index()
    detail["mean_pred"] = detail["weighted_pred"] / detail["weight"]
    detail["actual_rate"] = detail["weighted_bad"] / detail["weight"]
    detail.insert(0, "weighted", weight is not None)
    detail.insert(0, "dataset", dataset)
    return summary, detail


def build_core_decisions() -> pd.DataFrame:
    rows = []
    for index, (issue_type, scope) in enumerate([
        ("overfitting", "discrimination_and_dataset_gap"),
        ("ranking_issue", "bucket_lift_and_cumulative_lift"),
        ("data_shift", "overall_and_period_psi"),
        ("poor_calibration", "brier_logloss_reliability_slope"),
    ], start=1):
        rows.append({
            "decision_id": f"D{index:03d}", "issue_type": issue_type, "metric_scope": scope,
            "observed_value": "", "reference_value": "", "assessment": "",
            "recommended_action": "", "return_stage": "", "decision": "pending",
            "confirmed_by": "", "confirmed_at": "", "reason": "",
        })
    return pd.DataFrame(rows)


def _save_curve(table: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(table[x], table[y], marker="o")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_roc_and_ks(data: pd.DataFrame, dataset: str, output: Path) -> None:
    y, score = data["y_true"], data["y_pred_raw"]
    fpr, tpr, _ = roc_curve(y, score)
    auc = roc_auc_score(y, score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.legend()
    ax.set_title(f"ROC Curve - {dataset}")
    fig.tight_layout()
    fig.savefig(output / f"05_roc_curve_{dataset.lower()}.png", dpi=150)
    plt.close(fig)
    ordered = data.sort_values("y_pred_raw", ascending=False).reset_index(drop=True)
    ordered["cum_bad"] = ordered["y_true"].cumsum() / ordered["y_true"].sum()
    ordered["cum_good"] = (1 - ordered["y_true"]).cumsum() / (1 - ordered["y_true"]).sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ordered["cum_bad"], label="Bad")
    ax.plot(ordered["cum_good"], label="Good")
    ax.plot(ordered["cum_bad"] - ordered["cum_good"], label="KS")
    ax.legend()
    ax.set_title(f"KS Curve - {dataset}")
    fig.tight_layout()
    fig.savefig(output / f"05_ks_curve_{dataset.lower()}.png", dpi=150)
    plt.close(fig)


def save_bucket_lift(table: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(table["bucket"].astype(str), table["lift"])
    ax.axhline(1, linestyle="--", color="gray")
    ax.set_title(title)
    ax.set_xlabel("bucket")
    ax.set_ylabel("lift")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_reliability(detail: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(detail["mean_pred"], detail["actual_rate"], marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段 5：模型评估")
    parser.add_argument("--previous-manifest")
    parser.add_argument("--compatibility-mode", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--pred-train")
    parser.add_argument("--pred-test")
    parser.add_argument("--pred-oot")
    parser.add_argument("--evaluation-decisions")
    parser.add_argument("--rework-history")
    parser.add_argument("--time-col")
    parser.add_argument("--weight-col", default="sample_weight")
    parser.add_argument("--n-buckets", type=int, default=10)
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--sep", default=",")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_formal_entry(args.previous_manifest, args.compatibility_mode, 5)
    manifest = load_previous_manifest(args.previous_manifest, 4) if args.previous_manifest else None
    config_path = Path(args.config or (manifest or {}).get("config_path", ""))
    config = load_config(config_path) if config_path.exists() else {}
    declared = (manifest or {}).get("outputs", {})
    args.pred_train = args.pred_train or declared.get("04_pred_train")
    args.pred_test = args.pred_test or declared.get("04_pred_test")
    args.pred_oot = args.pred_oot or declared.get("04_pred_oot")
    if not args.pred_train or not args.pred_oot:
        raise ValueError("阶段 5 manifest 缺少 Train/OOT 预测")
    time_col = config.get("fields", {}).get("time_col", args.time_col)
    granularity = config.get("psi_period_granularity")
    if granularity not in {"week", "month", "quarter"}:
        raise ValueError("阶段 5 必须从配置读取已确认的 psi_period_granularity")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    datasets = {
        "Train": pd.read_csv(args.pred_train, sep=args.sep, encoding="utf-8-sig"),
        "OOT": pd.read_csv(args.pred_oot, sep=args.sep, encoding="utf-8-sig"),
    }
    if args.pred_test:
        datasets["Test"] = pd.read_csv(args.pred_test, sep=args.sep, encoding="utf-8-sig")
    for frame in datasets.values():
        validate_prediction_frame(frame, time_col)
    edges = fit_score_edges(datasets["Train"]["y_pred_raw"], args.n_buckets)
    (output / "05_bucket_edges.json").write_text(json.dumps(edges), encoding="utf-8")
    metrics, internal, fixed, calibration_summaries, calibration_details = [], [], [], [], []
    for name, frame in datasets.items():
        save_roc_and_ks(frame, name, output)
        scopes = [None] + ([args.weight_col] if args.weight_col in frame.columns else [])
        internal_edges = fit_score_edges(frame["y_pred_raw"], args.n_buckets)
        for scope_weight in scopes:
            metrics.append(discrimination_metrics(frame, name, scope_weight))
            internal.append(build_bucket_table(frame, internal_edges, name, "internal", scope_weight))
            fixed.append(build_bucket_table(frame, edges, name, "fixed_train_edges", scope_weight))
            cal_summary, cal_detail = calibration_metrics(frame, name, args.n_buckets, scope_weight)
            calibration_summaries.append(cal_summary)
            calibration_details.append(cal_detail)
        fixed_table = fixed[-len(scopes)]
        save_bucket_lift(fixed_table, output / f"05_lift_bucket_{name.lower()}.png", f"Bucket Lift - {name}")
        save_reliability(calibration_details[-len(scopes)], output / f"05_reliability_{name.lower()}.png",
                         f"Reliability - {name}")
        _save_curve(fixed_table, "cumulative_sample_pct", "cumulative_lift",
                    output / f"05_lift_cumulative_{name.lower()}.png", f"Cumulative Lift - {name}")
        _save_curve(fixed_table, "cumulative_sample_pct", "cumulative_bad_pct",
                    output / f"05_gains_{name.lower()}.png", f"Gains - {name}")
    pd.DataFrame(metrics).to_csv(output / "05_evaluation_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(internal, ignore_index=True).to_csv(output / "05_bucket_internal.csv", index=False, encoding="utf-8-sig")
    fixed_all = pd.concat(fixed, ignore_index=True)
    fixed_all.to_csv(output / "05_bucket_fixed.csv", index=False, encoding="utf-8-sig")
    comparison = fixed_all.loc[fixed_all["weighted"].eq(False)]
    fig, ax = plt.subplots(figsize=(8, 4))
    for dataset, group in comparison.groupby("dataset"):
        ax.plot(group["bucket"].astype(str), group["lift"], marker="o", label=dataset)
    ax.legend()
    ax.set_title("Train Fixed-Bin Lift Comparison")
    fig.tight_layout()
    fig.savefig(output / "05_lift_fixed_comparison.png", dpi=150)
    plt.close(fig)
    psi_details, psi_summaries, period_details, period_summaries = [], [], [], []
    scopes = [None] + ([args.weight_col] if args.weight_col in datasets["Train"].columns and args.weight_col in datasets["OOT"].columns else [])
    for scope_weight in scopes:
        psi, psi_value = psi_detail(datasets["Train"], datasets["OOT"], edges, "Train", "OOT", scope_weight)
        psi_details.append(psi)
        psi_summaries.append({"comparison": "Train_vs_OOT", "psi": psi_value, "weighted": bool(scope_weight)})
        period_detail, period_summary = period_psi(
            datasets["Train"], datasets["OOT"], edges, time_col, granularity, scope_weight,
        )
        period_details.append(period_detail)
        period_summaries.append(period_summary)
    pd.concat(psi_details, ignore_index=True).to_csv(output / "05_score_psi_detail.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(psi_summaries).to_csv(
        output / "05_score_psi_summary.csv", index=False, encoding="utf-8-sig",
    )
    period_detail = pd.concat(period_details, ignore_index=True)
    period_summary = pd.concat(period_summaries, ignore_index=True)
    period_detail.to_csv(output / "05_period_psi_detail.csv", index=False, encoding="utf-8-sig")
    period_summary.to_csv(output / "05_period_psi_summary.csv", index=False, encoding="utf-8-sig")
    _save_curve(period_summary, "period", "psi", output / "05_period_psi_trend.png", "Period PSI")
    pd.DataFrame(calibration_summaries).to_csv(output / "05_calibration_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(calibration_details, ignore_index=True).to_csv(output / "05_calibration_detail.csv", index=False, encoding="utf-8-sig")
    decisions = pd.read_csv(args.evaluation_decisions, encoding="utf-8-sig") if args.evaluation_decisions else build_core_decisions()
    if not args.evaluation_decisions:
        evidence = {
            "discrimination_and_dataset_gap": pd.DataFrame(metrics).to_json(orient="records", force_ascii=False),
            "bucket_lift_and_cumulative_lift": fixed_all.groupby(["dataset", "weighted"]).agg(
                max_ks=("cumulative_ks", "max"), max_cumulative_lift=("cumulative_lift", "max"),
            ).reset_index().to_json(orient="records", force_ascii=False),
            "overall_and_period_psi": json.dumps(
                {"overall": psi_summaries, "period_max": float(period_summary["psi"].max())},
                ensure_ascii=False,
            ),
            "brier_logloss_reliability_slope": pd.DataFrame(calibration_summaries).to_json(
                orient="records", force_ascii=False,
            ),
        }
        decisions["observed_value"] = decisions["metric_scope"].map(evidence)
    decisions.to_csv(output / "05_evaluation_decisions.csv", index=False, encoding="utf-8-sig")
    rework = pd.read_csv(args.rework_history, encoding="utf-8-sig") if args.rework_history else pd.DataFrame(
        columns=["iteration_id", "source_model_version", "issue_type", "evidence", "return_stage",
                 "action_taken", "new_model_version", "oot_usage_status", "reevaluation_result",
                 "confirmed_by", "confirmed_at"]
    )
    rework.to_csv(output / "05_rework_history.csv", index=False, encoding="utf-8-sig")
    pending = decisions.loc[decisions["decision"].eq("pending")]
    summary = pd.DataFrame([{"status": "pending" if not pending.empty else "completed",
                             "datasets": ",".join(datasets), "period_granularity": granularity}])
    artifacts = list(output.glob("05_*"))
    write_output_list(output / "05-output-list.xlsx", summary, artifacts, pending,
                      {"evaluation_summary": pd.DataFrame(metrics), "evaluation_decisions": decisions})
    outputs = {path.stem: path for path in output.glob("05_*")}
    outputs["05_output_list"] = output / "05-output-list.xlsx"
    status = "pending" if not pending.empty else "completed"
    next_stage = 6 if config.get("enable_stage6_scoring") is True else 7
    write_stage_manifest(output, 5, status, config_path, {"previous_manifest": args.previous_manifest or ""},
                         outputs, pending.to_dict("records"), next_stage)
    if not pending.empty:
        raise SystemExit("阶段 5 核心决策仍为 pending，请确认后重跑。")


if __name__ == "__main__":
    main()
