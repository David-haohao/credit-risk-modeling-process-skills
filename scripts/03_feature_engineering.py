#!/usr/bin/env python3
"""阶段 3：特征工程。"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import toad

from smoothed_woe import SmoothedWOETransformer
from stage_contracts import (
    load_config,
    load_previous_manifest,
    require_formal_entry,
    write_config,
    write_output_list,
    write_stage_manifest,
)


ADJUSTMENT_COLUMNS = [
    "source_feature",
    "action",
    "output_feature",
    "x_mid",
    "manual_breaks",
    "before_iv",
    "after_iv",
    "before_monotonic",
    "after_monotonic",
    "u_shape_candidate",
    "decision_reason",
    "confirmed_by",
    "confirmed_at",
]


def write_stage3_pending(output_dir: str, config_path: Path, previous_manifest: Optional[str], reason: str) -> None:
    output = Path(output_dir)
    pending = pd.DataFrame([{"issue_type": reason, "decision": "pending"}])
    artifacts = list(output.glob("03_*"))
    write_output_list(
        output / "03-output-list.xlsx",
        pd.DataFrame([{"status": "pending", "reason": reason}]),
        artifacts,
        pending,
    )
    outputs = {path.stem: path for path in output.glob("03_*")}
    outputs["03_output_list"] = output / "03-output-list.xlsx"
    write_stage_manifest(
        output,
        3,
        "pending",
        config_path,
        {"previous_manifest": previous_manifest or ""},
        outputs,
        pending.to_dict("records"),
        4,
    )


def apply_quality_actions(datasets: dict[str, pd.DataFrame], decisions: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    result = {name: frame.copy() for name, frame in datasets.items()}
    audit = []
    for row in decisions.to_dict("records"):
        decision = row.get("decision")
        if decision not in {"cap", "transform"}:
            continue
        feature = row["feature"]
        detail = json.loads(row.get("decision_detail") or "{}")
        action = "cap" if decision == "cap" else detail.get("action")
        if action == "cap":
            if "lower" not in detail and "upper" not in detail:
                raise ValueError(f"{feature} cap 必须提供 lower 或 upper")
            for frame in result.values():
                frame[feature] = frame[feature].clip(lower=detail.get("lower"), upper=detail.get("upper"))
        elif action in {"replace", "merge_categories"}:
            mapping = detail.get("mapping")
            if not isinstance(mapping, dict):
                raise ValueError(f"{feature} {action} 必须提供 mapping")
            for frame in result.values():
                frame[feature] = frame[feature].replace(mapping)
        else:
            raise ValueError(f"不支持的质量处理动作: {action}")
        audit.append({"feature": feature, "action": action, "decision_detail": json.dumps(detail, ensure_ascii=False)})
    return result, pd.DataFrame(audit)


def fast_iv(series: pd.Series, y: pd.Series, smooth: float = 0.5) -> float:
    valid_y = y.isin([0, 1])
    series = series.loc[valid_y]
    y = y.loc[valid_y]
    n_bad_total = y.sum()
    n_good_total = len(y) - n_bad_total
    if n_bad_total == 0 or n_good_total == 0:
        return 0.0
    try:
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 10:
            grouped = pd.qcut(series, q=10, duplicates="drop").astype("string")
        else:
            grouped = series.astype("string")
        grouped = grouped.fillna("__MISSING__")
    except (TypeError, ValueError):
        return 0.0
    table = pd.crosstab(grouped, y).reindex(columns=[0, 1], fill_value=0)
    n_groups = len(table)
    good_pct = (table[0] + smooth) / (n_good_total + smooth * n_groups)
    bad_pct = (table[1] + smooth) / (n_bad_total + smooth * n_groups)
    return float(((bad_pct - good_pct) * np.log(bad_pct / good_pct)).sum())


def compute_iv_table(data: pd.DataFrame, candidates: list[str], target_col: str) -> pd.DataFrame:
    iv_results = {col: fast_iv(data[col], data[target_col]) for col in candidates}
    return pd.DataFrame(list(iv_results.items()), columns=["column", "iv"]).sort_values("iv", ascending=False).reset_index(drop=True)


def build_high_iv_alerts(iv_table: pd.DataFrame, threshold: float = 1.0, stage_label: str = "fast_iv") -> pd.DataFrame:
    alerts = iv_table.loc[iv_table["iv"] >= threshold].copy()
    if alerts.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "iv_value",
                "iv_stage",
                "alert_reason",
                "possible_cause",
                "needs_manual_review",
                "decision",
            ]
        )
    alerts["feature"] = alerts["column"]
    alerts["iv_value"] = alerts["iv"]
    alerts["iv_stage"] = stage_label
    alerts["alert_reason"] = "high_iv"
    alerts["possible_cause"] = "可能为特征穿越、单箱 good 或 bad 为 0，或特征本身效果极好"
    alerts["needs_manual_review"] = True
    alerts["decision"] = "pending"
    return alerts[
        [
            "feature",
            "iv_value",
            "iv_stage",
            "alert_reason",
            "possible_cause",
            "needs_manual_review",
            "decision",
        ]
    ].reset_index(drop=True)


def report_iv_distribution(iv_table: pd.DataFrame) -> None:
    print("\n=== IV 分布 ===")
    for th in [0.10, 0.05, 0.03, 0.02, 0.015, 0.01, 0.005]:
        n = int((iv_table["iv"] >= th).sum())
        print(f"  IV >= {th:>5.3f}: {n:>4} 个特征")


def detect_rule_variables(data: pd.DataFrame, feature_cols: list[str], target_col: str, missing_threshold: float = 0.80, iv_threshold: float = 1.0) -> list[dict]:
    rule_candidates = []
    for col in feature_cols:
        missing_rate = data[col].isna().mean()
        if missing_rate < missing_threshold:
            continue
        notna_mask = data[col].notna()
        if notna_mask.sum() < 10:
            continue
        non_missing_iv = fast_iv(data.loc[notna_mask, col], data.loc[notna_mask, target_col])
        non_missing_bad_rate = data.loc[notna_mask, target_col].mean()
        if non_missing_iv > iv_threshold or non_missing_bad_rate in [0.0, 1.0]:
            rule_candidates.append(
                {
                    "feature": col,
                    "missing_rate": round(missing_rate, 4),
                    "non_missing_iv": round(non_missing_iv, 4),
                    "non_missing_n": int(notna_mask.sum()),
                    "non_missing_bad_rate": round(non_missing_bad_rate, 4),
                    "candidate_reason": "强分离变量",
                    "decision": "pending",
                    "decision_reason": "",
                    "confirmed_by": "",
                    "confirmed_at": "",
                }
            )
    return rule_candidates


def pre_filter_features(data: pd.DataFrame, feature_cols: list[str], missing_threshold: float = 0.95, min_variance: float = 1e-6) -> tuple[list[str], dict[str, list[str]]]:
    missing_rates = data[feature_cols].isnull().mean()
    high_missing = missing_rates[missing_rates > missing_threshold].index.tolist()
    low_var = []
    for col in feature_cols:
        notna = data[col].dropna()
        if len(notna) < 100:
            low_var.append(col)
        elif notna.nunique() <= 1:
            low_var.append(col)
        elif pd.api.types.is_numeric_dtype(notna) and notna.var() < min_variance:
            low_var.append(col)
    dropped = {"high_missing": high_missing, "low_var": low_var}
    all_dropped = set(high_missing + low_var)
    candidates = [c for c in feature_cols if c not in all_dropped]
    return candidates, dropped


def correlation_filter(data: pd.DataFrame, iv_selected: list[str], iv_table: pd.DataFrame, target_col: str, corr_threshold: float = 0.7) -> tuple[list[str], list[dict]]:
    numeric_cols = data[iv_selected].select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in iv_selected if c not in numeric_cols]
    if len(numeric_cols) < 2:
        return iv_selected, []
    corr_matrix = data[numeric_cols].fillna(0).corr().abs()
    iv_map = dict(zip(iv_table["column"], iv_table["iv"]))
    high_corr_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            if corr_matrix.iloc[i, j] >= corr_threshold:
                feat_a, feat_b = numeric_cols[i], numeric_cols[j]
                iv_a = iv_map.get(feat_a, 0)
                iv_b = iv_map.get(feat_b, 0)
                keep = feat_a if iv_a >= iv_b else feat_b
                drop = feat_b if keep == feat_a else feat_a
                high_corr_pairs.append(
                    {
                        "feat_a": feat_a,
                        "feat_b": feat_b,
                        "corr": round(corr_matrix.iloc[i, j], 4),
                        "keep_feature": keep,
                        "drop_feature": drop,
                        "reason": "keep_higher_iv",
                    }
                )
    drop_set = {pair["drop_feature"] for pair in high_corr_pairs}
    final_cols = [c for c in numeric_cols if c not in drop_set] + non_numeric_cols
    return final_cols, high_corr_pairs


def run_binning(
    data: pd.DataFrame,
    target_col: str,
    method: str = "chi",
    min_samples: float = 0.05,
    max_n_bins: int = 6,
    empty_separate: bool = True,
) -> tuple[toad.transform.Combiner, pd.DataFrame]:
    combiner = toad.transform.Combiner()
    combiner.fit(
        data.drop(columns=[target_col]),
        y=data[target_col],
        method=method,
        min_samples=min_samples,
        n_bins=max_n_bins,
        empty_separate=empty_separate,
    )
    binned_features = combiner.transform(data.drop(columns=[target_col])).reset_index(drop=True)
    binned_data = pd.concat([binned_features, data[[target_col]].reset_index(drop=True)], axis=1)
    return combiner, binned_data


def apply_manual_breaks(combiner: toad.transform.Combiner, manual_breaks: dict[str, list[float]]) -> None:
    if manual_breaks:
        combiner.update(manual_breaks)


def parse_manual_breaks(value: object) -> list[float]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("manual_breaks 必须是 JSON 数组")
    return [float(v) for v in parsed]


def apply_feature_adjustments(datasets: dict[str, pd.DataFrame], adjustments: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    adjusted = {name: data.copy() for name, data in datasets.items()}
    lineage: dict[str, dict] = {}
    for row in adjustments.to_dict("records"):
        source = row["source_feature"]
        action = row["action"]
        output = row.get("output_feature")
        output = source if pd.isna(output) or str(output).strip() == "" else str(output)
        if action == "keep":
            lineage[source] = {"source_feature": source, "transformation": "keep", "x_mid": np.nan}
            continue
        if action == "drop":
            for data in adjusted.values():
                data.drop(columns=[source], inplace=True)
            continue
        if action == "manual_bin":
            lineage[source] = {"source_feature": source, "transformation": "manual_bin", "x_mid": np.nan}
            continue
        if action != "u_shape_abs":
            raise ValueError(f"不支持的特征调整 action: {action}")
        x_mid = row.get("x_mid")
        if pd.isna(x_mid):
            raise ValueError(f"{source} 使用 u_shape_abs 时必须提供 x_mid")
        if output == source:
            output = f"{source}__abs_mid"
        for name, data in adjusted.items():
            if source not in data.columns:
                raise ValueError(f"{name} 缺少待转换特征 {source}")
            if output in data.columns and output != source:
                raise ValueError(f"{name} 已存在输出特征 {output}")
            data[output] = (pd.to_numeric(data[source], errors="coerce") - float(x_mid)).abs()
            data.drop(columns=[source], inplace=True)
        lineage[output] = {"source_feature": source, "transformation": "u_shape_abs", "x_mid": float(x_mid)}
    return adjusted, lineage


def run_woe(binned_data: pd.DataFrame, target_col: str, smooth: float = 0.5) -> tuple[SmoothedWOETransformer, pd.DataFrame]:
    transformer = SmoothedWOETransformer(smooth=smooth)
    data_woe = transformer.fit_transform(binned_data.drop(columns=[target_col]), binned_data[target_col])
    data_woe = pd.concat([data_woe.reset_index(drop=True), binned_data[[target_col]].reset_index(drop=True)], axis=1)
    return transformer, data_woe


def transform_woe(data: pd.DataFrame, feature_cols: list[str], target_col: str, combiner: toad.transform.Combiner, transformer: SmoothedWOETransformer) -> pd.DataFrame:
    binned = combiner.transform(data[feature_cols])
    transformed = transformer.transform(binned)
    return pd.concat([transformed.reset_index(drop=True), data[[target_col]].reset_index(drop=True)], axis=1)


def summarize_binning(binned_data: pd.DataFrame, target_col: str, transformer: SmoothedWOETransformer, stage: str, rules: Optional[dict] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    details = []
    summaries = []
    y = binned_data[target_col]
    n_good_total = int((y == 0).sum())
    n_bad_total = int((y == 1).sum())
    total_rows = len(binned_data)
    overall_bad_rate = (n_bad_total / total_rows) if total_rows else 0.0
    for feature in binned_data.columns:
        if feature == target_col:
            continue
        table = pd.crosstab(binned_data[feature], y).reindex(columns=[0, 1], fill_value=0).sort_index()
        n_bins = len(table)
        good_dist = (table[0] + transformer.smooth) / (n_good_total + transformer.smooth * n_bins)
        bad_dist = (table[1] + transformer.smooth) / (n_bad_total + transformer.smooth * n_bins)
        woe = np.log(bad_dist / good_dist)
        bad_rates = table[1] / table.sum(axis=1)
        diffs = np.diff(bad_rates.to_numpy(dtype=float))
        nonzero_signs = np.sign(diffs[np.abs(diffs) > 1e-12])
        monotonic = bool(len(nonzero_signs) <= 1 or np.all(nonzero_signs >= 0) or np.all(nonzero_signs <= 0))
        rates = bad_rates.to_numpy(dtype=float)
        min_position = int(np.argmin(rates))
        u_shape = bool(not monotonic and n_bins >= 3 and 0 < min_position < n_bins - 1 and rates[0] > rates[min_position] and rates[-1] > rates[min_position])
        summaries.append(
            {
                "feature": feature,
                "stage": stage,
                "iv": transformer.iv_values[feature],
                "is_monotonic": monotonic,
                "u_shape_candidate": u_shape,
            }
        )
        for bin_value in table.index:
            total_count = int(table.loc[bin_value].sum())
            good_count = int(table.loc[bin_value, 0])
            bad_count = int(table.loc[bin_value, 1])
            total_pct = total_count / total_rows if total_rows else 0.0
            good_pct = good_count / n_good_total if n_good_total else 0.0
            bad_pct = bad_count / n_bad_total if n_bad_total else 0.0
            lift = float(bad_rates.loc[bin_value] / overall_bad_rate) if overall_bad_rate else np.nan
            details.append(
                {
                    "feature": feature,
                    "stage": stage,
                    "bin": bin_value,
                    "binning_rule": json.dumps((rules or {}).get(feature, []), ensure_ascii=False, default=str),
                    "count": total_count,
                    "good_cnt": good_count,
                    "bad_cnt": bad_count,
                    "total_count": total_count,
                    "good_count": good_count,
                    "bad_count": bad_count,
                    "total_pct": float(total_pct),
                    "good_pct": float(good_pct),
                    "bad_pct": float(bad_pct),
                    "bad_rate": float(bad_rates.loc[bin_value]),
                    "lift": lift,
                    "woe": float(woe.loc[bin_value]),
                    "iv_component": float((bad_dist.loc[bin_value] - good_dist.loc[bin_value]) * woe.loc[bin_value]),
                }
            )
    return pd.DataFrame(details), pd.DataFrame(summaries)


def build_adjustment_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    candidates = summary.loc[~summary["is_monotonic"], ["feature", "u_shape_candidate"]].copy()
    if candidates.empty:
        return pd.DataFrame(columns=ADJUSTMENT_COLUMNS)
    candidates = candidates.rename(columns={"feature": "source_feature"})
    candidates["action"] = "pending"
    candidates["output_feature"] = candidates["source_feature"].astype(str) + "__abs_mid"
    candidates.loc[~candidates["u_shape_candidate"], "output_feature"] = ""
    candidates["x_mid"] = np.nan
    candidates["manual_breaks"] = ""
    candidates["before_iv"] = candidates["source_feature"].map(summary.set_index("feature")["iv"])
    candidates["after_iv"] = np.nan
    candidates["before_monotonic"] = False
    candidates["after_monotonic"] = np.nan
    candidates["decision_reason"] = ""
    candidates["confirmed_by"] = ""
    candidates["confirmed_at"] = ""
    return candidates[ADJUSTMENT_COLUMNS]


def export_final_features(data_train: pd.DataFrame, data_test: Optional[pd.DataFrame], data_oot: pd.DataFrame, final_cols: list[str], target_col: str, reserved_cols: list[str], output_dir: str, is_woe: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)
    suffix = "woe" if is_woe else "final"
    output_cols = [c for c in reserved_cols if c in data_train.columns] + final_cols + [target_col]
    output_cols = list(dict.fromkeys(output_cols))
    data_train[output_cols].to_csv(f"{output_dir}/03_train_{suffix}.csv", index=False, encoding="utf-8-sig")
    data_oot[output_cols].to_csv(f"{output_dir}/03_oot_{suffix}.csv", index=False, encoding="utf-8-sig")
    if data_test is not None:
        data_test[output_cols].to_csv(f"{output_dir}/03_test_{suffix}.csv", index=False, encoding="utf-8-sig")
    with open(f"{output_dir}/03_final_features.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(final_cols))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段 3：特征工程")
    parser.add_argument("--train")
    parser.add_argument("--previous-manifest")
    parser.add_argument("--compatibility-mode", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--test")
    parser.add_argument("--oot")
    parser.add_argument("--target-col")
    parser.add_argument("--time-col")
    parser.add_argument("--reserved-cols", nargs="*", default=["sample_id", "sample_weight"])
    parser.add_argument("--quality-decisions")
    parser.add_argument("--rule-decisions")
    parser.add_argument("--feature-adjustments")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--model-type", choices=["LR", "XGB", "LGB"], default="XGB")
    parser.add_argument("--iv-threshold", type=float, default=0.02)
    parser.add_argument("--corr-threshold", type=float, default=0.7)
    parser.add_argument("--missing-threshold", type=float, default=0.95)
    parser.add_argument("--rule-missing-threshold", type=float, default=0.80)
    parser.add_argument("--rule-iv-threshold", type=float, default=1.0)
    parser.add_argument("--pre-filter", action="store_true")
    parser.add_argument("--woe-smooth", type=float, default=0.5)
    parser.add_argument("--high-iv-threshold", type=float, default=1.0)
    parser.add_argument("--sep", default=",")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_formal_entry(args.previous_manifest, args.compatibility_mode, 3)
    manifest = load_previous_manifest(args.previous_manifest, 2) if args.previous_manifest else None
    config_path = Path(args.config or (manifest or {}).get("config_path", ""))
    config = load_config(config_path) if config_path.exists() else {}
    declared = (manifest or {}).get("outputs", {})
    args.train = args.train or declared.get("train")
    args.test = args.test or declared.get("test")
    args.oot = args.oot or declared.get("oot")
    args.quality_decisions = args.quality_decisions or declared.get("02_quality_decisions")
    fields = config.get("fields", {})
    stage3_config = config.get("stage3", {})
    args.target_col = fields.get("target_col", args.target_col)
    args.time_col = fields.get("time_col", args.time_col)
    args.model_type = config.get("model_type", args.model_type)
    if stage3_config.get("iv_threshold") not in {None, "pending"}:
        args.iv_threshold = float(stage3_config["iv_threshold"])
    if stage3_config.get("corr_threshold") not in {None, "pending"}:
        args.corr_threshold = float(stage3_config["corr_threshold"])
    args.high_iv_threshold = float(stage3_config.get("high_iv_threshold", args.high_iv_threshold))
    if not args.train or not args.oot or not args.target_col:
        raise ValueError("阶段 3 manifest/config 缺少 Train、OOT 或 target_col")
    os.makedirs(args.output_dir, exist_ok=True)

    train = pd.read_csv(args.train, sep=args.sep, encoding="utf-8-sig")
    if args.model_type in ("XGB", "LGB") and not args.test:
        raise ValueError("XGB/LGB 路径必须提供 Test")
    test = pd.read_csv(args.test, sep=args.sep, encoding="utf-8-sig") if args.test else None
    oot = pd.read_csv(args.oot, sep=args.sep, encoding="utf-8-sig")
    requested_reserved = args.reserved_cols + ([args.time_col] if args.time_col else [])
    reserved_cols = [c for c in requested_reserved if c in train.columns and c != args.target_col]
    active_features = [c for c in train.columns if c not in set(reserved_cols + [args.target_col])]
    original_candidate_features = active_features.copy()

    if args.quality_decisions:
        quality = pd.read_csv(args.quality_decisions, encoding="utf-8-sig")
        pending_leakage = quality[(quality["issue_type"] == "time_leakage") & (quality["decision"] == "pending")]
        if not pending_leakage.empty:
            raise ValueError("存在未确认的时间泄漏候选特征，不能继续阶段 3")
        custom_actions = quality[quality["decision"].isin(["transform", "cap"])]
        datasets = {"train": train, "oot": oot}
        if test is not None:
            datasets["test"] = test
        datasets, quality_action_audit = apply_quality_actions(datasets, custom_actions)
        train, oot, test = datasets["train"], datasets["oot"], datasets.get("test")
        quality_action_audit.to_csv(f"{args.output_dir}/03_quality_action_audit.csv", index=False, encoding="utf-8-sig")
        drop_quality = set(quality.loc[quality["decision"] == "drop", "feature"])
        active_features = [c for c in active_features if c not in drop_quality]

    rule_candidates = detect_rule_variables(train, active_features, args.target_col, args.rule_missing_threshold, args.rule_iv_threshold)
    if rule_candidates:
        rules = pd.DataFrame(rule_candidates)
        if args.rule_decisions:
            decisions = pd.read_csv(args.rule_decisions, encoding="utf-8-sig")
            decision_cols = ["feature", "decision", "decision_reason", "confirmed_by", "confirmed_at"]
            rules = rules.drop(columns=decision_cols[1:]).merge(decisions[decision_cols], on="feature", how="left")
            rules["decision"] = rules["decision"].fillna("pending")
        rules.to_csv(f"{args.output_dir}/03_rule_candidates.csv", index=False, encoding="utf-8-sig")
        if (rules["decision"] == "pending").any():
            write_stage3_pending(args.output_dir, config_path, args.previous_manifest, "rule_candidate_confirmation")
            return
        remove_rules = set(rules.loc[rules["decision"].isin(["rule", "drop"]), "feature"])
        active_features = [c for c in active_features if c not in remove_rules]

    if args.pre_filter:
        candidates, dropped = pre_filter_features(train, active_features, args.missing_threshold)
        pd.DataFrame(
            {
                "dropped_reason": ["high_missing"] * len(dropped["high_missing"]) + ["low_var"] * len(dropped["low_var"]),
                "column": dropped["high_missing"] + dropped["low_var"],
            }
        ).to_csv(f"{args.output_dir}/03_pre_filter_dropped.csv", index=False, encoding="utf-8-sig")
    else:
        candidates = active_features

    fast_iv_table = compute_iv_table(train, candidates, args.target_col)
    fast_iv_table.to_csv(f"{args.output_dir}/03_fast_iv_table.csv", index=False, encoding="utf-8-sig")
    high_iv_alerts = build_high_iv_alerts(fast_iv_table, args.high_iv_threshold, "fast_iv")

    lr_combiner = None
    lr_woe_transformer = None
    adjustment_requires_review = False
    final_cols: list[str]
    high_corr_pairs: list[dict] = []

    if args.model_type == "LR":
        candidates = fast_iv_table.loc[fast_iv_table["iv"] >= args.iv_threshold, "column"].tolist()
        if not candidates and not fast_iv_table.empty:
            candidates = [fast_iv_table.iloc[0]["column"]]
        feature_lineage = {feature: {"source_feature": feature, "transformation": "none", "x_mid": np.nan} for feature in candidates}
        lr_combiner, binned_candidates = run_binning(train[candidates + [args.target_col]], args.target_col)
        lr_woe_transformer, _ = run_woe(binned_candidates, args.target_col, smooth=args.woe_smooth)
        before_detail, before_summary = summarize_binning(binned_candidates, args.target_col, lr_woe_transformer, "before_adjustment", lr_combiner.export())
        adjustment_candidates = build_adjustment_candidates(before_summary)
        if not args.feature_adjustments:
            adjustment_candidates.to_csv(f"{args.output_dir}/03_feature_adjustments.csv", index=False, encoding="utf-8-sig")
        if not adjustment_candidates.empty and not args.feature_adjustments:
            before_detail.to_csv(f"{args.output_dir}/03_binning_detail.csv", index=False, encoding="utf-8-sig")
            before_summary.rename(columns={"feature": "column"}).to_csv(f"{args.output_dir}/03_iv_table.csv", index=False, encoding="utf-8-sig")
            write_stage3_pending(args.output_dir, config_path, args.previous_manifest, "lr_feature_adjustment_confirmation")
            return

        details_to_concat = [before_detail]
        after_summary = before_summary.copy()
        if args.feature_adjustments:
            adjustments = pd.read_csv(args.feature_adjustments, encoding="utf-8-sig")
            missing_cols = set(ADJUSTMENT_COLUMNS) - set(adjustments.columns)
            if missing_cols:
                raise ValueError(f"特征调整表缺少字段: {sorted(missing_cols)}")
            if (adjustments["action"] == "pending").any():
                raise ValueError("特征调整表仍存在 pending，不能继续阶段 3")
            adjustments.loc[(adjustments["action"] == "u_shape_abs") & adjustments["output_feature"].fillna("").eq(""), "output_feature"] = adjustments["source_feature"].astype(str) + "__abs_mid"
            datasets = {"train": train, "oot": oot}
            if test is not None:
                datasets["test"] = test
            adjusted_datasets, changed_lineage = apply_feature_adjustments(datasets, adjustments)
            train, oot, test = adjusted_datasets["train"], adjusted_datasets["oot"], adjusted_datasets.get("test")
            feature_lineage.update(changed_lineage)
            dropped = set(adjustments.loc[adjustments["action"] == "drop", "source_feature"])
            replaced = set(adjustments.loc[adjustments["action"] == "u_shape_abs", "source_feature"])
            candidates = [c for c in candidates if c not in dropped | replaced]
            for output_feature in adjustments.loc[adjustments["action"] == "u_shape_abs", "output_feature"].fillna(""):
                if output_feature:
                    candidates.append(output_feature)
            candidates = list(dict.fromkeys(candidates))
            lr_combiner, binned_candidates = run_binning(train[candidates + [args.target_col]], args.target_col)
            manual_breaks = {}
            rows_with_breaks = adjustments.loc[adjustments["action"].isin(["manual_bin", "u_shape_abs"]) & adjustments["manual_breaks"].fillna("").ne("")]
            for row in rows_with_breaks.to_dict("records"):
                feature = row["output_feature"] if row["action"] == "u_shape_abs" else row["source_feature"]
                manual_breaks[feature] = parse_manual_breaks(row["manual_breaks"])
            apply_manual_breaks(lr_combiner, manual_breaks)
            binned_candidates = lr_combiner.transform(train[candidates])
            binned_candidates[args.target_col] = train[args.target_col].values
            lr_woe_transformer, _ = run_woe(binned_candidates, args.target_col, smooth=args.woe_smooth)
            after_detail, after_summary = summarize_binning(binned_candidates, args.target_col, lr_woe_transformer, "after_adjustment", lr_combiner.export())
            details_to_concat.append(after_detail)
            after_iv_map = after_summary.set_index("feature")["iv"]
            after_monotonic_map = after_summary.set_index("feature")["is_monotonic"]
            adjustment_outputs = adjustments["output_feature"].where(adjustments["output_feature"].fillna("").ne(""), adjustments["source_feature"])
            adjustments["after_iv"] = adjustment_outputs.map(after_iv_map)
            adjustments["after_monotonic"] = adjustment_outputs.map(after_monotonic_map)
            adjusted_actions = adjustments["action"].isin(["manual_bin", "u_shape_abs"])
            adjustment_requires_review = bool((adjusted_actions & ~adjustments["after_monotonic"].fillna(False)).any())
            adjustments.to_csv(f"{args.output_dir}/03_feature_adjustments.csv", index=False, encoding="utf-8-sig")

        iv_table = lr_woe_transformer.iv_table().sort_values("iv", ascending=False).reset_index(drop=True)
        before_iv = before_summary.set_index("feature")["iv"]
        before_monotonic = before_summary.set_index("feature")["is_monotonic"]
        after_monotonic = after_summary.set_index("feature")["is_monotonic"]
        iv_table["source_feature"] = iv_table["column"].map(lambda c: feature_lineage.get(c, {}).get("source_feature", c))
        iv_table["transformation"] = iv_table["column"].map(lambda c: feature_lineage.get(c, {}).get("transformation", "none"))
        iv_table["x_mid"] = iv_table["column"].map(lambda c: feature_lineage.get(c, {}).get("x_mid", np.nan))
        iv_table["before_iv"] = iv_table["source_feature"].map(before_iv)
        iv_table["after_iv"] = iv_table["iv"]
        iv_table["before_monotonic"] = iv_table["source_feature"].map(before_monotonic)
        iv_table["after_monotonic"] = iv_table["column"].map(after_monotonic)
        iv_table["final_iv"] = iv_table["iv"]
        pd.concat(details_to_concat, ignore_index=True).to_csv(f"{args.output_dir}/03_binning_detail.csv", index=False, encoding="utf-8-sig")
        if adjustment_requires_review:
            write_stage3_pending(args.output_dir, config_path, args.previous_manifest, "lr_adjustment_reconfirmation")
            return
    else:
        iv_table = fast_iv_table.copy()
        iv_table["source_feature"] = iv_table["column"]
        iv_table["transformation"] = "none"
        iv_table["x_mid"] = np.nan
        iv_table["before_iv"] = iv_table["iv"]
        iv_table["after_iv"] = iv_table["iv"]
        iv_table["before_monotonic"] = np.nan
        iv_table["after_monotonic"] = np.nan
        iv_table["final_iv"] = iv_table["iv"]

    report_iv_distribution(iv_table)
    iv_selected = iv_table.loc[iv_table["iv"] >= args.iv_threshold, "column"].tolist()
    final_cols, high_corr_pairs = correlation_filter(train, iv_selected, iv_table, args.target_col, args.corr_threshold)
    if high_corr_pairs:
        pd.DataFrame(high_corr_pairs).to_csv(f"{args.output_dir}/03_high_corr_pairs.csv", index=False, encoding="utf-8-sig")

    selected_set = set(final_cols)
    iv_table["selected_or_dropped"] = iv_table["column"].map(lambda c: "selected" if c in selected_set else "dropped")
    iv_table["decision_reason"] = iv_table["selected_or_dropped"].map(
        {"selected": "final_feature_set", "dropped": "fast_iv_or_correlation_filter"}
    )
    iv_table.to_csv(f"{args.output_dir}/03_iv_table.csv", index=False, encoding="utf-8-sig")
    high_iv_alerts.to_csv(f"{args.output_dir}/03_high_iv_alerts.csv", index=False, encoding="utf-8-sig")

    if args.model_type == "LR":
        train_woe = transform_woe(train, final_cols, args.target_col, lr_combiner, lr_woe_transformer)
        oot_woe = transform_woe(oot, final_cols, args.target_col, lr_combiner, lr_woe_transformer)
        for col in reserved_cols:
            if col in train.columns:
                train_woe[col] = train[col].values
            if col in oot.columns:
                oot_woe[col] = oot[col].values
        with open(f"{args.output_dir}/03_combiner.pkl", "wb") as handle:
            pickle.dump(lr_combiner, handle)
        with open(f"{args.output_dir}/03_woe_transformer.pkl", "wb") as handle:
            pickle.dump(lr_woe_transformer, handle)
        export_final_features(train_woe, None, oot_woe, final_cols, args.target_col, reserved_cols, args.output_dir, is_woe=True)
    else:
        export_final_features(train, test, oot, final_cols, args.target_col, reserved_cols, args.output_dir, is_woe=False)

    output = Path(args.output_dir)
    feature_decisions = pd.DataFrame(
        [
            {
                "feature": feature,
                "final_status": "selected" if feature in selected_set else "dropped",
                "reason": "final_feature_set" if feature in selected_set else "feature_screening",
            }
            for feature in original_candidate_features
        ]
    )
    feature_decisions.to_csv(output / "03_feature_decisions.csv", index=False, encoding="utf-8-sig")
    config.setdefault("stage3", {}).update(
        {
            "iv_threshold": args.iv_threshold,
            "corr_threshold": args.corr_threshold,
            "woe_smooth": args.woe_smooth,
            "high_iv_threshold": args.high_iv_threshold,
        }
    )
    write_config(config_path, config)
    summary = pd.DataFrame([{"status": "completed", "model_type": args.model_type, "final_feature_count": len(final_cols)}])
    artifacts = list(output.glob("03_*"))
    write_output_list(output / "03-output-list.xlsx", summary, artifacts, extra_sheets={"feature_decisions": feature_decisions})
    outputs = {path.stem: path for path in output.glob("03_*")}
    outputs["03_output_list"] = output / "03-output-list.xlsx"
    write_stage_manifest(output, 3, "completed", config_path, {"previous_manifest": args.previous_manifest or ""}, outputs, [], 4)


if __name__ == "__main__":
    main()
