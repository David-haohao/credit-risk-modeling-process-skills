#!/usr/bin/env python3
"""阶段 1：数据准备与样本切分。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from stage_contracts import (
    load_config,
    load_previous_manifest,
    require_formal_entry,
    write_config,
    write_output_list,
    write_stage_manifest,
)


COMMON_SAMPLE_ID_CANDIDATES = [
    "apply_no",
    "application_no",
    "order_no",
    "app_no",
    "loan_no",
    "mobile",
    "mobile_md5",
    "phone_md5",
    "idcard_md5",
    "cert_no_md5",
    "id_no",
    "idcard",
    "user_id",
    "customer_id",
]


def normalize_datetime_series(series: pd.Series) -> pd.Series:
    non_null = series.dropna()
    if non_null.empty:
        return pd.to_datetime(series, errors="coerce")
    as_text = non_null.astype(str).str.strip()
    if as_text.str.fullmatch(r"\d{8}").all():
        return pd.to_datetime(series.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def split_oot_by_date(data: pd.DataFrame, time_col: str, oot_start_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    data[time_col] = normalize_datetime_series(data[time_col])
    oot_data = data[data[time_col] >= pd.Timestamp(oot_start_date)].copy()
    modeling_data = data[data[time_col] < pd.Timestamp(oot_start_date)].copy()
    return modeling_data, oot_data


def split_oot_by_recent_months(data: pd.DataFrame, time_col: str, months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    data[time_col] = normalize_datetime_series(data[time_col])
    max_date = data[time_col].max()
    cutoff_date = max_date - pd.DateOffset(months=months)
    oot_data = data[data[time_col] >= cutoff_date].copy()
    modeling_data = data[data[time_col] < cutoff_date].copy()
    return modeling_data, oot_data


def split_oot_by_proportion(data: pd.DataFrame, time_col: str, proportion: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < proportion < 1:
        raise ValueError("OOT 比例必须位于 0 和 1 之间")
    ordered = data.copy()
    ordered[time_col] = normalize_datetime_series(ordered[time_col])
    ordered = ordered.sort_values(time_col, kind="stable")
    boundary_index = max(0, int(np.floor(len(ordered) * (1 - proportion))))
    boundary_time = ordered.iloc[boundary_index][time_col]
    oot_data = ordered[ordered[time_col] >= boundary_time].copy()
    modeling_data = ordered[ordered[time_col] < boundary_time].copy()
    return modeling_data, oot_data


def split_train_test(data: pd.DataFrame, test_size: float = 0.3, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(data, test_size=test_size, random_state=random_state)


def full_sample(data: pd.DataFrame) -> pd.DataFrame:
    return data.copy()


def stratified_sample(data: pd.DataFrame, target_col: str, target_total: int, random_state: int = 42) -> pd.DataFrame:
    data = data.copy()
    bad = data[data[target_col] == 1]
    good = data[data[target_col] == 0]
    n_bad = len(bad)
    n_good = len(good)
    n_good_sample = max(0, target_total - n_bad)
    n_good_sample = min(n_good_sample, n_good)
    if n_good_sample == 0:
        raise ValueError("分层抽样目标样本量必须大于坏样本数量")
    good_sampled = good.sample(n=n_good_sample, random_state=random_state)
    sampled = pd.concat([bad.copy(), good_sampled], ignore_index=True)
    sampled["sample_weight"] = 1.0
    sampled.loc[sampled[target_col] == 1, "sample_weight"] = n_bad / len(bad)
    sampled.loc[sampled[target_col] == 0, "sample_weight"] = n_good / n_good_sample
    return sampled


def handle_grey_samples(data: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    grey_mask = ~data[target_col].isin([0.0, 1.0])
    grey_samples = data[grey_mask].copy()
    clean_data = data[~grey_mask].copy()
    clean_data[target_col] = clean_data[target_col].astype(int)
    return clean_data, grey_samples


def missing_report(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    cols = [c for c in data.columns if c != target_col]
    report = pd.DataFrame(
        {
            "column": cols,
            "missing_count": [data[c].isna().sum() for c in cols],
            "missing_rate": [data[c].isna().mean() for c in cols],
        }
    )
    report["missing_rate_pct"] = (report["missing_rate"] * 100).round(2)
    return report.sort_values("missing_rate", ascending=False).reset_index(drop=True)


def ensure_sample_id(data: pd.DataFrame, sample_id_col: Optional[str], source_cols: Optional[list[str]] = None) -> pd.DataFrame:
    data = data.copy()
    if sample_id_col:
        if sample_id_col not in data.columns:
            raise ValueError(f"样本唯一标识列不存在: {sample_id_col}")
        if data[sample_id_col].isna().any() or data[sample_id_col].duplicated().any():
            raise ValueError(f"样本唯一标识列必须非空且唯一: {sample_id_col}")
        data["sample_id"] = data[sample_id_col].astype(str)
        if sample_id_col != "sample_id":
            data = data.drop(columns=[sample_id_col])
        return data

    if source_cols:
        missing = [col for col in source_cols if col not in data.columns]
        if missing:
            raise ValueError(f"样本唯一标识组合字段不存在: {missing}")
        normalized = data[source_cols].astype("string").fillna("__MISSING__")
    else:
        normalized = data.astype("string").fillna("__MISSING__")

    content_hash = normalized.apply(
        lambda row: hashlib.sha256(
            json.dumps(row.tolist(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24],
        axis=1,
    )
    duplicate_order = content_hash.groupby(content_hash).cumcount()
    data["sample_id"] = [f"H{digest}-{order:04d}" for digest, order in zip(content_hash, duplicate_order)]
    return data


def discover_sample_id_candidates(columns: list[str]) -> list[str]:
    column_map = {col.lower(): col for col in columns}
    preferred = [column_map[key] for key in COMMON_SAMPLE_ID_CANDIDATES if key in column_map]
    if preferred:
        return preferred
    fuzzy = [
        col for col in columns
        if any(token in col.lower() for token in ["apply", "app", "mobile", "phone", "idcard", "cert", "order"])
    ]
    return fuzzy[:10]


def validate_stage1_confirmations(config: dict, sample_id_candidates: Optional[list[str]] = None) -> None:
    fields = config.get("fields", {})
    split = config.get("sample_split", {})
    missing = config.get("missing_definition", {})
    required_values = {
        "fields.target_col": fields.get("target_col"),
        "fields.time_col": fields.get("time_col"),
        "model_type": config.get("model_type"),
        "sample_split.oot_method": split.get("oot_method"),
        "sample_split.sample_method": split.get("sample_method"),
    }
    unresolved = [key for key, value in required_values.items() if value in {None, "", "pending"}]
    if split.get("confirmed_by_user") is not True:
        unresolved.append("sample_split.confirmed_by_user")
    if missing.get("confirmed_by_user") is not True:
        unresolved.append("missing_definition.confirmed_by_user")
    if unresolved:
        raise ValueError(f"阶段 1 缺少用户已确认配置: {unresolved}")

    has_explicit_id = bool(fields.get("sample_id_col") or fields.get("sample_id_source_cols"))
    if not has_explicit_id and fields.get("sample_id_method") != "content_hash_confirmed":
        candidates = sample_id_candidates or []
        suffix = f"，可候选字段: {candidates}" if candidates else ""
        raise ValueError(f"阶段 1 未确认 sample_id 方案，禁止默认使用 content_hash{suffix}")


def normalize_missing_values(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    missing = config.get("missing_definition", {})
    tokens = [token for token in missing.get("tokens", []) if token is not None]
    blank_as_missing = bool(missing.get("blank_as_missing", False))
    result = data.copy()
    for col in result.columns:
        if blank_as_missing and result[col].dtype == object:
            result.loc[result[col].astype(str).str.strip() == "", col] = np.nan
        if tokens:
            result[col] = result[col].replace(tokens, np.nan)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段 1：数据准备与样本切分")
    parser.add_argument("--input", help="兼容模式输入 CSV 路径")
    parser.add_argument("--previous-manifest", help="阶段 0 正式交接 manifest")
    parser.add_argument("--compatibility-mode", action="store_true", help="允许直接文件参数运行")
    parser.add_argument("--config", help="00_modeling_config.yaml")
    parser.add_argument("--target-col", help="Y_label 列名")
    parser.add_argument("--time-col", help="时间列名")
    parser.add_argument("--sample-id-col", help="原始样本唯一标识列")
    parser.add_argument("--output-dir", default="./output", help="输出目录")
    parser.add_argument("--model-type", choices=["LR", "XGB", "LGB"], default="XGB")
    parser.add_argument("--oot-method", choices=["date", "months", "proportion"], default="months")
    parser.add_argument("--oot-start-date")
    parser.add_argument("--oot-months", type=int, default=3)
    parser.add_argument("--oot-proportion", type=float)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--sample-method", choices=["full", "stratified"], default="full")
    parser.add_argument("--target-total", type=int, default=100000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sep", default=",")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_formal_entry(args.previous_manifest, args.compatibility_mode, 1)
    manifest = load_previous_manifest(args.previous_manifest, 0) if args.previous_manifest else None
    config_path = Path(args.config or (manifest or {}).get("config_path", ""))
    config = load_config(config_path) if config_path.exists() else {}
    args.input = args.input or (manifest or {}).get("inputs", {}).get("raw_data")
    fields = config.get("fields", {})
    split_config = config.get("sample_split", {})
    args.target_col = fields.get("target_col", args.target_col)
    args.time_col = fields.get("time_col", args.time_col)
    args.sample_id_col = fields.get("sample_id_col", args.sample_id_col)
    sample_id_source_cols = fields.get("sample_id_source_cols") or []
    sample_id_method = fields.get("sample_id_method")
    args.model_type = config.get("model_type", args.model_type)
    args.oot_method = split_config.get("oot_method", args.oot_method)
    args.oot_start_date = split_config.get("oot_start_date", args.oot_start_date)
    args.oot_months = split_config.get("oot_months", args.oot_months)
    args.oot_proportion = split_config.get("oot_proportion", args.oot_proportion)
    args.sample_method = split_config.get("sample_method", args.sample_method)
    if not args.input or not args.target_col or not args.time_col:
        raise ValueError("阶段 1 manifest/config 缺少原始数据、target_col 或 time_col")

    header = pd.read_csv(args.input, sep=args.sep, encoding="utf-8-sig", nrows=0)
    sample_id_candidates = discover_sample_id_candidates(header.columns.tolist())
    validate_stage1_confirmations(config, sample_id_candidates)

    df = pd.read_csv(args.input, sep=args.sep, encoding="utf-8-sig")
    df = normalize_missing_values(df, config)
    if sample_id_method == "content_hash_confirmed" and not args.sample_id_col and not sample_id_source_cols:
        pass
    df = ensure_sample_id(df, args.sample_id_col, sample_id_source_cols)

    clean_data, grey_samples = handle_grey_samples(df, args.target_col)
    if args.oot_method == "date":
        if not args.oot_start_date:
            raise ValueError("oot-method=date 时必须提供 oot_start_date")
        modeling_data, oot_data = split_oot_by_date(clean_data, args.time_col, args.oot_start_date)
    elif args.oot_method == "months":
        modeling_data, oot_data = split_oot_by_recent_months(clean_data, args.time_col, args.oot_months)
    else:
        if args.oot_proportion is None:
            raise ValueError("oot-method=proportion 时必须确认 oot_proportion")
        modeling_data, oot_data = split_oot_by_proportion(clean_data, args.time_col, args.oot_proportion)

    if args.model_type == "LR":
        train = modeling_data.copy()
        test = None
    else:
        train, test = split_train_test(modeling_data, test_size=args.test_size, random_state=args.random_state)

    if args.sample_method == "stratified":
        train = stratified_sample(train, args.target_col, args.target_total, args.random_state)
    else:
        train = full_sample(train)

    miss_report = missing_report(train, args.target_col)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train.to_csv(output / "01_train.csv", index=False, encoding="utf-8-sig")
    if test is not None:
        test.to_csv(output / "01_test.csv", index=False, encoding="utf-8-sig")
    oot_data.to_csv(output / "01_oot.csv", index=False, encoding="utf-8-sig")
    miss_report.to_csv(output / "01_missing_report.csv", index=False, encoding="utf-8-sig")
    if len(grey_samples) > 0:
        grey_samples.to_csv(output / "01_grey_samples.csv", index=False, encoding="utf-8-sig")

    split_audit = pd.DataFrame(
        [
            {
                "dataset": name,
                "rows": len(frame),
                "bad_rate": frame[args.target_col].mean(),
                "time_min": frame[args.time_col].min(),
                "time_max": frame[args.time_col].max(),
            }
            for name, frame in [("train", train), ("test", test), ("oot", oot_data)]
            if frame is not None
        ]
    )
    split_audit.to_csv(output / "01_split_audit.csv", index=False, encoding="utf-8-sig")
    id_audit = pd.DataFrame(
        [
            {
                "method": "source_column" if args.sample_id_col else ("source_columns" if sample_id_source_cols else "content_hash"),
                "null_count": int(df["sample_id"].isna().sum()),
                "duplicate_count": int(df["sample_id"].duplicated().sum()),
            }
        ]
    )
    id_audit.to_csv(output / "01_sample_id_audit.csv", index=False, encoding="utf-8-sig")
    sampling_audit = pd.DataFrame(
        [
            {
                "method": args.sample_method,
                "target_total": args.target_total,
                "train_rows": len(train),
                "has_sample_weight": "sample_weight" in train,
            }
        ]
    )
    sampling_audit.to_csv(output / "01_sampling_audit.csv", index=False, encoding="utf-8-sig")

    config.setdefault("sample_split", {}).update(
        {
            "oot_method": args.oot_method,
            "oot_start_date": args.oot_start_date,
            "oot_months": args.oot_months,
            "oot_proportion": args.oot_proportion,
            "sample_method": args.sample_method,
            "test_size": args.test_size,
            "random_state": args.random_state,
        }
    )
    config.setdefault("fields", {})["sample_id_col"] = "sample_id"
    if config_path:
        write_config(config_path, config)

    outputs = {
        "train": output / "01_train.csv",
        "oot": output / "01_oot.csv",
        "missing_report": output / "01_missing_report.csv",
        "split_audit": output / "01_split_audit.csv",
        "sample_id_audit": output / "01_sample_id_audit.csv",
        "sampling_audit": output / "01_sampling_audit.csv",
    }
    if test is not None:
        outputs["test"] = output / "01_test.csv"
    if len(grey_samples) > 0:
        outputs["grey"] = output / "01_grey_samples.csv"

    summary = pd.DataFrame([{"model_type": args.model_type, "oot_method": args.oot_method, "status": "completed"}])
    write_output_list(output / "01-output-list.xlsx", summary, list(outputs.values()))
    outputs["01_output_list"] = output / "01-output-list.xlsx"
    write_stage_manifest(output, 1, "completed", config_path, {"previous_manifest": args.previous_manifest or ""}, outputs, [], 2)


if __name__ == "__main__":
    main()
