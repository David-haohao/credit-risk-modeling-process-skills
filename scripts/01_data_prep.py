#!/usr/bin/env python3
"""阶段 1：数据准备与样本划分 —— OOT 切分、Train/Test 划分、抽样、灰样本处理、缺失率摸排。"""

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
    load_config, load_previous_manifest, require_formal_entry, write_config,
    write_output_list, write_stage_manifest,
)


# =============================================================================
# OOT 切分
# =============================================================================

def split_oot_by_date(
    data: pd.DataFrame, time_col: str, oot_start_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按固定日期切分 OOT。"""
    data = data.copy()
    data[time_col] = pd.to_datetime(data[time_col])

    oot_data = data[data[time_col] >= pd.Timestamp(oot_start_date)].copy()
    modeling_data = data[data[time_col] < pd.Timestamp(oot_start_date)].copy()

    print(f"建模样本: {len(modeling_data)} 条（{modeling_data[time_col].min().date()} ~ {modeling_data[time_col].max().date()}）")
    print(f"OOT 样本:  {len(oot_data)} 条（{oot_data[time_col].min().date()} ~ {oot_data[time_col].max().date()}）")

    return modeling_data, oot_data


def split_oot_by_recent_months(
    data: pd.DataFrame, time_col: str, months: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """取最近 N 个月的样本作为 OOT。"""
    data = data.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    max_date = data[time_col].max()
    cutoff_date = max_date - pd.DateOffset(months=months)

    oot_data = data[data[time_col] >= cutoff_date].copy()
    modeling_data = data[data[time_col] < cutoff_date].copy()

    print(f"数据最大日期: {max_date.date()}")
    print(f"OOT 起始日期: {cutoff_date.date()}")
    print(f"建模样本: {len(modeling_data)} 条")
    print(f"OOT 样本:  {len(oot_data)} 条")

    return modeling_data, oot_data


def split_oot_by_proportion(
    data: pd.DataFrame, time_col: str, proportion: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按时间排序取末尾样本比例；边界同一时间点整体进入 OOT。"""
    if not 0 < proportion < 1:
        raise ValueError("OOT 比例必须位于 0 和 1 之间")
    ordered = data.copy()
    ordered[time_col] = pd.to_datetime(ordered[time_col])
    ordered = ordered.sort_values(time_col, kind="stable")
    boundary_index = max(0, int(np.floor(len(ordered) * (1 - proportion))))
    boundary_time = ordered.iloc[boundary_index][time_col]
    oot_data = ordered[ordered[time_col] >= boundary_time].copy()
    modeling_data = ordered[ordered[time_col] < boundary_time].copy()
    return modeling_data, oot_data


# =============================================================================
# 建模样本内部随机切分
# =============================================================================

def split_train_test(
    data: pd.DataFrame, test_size: float = 0.3, random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """建模样本内部随机切 Train 和 Test（7:3 默认）。"""
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    print(f"训练集: {len(train)} 条")
    print(f"测试集: {len(test)} 条")
    return train, test


# =============================================================================
# 抽样
# =============================================================================

def full_sample(data: pd.DataFrame) -> pd.DataFrame:
    """全量抽样：不做任何抽样，直接返回副本。"""
    return data.copy()


def stratified_sample(
    data: pd.DataFrame, target_col: str, target_total: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """分层抽样：坏样本全取，好样本按比例抽取。返回带 sample_weight 的样本。"""
    data = data.copy()
    bad = data[data[target_col] == 1]
    good = data[data[target_col] == 0]

    n_bad = len(bad)
    n_good = len(good)
    n_total = n_bad + n_good

    bad_sampled = bad.copy()
    n_good_sample = max(0, target_total - n_bad)
    n_good_sample = min(n_good_sample, n_good)
    if n_good_sample == 0:
        raise ValueError("分层抽样目标样本量必须大于坏样本数量，以保留部分好样本")

    good_sampled = good.sample(n=n_good_sample, random_state=random_state)
    sampled = pd.concat([bad_sampled, good_sampled], ignore_index=True)

    sampled["sample_weight"] = 1.0
    sampled.loc[sampled[target_col] == 1, "sample_weight"] = n_bad / len(bad_sampled)
    sampled.loc[sampled[target_col] == 0, "sample_weight"] = n_good / n_good_sample

    print(f"原始样本: {n_total}（好: {n_good}, 坏: {n_bad}, 坏账率: {n_bad/n_total:.4%}）")
    print(f"抽样后: {len(sampled)}（好: {len(good_sampled)}, 坏: {len(bad_sampled)}）")
    print(f"好样本权重: {n_good / n_good_sample:.2f}（每个好样本代表 {n_good / n_good_sample:.1f} 个原始好样本）")
    print(f"坏样本权重: {n_bad / len(bad_sampled):.2f}")

    return sampled


# =============================================================================
# 灰样本处理
# =============================================================================

def handle_grey_samples(
    data: pd.DataFrame, target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """识别并分离灰样本（Y 非 0 非 1）。"""
    grey_mask = ~data[target_col].isin([0.0, 1.0])
    grey_samples = data[grey_mask].copy()
    clean_data = data[~grey_mask].copy()
    clean_data[target_col] = clean_data[target_col].astype(int)

    print(f"原始样本: {len(data)}")
    print(f"灰样本: {len(grey_samples)} ({len(grey_samples)/len(data):.2%})")
    if len(grey_samples) > 0:
        print(f"灰样本取值: {grey_samples[target_col].unique()}")
    print(f"清洗后样本: {len(clean_data)}")

    return clean_data, grey_samples


# =============================================================================
# 缺失率摸排
# =============================================================================

def missing_report(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """输出每列的缺失率，排除目标列。仅记录，不做剔除。"""
    cols = [c for c in data.columns if c != target_col]
    report = pd.DataFrame({
        "column": cols,
        "missing_count": [data[c].isna().sum() for c in cols],
        "missing_rate": [data[c].isna().mean() for c in cols],
    })
    report["missing_rate_pct"] = (report["missing_rate"] * 100).round(2)
    report = report.sort_values("missing_rate", ascending=False)
    return report.reset_index(drop=True)


def ensure_sample_id(
    data: pd.DataFrame, sample_id_col: Optional[str], source_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """固化统一 sample_id；未指定主键时使用原始行号生成。"""
    data = data.copy()
    if sample_id_col:
        if sample_id_col not in data.columns:
            raise ValueError(f"样本唯一标识列不存在: {sample_id_col}")
        if data[sample_id_col].isna().any() or data[sample_id_col].duplicated().any():
            raise ValueError(f"样本唯一标识列必须非空且唯一: {sample_id_col}")
        data["sample_id"] = data[sample_id_col].astype(str)
        if sample_id_col != "sample_id":
            data = data.drop(columns=[sample_id_col])
    else:
        if source_cols:
            missing = [col for col in source_cols if col not in data.columns]
            if missing:
                raise ValueError(f"样本唯一标识组合字段不存在: {missing}")
        normalized = data[source_cols].astype("string").fillna("__MISSING__") if source_cols else data.astype("string").fillna("__MISSING__")
        content_hash = normalized.apply(
            lambda row: hashlib.sha256(
                json.dumps(row.tolist(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:24],
            axis=1,
        )
        duplicate_order = content_hash.groupby(content_hash).cumcount()
        data["sample_id"] = [
            f"H{digest}-{order:04d}" for digest, order in zip(content_hash, duplicate_order)
        ]
    return data


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="阶段 1：数据准备与样本划分"
    )
    parser.add_argument("--input", help="兼容模式输入 CSV 路径")
    parser.add_argument("--previous-manifest", help="阶段 0 正式交接 manifest")
    parser.add_argument("--compatibility-mode", action="store_true", help="允许直接文件参数运行")
    parser.add_argument("--config", help="00_modeling_config.yaml")
    parser.add_argument("--target-col", help="Y_label 列名")
    parser.add_argument("--time-col", help="时间列名")
    parser.add_argument("--sample-id-col", help="原始样本唯一标识列；未提供时按原始行号生成 sample_id")
    parser.add_argument("--output-dir", default="./output", help="输出目录")
    parser.add_argument("--model-type", choices=["LR", "XGB", "LGB"], default="XGB",
                        help="模型类型：LR 两段划分，XGB/LGB 三段划分")
    parser.add_argument("--oot-method", choices=["date", "months", "proportion"], default="months",
                        help="OOT 切分方式")
    parser.add_argument("--oot-start-date", help="OOT 起始日期（oot-method=date 时使用）")
    parser.add_argument("--oot-months", type=int, default=3,
                        help="OOT 最近 N 个月（oot-method=months 时使用）")
    parser.add_argument("--oot-proportion", type=float, help="按时间排序取末尾样本比例")
    parser.add_argument("--test-size", type=float, default=0.3,
                        help="Test 集比例（默认 0.3）")
    parser.add_argument("--sample-method", choices=["full", "stratified"], default="full",
                        help="抽样方式")
    parser.add_argument("--target-total", type=int, default=100000,
                        help="分层抽样目标样本量")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sep", default=",", help="CSV 分隔符")
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
    args.model_type = config.get("model_type", args.model_type)
    args.oot_method = split_config.get("oot_method", args.oot_method)
    args.oot_start_date = split_config.get("oot_start_date", args.oot_start_date)
    args.oot_months = split_config.get("oot_months", args.oot_months)
    args.oot_proportion = split_config.get("oot_proportion", args.oot_proportion)
    if not args.input or not args.target_col or not args.time_col:
        raise ValueError("阶段 1 manifest/config 缺少原始数据、target_col 或 time_col")

    df = pd.read_csv(args.input, sep=args.sep, encoding="utf-8-sig")
    df = ensure_sample_id(df, args.sample_id_col, sample_id_source_cols)
    print(f"加载数据: {len(df)} 行, {len(df.columns)} 列")

    # 灰样本处理
    clean_data, grey_samples = handle_grey_samples(df, args.target_col)

    # OOT 切分
    if args.oot_method == "date":
        if not args.oot_start_date:
            print("错误: oot-method=date 时必须提供 --oot-start-date")
            sys.exit(1)
        modeling_data, oot_data = split_oot_by_date(clean_data, args.time_col, args.oot_start_date)
    elif args.oot_method == "months":
        modeling_data, oot_data = split_oot_by_recent_months(clean_data, args.time_col, args.oot_months)
    else:
        if args.oot_proportion is None:
            raise ValueError("oot-method=proportion 时必须确认 oot_proportion")
        modeling_data, oot_data = split_oot_by_proportion(clean_data, args.time_col, args.oot_proportion)

    # 建模样本内部切分：LR 使用 Train + OOT；树模型使用 Train + Test + OOT
    if args.model_type == "LR":
        train = modeling_data.copy()
        test = None
        print(f"训练集: {len(train)} 条（LR 模式不额外切分 Test）")
    else:
        train, test = split_train_test(
            modeling_data, test_size=args.test_size, random_state=args.random_state,
        )

    # 抽样
    if args.sample_method == "stratified":
        train = stratified_sample(train, args.target_col, args.target_total, args.random_state)

    # 缺失率报告
    miss_report = missing_report(train, args.target_col)

    # 导出
    import os as _os
    _os.makedirs(args.output_dir, exist_ok=True)

    train.to_csv(f"{args.output_dir}/01_train.csv", index=False, encoding="utf-8-sig")
    if test is not None:
        test.to_csv(f"{args.output_dir}/01_test.csv", index=False, encoding="utf-8-sig")
    oot_data.to_csv(f"{args.output_dir}/01_oot.csv", index=False, encoding="utf-8-sig")
    miss_report.to_csv(f"{args.output_dir}/01_missing_report.csv", index=False, encoding="utf-8-sig")

    if len(grey_samples) > 0:
        grey_samples.to_csv(f"{args.output_dir}/01_grey_samples.csv", index=False, encoding="utf-8-sig")

    output = Path(args.output_dir)
    split_audit = pd.DataFrame([
        {"dataset": name, "rows": len(frame), "bad_rate": frame[args.target_col].mean(),
         "time_min": frame[args.time_col].min(), "time_max": frame[args.time_col].max()}
        for name, frame in [("train", train), ("test", test), ("oot", oot_data)]
        if frame is not None
    ])
    split_audit.to_csv(output / "01_split_audit.csv", index=False, encoding="utf-8-sig")
    id_audit = pd.DataFrame([{
        "method": "source_column" if args.sample_id_col else "content_hash",
        "null_count": int(df["sample_id"].isna().sum()),
        "duplicate_count": int(df["sample_id"].duplicated().sum()),
    }])
    id_audit.to_csv(output / "01_sample_id_audit.csv", index=False, encoding="utf-8-sig")
    sampling_audit = pd.DataFrame([{
        "method": args.sample_method, "target_total": args.target_total,
        "train_rows": len(train), "has_sample_weight": "sample_weight" in train,
    }])
    sampling_audit.to_csv(output / "01_sampling_audit.csv", index=False, encoding="utf-8-sig")
    config.setdefault("sample_split", {}).update({
        "oot_method": args.oot_method, "oot_start_date": args.oot_start_date,
        "oot_months": args.oot_months, "oot_proportion": args.oot_proportion,
        "test_size": args.test_size, "random_state": args.random_state,
    })
    config.setdefault("fields", {})["sample_id_col"] = "sample_id"
    if config_path:
        write_config(config_path, config)
    outputs = {
        "train": output / "01_train.csv", "oot": output / "01_oot.csv",
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

    print(f"\n输出已保存至: {args.output_dir}")
    print(f"  train: {len(train)} 条")
    if test is not None:
        print(f"  test:  {len(test)} 条")
    print(f"  oot:   {len(oot_data)} 条")
    print(f"  灰样本: {len(grey_samples)} 条")


if __name__ == "__main__":
    main()
