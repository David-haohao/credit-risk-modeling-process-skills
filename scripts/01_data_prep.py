#!/usr/bin/env python3
"""阶段 1：数据准备与样本划分 —— OOT 切分、Train/Test 划分、抽样、灰样本处理、缺失率摸排。"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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


def ensure_sample_id(data: pd.DataFrame, sample_id_col: Optional[str]) -> pd.DataFrame:
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
        data["sample_id"] = [f"S{i:012d}" for i in range(len(data))]
    return data


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="阶段 1：数据准备与样本划分"
    )
    parser.add_argument("--input", required=True, help="输入 CSV 路径")
    parser.add_argument("--target-col", required=True, help="Y_label 列名")
    parser.add_argument("--time-col", required=True, help="时间列名")
    parser.add_argument("--sample-id-col", help="原始样本唯一标识列；未提供时按原始行号生成 sample_id")
    parser.add_argument("--output-dir", default="./output", help="输出目录")
    parser.add_argument("--model-type", choices=["LR", "XGB", "LGB"], default="XGB",
                        help="模型类型：LR 两段划分，XGB/LGB 三段划分")
    parser.add_argument("--oot-method", choices=["date", "months"], default="months",
                        help="OOT 切分方式")
    parser.add_argument("--oot-start-date", help="OOT 起始日期（oot-method=date 时使用）")
    parser.add_argument("--oot-months", type=int, default=3,
                        help="OOT 最近 N 个月（oot-method=months 时使用）")
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

    df = pd.read_csv(args.input, sep=args.sep, encoding="utf-8-sig")
    df = ensure_sample_id(df, args.sample_id_col)
    print(f"加载数据: {len(df)} 行, {len(df.columns)} 列")

    # 灰样本处理
    clean_data, grey_samples = handle_grey_samples(df, args.target_col)

    # OOT 切分
    if args.oot_method == "date":
        if not args.oot_start_date:
            print("错误: oot-method=date 时必须提供 --oot-start-date")
            sys.exit(1)
        modeling_data, oot_data = split_oot_by_date(clean_data, args.time_col, args.oot_start_date)
    else:
        modeling_data, oot_data = split_oot_by_recent_months(clean_data, args.time_col, args.oot_months)

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

    print(f"\n输出已保存至: {args.output_dir}")
    print(f"  train: {len(train)} 条")
    if test is not None:
        print(f"  test:  {len(test)} 条")
    print(f"  oot:   {len(oot_data)} 条")
    print(f"  灰样本: {len(grey_samples)} 条")


if __name__ == "__main__":
    main()
