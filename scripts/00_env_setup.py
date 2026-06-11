#!/usr/bin/env python3
"""阶段 0：环境初始化与数据预检 —— 依赖检查、数据加载、缺失值探测、数据规模预检。"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from stage_contracts import load_config, write_config, write_output_list, write_stage_manifest


# =============================================================================
# 依赖检查
# =============================================================================

REQUIRED_PACKAGES: dict[str, str] = {
    "pandas": "1.0+",
    "numpy": "1.18+",
    "matplotlib": "3.0+",
    "seaborn": "0.10+",
    "plotly": "5.0+",
    "toad": "0.1.0+",
    "xgboost": "1.5+",
    "lightgbm": "3.0+",
    "optuna": "2.0+",
    "sklearn": "0.24+",
    "scipy": "1.5+",
    "statsmodels": "0.12+",
    "yaml": "6.0+",
}
INSTALL_NAMES = {"yaml": "PyYAML", "sklearn": "scikit-learn"}


def check_packages() -> bool:
    """检查所有依赖包，返回是否全部就绪。"""
    missing: list[str] = []
    for pkg, version in REQUIRED_PACKAGES.items():
        try:
            mod = importlib.import_module(pkg)
            actual = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {pkg} == {actual}")
        except ImportError:
            print(f"  [MISSING] {pkg} ({version})")
            missing.append(INSTALL_NAMES.get(pkg, pkg))

    if missing:
        print(f"\n缺少以下包，请先安装：")
        print(f"  pip install {' '.join(missing)}")
        return False
    else:
        print(f"\n所有依赖包已就绪。")
        return True


# =============================================================================
# 中文字体配置
# =============================================================================

def setup_chinese_font() -> None:
    """配置 matplotlib 中文字体与 seaborn 样式。"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        print("注意：未检测到中文字体，图表可能显示异常。")

    sns.set_style("whitegrid")


# =============================================================================
# 数据加载
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """按后缀加载 CSV / Excel / Parquet。"""
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath, encoding="utf-8-sig")
    elif filepath.endswith((".xlsx", ".xls")):
        return pd.read_excel(filepath)
    elif filepath.endswith(".parquet"):
        return pd.read_parquet(filepath)
    else:
        raise ValueError(f"不支持的文件格式: {filepath}")


# =============================================================================
# 数据规模预检
# =============================================================================

def data_scale_check(filepath: str) -> tuple[int, int, float]:
    """检查数据规模，输出性能预警。"""
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"文件大小: {file_size_mb:.0f} MB")

    if filepath.endswith(".csv"):
        df_head = pd.read_csv(filepath, nrows=5, encoding="utf-8-sig")
        with open(filepath, "r", encoding="utf-8-sig") as f:
            n_rows = sum(1 for _ in f) - 1
    else:
        data = load_data(filepath)
        df_head = data.head()
        n_rows = len(data)

    n_cols = len(df_head.columns)
    n_features = max(0, n_cols - 4)
    print(f"列数: {n_cols}")
    print(f"行数: {n_rows}")
    if n_rows > 200_000 or n_features > 200:
        print("⚠️ 预警: 数据规模较大，请评估运行时间与内存。")
        print("  模型类型应依据业务可解释性、有效样本量和 bad 样本数量确认。")
        print("  预筛选、分批处理或采样加速应根据实际性能测试决定。")

    return n_rows, n_cols, file_size_mb


# =============================================================================
# 缺失值探测与统一
# =============================================================================

def detect_missing_markers(data: pd.DataFrame) -> None:
    """探测 NaN 数量和常见特殊值（-9999, -999, -99999）的分布。"""
    print("=== NaN 数量（前 20 列）===")
    nan_counts = data.isna().sum()
    print(nan_counts[nan_counts > 0].head(20))

    print("\n=== 特殊值检查 ===")
    special_markers = [-9999, -999, -99999]
    for col in data.select_dtypes(include=[np.number]).columns[:30]:
        for marker in special_markers:
            count = (data[col] == marker).sum()
            if count > 0:
                print(f"[WARN] {col}: {marker} 出现 {count} 次")


def unify_missing(df: pd.DataFrame, markers: list[Any]) -> pd.DataFrame:
    """将特殊缺失标识统一转换为 NaN。"""
    df = df.copy()
    for col in df.columns:
        for marker in markers:
            df.loc[df[col] == marker, col] = np.nan
    return df


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="阶段 0：环境初始化与数据预检"
    )
    parser.add_argument("--filepath", help="数据文件路径（csv/xlsx/parquet）")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--config-template", action="store_true", help="生成待确认配置模板")
    parser.add_argument("--check-packages", action="store_true", help="检查依赖包")
    parser.add_argument("--missing-markers", nargs="*", type=float,
                        default=[-9999, -999], help="特殊缺失值标识，默认 -9999 -999")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "00_modeling_config.yaml"
    if args.config_template or not config_path.exists():
        write_config(config_path, {
            "project": {"name": "pending", "created_at": "pending"},
            "paths": {"raw_data_path": args.filepath or "pending", "code_dir": "pending",
                      "intermediate_dir": str(output.resolve()), "report_dir": "pending"},
            "fields": {"target_col": "pending", "time_col": "pending", "sample_id_col": None,
                       "sample_id_source_cols": [], "sample_weight_col": "sample_weight"},
            "model_type": "pending", "enable_stage6_scoring": "pending",
            "psi_period_granularity": "pending", "sample_split": {"oot_method": "pending"},
            "stage3": {"iv_threshold": "pending", "corr_threshold": 0.7},
            "scoring": {"base_score": "pending", "base_odds": "pending", "pdo": "pending"},
        })
        if not args.filepath:
            print(f"配置模板已生成: {config_path}")
            return
    if not args.filepath:
        raise ValueError("执行数据预检时必须提供 --filepath")

    if args.check_packages:
        ok = check_packages()
        if not ok:
            sys.exit(1)

    setup_chinese_font()

    n_rows, n_cols, file_size_mb = data_scale_check(args.filepath)

    df = load_data(args.filepath)
    detect_missing_markers(df)

    if args.missing_markers:
        df = unify_missing(df, args.missing_markers)
        print(f"\n已将 {args.missing_markers} 统一转换为 NaN")
    environment = pd.DataFrame([
        {"package": pkg, "required_version": version,
         "installed": importlib.util.find_spec(pkg) is not None}
        for pkg, version in REQUIRED_PACKAGES.items()
    ])
    environment.to_csv(output / "00_environment_check.csv", index=False, encoding="utf-8-sig")
    profile = pd.DataFrame([{
        "raw_data_path": str(Path(args.filepath).resolve()), "rows": n_rows,
        "columns": n_cols, "file_size_mb": file_size_mb, "column": column,
        "dtype": str(df[column].dtype), "non_missing_count": int(df[column].notna().sum()),
        "missing_count": int(df[column].isna().sum()), "unique": int(df[column].nunique(dropna=True)),
    } for column in df.columns])
    profile.to_csv(output / "00_data_profile.csv", index=False, encoding="utf-8-sig")
    config = load_config(config_path)
    checks = {
        "paths.raw_data_path": config.get("paths", {}).get("raw_data_path"),
        "paths.code_dir": config.get("paths", {}).get("code_dir"),
        "paths.intermediate_dir": config.get("paths", {}).get("intermediate_dir"),
        "paths.report_dir": config.get("paths", {}).get("report_dir"),
        "fields.target_col": config.get("fields", {}).get("target_col"),
        "fields.time_col": config.get("fields", {}).get("time_col"),
        "model_type": config.get("model_type"),
        "enable_stage6_scoring": config.get("enable_stage6_scoring"),
        "psi_period_granularity": config.get("psi_period_granularity"),
        "sample_split.oot_method": config.get("sample_split", {}).get("oot_method"),
    }
    required_pending = [{"config_item": key, "decision": "pending"} for key, value in checks.items()
                        if value in {None, "", "pending"}]
    outputs = {"config": config_path, "environment_check": output / "00_environment_check.csv",
               "data_profile": output / "00_data_profile.csv"}
    status = "pending" if required_pending else "completed"
    summary = pd.DataFrame([{"status": status, "raw_data_path": args.filepath}])
    write_output_list(output / "00-output-list.xlsx", summary, list(outputs.values()), pd.DataFrame(required_pending))
    outputs["00_output_list"] = output / "00-output-list.xlsx"
    write_stage_manifest(output, 0, status, config_path, {"raw_data": args.filepath}, outputs, required_pending, 1)


if __name__ == "__main__":
    main()
