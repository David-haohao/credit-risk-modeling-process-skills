#!/usr/bin/env python3
"""阶段 7：汇总阶段 0-6 产物并生成最终报告文件清单和 HTML 报告。"""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Optional

import pandas as pd


REQUIRED_PATTERNS = [
    "00_modeling_config.yaml",
    "00-output-list.xlsx",
    "01-output-list.xlsx",
    "02-output-list.xlsx",
    "03-output-list.xlsx",
    "04-output-list.xlsx",
    "05-output-list.xlsx",
    "05_evaluation_decisions.csv",
]

STAGE6_REQUIRED_PATTERNS = [
    "06-output-list.xlsx",
    "06_scoring_decisions.csv",
    "06_scoring_parameters.json",
]

DECISION_FILES = {
    "05": "05_evaluation_decisions.csv",
    "06": "06_scoring_decisions.csv",
}

INVENTORY_COLUMNS = [
    "artifact",
    "relative_path",
    "suffix",
    "size_bytes",
    "included_in_report",
]


def build_inventory(artifact_dir: Path) -> pd.DataFrame:
    """递归登记阶段 0-6 产物。"""
    rows = []
    for path in sorted(artifact_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("07_"):
            continue
        rows.append({
            "artifact": path.name,
            "relative_path": path.relative_to(artifact_dir).as_posix(),
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "included_in_report": path.suffix.lower() in {".png", ".html"},
        })
    return pd.DataFrame(rows, columns=INVENTORY_COLUMNS)


def read_stage6_scope(config_path: Path) -> Optional[bool]:
    """读取阶段 0 配置中的 enable_stage6_scoring。"""
    if not config_path.exists():
        return None
    for line in config_path.read_text(encoding="utf-8-sig").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == "enable_stage6_scoring":
            normalized = value.split("#", 1)[0].strip().strip("'\"").lower()
            if normalized in {"true", "yes", "1"}:
                return True
            if normalized in {"false", "no", "0"}:
                return False
    return None


def read_confirmed_decisions(
    artifact_dir: Path, stage6_enabled: Optional[bool],
) -> tuple[list[tuple[str, pd.DataFrame]], list[dict]]:
    """读取上游已确认决策，不从模型指标推断结论。"""
    confirmed = []
    issues = []
    stages = ["05"] + (["06"] if stage6_enabled is True else [])
    for stage in stages:
        filename = DECISION_FILES[stage]
        paths = list(artifact_dir.rglob(filename))
        if not paths:
            continue
        if len(paths) > 1:
            issues.append({
                "issue_id": f"RDEC{stage}",
                "issue_type": "inconsistent_value",
                "source_stage": stage,
                "artifact": filename,
                "description": "发现多个同名决策表，无法确定最终版本",
                "decision": "pending",
                "reason": "",
                "confirmed_by": "",
                "confirmed_at": "",
            })
            continue
        path = paths[0]
        try:
            decisions = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as exc:
            issues.append({
                "issue_id": f"RDEC{stage}",
                "issue_type": "inconsistent_value",
                "source_stage": stage,
                "artifact": filename,
                "description": f"决策表无法读取: {exc}",
                "decision": "pending",
                "reason": "",
                "confirmed_by": "",
                "confirmed_at": "",
            })
            continue
        if "decision" not in decisions.columns:
            issues.append({
                "issue_id": f"RDEC{stage}",
                "issue_type": "inconsistent_value",
                "source_stage": stage,
                "artifact": filename,
                "description": "决策表缺少 decision 字段",
                "decision": "pending",
                "reason": "",
                "confirmed_by": "",
                "confirmed_at": "",
            })
            continue
        decision_values = decisions["decision"].fillna("pending").astype(str).str.strip().str.lower()
        if decision_values.eq("pending").any():
            issues.append({
                "issue_id": f"RPENDING{stage}",
                "issue_type": "unconfirmed_decision",
                "source_stage": stage,
                "artifact": filename,
                "description": "决策表仍存在 pending",
                "decision": "pending",
                "reason": "",
                "confirmed_by": "",
                "confirmed_at": "",
            })
        confirmed_rows = decisions.loc[~decision_values.eq("pending")].copy()
        if confirmed_rows.empty:
            issues.append({
                "issue_id": f"RCONFIRMED{stage}",
                "issue_type": "unconfirmed_decision",
                "source_stage": stage,
                "artifact": filename,
                "description": "决策表没有已确认记录",
                "decision": "pending",
                "reason": "",
                "confirmed_by": "",
                "confirmed_at": "",
            })
        else:
            confirmed.append((stage, confirmed_rows))
    return confirmed, issues


def find_report_issues(
    inventory: pd.DataFrame, stage6_enabled: Optional[bool], decision_issues: list[dict],
) -> pd.DataFrame:
    """检查最终报告必需产物是否存在。"""
    available = set(inventory["artifact"]) if not inventory.empty else set()
    rows = []
    required = REQUIRED_PATTERNS + (STAGE6_REQUIRED_PATTERNS if stage6_enabled is True else [])
    for index, artifact in enumerate(required, start=1):
        if artifact not in available:
            rows.append({
                "issue_id": f"R{index:03d}",
                "issue_type": "missing_required_input",
                "source_stage": artifact[:2],
                "artifact": artifact,
                "description": "最终报告必需输入缺失",
                "decision": "pending",
                "reason": "",
                "confirmed_by": "",
                "confirmed_at": "",
            })
    if stage6_enabled is None:
        rows.append({
            "issue_id": "RSCOPE",
            "issue_type": "unconfirmed_decision",
            "source_stage": "00",
            "artifact": "00_modeling_config.yaml",
            "description": "enable_stage6_scoring 缺失或无法识别",
            "decision": "pending",
            "reason": "",
            "confirmed_by": "",
            "confirmed_at": "",
        })
    rows.extend(decision_issues)
    return pd.DataFrame(rows)


def render_html(
    inventory: pd.DataFrame, issues: pd.DataFrame, artifact_dir: Path,
    stage6_enabled: Optional[bool], confirmed_decisions: list[tuple[str, pd.DataFrame]],
    output_path: Path,
) -> None:
    """生成不重新计算指标的最终报告与交付清单。"""
    body = ["<h1>A卡建模报告</h1>"]
    body.append("<h2>最终结论与建议</h2>")
    body.append("<p>本节仅汇总上游已确认决策，不根据模型指标推断新结论。</p>")
    if confirmed_decisions:
        for stage, decisions in confirmed_decisions:
            body.append(f"<h3>阶段 {stage} 已确认决策</h3>")
            body.append(decisions.to_html(index=False, escape=True))
    else:
        body.append("<p>暂无可展示的已确认决策，最终交付需暂停。</p>")
    if stage6_enabled is False:
        body.append("<p>本项目已在阶段0确认不执行阶段6概率校准与评分转换。</p>")
    for stage in range(7):
        stage_files = inventory[inventory["relative_path"].str.match(fr"(^|.*/)0{stage}[_-]")]
        body.append(f"<h2>阶段 {stage}</h2>")
        if stage_files.empty:
            body.append("<p>未发现该阶段可展示产物。</p>")
            continue
        body.append(stage_files.to_html(index=False, escape=True))
        for relative_path in stage_files.loc[
            stage_files["suffix"].eq(".png"), "relative_path"
        ]:
            image_uri = (artifact_dir / relative_path).resolve().as_uri()
            body.append(
                f"<p><img src='{html.escape(image_uri)}' "
                "style='max-width:100%;height:auto'></p>"
            )
    body.append("<h2>交付文件清单</h2>")
    body.append(inventory.to_html(index=False, escape=True))
    body.append("<h2>报告问题</h2>")
    body.append("<p>无阻断问题。</p>" if issues.empty else issues.to_html(index=False, escape=True))
    document = "<!doctype html><html><head><meta charset='utf-8'><title>A卡建模报告</title></head><body>"
    document += "".join(body) + "</body></html>"
    output_path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段 7：最终报告汇总与交付")
    parser.add_argument("--artifact-dir", required=True, help="阶段 0-6 过程文件目录")
    parser.add_argument("--report-dir", required=True, help="阶段 0 确认的最终报告目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_inventory(artifact_dir)
    stage6_enabled = read_stage6_scope(artifact_dir / "00_modeling_config.yaml")
    confirmed_decisions, decision_issues = read_confirmed_decisions(artifact_dir, stage6_enabled)
    issues = find_report_issues(inventory, stage6_enabled, decision_issues)
    inventory.to_csv(report_dir / "07_artifact_inventory.csv", index=False, encoding="utf-8-sig")
    issues.to_csv(report_dir / "07_report_issues.csv", index=False, encoding="utf-8-sig")
    render_html(
        inventory, issues, artifact_dir, stage6_enabled, confirmed_decisions,
        report_dir / "07_A卡建模报告.html",
    )
    with pd.ExcelWriter(report_dir / "07-output-list.xlsx") as writer:
        inventory.to_excel(writer, sheet_name="artifact_inventory", index=False)
        issues.to_excel(writer, sheet_name="report_issues", index=False)

    if not issues.empty:
        raise SystemExit("最终报告存在必需输入缺失，请确认 07_report_issues.csv 后再交付。")

    print(f"最终报告已生成至: {report_dir}")


if __name__ == "__main__":
    main()
