#!/usr/bin/env python3
"""阶段 7：汇总阶段 0-6 产物并生成最终报告文件清单和 HTML 报告。"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

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
    "06-output-list.xlsx",
    "06_scoring_decisions.csv",
    "06_scoring_parameters.json",
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
    return pd.DataFrame(rows)


def find_report_issues(inventory: pd.DataFrame) -> pd.DataFrame:
    """检查最终报告必需产物是否存在。"""
    available = set(inventory["artifact"]) if not inventory.empty else set()
    rows = []
    for index, artifact in enumerate(REQUIRED_PATTERNS, start=1):
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
    return pd.DataFrame(rows)


def render_html(
    inventory: pd.DataFrame, issues: pd.DataFrame, artifact_dir: Path, output_path: Path,
) -> None:
    """生成不重新计算指标的最终报告与交付清单。"""
    body = ["<h1>A卡建模报告</h1>"]
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
    issues = find_report_issues(inventory)
    inventory.to_csv(report_dir / "07_artifact_inventory.csv", index=False, encoding="utf-8-sig")
    issues.to_csv(report_dir / "07_report_issues.csv", index=False, encoding="utf-8-sig")
    render_html(inventory, issues, artifact_dir, report_dir / "07_A卡建模报告.html")
    with pd.ExcelWriter(report_dir / "07-output-list.xlsx") as writer:
        inventory.to_excel(writer, sheet_name="artifact_inventory", index=False)
        issues.to_excel(writer, sheet_name="report_issues", index=False)

    if not issues.empty:
        raise SystemExit("最终报告存在必需输入缺失，请确认 07_report_issues.csv 后再交付。")

    print(f"最终报告已生成至: {report_dir}")


if __name__ == "__main__":
    main()
