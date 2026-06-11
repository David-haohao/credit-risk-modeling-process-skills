"""Shared configuration, manifest, and output-list contracts for stages 0-7."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any, Optional

import pandas as pd


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="minutes")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8-sig")
    try:
        import yaml
    except ImportError:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("读取 YAML 配置需要安装 PyYAML") from exc
    result = yaml.safe_load(text)
    if not isinstance(result, dict):
        raise ValueError("00_modeling_config.yaml 顶层必须为对象")
    return result


def write_config(path: str | Path, config: dict[str, Any]) -> Path:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config["config_version"] = int(config.get("config_version", 0)) + 1
    config["updated_at"] = now_iso()
    try:
        import yaml
    except ImportError:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8",
        )
    return config_path


def require_formal_entry(
    previous_manifest: Optional[str | Path], compatibility_mode: bool, stage: int,
) -> None:
    if stage > 0 and not previous_manifest and not compatibility_mode:
        raise ValueError(f"阶段 {stage} 正式运行必须提供 --previous-manifest")


def load_previous_manifest(path: str | Path, expected_stage: Optional[int] = None) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if expected_stage is not None and manifest.get("stage") != expected_stage:
        raise ValueError(f"上一阶段 manifest 应为阶段 {expected_stage}")
    if manifest.get("status") != "completed":
        raise ValueError("上一阶段尚未完成")
    if manifest.get("pending_decisions"):
        raise ValueError("上一阶段仍存在 pending 决策")
    return manifest


def resolve_manifest_artifact(manifest: dict[str, Any], key: str) -> Path:
    value = manifest.get("outputs", {}).get(key)
    if not value:
        raise ValueError(f"上一阶段 manifest 缺少输出: {key}")
    return Path(value)


def snapshot_config(config_path: str | Path, output_dir: str | Path, stage: int) -> Path:
    source = Path(config_path)
    target = Path(output_dir) / f"{stage:02d}_modeling_config_snapshot.yaml"
    shutil.copyfile(source, target)
    return target.resolve()


def write_stage_manifest(
    output_dir: str | Path,
    stage: int,
    status: str,
    config_path: str | Path,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    pending_decisions: list[Any],
    next_stage: Optional[int],
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_config(config_path, output, stage)
    declared_outputs = {}
    previous_manifest = inputs.get("previous_manifest")
    if previous_manifest and Path(str(previous_manifest)).is_file():
        previous = json.loads(Path(str(previous_manifest)).read_text(encoding="utf-8"))
        declared_outputs.update(previous.get("outputs", {}))
    declared_outputs.update({k: str(Path(v).resolve()) for k, v in outputs.items()})
    manifest = {
        "stage": stage,
        "status": status,
        "created_at": now_iso(),
        "config_snapshot": str(snapshot),
        "inputs": {k: str(v) for k, v in inputs.items()},
        "outputs": declared_outputs,
        "pending_decisions": pending_decisions,
        "next_stage": next_stage,
    }
    path = output / f"{stage:02d}_stage_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_output_list(
    path: str | Path,
    summary: pd.DataFrame,
    artifacts: list[str | Path],
    pending_decisions: Optional[pd.DataFrame] = None,
    extra_sheets: Optional[dict[str, pd.DataFrame]] = None,
) -> None:
    inventory = pd.DataFrame([{
        "file": Path(item).name,
        "path": str(Path(item).resolve()),
        "exists": Path(item).exists(),
    } for item in artifacts])
    pending = pending_decisions if pending_decisions is not None else pd.DataFrame()
    with pd.ExcelWriter(path) as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        inventory.to_excel(writer, sheet_name="file_inventory", index=False)
        pending.to_excel(writer, sheet_name="pending_decisions", index=False)
        for name, frame in (extra_sheets or {}).items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
