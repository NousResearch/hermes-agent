"""Tool Output Verbosity conventions and formatters.

Supports MINIMAL, SUMMARY (default for durable agent workflows), and FULL.
Automatically offloads full payloads to the ArtifactStore when verbosity is summary.
"""

from __future__ import annotations

from enum import Enum
import json
import logging
from typing import Any, Dict, Optional, Union

from workstation.artifacts import ArtifactRef, ArtifactStore

logger = logging.getLogger(__name__)


class VerbosityLevel(str, Enum):
    MINIMAL = "minimal"
    SUMMARY = "summary"
    FULL = "full"


def normalize_verbosity(val: Optional[Union[str, VerbosityLevel]]) -> VerbosityLevel:
    if isinstance(val, VerbosityLevel):
        return val
    if not val:
        return VerbosityLevel.SUMMARY
    lowered = str(val).strip().lower()
    if lowered == "minimal":
        return VerbosityLevel.MINIMAL
    if lowered == "full":
        return VerbosityLevel.FULL
    return VerbosityLevel.SUMMARY


def format_tool_output(
    data: Any,
    *,
    verbosity: Union[str, VerbosityLevel] = VerbosityLevel.SUMMARY,
    task_id: Optional[str] = None,
    name: str = "tool_output",
    artifact_store: Optional[ArtifactStore] = None,
    signals: Optional[Dict[str, Any]] = None,
    max_summary_chars: int = 500,
) -> Union[str, Dict[str, Any]]:
    """Format tool output according to requested verbosity level."""
    level = normalize_verbosity(verbosity)

    if level == VerbosityLevel.FULL:
        if isinstance(data, (dict, list)):
            return json.dumps(data, ensure_ascii=False, indent=2)
        return str(data)

    # For MINIMAL
    if level == VerbosityLevel.MINIMAL:
        if isinstance(data, dict):
            status = data.get("status") or data.get("success") or "ok"
            return json.dumps({"status": status, "keys": list(data.keys())[:5]}, ensure_ascii=False)
        if isinstance(data, list):
            return json.dumps({"status": "ok", "count": len(data)}, ensure_ascii=False)
        return json.dumps({"status": "ok", "length": len(str(data))}, ensure_ascii=False)

    # For SUMMARY (default): offload large payloads to ArtifactStore if task_id provided
    raw_text = json.dumps(data, ensure_ascii=False) if isinstance(data, (dict, list)) else str(data)
    length = len(raw_text)

    artifact_ref_str = None
    if task_id and length > max_summary_chars:
        store = artifact_store or ArtifactStore()
        ref = store.store(task_id=task_id, name=f"{name}.json", content=data)
        artifact_ref_str = ref.ref

    summary_payload: Dict[str, Any] = {
        "status": "ready" if not isinstance(data, dict) else data.get("status", "ready"),
        "text_length": length,
    }

    if isinstance(data, dict):
        # Extract small preview metrics
        summary_payload["metrics"] = {k: v for k, v in data.items() if isinstance(v, (int, float, bool)) and len(k) < 30}
    elif isinstance(data, list):
        summary_payload["item_count"] = len(data)

    if signals:
        summary_payload["signals"] = signals

    if artifact_ref_str:
        summary_payload["artifact_ref"] = artifact_ref_str
        summary_payload["hint"] = "Full content saved to artifact. Pass verbosity='full' or read artifact_ref if inspection needed."
    elif length <= max_summary_chars:
        summary_payload["preview"] = data

    return json.dumps(summary_payload, ensure_ascii=False)
