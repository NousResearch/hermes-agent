"""Dream weekly report — deterministic REPORT.md/run.json writer.

Dream (the weekly memory-hygiene/learning/self-observation cron job) is a
plain prompt-driven cron job, not a bespoke Python orchestrator like
``agent/curator.py``. This tool is the reliability lever for its report: the
LLM supplies *structured* fields describing what it did, and this module
owns rendering — formatting is guaranteed consistent even though the
*content* (what actually changed) still comes from the model's own turn.

Mirrors ``agent/curator.py``'s report-directory shape
(``{stamp}/REPORT.md`` + ``run.json`` under ``logs/<name>/``) without
sharing its diffing code, which is tightly coupled to skill-archival
semantics and not a fit here.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


DREAM_REPORT_SCHEMA = {
    "name": "dream_report",
    "description": (
        "Write this week's Dream report (REPORT.md + run.json under "
        "~/.hermes/logs/dream/). Call this exactly once, as the LAST tool "
        "call of a Dream run, after hygiene, learning, the Langfuse review, "
        "and any skill fixes are done. All fields are optional — omit or "
        "pass an empty list for anything that didn't happen this run (e.g. "
        "no skill fixes most weeks is expected and correct)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_usage_before": {
                "type": "string",
                "description": (
                    "MEMORY.md usage before this run, transcribed from the "
                    "'MEMORY (your personal notes)' header in your system "
                    "prompt, e.g. '62% — 1,364/2,200 chars'."
                ),
            },
            "memory_usage_after": {
                "type": "string",
                "description": (
                    "MEMORY.md usage after hygiene/learning, transcribed "
                    "from the memory() tool's own 'usage' response field."
                ),
            },
            "user_usage_before": {
                "type": "string",
                "description": "USER.md usage before this run, same source as memory_usage_before.",
            },
            "user_usage_after": {
                "type": "string",
                "description": "USER.md usage after this run, same source as memory_usage_after.",
            },
            "memory_changes": {
                "type": "array",
                "description": "Changes made to MEMORY.md this run.",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": ["add", "remove", "merge"]},
                        "text": {"type": "string"},
                        "merged_from": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["op", "text"],
                },
            },
            "user_changes": {
                "type": "array",
                "description": "Changes made to USER.md this run. Same shape as memory_changes.",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": ["add", "remove", "merge"]},
                        "text": {"type": "string"},
                        "merged_from": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["op", "text"],
                },
            },
            "skill_changes": {
                "type": "array",
                "description": (
                    "Targeted skill fixes made this run via skill_manage. "
                    "Evidence-gated — most weeks this is empty."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "action": {"type": "string", "enum": ["patch", "create", "edit"]},
                        "reason": {"type": "string"},
                        "evidence": {"type": "string"},
                    },
                    "required": ["name", "action", "reason"],
                },
            },
            "langfuse_findings": {
                "type": "array",
                "description": "Anomalies found via langfuse_query this run, kept for the record even if not acted on.",
                "items": {
                    "type": "object",
                    "properties": {
                        "trace_id": {"type": "string"},
                        "timestamp": {"type": "string"},
                        "symptom": {"type": "string"},
                    },
                    "required": ["trace_id", "symptom"],
                },
            },
            "flagged_for_code_fix": {
                "type": "array",
                "description": (
                    "Issues found that need an actual code change, not a memory/skill edit. "
                    "Not acted on automatically — this is a record for a human (or a future "
                    "automated pass) to pick up."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "issue": {"type": "string"},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                        "suspected_area": {"type": "string"},
                    },
                    "required": ["issue"],
                },
            },
            "summary": {
                "type": "string",
                "description": (
                    "3-5 plain-text lines summarizing the run. If Dream's delivery is "
                    "not 'local', this is sent to the user verbatim as the delivered message "
                    "— write it for a human reading a notification, not a log."
                ),
            },
        },
    },
}


def _dream_reports_root() -> Path:
    """Directory where Dream run reports are written.

    Lives under ``~/.hermes/logs/dream/`` alongside ``agent.log`` and the
    sibling ``logs/curator/`` reports dir — operational telemetry, not
    mixed into ``~/.hermes/skills/`` or the memory stores it's reporting on.
    """
    root = get_hermes_home() / "logs" / "dream"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.debug("Dream reports dir create failed: %s", e)
    return root


def _bullet_changes(lines: List[str], heading: str, changes: List[Dict[str, Any]], empty_text: str) -> None:
    lines.append(f"## {heading}\n")
    if not changes:
        lines.append(f"_{empty_text}_\n")
        return
    for c in changes:
        if not isinstance(c, dict):
            continue
        op = c.get("op") or c.get("action") or "?"
        text = c.get("text") or c.get("reason") or ""
        line = f"- **{op}**: {text}"
        merged_from = c.get("merged_from")
        if merged_from:
            line += f" _(merged from: {', '.join(merged_from)})_"
        evidence = c.get("evidence")
        if evidence:
            line += f" — evidence: {evidence}"
        name = c.get("name")
        if name:
            line = f"- **{op}** `{name}`: {text}" + (f" — evidence: {evidence}" if evidence else "")
        lines.append(line)
    lines.append("")


def _render_dream_report_markdown(payload: Dict[str, Any]) -> str:
    """Render the human-readable Dream report."""
    lines: List[str] = []
    generated_at = payload.get("generated_at", "")
    lines.append(f"# Dream run — {generated_at}\n")

    lines.append("## Memory\n")
    lines.append(f"- MEMORY.md: {payload.get('memory_usage_before') or '(not recorded)'} → "
                 f"{payload.get('memory_usage_after') or '(not recorded)'}")
    lines.append(f"- USER.md: {payload.get('user_usage_before') or '(not recorded)'} → "
                 f"{payload.get('user_usage_after') or '(not recorded)'}")
    lines.append("")
    _bullet_changes(lines, "Memory changes", payload.get("memory_changes") or [], "No memory changes this run.")
    _bullet_changes(lines, "User-profile changes", payload.get("user_changes") or [], "No user-profile changes this run.")

    lines.append("## Skills\n")
    skill_changes = payload.get("skill_changes") or []
    if not skill_changes:
        lines.append("_No skill changes this run._\n")
    else:
        for c in skill_changes:
            if not isinstance(c, dict):
                continue
            name = c.get("name", "?")
            action = c.get("action", "?")
            reason = c.get("reason", "")
            evidence = c.get("evidence", "")
            line = f"- **{action}** `{name}`: {reason}"
            if evidence:
                line += f" — evidence: {evidence}"
            lines.append(line)
        lines.append("")

    lines.append("## Langfuse findings\n")
    findings = payload.get("langfuse_findings") or []
    if not findings:
        lines.append("_No anomalies found in the past week._\n")
    else:
        for f in findings:
            if not isinstance(f, dict):
                continue
            trace_id = f.get("trace_id", "?")
            ts = f.get("timestamp", "")
            symptom = f.get("symptom", "")
            lines.append(f"- `{trace_id}` ({ts}): {symptom}")
        lines.append("")

    # Stable header — a future automated pass over these reports would grep
    # for this exact section, so don't rename it casually.
    lines.append("## Flagged for code-level fix\n")
    flagged = payload.get("flagged_for_code_fix") or []
    if not flagged:
        lines.append("_None this run._\n")
    else:
        for item in flagged:
            if not isinstance(item, dict):
                continue
            issue = item.get("issue", "?")
            area = item.get("suspected_area", "")
            evidence = item.get("evidence") or []
            lines.append(f"- {issue}" + (f" _(suspected: {area})_" if area else ""))
            for e in evidence:
                lines.append(f"  - evidence: {e}")
        lines.append("")

    lines.append("## Summary\n")
    lines.append(payload.get("summary") or "_(no summary provided)_")
    lines.append("")

    return "\n".join(lines) + "\n"


def dream_report(
    memory_usage_before: Optional[str] = None,
    memory_usage_after: Optional[str] = None,
    user_usage_before: Optional[str] = None,
    user_usage_after: Optional[str] = None,
    memory_changes: Optional[List[Dict[str, Any]]] = None,
    user_changes: Optional[List[Dict[str, Any]]] = None,
    skill_changes: Optional[List[Dict[str, Any]]] = None,
    langfuse_findings: Optional[List[Dict[str, Any]]] = None,
    flagged_for_code_fix: Optional[List[Dict[str, Any]]] = None,
    summary: Optional[str] = None,
) -> str:
    """Write this week's Dream report. See DREAM_REPORT_SCHEMA for field docs."""
    root = _dream_reports_root()
    started_at = datetime.now(timezone.utc)
    stamp = started_at.strftime("%Y%m%d-%H%M%S")
    run_dir = root / stamp
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = root / f"{stamp}-{suffix}"
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as e:
        logger.warning("Dream report dir create failed: %s", e)
        return json.dumps({"success": False, "error": f"could not create report dir: {e}"}, ensure_ascii=False)

    payload = {
        "generated_at": started_at.isoformat(),
        "memory_usage_before": memory_usage_before,
        "memory_usage_after": memory_usage_after,
        "user_usage_before": user_usage_before,
        "user_usage_after": user_usage_after,
        "memory_changes": memory_changes or [],
        "user_changes": user_changes or [],
        "skill_changes": skill_changes or [],
        "langfuse_findings": langfuse_findings or [],
        "flagged_for_code_fix": flagged_for_code_fix or [],
        "summary": summary or "",
    }

    try:
        (run_dir / "run.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except OSError as e:
        logger.warning("Dream run.json write failed: %s", e)

    try:
        md = _render_dream_report_markdown(payload)
        (run_dir / "REPORT.md").write_text(md, encoding="utf-8")
    except OSError as e:
        logger.warning("Dream REPORT.md write failed: %s", e)

    return json.dumps(
        {
            "success": True,
            "report_dir": str(run_dir),
            "message": "Report written. This is the last step of a Dream run — do not call any more tools.",
        },
        ensure_ascii=False,
    )


def check_dream_report_requirements() -> bool:
    """Dream report tool has no external requirements -- always available."""
    return True


# --- Registry ---
from tools.registry import registry

registry.register(
    name="dream_report",
    toolset="dream",
    schema=DREAM_REPORT_SCHEMA,
    handler=lambda args, **kw: dream_report(
        memory_usage_before=args.get("memory_usage_before"),
        memory_usage_after=args.get("memory_usage_after"),
        user_usage_before=args.get("user_usage_before"),
        user_usage_after=args.get("user_usage_after"),
        memory_changes=args.get("memory_changes"),
        user_changes=args.get("user_changes"),
        skill_changes=args.get("skill_changes"),
        langfuse_findings=args.get("langfuse_findings"),
        flagged_for_code_fix=args.get("flagged_for_code_fix"),
        summary=args.get("summary"),
    ),
    check_fn=check_dream_report_requirements,
    emoji="🌙",
)
