#!/usr/bin/env python3
"""
Read-only audit for recursive self-improvement across knowledge + memory.

Complements the self-improving skill: surfaces tier health, knowledge↔HOT
reconciliation hints, and a session loop checklist. Does **not** write
.learnings/ or artifacts (Town-Hermes feedback protocol FROZEN).

CLI:
    python -m tools.self_improvement_audit
    python -m tools.self_improvement_audit --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

HOT_LINE_CAP = 100
WARM_LINE_SOFT_CAP = 200

# Operational anchors often appear in knowledge layer but must stay in HOT memory.
_RECONCILE_ANCHORS = (
    "architecture freeze",
    "h20d",
    "town-hermes feedback",
    "feedback protocol",
    "spec 100",
    "spec 095",
    "spec 087",
    "spec 088",
    "ruleset",
    "8887576e",
    "deepseek",
    "13f",
    "q1 2026",
)

_RECURSIVE_LOOP = [
    "Session start: fleet_context_snapshot(summary=True) or self_improvement_snapshot().",
    "Load HOT: learnings_read(file='memory.md').",
    "Load knowledge: knowledge_read(artifact='latest_state'), held_spec_ledger, contradiction_ledger.",
    "During work: log corrections to corrections.md (append-only, operator or agent).",
    "After significant work: self-reflection block (CONTEXT / REFLECTION / LESSON).",
    "Before editing memory.md: run self_improvement_snapshot(); apply only operator-approved diffs.",
    "Reconcile: knowledge layer = operational truth; HOT memory = bootstrap summary (<=100 lines).",
    "Never auto-sync Town↔Hermes memory until feedback protocol unfreezes.",
]


@dataclass
class TierReport:
    tier: str
    path: str
    present: bool
    line_count: int = 0
    over_cap: bool = False
    modified: str = ""


@dataclass
class ReconcileHint:
    anchor: str
    in_knowledge: bool
    in_hot_memory: bool
    action: str


def _file_mtime_iso(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return ""


def _read_text(path: Optional[Path]) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _line_count(text: str) -> int:
    if not text:
        return 0
    return len(text.splitlines())


def _parse_last_updated(text: str) -> Optional[str]:
    for line in text.splitlines()[:20]:
        m = re.search(r"last\s+updated:\s*(\d{4}-\d{2}-\d{2})", line, re.I)
        if m:
            return m.group(1)
    return None


def audit_hot_memory(path: Path) -> TierReport:
    text = _read_text(path)
    lines = _line_count(text)
    return TierReport(
        tier="HOT",
        path=str(path),
        present=path.is_file(),
        line_count=lines,
        over_cap=lines > HOT_LINE_CAP,
        modified=_file_mtime_iso(path) if path.is_file() else "",
    )


def audit_warm_tiers(learnings_dir: Path) -> List[TierReport]:
    reports: List[TierReport] = []
    for sub in ("projects", "domains"):
        base = learnings_dir / sub
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*.md")):
            if f.name.startswith("."):
                continue
            lines = _line_count(_read_text(f))
            reports.append(
                TierReport(
                    tier="WARM",
                    path=str(f.relative_to(learnings_dir)),
                    present=True,
                    line_count=lines,
                    over_cap=lines > WARM_LINE_SOFT_CAP,
                    modified=_file_mtime_iso(f),
                )
            )
    return reports


def audit_corrections_log(path: Path) -> Dict[str, Any]:
    text = _read_text(path)
    if not path.is_file():
        return {"present": False, "path": str(path), "entries": 0}
    # Rough entry count: lines starting with ## or CONTEXT:
    entries = sum(
        1
        for line in text.splitlines()
        if line.startswith("## ") or line.startswith("CONTEXT:")
    )
    return {
        "present": True,
        "path": str(path),
        "line_count": _line_count(text),
        "entries_approx": entries,
        "modified": _file_mtime_iso(path),
    }


def reconcile_knowledge_memory(
    hot_text: str,
    knowledge_texts: Sequence[str],
) -> List[ReconcileHint]:
    combined_k = "\n".join(knowledge_texts).lower()
    hot_l = hot_text.lower()
    hints: List[ReconcileHint] = []
    for anchor in _RECONCILE_ANCHORS:
        in_k = anchor in combined_k
        in_h = anchor in hot_l
        if in_k and not in_h:
            action = "Consider adding to HOT memory.md (operator-approved)."
        elif in_h and not in_k:
            action = "HOT-only context; verify knowledge layer is stale."
        elif in_k and in_h:
            action = "Aligned."
        else:
            continue
        hints.append(
            ReconcileHint(
                anchor=anchor,
                in_knowledge=in_k,
                in_hot_memory=in_h,
                action=action,
            )
        )
    return hints


def build_self_improvement_audit(
    *,
    learnings_dir: Optional[Path] = None,
    latest_state_path: Optional[Path] = None,
    held_spec_path: Optional[Path] = None,
    contradiction_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build a read-only self-improvement audit dict."""
    as_of = datetime.now(timezone.utc).isoformat()
    learnings_dir = learnings_dir or Path(".learnings")

    hot_path = learnings_dir / "memory.md"
    hot = audit_hot_memory(hot_path)
    hot_text = _read_text(hot_path if hot.present else None)

    warm = audit_warm_tiers(learnings_dir) if learnings_dir.is_dir() else []
    corrections = audit_corrections_log(learnings_dir / "corrections.md")

    k_latest = _read_text(latest_state_path)
    k_held = _read_text(held_spec_path)
    k_contra = _read_text(contradiction_path)

    reconcile = reconcile_knowledge_memory(hot_text, [k_latest, k_held, k_contra])

    proposals: List[str] = []
    if hot.over_cap:
        proposals.append(
            f"HOT memory is {hot.line_count} lines (cap {HOT_LINE_CAP}); "
            "demote oldest sections to projects/ or domains/."
        )
    for w in warm:
        if w.over_cap:
            proposals.append(
                f"WARM file {w.path} is {w.line_count} lines (soft cap {WARM_LINE_SOFT_CAP}); compact or archive."
            )
    for hint in reconcile:
        if hint.action.startswith("Consider"):
            proposals.append(f"Reconcile '{hint.anchor}': {hint.action}")
    if not corrections.get("present"):
        proposals.append(
            "Create .learnings/corrections.md for append-only correction log (see self-improving skill)."
        )

    hot_updated = _parse_last_updated(hot_text)
    k_generated = None
    for text in (k_latest, k_held):
        m = re.search(r"generated:\s*(\d{4}-\d{2}-\d{2})", text, re.I)
        if m:
            k_generated = m.group(1)
            break
    if hot_updated and k_generated and hot_updated < k_generated:
        proposals.append(
            f"HOT memory last updated {hot_updated} but knowledge artifact is {k_generated}; "
            "refresh HOT after operator review."
        )

    return {
        "as_of": as_of,
        "writes_allowed": False,
        "governance_note": "Town-Hermes automated memory sync is FROZEN; proposals require operator approval.",
        "paths": {
            "learnings_dir": str(learnings_dir) if learnings_dir.is_dir() else None,
            "hot_memory": str(hot_path),
            "latest_state": str(latest_state_path) if latest_state_path else None,
            "held_spec_ledger": str(held_spec_path) if held_spec_path else None,
            "contradiction_ledger": str(contradiction_path) if contradiction_path else None,
        },
        "hot_tier": {
            "present": hot.present,
            "line_count": hot.line_count,
            "cap": HOT_LINE_CAP,
            "over_cap": hot.over_cap,
            "last_updated": hot_updated,
            "modified": hot.modified,
        },
        "warm_tiers": [
            {
                "path": w.path,
                "line_count": w.line_count,
                "over_cap": w.over_cap,
                "modified": w.modified,
            }
            for w in warm
        ],
        "corrections_log": corrections,
        "knowledge_present": {
            "latest_state": bool(k_latest),
            "held_spec_ledger": bool(k_held),
            "contradiction_ledger": bool(k_contra),
        },
        "reconciliation": [
            {
                "anchor": h.anchor,
                "in_knowledge": h.in_knowledge,
                "in_hot_memory": h.in_hot_memory,
                "action": h.action,
            }
            for h in reconcile
        ],
        "proposals": proposals,
        "recursive_loop": list(_RECURSIVE_LOOP),
        "recommended_mcp_calls": [
            "self_improvement_snapshot()",
            "fleet_context_snapshot(summary=True)",
            "learnings_read(file='memory.md')",
            "knowledge_read(artifact='latest_state')",
            "knowledge_read(artifact='held_spec_ledger')",
            "skills_read(name='autonomous-ai-agents/self-improving')",
        ],
    }


def format_audit_report(audit: Dict[str, Any]) -> str:
    lines = [
        "Self-improvement audit (read-only)",
        f"  As of: {audit.get('as_of', '')}",
        f"  Writes allowed: {audit.get('writes_allowed')}",
        "",
        "HOT memory:",
        f"  {audit.get('hot_tier', {})}",
    ]
    props = audit.get("proposals") or []
    if props:
        lines.append("")
        lines.append(f"  {len(props)} proposal(s):")
        for p in props:
            lines.append(f"    - {p}")
    else:
        lines.append("")
        lines.append("  No proposals — tiers look healthy.")
    lines.append("")
    lines.append("Recursive loop:")
    for step in audit.get("recursive_loop") or []:
        lines.append(f"  - {step}")
    lines.append("")
    lines.append(f"  {audit.get('governance_note', '')}")
    return "\n".join(lines)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only self-improvement audit.")
    parser.add_argument(
        "--learnings-dir",
        type=Path,
        default=Path(".learnings"),
        help="Path to .learnings directory",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args(argv)

    repo = Path.cwd()
    audit = build_self_improvement_audit(
        learnings_dir=args.learnings_dir,
        latest_state_path=repo / "artifacts/ops/knowledge_layer/latest_state.md",
        held_spec_path=repo / "artifacts/ops/held_spec_ledger/latest.md",
        contradiction_path=repo / "artifacts/ops/contradiction_ledger/latest.md",
    )
    if args.json:
        print(json.dumps(audit, indent=2))
    else:
        print(format_audit_report(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
