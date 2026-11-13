"""Scan baseline for background-review ``skill_manage`` writes: only findings a write introduces
block it. ``scan_skill`` is reached lazily through ``tools.skill_manager_tool`` so that module
stays the patch seam; ``_guarded_write`` there owns the write and rollback."""

import contextvars as _ctxvars
import hashlib
import logging
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Optional

from tools.skill_manager_guards import _is_background_review
from tools.skills_guard import _determine_verdict

logger = logging.getLogger("tools.skill_manager_tool")

# Set while replaying an approved write that the background-review fork staged.
_replaying_background_write: _ctxvars.ContextVar[bool] = _ctxvars.ContextVar(
    "replaying_background_write", default=False)


def _background_origin() -> bool:
    return _is_background_review() or _replaying_background_write.get()


def _finding_keys(skill_dir: Path, findings) -> list[tuple[str, str, str, str]]:
    """(file, pattern_id, severity, sha256 of the full source line) per finding. Not the line number
    (an edit above a finding moves it) and not ``f.match``, which the scanner truncates on long lines,
    so a kept prefix with a new tail would match. Severity is part of the key because the scanner
    derives it from context (an unchanged line under a renamed denylist owner escalates), so an
    escalated finding counts as introduced. Findings without a source line key on their match."""
    lines: dict[str, list[bytes]] = {}
    keys = []
    for f in findings:
        if f.line > 0 and f.file not in lines:
            try:
                lines[f.file] = re.split(rb"\r\n|\r|\n", (skill_dir / f.file).read_bytes())
            except OSError:
                lines[f.file] = []
        src = lines.get(f.file, [])
        text = src[f.line - 1].strip() if 0 < f.line <= len(src) else f.match.encode("utf-8", "replace")
        keys.append((f.file, f.pattern_id, f.severity, hashlib.sha256(text).hexdigest()))
    return keys


def _scan_baseline(skill_dir: Path) -> tuple[Optional[Counter], Optional[str]]:
    """Pre-write findings for a background-origin write, so only findings the write introduces block
    it (bundled skills already carry flagged examples the fork did not write). (baseline, error)."""
    if not _background_origin():
        return None, None
    from tools import skill_manager_tool
    try:
        findings = skill_manager_tool.scan_skill(skill_dir, source="agent-created").findings
        return Counter(_finding_keys(skill_dir, findings)), None
    except Exception as e:  # a required scan that cannot finish must not allow the write
        logger.warning("Security scan failed for %s: %s", skill_dir, e, exc_info=True)
        return None, f"Security scan could not complete, so this skill was blocked: {e}"


def _introduced_findings(skill_dir: Path, result, baseline: Counter):
    remaining = Counter(baseline)
    new = []
    for f, key in zip(result.findings, _finding_keys(skill_dir, result.findings)):
        if remaining[key] > 0:
            remaining[key] -= 1
        else:
            new.append(f)
    return replace(result, findings=new, verdict=_determine_verdict(new))
