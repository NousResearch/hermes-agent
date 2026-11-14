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


def _same_file_key(path: Path) -> Optional[tuple[int, int]]:
    try:
        st = path.stat()
    except (OSError, RuntimeError):  # missing, dangling or looping: cannot share the target's inode
        return None
    return st.st_dev, st.st_ino


def _aliases(path: Path, target: Path, real: Path, key: Optional[tuple[int, int]]) -> bool:
    if path == target:
        return False
    try:
        if path.resolve() == real:
            return True
    except (OSError, RuntimeError):  # resolve() raises RuntimeError on a loop on 3.11/3.12
        return False
    return key is not None and _same_file_key(path) == key


def _scan_scope(skill_dir: Path, target: Path) -> list[Path]:
    """Files a background-origin write's two scans cover: *target* plus every in-skill path that is the
    same file: a symlink resolving to it (scanned under its own name's rules) or a hardlink sharing its
    inode. The normal atomic replace gives the target a new inode, so a hardlinked sibling keeps the old
    bytes and cancels against the baseline, but the EXDEV/EBUSY copy fallback and the Windows in-place
    rewrite write through the existing inode and change the sibling too. Every other file reads the same
    before and after the write, so its findings would cancel; scanning it only cost time. Structural
    checks still cover the whole dir."""
    real = target.resolve()  # the atomic write lands here when *target* is an in-skill symlink
    key = _same_file_key(target)  # None for a new target: nothing can be hardlinked to it yet
    return [target] + [f for f in skill_dir.rglob("*") if _aliases(f, target, real, key)]


def _scan_baseline(skill_dir: Path, target: Path) -> tuple[Optional[list[Path]], Optional[Counter], Optional[str]]:
    """Pre-write findings over the scan scope of *target* for a background-origin write, so only findings
    the write introduces block it (bundled skills already carry flagged examples the fork did not write).
    (scope, baseline, error); scope None (full scan) outside background origin."""
    if not _background_origin():
        return None, None, None
    from tools import skill_manager_tool
    try:
        scope = _scan_scope(skill_dir, target)
        findings = skill_manager_tool.scan_skill(skill_dir, source="agent-created", files=scope).findings
        return scope, Counter(_finding_keys(skill_dir, findings)), None
    except Exception as e:  # a required scan that cannot finish must not allow the write
        logger.warning("Security scan failed for %s: %s", skill_dir, e, exc_info=True)
        return None, None, f"Security scan could not complete, so this skill was blocked: {e}"



def _introduced_findings(skill_dir: Path, result, baseline: Counter):
    remaining = Counter(baseline)
    new = []
    for f, key in zip(result.findings, _finding_keys(skill_dir, result.findings)):
        if remaining[key] > 0:
            remaining[key] -= 1
        else:
            new.append(f)
    return replace(result, findings=new, verdict=_determine_verdict(new))
