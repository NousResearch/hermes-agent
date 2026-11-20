"""Skills registry↔disk reconciliation — the ORCA-06 desync detector.

Incident (2026-10-26, ORCA-06): ``~/.hermes/skills/orca_orchestrator`` was raw-deleted
(``rm -rf`` from a foreground CLI session) while ``skills/.usage.json`` still listed it
``state: active``. Nothing reconciled the usage registry against on-disk skill dirs, so
the desync went unnoticed until a dispatched worker hit it. ``hermes skills doctor``
ports the host-proven check (~/.hermes/scripts/skills-health-check.py) into the repo as
a first-class command.

Why NAME-based reconciliation (not registry-key ↔ dir-path): registry keys mix flat skill
names (``orca_orchestrator``), category paths (``devops/kanban-worker``) and legacy forms
(``./spec-driven-development``, ``agent-skills:using-agent-skills``); and the registry is
a cumulative ledger — hub-viewed skills that were never installed locally are normal
entries, so most keys legitimately have no local dir.

Signals:
  CRITICAL — registry entry with state ``active``|``stale`` AND (pinned OR used/viewed
             within ``LOOKBACK_DAYS``) whose skill resolves to NO on-disk dir and NO
             ``.archive`` copy. Pinned entries are always checked regardless of age:
             pin is a protection promise, and a pinned entry with no dir is a desync
             by definition.
  INFO     — on-disk skill dir (SKILL.md present) with no registry entry at all
             (fresh skills register on first use/view — normal).

Stdlib-only by design (plus ``hermes_constants`` for home resolution): the check must
run under cron/no-agent contexts and any interpreter, exactly like the host script it
ports.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

LOOKBACK_DAYS = 14
CRITICAL_STATES = ("active", "stale")


def norm_key(key: str) -> str:
    """Normalize a registry key: strip legacy ``./`` prefixes, map legacy ``:`` to ``/``."""
    return key.lstrip("./").replace(":", "/")


def parse_ts(ts: Optional[str]) -> Optional[dt.datetime]:
    """Parse an ISO-8601 timestamp (tolerating a trailing ``Z``); None when absent/invalid."""
    if not ts:
        return None
    try:
        return dt.datetime.fromisoformat(str(ts))
    except ValueError:
        return None


def _frontmatter_name(skill_md: Path) -> Optional[str]:
    """The ``name:`` frontmatter field of a SKILL.md (first 4 KiB); None when absent."""
    try:
        head = skill_md.read_text(encoding="utf-8", errors="ignore")[:4096]
    except OSError:
        return None
    m = re.search(r"^name:\s*(.+)$", head, re.MULTILINE)
    return m.group(1).strip().strip("\"'") if m else None


def build_disk_index(skills_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Walk *skills_root* for SKILL.md-bearing dirs → ``(name → relpath, relpath → name)``.

    Top-level dot-directories (``.archive``, ``.curator_backups``, ``.hub``, ...) are
    platform bookkeeping, not skills, and are skipped.
    """
    by_name: dict[str, str] = {}
    by_path: dict[str, str] = {}
    for root, dirs, files in os.walk(skills_root, followlinks=True):
        rel = os.path.relpath(root, skills_root)
        if rel != "." and rel.split(os.sep)[0].startswith("."):
            dirs[:] = []
            continue
        if "SKILL.md" in files:
            name = _frontmatter_name(Path(root) / "SKILL.md") or os.path.basename(root)
            by_name.setdefault(name, rel)
            by_path[rel] = name
    return by_name, by_path


def archived_names(skills_root: Path) -> set[str]:
    """Skill names recoverable from ``.archive`` (top-level entries + nested legacy archives)."""
    names: set[str] = set()
    archive_root = skills_root / ".archive"
    if not archive_root.is_dir():
        return names
    for entry in os.scandir(archive_root):
        if not entry.is_dir():
            continue
        names.add(entry.name)
        for r, _d, f in os.walk(entry.path):
            if "SKILL.md" in f:
                names.add(os.path.basename(r))
    return names


def reconcile(
    skills_root: Path,
    registry: dict[str, Any],
    *,
    lookback_days: int = LOOKBACK_DAYS,
    now: Optional[dt.datetime] = None,
) -> dict[str, Any]:
    """Name-based registry↔disk reconciliation over *registry* (the ``.usage.json`` map).

    Returns the report body: ``critical`` desyncs and ``info_unregistered_dirs`` plus
    entry/dir counts. Pure w.r.t. disk state except for the walk of *skills_root* —
    callers pass an explicit ``now`` for deterministic tests.
    """
    now = now or dt.datetime.now(dt.UTC)
    by_name, by_path = build_disk_index(skills_root)
    archived = archived_names(skills_root)

    critical: list[dict[str, Any]] = []
    for key, rec in registry.items():
        if not isinstance(rec, dict) or rec.get("archived_at"):
            continue  # archived skills legitimately live only in .archive
        nk = norm_key(key)
        on_disk = (
            nk in by_path
            or key in by_name
            or nk in by_name
            or os.path.basename(nk) in by_name
            or key in archived
            or os.path.basename(nk) in archived
        )
        if on_disk:
            continue
        pinned = bool(rec.get("pinned"))
        # Real activity only — created_at seeds on first hub sighting and is not
        # evidence the skill was ever installed locally (curator seeds never-active
        # records at now).
        last = parse_ts(rec.get("last_used_at")) or parse_ts(rec.get("last_viewed_at"))
        age_days = (now - last).days if last else None
        recent = age_days is not None and age_days <= lookback_days
        if rec.get("state") in CRITICAL_STATES and (pinned or recent):
            critical.append({
                "key": key,
                "state": rec.get("state"),
                "pinned": pinned,
                "use_count": rec.get("use_count", 0),
                "last_used_at": rec.get("last_used_at"),
                "age_days": age_days,
            })

    disk_names = set(by_name) | {os.path.basename(p) for p in by_path}
    reg_names = {norm_key(k) for k in registry} | set(registry)
    info_unregistered: list[str] = []
    for name in sorted(disk_names - reg_names):
        if name not in registry:
            info_unregistered.append(by_name.get(name) or name)

    return {
        "registry_entries": len(registry),
        "disk_skill_dirs": len(by_path),
        "critical": critical,
        "info_unregistered_dirs": info_unregistered,
    }


def default_skills_root() -> Path:
    """The active profile's skills dir (HERMES_HOME-aware)."""
    return get_hermes_home() / "skills"


def doctor_report(skills_root: Optional[Path] = None, *,
                  now: Optional[dt.datetime] = None) -> dict[str, Any]:
    """Full report for *skills_root* (default: the active profile's skills dir).

    ``registry_present`` False (or ``registry_error`` set) means reconciliation could not
    run at all — the CLI treats that as a failure too, mirroring the host script.
    """
    skills_root = Path(skills_root) if skills_root is not None else default_skills_root()
    usage_path = skills_root / ".usage.json"
    report: dict[str, Any] = {
        "ts_utc": dt.datetime.now(dt.UTC).isoformat(),
        "skills_root": str(skills_root),
        "registry_present": usage_path.is_file(),
        "registry_error": None,
        "registry_entries": 0,
        "disk_skill_dirs": 0,
        "critical": [],
        "info_unregistered_dirs": [],
    }
    if not report["registry_present"]:
        return report
    try:
        registry = json.loads(usage_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as e:
        report["registry_error"] = f"{type(e).__name__}: {e}"
        return report
    if not isinstance(registry, dict):
        report["registry_error"] = "registry is not a JSON object"
        return report
    report.update(reconcile(skills_root, registry, now=now))
    return report
