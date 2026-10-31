"""Tests for tools/skills_health.py — ORCA-06 registry↔disk reconciliation.

The CRITICAL rule under test is the ORCA-06 signature (2026-10-26): a registry entry
with state active|stale AND (pinned OR used/viewed within the lookback window) whose
skill resolves to NO on-disk dir and NO .archive copy — while hub-viewed-only and
archived entries stay excluded.
"""

import datetime as dt
import json
import os
from pathlib import Path

from tools.skills_health import (
    LOOKBACK_DAYS,
    archived_names,
    build_disk_index,
    doctor_report,
    norm_key,
    parse_ts,
    reconcile,
)

NOW = dt.datetime(2026, 10, 26, 12, 0, tzinfo=dt.UTC)


def iso(days_ago: float = 0) -> str:
    return (NOW - dt.timedelta(days=days_ago)).isoformat()


def make_skill(root: Path, *parts: str, fm_name: str | None = None) -> Path:
    """Create a SKILL.md-bearing skill dir under *root*; returns the dir path."""
    d = root.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    name = fm_name or parts[-1]
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test skill\n---\n\n# {name}\n", encoding="utf-8")
    return d


def rec(state="active", pinned=False, used_days_ago: float | None = 1,
        viewed_days_ago: float | None = 1, **extra) -> dict:
    """A .usage.json record; activity timestamps default to 1 day ago (recent)."""
    r = {
        "state": state, "pinned": pinned, "use_count": 3, "view_count": 3,
        "last_used_at": iso(used_days_ago) if used_days_ago is not None else None,
        "last_viewed_at": iso(viewed_days_ago) if viewed_days_ago is not None else None,
        "created_at": iso(30), "archived_at": None,
    }
    r.update(extra)
    return r


# --- key/timestamp normalization -------------------------------------------

def test_norm_key_handles_legacy_forms():
    assert norm_key("flat-name") == "flat-name"
    assert norm_key("./spec-driven-development") == "spec-driven-development"
    assert norm_key("agent-skills:using-agent-skills") == "agent-skills/using-agent-skills"
    assert norm_key("./devops:kanban-worker") == "devops/kanban-worker"


def test_parse_ts():
    assert parse_ts(None) is None
    assert parse_ts("") is None
    assert parse_ts("not-a-date") is None
    assert parse_ts("2026-10-26T04:08:34Z") == dt.datetime(2026, 10, 26, 4, 8, 34,
                                                           tzinfo=dt.UTC)
    assert parse_ts("2026-10-26T04:08:34+00:00") == dt.datetime(2026, 10, 26, 4, 8, 34,
                                                                tzinfo=dt.UTC)


# --- disk index --------------------------------------------------------------

def test_build_disk_index_uses_frontmatter_name_and_skips_dot_dirs(tmp_path):
    make_skill(tmp_path, "devops", "kanban-worker", fm_name="kanban-worker")
    make_skill(tmp_path, "plain-skill")
    (tmp_path / ".archive" / "old-skill").mkdir(parents=True)
    (tmp_path / ".archive" / "old-skill" / "SKILL.md").write_text("name: old-skill\n")
    (tmp_path / ".curator_backups" / "blobs").mkdir(parents=True)

    by_name, by_path = build_disk_index(tmp_path)

    assert by_name["kanban-worker"] == f"devops{os.sep}kanban-worker"
    assert by_name["plain-skill"] == "plain-skill"
    assert "old-skill" not in by_name  # .archive is bookkeeping, not a live skill
    assert len(by_path) == 2


def test_archived_names_includes_nested_legacy_archives(tmp_path):
    (tmp_path / ".archive" / "gone").mkdir(parents=True)
    (tmp_path / ".archive" / "gone" / "SKILL.md").write_text("name: gone\n")
    (tmp_path / ".archive" / "legacy" / "nested" / "deep").mkdir(parents=True)
    (tmp_path / ".archive" / "legacy" / "nested" / "deep" / "SKILL.md").write_text("n\n")

    names = archived_names(tmp_path)
    assert names == {"gone", "legacy", "deep"}  # "nested" holds no SKILL.md


# --- CRITICAL rule (the ORCA-06 signature) -----------------------------------

def test_critical_active_pinned_missing_dir(tmp_path):
    """Acceptance signature: active + pinned + no dir + no archive copy = CRITICAL."""
    make_skill(tmp_path, "healthy")
    registry = {"healthy": rec(), "gone-pinned": rec(pinned=True)}

    report = reconcile(tmp_path, registry, now=NOW)

    assert [c["key"] for c in report["critical"]] == ["gone-pinned"]
    assert report["critical"][0]["pinned"] is True
    assert report["critical"][0]["state"] == "active"


def test_pinned_checked_regardless_of_activity_age(tmp_path):
    """Pin is a protection promise: a pinned entry is checked even with no activity at all."""
    registry = {"ancient-pinned": rec(pinned=True, used_days_ago=400, viewed_days_ago=400)}
    assert reconcile(tmp_path, registry, now=NOW)["critical"][0]["key"] == "ancient-pinned"

    # No timestamps at all (curator-seeded record) — pinned still forces the check.
    registry = {"phantom": {"state": "active", "pinned": True, "use_count": 0,
                            "created_at": iso(30), "archived_at": None}}
    assert reconcile(tmp_path, registry, now=NOW)["critical"][0]["key"] == "phantom"


def test_active_recent_unpinned_missing_dir_is_critical(tmp_path):
    registry = {"recent-gone": rec(used_days_ago=2)}
    assert reconcile(tmp_path, registry, now=NOW)["critical"][0]["key"] == "recent-gone"


def test_stale_state_included_in_critical(tmp_path):
    registry = {"stale-gone": rec(state="stale", used_days_ago=3)}
    assert reconcile(tmp_path, registry, now=NOW)["critical"][0]["key"] == "stale-gone"


def test_active_old_unpinned_missing_dir_not_critical(tmp_path):
    """Aged-out, unpinned entries (typical hub-viewed-only ledger rows) are excluded."""
    registry = {"aged-out": rec(used_days_ago=LOOKBACK_DAYS + 1)}
    assert reconcile(tmp_path, registry, now=NOW)["critical"] == []


def test_created_at_seed_is_not_activity(tmp_path):
    """Curator seeds never-active records with created_at=now — not evidence of local use."""
    registry = {"hub-seed": {"state": "active", "pinned": False, "use_count": 0,
                             "view_count": 0, "created_at": iso(0), "archived_at": None}}
    assert reconcile(tmp_path, registry, now=NOW)["critical"] == []


def test_archived_state_and_archived_at_excluded(tmp_path):
    registry = {
        "archived-flag": rec(state="archived", used_days_ago=0),
        "archived-at-set": rec(used_days_ago=0, archived_at=iso(5)),
    }
    assert reconcile(tmp_path, registry, now=NOW)["critical"] == []


def test_archive_copy_counts_as_valid_location(tmp_path):
    """state active|stale + pinned + missing live dir is fine when .archive holds a copy."""
    (tmp_path / ".archive" / "moved-to-archive").mkdir(parents=True)
    (tmp_path / ".archive" / "moved-to-archive" / "SKILL.md").write_text(
        "name: moved-to-archive\n", encoding="utf-8")
    registry = {"moved-to-archive": rec(pinned=True, used_days_ago=0)}
    assert reconcile(tmp_path, registry, now=NOW)["critical"] == []


# --- name-based key resolution ------------------------------------------------

def test_legacy_and_path_key_forms_resolve_to_disk(tmp_path):
    make_skill(tmp_path, "devops", "kanban-worker", fm_name="kanban-worker")
    make_skill(tmp_path, "agent-skills", "using-agent-skills")
    make_skill(tmp_path, "flat-skill")
    registry = {
        "kanban-worker": rec(),                       # flat frontmatter name
        "devops/kanban-worker": rec(),                # category path
        "./devops/kanban-worker": rec(),              # legacy "./" prefix
        "agent-skills:using-agent-skills": rec(),     # legacy ":" separator
        "flat-skill": rec(),
    }
    assert reconcile(tmp_path, registry, now=NOW)["critical"] == []


def test_key_resolves_via_frontmatter_name_when_dirname_differs(tmp_path):
    """Dir basename and frontmatter name can differ; either resolves the registry key."""
    make_skill(tmp_path, "some-dirname", fm_name="frontmatter-name")
    for key in ("frontmatter-name", "some-dirname"):
        assert reconcile(tmp_path, {key: rec()}, now=NOW)["critical"] == []


def test_hub_viewed_only_ledger_rows_are_normal(tmp_path):
    """~160 of 367 live keys legitimately have no local dir: hub-viewed, aged out."""
    registry = {f"hub-only-{i}": rec(used_days_ago=100 + i) for i in range(5)}
    report = reconcile(tmp_path, registry, now=NOW)
    assert report["critical"] == []
    assert report["registry_entries"] == 5


def test_recently_viewed_but_never_used_still_counts(tmp_path):
    registry = {"viewed-only": rec(used_days_ago=None, viewed_days_ago=1)}
    assert reconcile(tmp_path, registry, now=NOW)["critical"][0]["key"] == "viewed-only"


# --- INFO + report integration -------------------------------------------------

def test_info_unregistered_dirs(tmp_path):
    make_skill(tmp_path, "unregistered-fresh")
    make_skill(tmp_path, "registered")
    registry = {"registered": rec()}
    report = reconcile(tmp_path, registry, now=NOW)
    assert report["info_unregistered_dirs"] == ["unregistered-fresh"]
    assert report["disk_skill_dirs"] == 2


def test_doctor_report_missing_registry(tmp_path):
    make_skill(tmp_path, "skill")
    report = doctor_report(tmp_path)
    assert report["registry_present"] is False
    assert report["critical"] == []


def test_doctor_report_corrupt_registry(tmp_path):
    (tmp_path / ".usage.json").write_text("{not json", encoding="utf-8")
    report = doctor_report(tmp_path)
    assert report["registry_present"] is True
    assert report["registry_error"] is not None
    assert report["critical"] == []


def test_doctor_report_full_pass(tmp_path):
    make_skill(tmp_path, "healthy")
    (tmp_path / ".usage.json").write_text(json.dumps({
        "healthy": rec(),
        "gone-pinned": rec(pinned=True, used_days_ago=400),
    }), encoding="utf-8")
    report = doctor_report(tmp_path, now=NOW)
    assert report["registry_present"] is True
    assert report["registry_error"] is None
    assert report["disk_skill_dirs"] == 1
    assert [c["key"] for c in report["critical"]] == ["gone-pinned"]
    assert report["critical"][0]["age_days"] == 400
