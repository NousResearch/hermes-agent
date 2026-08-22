"""Tests for the guarded destructive-admin archive agent tool.

Covers ``kanban_admin_archive`` (tools/kanban_tools.py) and its config gate
``kanban.admin_profiles`` (hermes_cli/config_defaults.py). The tool is
security-sensitive and archive-only:

  - Absent unless ALL hold: not a delegate_task child, not a dispatcher task
    worker, the active profile is normalized-allowlisted in
    ``kanban.admin_profiles``, and the profile's config has the kanban
    toolset active. ``HERMES_KANBAN_TASK`` alone never exposes it; an
    empty/default allowlist exposes nobody.
  - The handler rechecks authorization at runtime and explicitly names the
    ``kanban.admin_profiles`` key in refusals; delegated children are
    explicitly refused.
  - There is deliberately NO unarchive agent tool or schema.

All board DB traffic resolves under pytest ``tmp_path`` scratch boards; no
live board is ever touched. Every test controls the registry ``check_fn``
TTL cache deterministically via ``invalidate_check_fn_cache()`` after any
config/env change.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Harness: config + scratch boards under tmp_path
# ---------------------------------------------------------------------------

def _write_config(home: Path, *, toolsets, admin_profiles):
    """Write a minimal config.yaml with the kanban toolset + allowlist."""
    home.mkdir(parents=True, exist_ok=True)
    ts = "\n".join(f"    - {t}" for t in toolsets)
    ap = "\n".join(f"    - {p!r}" for p in admin_profiles)
    (home / "config.yaml").write_text(
        f"toolsets:\n{ts}\n"
        f"kanban:\n  admin_profiles:\n{ap}\n"
    )


@pytest.fixture
def allowlisted_env(monkeypatch, tmp_path):
    """An allowlisted orchestrator: kanban toolset + allowlisted profile +
    unset HERMES_KANBAN_TASK (not a dispatcher worker). Returns the home."""
    home = tmp_path / ".hermes"
    _write_config(home, toolsets=["kanban"], admin_profiles=["adminpro"])
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "adminpro")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db(board="recovery")
    kb.init_db(board="alt")
    return home


def _check():
    from tools import kanban_tools as kt
    return kt._check_kanban_admin_archive()


def _schema(tool_name="kanban_admin_archive"):
    """Return the tool's schema via the real registry gating (check_fn)."""
    from tools.registry import invalidate_check_fn_cache, registry
    invalidate_check_fn_cache()
    return registry.get_definitions({tool_name}, quiet=True)


def _id(db, title):
    return kb.create_task(db, title=title, initial_status="blocked")


def build_dm(db):
    """A->B, A->C, B->D, C->D, all todo (closed dominated closure)."""
    a = _id(db, "dm A")
    b = kb.create_task(db, title="dm B", parents=[a])
    c = kb.create_task(db, title="dm C", parents=[a])
    d = kb.create_task(db, title="dm D", parents=[b, c])
    db.execute("UPDATE tasks SET status='todo' WHERE id=?", (a,))
    return {"a": a, "b": b, "c": c, "d": d}


def _statuses(db):
    return {
        row["id"]: row["status"]
        for row in db.execute("SELECT id, status FROM tasks").fetchall()
    }


# ---------------------------------------------------------------------------
# Gating: schema absent / present
# ---------------------------------------------------------------------------

def test_archive_tool_absent_by_default_empty_allowlist(monkeypatch, tmp_path):
    """Default config (admin_profiles=[] / absent) exposes the tool to nobody,
    even with the kanban toolset active and an orchestrator actor."""
    home = tmp_path / ".hermes"
    _write_config(home, toolsets=["kanban"], admin_profiles=[])
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "someone")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert _check() is False
    assert _schema() == []


def test_archive_tool_absent_when_not_allowlisted(monkeypatch, tmp_path):
    """A profile not on the allowlist must not see the tool, even with the
    kanban toolset active."""
    home = tmp_path / ".hermes"
    _write_config(home, toolsets=["kanban"], admin_profiles=["techlead"])
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "someother")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert _check() is False
    assert _schema() == []


def test_archive_tool_present_for_allowlisted_orchestrator(allowlisted_env):
    """An allowlisted, kanban-toolset orchestrator (no HERMES_KANBAN_TASK)
    sees the tool in the real registry schema."""
    assert _check() is True
    schema = _schema()
    assert len(schema) == 1
    fn = schema[0]["function"]
    assert fn["name"] == "kanban_admin_archive"


def test_hermes_kanban_task_alone_never_exposes(monkeypatch, tmp_path):
    """HERMES_KANBAN_TASK set (a dispatcher worker) never grants the tool,
    even when the actor is allowlisted."""
    home = tmp_path / ".hermes"
    _write_config(home, toolsets=["kanban"], admin_profiles=["adminpro"])
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "adminpro")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker_task")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert _check() is False
    assert _schema() == []


def test_archive_tool_absent_when_kanban_toolset_inactive(monkeypatch, tmp_path):
    """Without the kanban toolset in config, the tool is hidden even for an
    allowlisted profile."""
    home = tmp_path / ".hermes"
    _write_config(home, toolsets=["hermes-cli"], admin_profiles=["adminpro"])
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "adminpro")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert _check() is False
    assert _schema() == []


def test_archive_tool_absent_for_delegated_child(monkeypatch, allowlisted_env):
    """A delegate_task child must not see the tool even when allowlisted."""
    import tools.kanban_tools as kt
    monkeypatch.setattr(kt, "_is_delegated_child_context", lambda: True)
    assert _check() is False
    assert _schema() == []


# ---------------------------------------------------------------------------
# Allowlist normalization
# ---------------------------------------------------------------------------

def test_allowlist_case_and_whitespace_normalization(monkeypatch, tmp_path):
    """Both directions of the allowlist comparison are case-insensitive and
    whitespace-stripped, so ``" AdminPro "`` matches active profile
    ``"adminpro"`` and vice versa."""
    home = tmp_path / ".hermes"
    _write_config(home, toolsets=["kanban"], admin_profiles=["  AdminPro  "])
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "adminpro")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from tools import kanban_tools as kt
    assert kt._normalize_admin_profile("  AdminPro  ") == "adminpro"
    assert kt._normalize_admin_profile(None) == ""
    assert kt._admin_actor_authorized() is True
    assert _check() is True


def test_empty_allowlist_entry_never_authorizes(monkeypatch, tmp_path):
    """'' / None allowlist entries must be dropped, never matching anybody."""
    from tools import kanban_tools as kt
    assert kt._normalize_admin_profile("") == ""
    assert kt._normalize_admin_profile(None) == ""
    # Directly exercise the set-builder on a mixed list via _admin_allowlist
    # after writing a config with a blank entry.
    home = tmp_path / ".hermes"
    _write_config(home, toolsets=["kanban"], admin_profiles=["", "adminpro"])
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "adminpro")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert _admin_allowlist_set() == {"adminpro"}


def _admin_allowlist_set():
    from tools import kanban_tools as kt
    return kt._admin_allowlist()


# ---------------------------------------------------------------------------
# Handler refusal / authorization recheck
# ---------------------------------------------------------------------------

def test_handler_rejects_non_allowlisted_actor_naming_config_key(
    monkeypatch, tmp_path
):
    """A non-allowlisted actor calling the handler gets a structured refusal
    that explicitly names the ``kanban.admin_profiles`` config key."""
    home = tmp_path / ".hermes"
    _write_config(home, toolsets=["kanban"], admin_profiles=["techlead"])
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "intruder")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db(board="recovery")

    from tools import kanban_tools as kt
    out = kt._handle_admin_archive({
        "root_ids": ["t_x"],
        "reason": "cleanup",
    })
    d = json.loads(out)
    assert "error" in d
    assert "kanban.admin_profiles" in d["error"]
    # The actor is the profile, and it is named in the refusal.
    assert "intruder" in d["error"]


def test_handler_refuses_delegated_child(monkeypatch, allowlisted_env):
    """Delegated children are explicitly refused in the handler, regardless
    of allowlist status (defense in depth over the schema gate)."""
    from tools import kanban_tools as kt
    monkeypatch.setattr(kt, "_is_delegated_child_context", lambda: True)
    out = kt._handle_admin_archive({
        "root_ids": ["t_x"],
        "reason": "cleanup",
    })
    d = json.loads(out)
    assert "error" in d
    assert "delegate_task" in d["error"]


def test_handler_validates_root_ids_and_reason(allowlisted_env):
    from tools import kanban_tools as kt
    # Empty / non-string root_ids.
    for bad in ([] , None, ["  "], [123]):
        d = json.loads(kt._handle_admin_archive({
            "root_ids": bad, "reason": "cleanup",
        }))
        assert "error" in d and "root_ids" in d["error"], bad
    # Missing reason.
    d = json.loads(kt._handle_admin_archive({"root_ids": ["t_x"]}))
    assert "error" in d and "reason" in d["error"]


def test_handler_unknown_root_all_or_nothing(allowlisted_env):
    with kb.connect_closing(board="recovery") as db:
        built = build_dm(db)
    from tools import kanban_tools as kt
    bogus = "t_doesnotexist"
    out = kt._handle_admin_archive({
        "root_ids": [built["a"], bogus],
        "reason": "cleanup",
        "dry_run": True,
    })
    d = json.loads(out)
    assert d.get("error")
    assert bogus in d["error"]
    # All-or-nothing: nothing archived even though a valid sibling id was
    # supplied.
    with kb.connect_closing(board="recovery") as db:
        st = _statuses(db)
        assert st[built["a"]] == "todo"


# ---------------------------------------------------------------------------
# tmp_path board scoping / dry-run / execution
# ---------------------------------------------------------------------------

def test_dry_run_deterministic_and_non_mutating(allowlisted_env):
    with kb.connect_closing(board="recovery") as db:
        built = build_dm(db)
    from tools import kanban_tools as kt
    a1 = json.loads(kt._handle_admin_archive({
        "root_ids": [built["a"]], "reason": "cleanup", "dry_run": True,
        "board": "recovery",
    }))
    a2 = json.loads(kt._handle_admin_archive({
        "root_ids": [built["a"]], "reason": "cleanup", "dry_run": True,
        "board": "recovery",
    }))
    # Deterministic: byte-identical plans.
    assert a1 == a2
    assert a1["dry_run"] is True
    assert a1["archive_group_id"] is None
    assert sorted(a1["root_ids"]) == [built["a"]]
    with kb.connect_closing(board="recovery") as db:
        st = _statuses(db)
        for tid in (built["a"], built["b"], built["c"], built["d"]):
            assert st[tid] == "todo"
        assert db.execute(
            "SELECT COUNT(*) AS n FROM task_events "
            "WHERE kind='admin_archived'"
        ).fetchone()["n"] == 0


def test_execution_archives_closure_and_names_group(allowlisted_env):
    with kb.connect_closing(board="recovery") as db:
        built = build_dm(db)
    from tools import kanban_tools as kt
    out = json.loads(kt._handle_admin_archive({
        "root_ids": [built["a"]], "reason": "cleanup", "board": "recovery",
    }))
    assert out.get("error") is None
    assert out["archive_group_id"].startswith("ag_")
    assert sorted(out["archived_ids"]) == sorted(
        [built["a"], built["b"], built["c"], built["d"]]
    )
    with kb.connect_closing(board="recovery") as db:
        st = _statuses(db)
        for tid in (built["a"], built["b"], built["c"], built["d"]):
            assert st[tid] == "archived"


def test_board_scoping_uses_requested_board(allowlisted_env):
    """Data on the 'alt' board must be untouched when archiving on
    'recovery' (and vice-versa), proving the tool resolves per-board DBs."""
    with kb.connect_closing(board="recovery") as db:
        built = build_dm(db)
    with kb.connect_closing(board="alt") as db:
        alt_ids = [_id(db, "alt task")]
    from tools import kanban_tools as kt
    out = json.loads(kt._handle_admin_archive({
        "root_ids": [built["a"]], "reason": "cleanup", "board": "recovery",
    }))
    assert out.get("error") is None
    with kb.connect_closing(board="recovery") as db:
        assert _statuses(db)[built["a"]] == "archived"
    with kb.connect_closing(board="alt") as db:
        assert _statuses(db)[alt_ids[0]] == "blocked"


# ---------------------------------------------------------------------------
# No unarchive surface; existing lifecycle registrations unchanged
# ---------------------------------------------------------------------------

def test_no_unarchive_agent_tool_or_schema():
    from tools.registry import registry
    names = registry.get_tool_names_for_toolset("kanban")
    assert not any("unarchive" in n for n in names)
    # No entry anywhere in the registry carries an unarchive tool name.
    all_names = {e.name for e in registry.get_all_entries()}
    assert not any("unarchive" in n for n in all_names)
    # The archive tool's schema carries no unarchive/restore field.
    schema = _schema()
    if schema:
        fn = schema[0]["function"]
        for key in ("unarchive", "restore"):
            assert key not in fn


def test_existing_lifecycle_registrations_unchanged():
    """Adding the admin tool must not alter the existing lifecycle toolset
    registrations (names + gating survived recovery)."""
    from tools.registry import registry
    names = set(registry.get_tool_names_for_toolset("kanban"))
    for expected in ("kanban_show", "kanban_complete", "kanban_block",
                     "kanban_heartbeat", "kanban_comment", "kanban_create",
                     "kanban_link", "kanban_attach", "kanban_attach_url",
                     "kanban_attachments", "kanban_unblock",
                     "kanban_request_review", "kanban_request_changes"):
        assert expected in names, expected
    # kanban_admin_archive is present, and it is the only admin_* tool.
    assert "kanban_admin_archive" in names
    admin_tools = [n for n in names if n.startswith("kanban_admin_")]
    assert admin_tools == ["kanban_admin_archive"]


# ---------------------------------------------------------------------------
# Deterministic config-cache handling
# ---------------------------------------------------------------------------

def test_registry_check_fn_cache_respects_allowlist_change(monkeypatch, tmp_path):
    """The check_fn TTL cache must not leak a stale authorization across
    config changes: after invalidate_check_fn_cache() the new allowlist is
    reflected."""
    home = tmp_path / ".hermes"
    _write_config(home, toolsets=["kanban"], admin_profiles=["techlead"])
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "adminpro")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Not allowlisted -> absent.
    assert _check() is False
    assert _schema() == []

    # Add adminpro to the allowlist -> present after cache invalidate.
    _write_config(home, toolsets=["kanban"], admin_profiles=["techlead", "adminpro"])
    from tools import kanban_tools as kt
    assert kt._admin_actor_authorized() is True
    assert _check() is True
    assert len(_schema()) == 1
