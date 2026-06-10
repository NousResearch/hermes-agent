"""Tests for the memory/skill write-approval gate (tools/write_approval.py)
and the shared slash-command handlers (hermes_cli/write_approval_commands.py).

Covers the tri-state write_mode (on/off/approve) for both subsystems, the
foreground-vs-background staging split, pending store CRUD, and the
list/approve/reject/diff/mode subcommand dispatch.
"""

import json
import os
import tempfile
import shutil

import pytest


@pytest.fixture
def hermes_home(monkeypatch):
    d = tempfile.mkdtemp(prefix="hermes_wa_test_")
    home = os.path.join(d, ".hermes")
    os.makedirs(home)
    monkeypatch.setenv("HERMES_HOME", home)
    yield home
    shutil.rmtree(d, ignore_errors=True)


def _set_mode(subsystem, mode):
    import hermes_cli.config as cfg
    c = cfg.load_config()
    c.setdefault(subsystem, {})["write_mode"] = mode
    cfg.save_config(c)


# ---------------------------------------------------------------------------
# Mode resolution
# ---------------------------------------------------------------------------

def test_default_write_mode_is_on(hermes_home):
    from tools import write_approval as wa
    assert wa.get_write_mode("memory") == "on"
    assert wa.get_write_mode("skills") == "on"


def test_invalid_subsystem_returns_on(hermes_home):
    from tools import write_approval as wa
    assert wa.get_write_mode("bogus") == "on"


def test_normalize_mode_handles_yaml_bool():
    from tools import write_approval as wa
    assert wa._normalize_mode(False) == "off"
    assert wa._normalize_mode(True) == "on"
    assert wa._normalize_mode("approve") == "approve"
    assert wa._normalize_mode("garbage") == "on"


# ---------------------------------------------------------------------------
# Memory gate
# ---------------------------------------------------------------------------

def test_memory_off_blocks_write(hermes_home):
    from tools.memory_tool import memory_tool, MemoryStore
    _set_mode("memory", "off")
    store = MemoryStore(); store.load_from_disk()
    r = json.loads(memory_tool("add", "user", "should not save", store=store))
    assert r["success"] is False
    assert "disabled" in r["error"].lower()
    assert store.user_entries == []


def test_memory_on_allows_write(hermes_home):
    from tools.memory_tool import memory_tool, MemoryStore
    _set_mode("memory", "on")
    store = MemoryStore(); store.load_from_disk()
    r = json.loads(memory_tool("add", "user", "save me", store=store))
    assert r["success"] is True
    assert r["entry_count"] == 1


def test_memory_approve_no_interactive_stages(hermes_home):
    # No approval callback registered and not a gateway context → stage.
    from tools.memory_tool import memory_tool, MemoryStore
    from tools import write_approval as wa
    _set_mode("memory", "approve")
    store = MemoryStore(); store.load_from_disk()
    r = json.loads(memory_tool("add", "memory", "stage me", store=store))
    assert r.get("staged") is True
    assert r.get("pending_id")
    # Not written to the live store yet.
    assert store.memory_entries == []
    pend = wa.list_pending("memory")
    assert len(pend) == 1
    assert pend[0]["id"] == r["pending_id"]


def test_memory_approve_then_apply(hermes_home):
    from tools.memory_tool import memory_tool, MemoryStore, apply_memory_pending
    from tools import write_approval as wa
    _set_mode("memory", "approve")
    store = MemoryStore(); store.load_from_disk()
    r = json.loads(memory_tool("add", "user", "approved entry", store=store))
    pid = r["pending_id"]
    rec = wa.get_pending("memory", pid)
    result = apply_memory_pending(rec["payload"], store)
    assert result["success"] is True
    assert "approved entry" in store.user_entries[0]


# ---------------------------------------------------------------------------
# Skill gate
# ---------------------------------------------------------------------------

_SKILL = (
    "---\nname: test-skill\ndescription: A test skill\nversion: 1.0.0\n---\n"
    "# Test\nbody\n"
)


def test_skill_off_blocks_create(hermes_home):
    from tools.skill_manager_tool import skill_manage
    _set_mode("skills", "off")
    r = json.loads(skill_manage("create", "blocked-skill", content=_SKILL))
    assert r["success"] is False
    assert "disabled" in r["error"].lower()


def test_skill_approve_always_stages(hermes_home):
    # Skills stage even in the foreground (too big to review inline).
    from tools.skill_manager_tool import skill_manage
    from tools import write_approval as wa
    _set_mode("skills", "approve")
    r = json.loads(skill_manage("create", "staged-skill", content=_SKILL))
    assert r.get("staged") is True
    assert "staged-skill" in r.get("gist", "")
    assert wa.pending_count("skills") == 1


def test_skill_approve_then_apply_writes_file(hermes_home):
    # SKILLS_DIR is resolved at import time, so reload the skill module under
    # this test's HERMES_HOME to exercise the real on-disk write path.
    import importlib
    import tools.skill_manager_tool as smt
    importlib.reload(smt)
    from tools import write_approval as wa
    _set_mode("skills", "approve")
    r = json.loads(smt.skill_manage("create", "applied-skill", content=_SKILL))
    rec = wa.get_pending("skills", r["pending_id"])
    res = json.loads(smt.apply_skill_pending(rec["payload"]))
    assert res["success"] is True
    assert smt._find_skill("applied-skill") is not None


def test_skill_create_diff_is_full_content(hermes_home):
    from tools.skill_manager_tool import skill_manage
    from tools import write_approval as wa
    _set_mode("skills", "approve")
    r = json.loads(skill_manage("create", "diff-skill", content=_SKILL))
    rec = wa.get_pending("skills", r["pending_id"])
    diff = wa.skill_pending_diff(rec)
    assert "name: test-skill" in diff


# ---------------------------------------------------------------------------
# Pending store CRUD
# ---------------------------------------------------------------------------

def test_pending_store_roundtrip(hermes_home):
    from tools import write_approval as wa
    rec = wa.stage_write("memory", {"action": "add", "target": "user", "content": "x"},
                         summary="add x", origin="foreground")
    assert wa.pending_count("memory") == 1
    got = wa.get_pending("memory", rec["id"])
    assert got["payload"]["content"] == "x"
    assert wa.discard_pending("memory", rec["id"]) is True
    assert wa.pending_count("memory") == 0
    assert wa.get_pending("memory", rec["id"]) is None


# ---------------------------------------------------------------------------
# Shared command handler
# ---------------------------------------------------------------------------

def test_handle_pending_list_empty(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    out = handle_pending_subcommand(wa.MEMORY, ["pending"])
    assert "No pending memory" in out


def test_handle_approve_all(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools.memory_tool import MemoryStore
    from tools import write_approval as wa
    store = MemoryStore(); store.load_from_disk()
    wa.stage_write("memory", {"action": "add", "target": "user", "content": "a"},
                   summary="a", origin="foreground")
    wa.stage_write("memory", {"action": "add", "target": "user", "content": "b"},
                   summary="b", origin="foreground")
    out = handle_pending_subcommand(wa.MEMORY, ["approve", "all"], memory_store=store)
    assert "Approved 2" in out
    assert wa.pending_count("memory") == 0
    assert len(store.user_entries) == 2


def test_handle_reject(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    rec = wa.stage_write("skills", {"action": "create", "name": "s"},
                         summary="create s", origin="background_review")
    out = handle_pending_subcommand(wa.SKILLS, ["reject", rec["id"]])
    assert "Rejected" in out
    assert wa.pending_count("skills") == 0


def test_handle_mode_set(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    captured = {}
    out = handle_pending_subcommand(
        wa.MEMORY, ["mode", "approve"],
        set_mode_fn=lambda m: captured.update(mode=m),
    )
    assert captured["mode"] == "approve"
    assert "approve" in out


def test_handle_mode_invalid(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    out = handle_pending_subcommand(wa.MEMORY, ["mode", "bogus"],
                                    set_mode_fn=lambda m: None)
    assert "Invalid mode" in out


def test_handle_unknown_subcommand_returns_none(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    # An unrecognized /skills subcommand (e.g. 'search') must return None so
    # the CLI falls through to the skills hub.
    out = handle_pending_subcommand(wa.SKILLS, ["search", "foo"])
    assert out is None
