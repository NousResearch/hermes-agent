"""Tests for the memory/skill write-approval gate (tools/write_approval.py)
and the shared slash-command handlers (hermes_cli/write_approval_commands.py).

Covers the boolean write_approval gate (off by default = write freely; on =
require approval) for both subsystems, the foreground-vs-background staging
split, pending store CRUD, and the list/approve/reject/diff/approval
subcommand dispatch.
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


def _set_approval(subsystem, enabled):
    import hermes_cli.config as cfg
    c = cfg.load_config()
    c.setdefault(subsystem, {})["write_approval"] = enabled
    cfg.save_config(c)


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def test_default_gate_is_off(hermes_home):
    from tools import write_approval as wa
    # Default: gate off → writes flow freely.
    assert wa.write_approval_enabled("memory") is False
    assert wa.write_approval_enabled("skills") is False


def test_invalid_subsystem_is_off(hermes_home):
    from tools import write_approval as wa
    assert wa.write_approval_enabled("bogus") is False


def test_normalize_enabled_coerces_values():
    from tools import write_approval as wa
    # Real bools pass through.
    assert wa._normalize_enabled(True) is True
    assert wa._normalize_enabled(False) is False
    # Truthy strings → True (incl. legacy 'approve').
    assert wa._normalize_enabled("on") is True
    assert wa._normalize_enabled("approve") is True
    assert wa._normalize_enabled("true") is True
    # Everything else → False (gate off is the safe default).
    assert wa._normalize_enabled("off") is False
    assert wa._normalize_enabled("garbage") is False
    assert wa._normalize_enabled(None) is False


# ---------------------------------------------------------------------------
# Memory gate
# ---------------------------------------------------------------------------

def test_memory_gate_off_allows_write(hermes_home):
    # Default (gate off) → write straight through, no staging.
    from tools.memory_tool import memory_tool, MemoryStore
    from tools import write_approval as wa
    store = MemoryStore(); store.load_from_disk()
    r = json.loads(memory_tool("add", "user", "save me", store=store))
    assert r["success"] is True
    assert r["entry_count"] == 1
    assert wa.pending_count("memory") == 0


def test_cli_memory_approve_without_live_agent_uses_fresh_store(hermes_home, capsys):
    """#46783: ``/memory approve`` from a context with no live agent (e.g. the
    Desktop GUI) passed ``memory_store=None`` into the shared handler, which
    returned "memory store unavailable" and applied nothing. The CLI handler must
    fall back to a freshly loaded on-disk store, like the gateway path does."""
    import json
    from tools.memory_tool import memory_tool, MemoryStore
    from tools import write_approval as wa
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    _set_approval("memory", True)
    staging = MemoryStore(); staging.load_from_disk()
    r = json.loads(memory_tool("add", "memory", "remember the launch date", store=staging))
    assert r.get("pending_id"), r
    assert wa.pending_count("memory") == 1

    # Bare CLI handler with no live agent → store resolves to None pre-fix.
    handler = CLICommandsMixin.__new__(CLICommandsMixin)
    handler.agent = None
    handler._handle_memory_command("/memory approve all")

    out = capsys.readouterr().out
    assert "memory store unavailable" not in out, out
    assert "Approved 1" in out, out
    assert wa.pending_count("memory") == 0
    # The approved write landed in a freshly loaded on-disk store (MEMORY.md).
    reloaded = MemoryStore(); reloaded.load_from_disk()
    assert any("remember the launch date" in e for e in reloaded.memory_entries)


def test_load_on_disk_store_honors_configured_limits_and_permissions(hermes_home, monkeypatch):
    """Fresh approval stores must match the live agent's limits and target gates."""
    from tools.memory_tool import load_on_disk_store

    # Config override path: helper picks up configured limits and store flags.
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "memory": {
                "memory_char_limit": 999,
                "user_char_limit": 444,
                "memory_enabled": False,
                "user_profile_enabled": True,
            }
        },
    )
    store = load_on_disk_store()
    assert store.memory_char_limit == 999
    assert store.user_char_limit == 444
    assert store.memory_enabled is False
    assert store.user_profile_enabled is True

    # Failure path: config raises → defaults, never blows up.
    def _boom():
        raise RuntimeError("no config")

    monkeypatch.setattr("hermes_cli.config.load_config", _boom)
    fallback = load_on_disk_store()
    assert fallback.memory_char_limit == 2200
    assert fallback.user_char_limit == 1375
    assert fallback.memory_enabled is True
    assert fallback.user_profile_enabled is True


# ---------------------------------------------------------------------------
# Skill gate
# ---------------------------------------------------------------------------

_SKILL = (
    "---\nname: test-skill\ndescription: A test skill\nversion: 1.0.0\n---\n"
    "# Test\nbody\n"
)


# ---------------------------------------------------------------------------
# Pending store CRUD
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shared command handler
# ---------------------------------------------------------------------------


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


def test_handle_approval_on(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    captured = {}
    out = handle_pending_subcommand(
        wa.MEMORY, ["approval", "on"],
        set_mode_fn=lambda enabled: captured.update(enabled=enabled),
    )
    assert captured["enabled"] is True
    assert "on" in out


def test_handle_approval_off(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    captured = {}
    out = handle_pending_subcommand(
        wa.SKILLS, ["approval", "off"],
        set_mode_fn=lambda enabled: captured.update(enabled=enabled),
    )
    assert captured["enabled"] is False
    assert "off" in out


# ---------------------------------------------------------------------------
# Inline (interactive CLI) approval path — regression for the bug where the
# per-thread approval callback was never passed to prompt_dangerous_approval,
# so every gated foreground memory write was silently denied.
# ---------------------------------------------------------------------------

@pytest.fixture
def approval_callback_cleanup():
    yield
    from tools.terminal_tool import set_approval_callback
    set_approval_callback(None)


def _stage_memory_add(content, target="memory"):
    """Stage one add payload exactly as memory_tool's gate path would."""
    from tools import write_approval as wa
    return wa.stage_write(
        wa.MEMORY, {"action": "add", "target": target, "content": content, "old_text": None},
        summary=f"add to {target}: {content[:120]}", origin="foreground")


def test_memory_gate_on_stages_even_with_approval_callback(hermes_home, approval_callback_cleanup):
    # #44963: memory writes use the pending-review flow, never the generic timed
    # approval box. A registered CLI approval callback must NOT be invoked.
    from tools.memory_tool import memory_tool, MemoryStore
    from tools.terminal_tool import set_approval_callback
    from tools import write_approval as wa
    _set_approval("memory", True)

    calls = []
    set_approval_callback(lambda command, description, **kw: calls.append((command, description)) or "once")

    store = MemoryStore(); store.load_from_disk()
    r = json.loads(memory_tool("add", "memory", "approved fact", store=store))
    assert r["success"] is True
    assert r.get("staged") is True
    assert r.get("pending_id")
    assert store.memory_entries == []
    assert wa.pending_count("memory") == 1
    assert calls == []


def test_memory_review_shows_record_with_abcde_menu(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    rec = _stage_memory_add("specific fact", target="user")
    out = handle_pending_subcommand(wa.MEMORY, ["review"])
    assert out is not None
    assert "MEMORY WRITE APPROVAL" in out
    assert f"Pending ID: {rec['id']}" in out
    assert "Action: add to USER" in out
    assert "specific fact" in out
    assert "A) Approve" in out
    assert "E) Review one-by-one" in out


def test_memory_review_renders_batch_operations(hermes_home):
    from tools import write_approval as wa
    # Sweeper finding on #44966: a staged batch is stored as
    # {action: "batch", operations: [...]} and must render every op, not just
    # the one-line summary.
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    wa.stage_write(wa.MEMORY,
                   {"action": "batch", "target": "memory",
                    "operations": [{"action": "add", "content": "op one"},
                                   {"action": "replace", "old_text": "stale", "content": "fresh"},
                                   {"action": "remove", "old_text": "gone"}]},
                   summary="apply 3 op(s) to memory", origin="foreground")
    out = handle_pending_subcommand(wa.MEMORY, ["review"])
    assert out is not None
    assert "- add: op one" in out
    assert "- replace: stale -> fresh" in out
    assert "- remove: gone" in out


def test_memory_review_empty_queue(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    assert handle_pending_subcommand(wa.MEMORY, ["review"]) == "No pending memory writes."


def test_memory_review_missing_id(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    _stage_memory_add("present fact")
    out = handle_pending_subcommand(wa.MEMORY, ["review", "nope"])
    assert out is not None
    assert "No pending memory write with id 'nope'" in out


def test_memory_edit_updates_pending_content(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    rec = _stage_memory_add("old fact")
    out = handle_pending_subcommand(wa.MEMORY, ["edit", rec["id"], "new", "fact"])
    assert out is not None
    assert "Updated pending memory write" in out
    updated = wa.get_pending(wa.MEMORY, rec["id"])
    assert updated["payload"]["content"] == "new fact"
    assert "new fact" in updated["summary"]


def test_memory_edit_rejects_empty_content(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    rec = _stage_memory_add("unchanged fact")
    out = handle_pending_subcommand(wa.MEMORY, ["edit", rec["id"]])
    assert out is not None
    assert "Usage" in out
    assert wa.get_pending(wa.MEMORY, rec["id"])["payload"]["content"] == "unchanged fact"


def test_memory_edit_rejects_batch_and_remove(hermes_home):
    from tools import write_approval as wa
    # Only single-op add/replace records are editable; batch/remove must be
    # rejected-and-reissued, never half-edited.
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    batch = wa.stage_write(wa.MEMORY, {"action": "batch", "target": "memory",
                                       "operations": [{"action": "add", "content": "x"}]},
                           summary="apply 1 op(s) to memory", origin="foreground")
    rem = wa.stage_write(wa.MEMORY, {"action": "remove", "target": "memory",
                                     "content": None, "old_text": "victim"},
                         summary="remove from memory: victim", origin="foreground")
    for rec in (batch, rem):
        out = handle_pending_subcommand(wa.MEMORY, ["edit", rec["id"], "whatever"])
        assert out is not None
        assert "cannot be edited" in out, out
    assert wa.get_pending(wa.MEMORY, batch["id"])["payload"]["action"] == "batch"


def test_memory_abcde_aliases_drive_approve_and_reject(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore
    store = MemoryStore(); store.load_from_disk()
    rec = _stage_memory_add("alias fact")
    assert "MEMORY WRITE APPROVAL" in handle_pending_subcommand(wa.MEMORY, ["e"])
    assert "pending writes" in handle_pending_subcommand(wa.MEMORY, ["c"])
    assert "Approved 1" in handle_pending_subcommand(wa.MEMORY, ["a", rec["id"]], memory_store=store)
    rec2 = _stage_memory_add("reject me")
    assert "Rejected" in handle_pending_subcommand(wa.MEMORY, ["b", rec2["id"]])
    _stage_memory_add("reject all me")
    assert "Rejected 1" in handle_pending_subcommand(wa.MEMORY, ["d"])
    assert wa.pending_count(wa.MEMORY) == 0


def test_memory_pending_list_is_labeled(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    _stage_memory_add("remember carefully")
    out = handle_pending_subcommand(wa.MEMORY, ["pending"])
    assert out is not None
    assert "MEMORY WRITE APPROVAL" in out
    assert "add to MEMORY" in out
    assert "/memory approve" in out


def test_memory_reject_all_alias_with_id_does_not_wipe_queue(hermes_home):
    # Cross-vendor review catch: '/memory d <id>' must not silently purge the
    # whole queue when the user plausibly meant "drop <id>" — point at /memory b.
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    rec1 = _stage_memory_add("keep me one")
    rec2 = _stage_memory_add("keep me two")
    out = handle_pending_subcommand(wa.MEMORY, ["d", rec2["id"]])
    assert out is not None
    assert "rejects ALL" in out
    assert f"/memory b {rec2['id']}" in out
    assert wa.pending_count(wa.MEMORY) == 2  # nothing purged


def test_memory_invalid_params_rejected_before_staging(hermes_home):
    # Param validation must run BEFORE the gate so a broken write is rejected
    # immediately instead of staged and failing at approve time.
    from tools.memory_tool import memory_tool, MemoryStore
    from tools import write_approval as wa
    _set_approval("memory", True)
    store = MemoryStore(); store.load_from_disk()
    r = json.loads(memory_tool("add", "memory", None, store=store))
    assert r["success"] is False
    assert wa.pending_count("memory") == 0


class TestSkillGist:
    """skill_gist builds a heuristic one-line summary for a pending skill write.

    Pure, no model call — every branch is verifiable from the function source.
    """

    def test_create_with_frontmatter_description(self):
        from tools import write_approval as wa
        content = "---\ndescription: My cool skill\n---\nprint('hi')\n"
        assert (
            wa.skill_gist("create", "demo", content=content)
            == f"create 'demo' — My cool skill ({len(content)} chars)"
        )

    def test_edit_without_description_uses_size_only(self):
        from tools import write_approval as wa
        content = "no frontmatter here"
        assert (
            wa.skill_gist("edit", "demo", content=content)
            == f"rewrite 'demo' ({len(content)} chars)"
        )


    def test_file_actions_and_unknown_fallback(self):
        from tools import write_approval as wa
        assert wa.skill_gist("write_file", "demo", file_path="a.py") == "write a.py in 'demo'"
        assert wa.skill_gist("remove_file", "demo", file_path="a.py") == "remove a.py from 'demo'"
        assert wa.skill_gist("delete", "demo") == "delete skill 'demo'"
        assert wa.skill_gist("unknown", "demo") == "unknown 'demo'"
