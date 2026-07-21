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
import threading
from pathlib import Path

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


def test_background_memory_stages_even_when_gate_is_off(hermes_home):
    """Autonomous review memory proposals need approval regardless of config."""
    from tools.memory_tool import memory_tool, MemoryStore
    from tools import write_approval as wa
    from tools.skill_provenance import (
        BACKGROUND_REVIEW,
        reset_current_write_origin,
        set_current_write_origin,
    )

    store = MemoryStore(); store.load_from_disk()
    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        result = json.loads(memory_tool("add", "memory", "review proposal", store=store))
    finally:
        reset_current_write_origin(token)

    assert result.get("staged") is True
    assert "require approval" in result["message"]
    assert store.memory_entries == []
    pending = wa.get_pending("memory", result["pending_id"])
    assert pending["origin"] == BACKGROUND_REVIEW


def test_memory_gate_on_no_interactive_stages(hermes_home):
    # Gate on, no approval callback / not a gateway context → stage.
    from tools.memory_tool import memory_tool, MemoryStore
    from tools import write_approval as wa
    _set_approval("memory", True)
    store = MemoryStore(); store.load_from_disk()
    r = json.loads(memory_tool("add", "memory", "stage me", store=store))
    assert r.get("staged") is True
    assert r.get("pending_id")
    # Not written to the live store yet.
    assert store.memory_entries == []
    pend = wa.list_pending("memory")
    assert len(pend) == 1
    assert pend[0]["id"] == r["pending_id"]


def test_memory_gate_on_then_apply(hermes_home):
    from tools.memory_tool import memory_tool, MemoryStore, apply_memory_pending
    from tools import write_approval as wa
    _set_approval("memory", True)
    store = MemoryStore(); store.load_from_disk()
    r = json.loads(memory_tool("add", "user", "approved entry", store=store))
    pid = r["pending_id"]
    rec = wa.get_pending("memory", pid)
    result = apply_memory_pending(rec["payload"], store)
    assert result["success"] is True
    assert "approved entry" in store.user_entries[0]


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


def test_load_on_disk_store_honors_configured_char_limits(hermes_home, monkeypatch):
    """load_on_disk_store() must read memory.memory_char_limit /
    user_char_limit from config so approvals applied without a live agent
    enforce the SAME caps as the live agent (agent_init.py). Falls back to
    defaults when config can't be loaded.
    """
    from tools.memory_tool import load_on_disk_store

    # Config override path: helper picks up the configured limits.
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"memory_char_limit": 999, "user_char_limit": 444}},
    )
    store = load_on_disk_store()
    assert store.memory_char_limit == 999
    assert store.user_char_limit == 444

    # Failure path: config raises → defaults, never blows up.
    def _boom():
        raise RuntimeError("no config")

    monkeypatch.setattr("hermes_cli.config.load_config", _boom)
    fallback = load_on_disk_store()
    assert fallback.memory_char_limit == 2200
    assert fallback.user_char_limit == 1375


# ---------------------------------------------------------------------------
# Skill gate
# ---------------------------------------------------------------------------

_SKILL = (
    "---\nname: test-skill\ndescription: A test skill\nversion: 1.0.0\n---\n"
    "# Test\nbody\n"
)


def test_skill_gate_off_allows_create(hermes_home):
    # Default (gate off) → skill is created normally, not staged.
    import importlib
    import tools.skill_manager_tool as smt
    importlib.reload(smt)
    from tools import write_approval as wa
    r = json.loads(smt.skill_manage("create", "free-skill", content=_SKILL))
    assert r.get("success") is True
    assert wa.pending_count("skills") == 0


def test_background_skill_stages_even_when_gate_is_off(hermes_home):
    """The shared background origin protects curator and review skill writes."""
    import importlib
    import tools.skill_manager_tool as smt
    importlib.reload(smt)
    from tools import write_approval as wa
    from tools.skill_provenance import (
        BACKGROUND_REVIEW,
        reset_current_write_origin,
        set_current_write_origin,
    )

    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        result = json.loads(smt.skill_manage("create", "background-skill", content=_SKILL))
    finally:
        reset_current_write_origin(token)

    assert result.get("staged") is True
    assert "require approval" in result["message"]
    assert smt._find_skill("background-skill") is None
    pending = wa.get_pending("skills", result["pending_id"])
    assert pending["origin"] == BACKGROUND_REVIEW


def test_skill_gate_on_always_stages(hermes_home):
    # Skills stage even in the foreground (too big to review inline).
    from tools.skill_manager_tool import skill_manage
    from tools import write_approval as wa
    _set_approval("skills", True)
    r = json.loads(skill_manage("create", "staged-skill", content=_SKILL))
    assert r.get("staged") is True
    assert "staged-skill" in r.get("gist", "")
    assert wa.pending_count("skills") == 1


def test_skill_gate_on_then_apply_writes_file(hermes_home):
    # SKILLS_DIR is resolved at import time, so reload the skill module under
    # this test's HERMES_HOME to exercise the real on-disk write path.
    import importlib
    import tools.skill_manager_tool as smt
    importlib.reload(smt)
    from tools import write_approval as wa
    _set_approval("skills", True)
    r = json.loads(smt.skill_manage("create", "applied-skill", content=_SKILL))
    rec = wa.get_pending("skills", r["pending_id"])
    res = json.loads(smt.apply_skill_pending(rec["payload"]))
    assert res["success"] is True
    assert smt._find_skill("applied-skill") is not None


def test_skill_create_diff_is_full_content(hermes_home):
    from tools.skill_manager_tool import skill_manage
    from tools import write_approval as wa
    _set_approval("skills", True)
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


def test_stage_write_failure_is_explicit_and_leaves_no_record(hermes_home, monkeypatch):
    from tools import write_approval as wa

    def _fail(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("utils.atomic_json_write", _fail)
    assert wa.stage_write(
        "memory", {"action": "add", "target": "memory", "content": "x"},
        summary="x", origin="foreground",
    ) is None
    assert wa.pending_count("memory") == 0


def test_stage_does_not_follow_existing_pending_leaf_symlink(hermes_home, monkeypatch):
    from tools import write_approval as wa

    pending_id = "0123abcd"
    pending_dir = Path(hermes_home) / "pending" / "memory"
    pending_dir.mkdir(parents=True)
    external_target = Path(hermes_home).parent / "outside-pending.json"
    external_target.write_text('{"sentinel": true}', encoding="utf-8")
    leaf = pending_dir / f"{pending_id}.json"
    try:
        leaf.symlink_to(external_target)
    except OSError:
        pytest.skip("Windows account cannot create a symlink")

    class _FixedUUID:
        def __init__(self, value):
            self.hex = value

    ids = iter((_FixedUUID(pending_id + "0" * 24), _FixedUUID("1" * 32)))
    monkeypatch.setattr(wa.uuid, "uuid4", lambda: next(ids))

    assert wa.stage_write(
        "memory", {"action": "add", "target": "memory", "content": "x"},
        summary="x", origin="foreground",
    ) is None
    assert json.loads(external_target.read_text(encoding="utf-8")) == {"sentinel": True}
    assert leaf.is_symlink()


@pytest.mark.parametrize("redirect_name", ["memory", "pending"])
def test_stage_rejects_pending_junction(hermes_home, monkeypatch, redirect_name):
    from tools import write_approval as wa

    monkeypatch.setattr(
        Path, "is_junction", lambda path: path.name == redirect_name, raising=False,
    )
    assert wa.stage_write(
        "memory", {"action": "add", "target": "memory", "content": "x"},
        summary="x", origin="foreground",
    ) is None
    assert wa.pending_count("memory") == 0


def test_pending_redirect_detects_windows_reparse_attribute(tmp_path, monkeypatch):
    from tools import write_approval as wa

    class _ReparsePoint:
        st_file_attributes = 0x400

    monkeypatch.setattr(Path, "is_symlink", lambda _path: False)
    monkeypatch.setattr(Path, "lstat", lambda _path: _ReparsePoint())
    assert wa._is_path_redirect(tmp_path) is True


def test_memory_stage_failure_returns_tool_error(hermes_home, monkeypatch):
    from tools.memory_tool import memory_tool, MemoryStore
    from tools import write_approval as wa

    _set_approval("memory", True)
    monkeypatch.setattr(wa, "stage_write", lambda *_args, **_kwargs: None)
    store = MemoryStore(); store.load_from_disk()
    result = json.loads(memory_tool("add", "memory", "not persisted", store=store))

    assert result["success"] is False
    assert result.get("staged") is None
    assert result.get("pending_id") is None
    assert store.memory_entries == []


def test_memory_batch_stage_failure_returns_tool_error(hermes_home, monkeypatch):
    from tools.memory_tool import memory_tool, MemoryStore
    from tools import write_approval as wa

    _set_approval("memory", True)
    monkeypatch.setattr(wa, "stage_write", lambda *_args, **_kwargs: None)
    store = MemoryStore(); store.load_from_disk()
    result = json.loads(memory_tool(
        target="memory", operations=[{"action": "add", "content": "not persisted"}],
        store=store,
    ))

    assert result["success"] is False
    assert result.get("staged") is None
    assert result.get("pending_id") is None
    assert store.memory_entries == []


def test_skill_stage_failure_returns_tool_error(hermes_home, monkeypatch):
    import importlib
    import tools.skill_manager_tool as smt
    importlib.reload(smt)
    from tools import write_approval as wa

    _set_approval("skills", True)
    monkeypatch.setattr(wa, "stage_write", lambda *_args, **_kwargs: None)
    result = json.loads(smt.skill_manage("create", "not-persisted", content=_SKILL))

    assert result["success"] is False
    assert result.get("staged") is None
    assert result.get("pending_id") is None
    assert smt._find_skill("not-persisted") is None


@pytest.mark.parametrize("pending_id", ["", "../outside", "..\\outside", "C:\\outside", "abcd.json", "ABCDEF12", "abc"])
def test_pending_id_rejects_path_like_input(hermes_home, pending_id):
    from tools import write_approval as wa

    assert wa.valid_pending_id(pending_id) is False
    assert wa.get_pending("memory", pending_id) is None
    assert wa.discard_pending("memory", pending_id) is False
    assert wa.claim_pending("memory", pending_id) is None


def test_list_and_get_skip_pending_leaf_symlink(hermes_home, monkeypatch):
    from tools import write_approval as wa

    pending_id = "0123abcd"
    path = Path(hermes_home) / "pending" / "memory" / f"{pending_id}.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"id": pending_id, "subsystem": "memory", "payload": {"action": "add"}}),
        encoding="utf-8",
    )
    real_is_symlink = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda candidate: candidate == path or real_is_symlink(candidate))

    assert wa.list_pending("memory") == []
    assert wa.get_pending("memory", pending_id) is None


@pytest.mark.parametrize(
    "record",
    [
        {"id": "deadbeef", "subsystem": "memory", "payload": {"action": "add"}},
        {"id": "0123abcd", "subsystem": "skills", "payload": {"action": "add"}},
        {"id": "0123abcd", "subsystem": "memory", "payload": []},
    ],
)
def test_list_and_get_skip_mismatched_pending_records(hermes_home, record):
    from tools import write_approval as wa

    pending_id = "0123abcd"
    path = Path(hermes_home) / "pending" / "memory" / f"{pending_id}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record), encoding="utf-8")

    assert wa.list_pending("memory") == []
    assert wa.get_pending("memory", pending_id) is None


def test_approve_claims_record_once_under_concurrency(hermes_home, monkeypatch):
    from hermes_cli import write_approval_commands as commands
    from tools import write_approval as wa

    record = wa.stage_write(
        "memory", {"action": "add", "target": "memory", "content": "x"},
        summary="x", origin="foreground",
    )
    assert record is not None
    claimed = threading.Event()
    release = threading.Event()
    calls = []

    def _apply(*_args, **_kwargs):
        calls.append(1)
        claimed.set()
        assert release.wait(timeout=5)
        return True, ""

    monkeypatch.setattr(commands, "_apply_one", _apply)
    outputs = []

    def _approve():
        outputs.append(commands.handle_pending_subcommand("memory", ["approve", record["id"]]))

    first = threading.Thread(target=_approve)
    first.start()
    assert claimed.wait(timeout=5)
    second = threading.Thread(target=_approve)
    second.start()
    second.join(timeout=5)
    release.set()
    first.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert calls == [1]
    assert wa.pending_count("memory") == 0
    assert sum("Approved 1" in output for output in outputs) == 1


def test_failed_approval_releases_claim_for_retry(hermes_home, monkeypatch):
    from hermes_cli import write_approval_commands as commands
    from tools import write_approval as wa

    record = wa.stage_write(
        "memory", {"action": "add", "target": "memory", "content": "x"},
        summary="x", origin="foreground",
    )
    assert record is not None
    monkeypatch.setattr(commands, "_apply_one", lambda *_args, **_kwargs: (False, "apply failed"))

    output = commands.handle_pending_subcommand("memory", ["approve", record["id"]])
    assert "apply failed" in output
    assert wa.get_pending("memory", record["id"]) is not None
    assert wa.pending_count("memory") == 1


def test_release_claim_never_overwrites_a_new_pending_record(hermes_home):
    from tools import write_approval as wa

    record = wa.stage_write(
        "memory", {"action": "add", "target": "memory", "content": "old"},
        summary="old", origin="foreground",
    )
    assert record is not None
    claimed = wa.claim_pending("memory", record["id"])
    assert claimed is not None
    _old, claim_path = claimed

    destination = os.path.join(hermes_home, "pending", "memory", f"{record['id']}.json")
    replacement = {
        "id": record["id"], "subsystem": "memory", "payload": {"action": "add"},
        "summary": "new", "created_at": 0,
    }
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(replacement, handle)

    assert wa.release_claim("memory", record["id"], claim_path) is False
    assert json.loads(open(destination, encoding="utf-8").read())["summary"] == "new"
    assert claim_path.exists()


@pytest.mark.parametrize(
    "foreign_subsystem,foreign_id",
    [("skills", "0123abcd"), ("memory", "deadbeef")],
)
def test_release_claim_rejects_foreign_claim_path(hermes_home, foreign_subsystem, foreign_id):
    from tools import write_approval as wa

    record = wa.stage_write(
        "memory", {"action": "add", "target": "memory", "content": "old"},
        summary="old", origin="foreground",
    )
    assert record is not None
    claimed = wa.claim_pending("memory", record["id"])
    assert claimed is not None
    _old, claim_path = claimed
    foreign = Path(hermes_home) / "pending" / foreign_subsystem / f".{foreign_id}.token.claim"
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_text("foreign", encoding="utf-8")

    assert wa.release_claim("memory", record["id"], foreign) is False
    assert wa.get_pending("memory", record["id"]) is None
    assert claim_path.exists()
    assert foreign.exists()


def test_release_claim_reports_published_retry_when_cleanup_fails(hermes_home, monkeypatch):
    from tools import write_approval as wa

    record = wa.stage_write(
        "memory", {"action": "add", "target": "memory", "content": "old"},
        summary="old", origin="foreground",
    )
    assert record is not None
    claimed = wa.claim_pending("memory", record["id"])
    assert claimed is not None
    _old, claim_path = claimed
    real_unlink = Path.unlink

    def _fail_only_claim(path, *args, **kwargs):
        if path == claim_path:
            raise OSError("cleanup blocked")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _fail_only_claim)
    assert wa.release_claim("memory", record["id"], claim_path) is None
    assert wa.get_pending("memory", record["id"]) is not None
    assert claim_path.exists()


def test_failed_approval_keeps_retry_available_when_claim_cleanup_fails(hermes_home, monkeypatch):
    from hermes_cli import write_approval_commands as commands
    from tools import write_approval as wa

    record = wa.stage_write(
        "memory", {"action": "add", "target": "memory", "content": "x"},
        summary="x", origin="foreground",
    )
    assert record is not None
    monkeypatch.setattr(commands, "_apply_one", lambda *_args, **_kwargs: (False, "apply failed"))
    monkeypatch.setattr(wa, "release_claim", lambda *_args, **_kwargs: None)

    output = commands.handle_pending_subcommand("memory", ["approve", record["id"]])
    assert "retry remains available" in output
    assert "retry is held" not in output


def test_reject_cleanup_failure_reports_manual_recovery(hermes_home, monkeypatch):
    from hermes_cli import write_approval_commands as commands
    from tools import write_approval as wa

    record = wa.stage_write(
        "memory", {"action": "add", "target": "memory", "content": "x"},
        summary="x", origin="foreground",
    )
    assert record is not None
    monkeypatch.setattr(wa, "complete_claim", lambda _claim: False)

    output = commands.handle_pending_subcommand("memory", ["reject", record["id"]])
    assert "manual recovery" in output
    assert wa.pending_count("memory") == 0


def test_claim_rejects_mismatched_subsystem_record(hermes_home):
    from tools import write_approval as wa

    pending_dir = os.path.join(hermes_home, "pending", "memory")
    os.makedirs(pending_dir)
    pending_id = "0123abcd"
    with open(os.path.join(pending_dir, f"{pending_id}.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {"id": pending_id, "subsystem": "skills", "payload": {"action": "add"}},
            handle,
        )

    assert wa.claim_pending("memory", pending_id) is None
    assert wa.pending_count("memory") == 0


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


def test_handle_mode_alias_still_works(hermes_home):
    # 'mode' is kept as a back-compat alias for 'approval'.
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    captured = {}
    out = handle_pending_subcommand(
        wa.MEMORY, ["mode", "on"],
        set_mode_fn=lambda enabled: captured.update(enabled=enabled),
    )
    assert captured["enabled"] is True
    assert "on" in out


def test_handle_approval_invalid(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    out = handle_pending_subcommand(wa.MEMORY, ["approval", "bogus"],
                                    set_mode_fn=lambda enabled: None)
    assert "Invalid value" in out


def test_handle_unknown_subcommand_returns_none(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    # An unrecognized /skills subcommand (e.g. 'search') must return None so
    # the CLI falls through to the skills hub.
    out = handle_pending_subcommand(wa.SKILLS, ["search", "foo"])
    assert out is None


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


def test_memory_inline_approve_writes(hermes_home, approval_callback_cleanup):
    from tools.memory_tool import memory_tool, MemoryStore
    from tools.terminal_tool import set_approval_callback
    from tools import write_approval as wa
    _set_approval("memory", True)

    calls = []
    def approve_cb(command, description, **kw):
        calls.append((command, description))
        return "once"
    set_approval_callback(approve_cb)

    store = MemoryStore(); store.load_from_disk()
    r = json.loads(memory_tool("add", "memory", "approved fact", store=store))
    assert r["success"] is True
    assert r.get("staged") is None  # real write, not staged
    assert store.memory_entries == ["approved fact"]
    assert wa.pending_count("memory") == 0
    # The registered callback must actually be invoked (not the input() path).
    assert len(calls) == 1
    assert "approved fact" in calls[0][0]


def test_memory_inline_deny_blocks(hermes_home, approval_callback_cleanup):
    from tools.memory_tool import memory_tool, MemoryStore
    from tools.terminal_tool import set_approval_callback
    from tools import write_approval as wa
    _set_approval("memory", True)
    set_approval_callback(lambda command, description, **kw: "deny")

    store = MemoryStore(); store.load_from_disk()
    r = json.loads(memory_tool("add", "memory", "denied fact", store=store))
    assert r["success"] is False
    assert "denied" in r["error"].lower()
    assert store.memory_entries == []
    assert wa.pending_count("memory") == 0  # denied, not staged


def test_memory_inline_callback_error_stages(hermes_home, approval_callback_cleanup):
    # If the prompt machinery fails, fall back to staging — never drop silently.
    from tools.memory_tool import memory_tool, MemoryStore
    from tools.terminal_tool import set_approval_callback
    from tools import write_approval as wa
    _set_approval("memory", True)
    def broken_cb(command, description, **kw):
        raise RuntimeError("boom")
    set_approval_callback(broken_cb)

    store = MemoryStore(); store.load_from_disk()
    r = json.loads(memory_tool("add", "memory", "fallback fact", store=store))
    assert r.get("staged") is True
    assert wa.pending_count("memory") == 1


def test_gateway_context_stages_not_prompts(hermes_home, monkeypatch):
    # A gateway session has no per-thread CLI callback; the dangerous-command
    # /approve round-trip lives in the pending-queue machinery which the gate
    # does not use. The gate must stage, never attempt an inline prompt
    # (which would hit the input() fallback and silently deny).
    from tools.memory_tool import memory_tool, MemoryStore
    from tools import write_approval as wa
    _set_approval("memory", True)
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")

    store = MemoryStore(); store.load_from_disk()
    r = json.loads(memory_tool("add", "memory", "gateway fact", store=store))
    assert r.get("staged") is True
    assert store.memory_entries == []
    assert wa.pending_count("memory") == 1


def test_skills_never_prompt_inline_even_with_callback(hermes_home, approval_callback_cleanup):
    # Skills always stage — even when an interactive callback is registered.
    from tools.skill_manager_tool import skill_manage
    from tools.terminal_tool import set_approval_callback
    from tools import write_approval as wa
    _set_approval("skills", True)

    calls = []
    set_approval_callback(lambda c, d, **kw: calls.append(1) or "once")

    r = json.loads(skill_manage(
        action="create", name="test-inline-skill",
        content="---\nname: test-inline-skill\ndescription: x\n---\nbody\n"))
    assert r.get("staged") is True
    assert calls == []  # never prompted
    assert wa.pending_count("skills") == 1


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

    def test_large_content_reports_kb(self):
        from tools import write_approval as wa
        content = "x" * 2048  # >= 1024 bytes -> KB rounding
        assert wa.skill_gist("create", "big", content=content) == "create 'big' (3 KB)"

    def test_create_without_content_falls_through(self):
        from tools import write_approval as wa
        assert wa.skill_gist("create", "demo") == "create 'demo'"

    def test_patch_counts_lines(self):
        from tools import write_approval as wa
        assert (
            wa.skill_gist("patch", "demo", file_path="SKILL.md",
                          old_string="a\nb", new_string="x\ny\nz")
            == "patch 'demo' SKILL.md (+3/-2 lines)"
        )

    def test_patch_defaults_target_and_empty_strings(self):
        from tools import write_approval as wa
        assert wa.skill_gist("patch", "demo") == "patch 'demo' SKILL.md (+0/-0 lines)"

    def test_file_actions_and_unknown_fallback(self):
        from tools import write_approval as wa
        assert wa.skill_gist("write_file", "demo", file_path="a.py") == "write a.py in 'demo'"
        assert wa.skill_gist("remove_file", "demo", file_path="a.py") == "remove a.py from 'demo'"
        assert wa.skill_gist("delete", "demo") == "delete skill 'demo'"
        assert wa.skill_gist("unknown", "demo") == "unknown 'demo'"
