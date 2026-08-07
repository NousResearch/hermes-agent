"""Tests for the memory/skill write-approval gate (tools/write_approval.py)
and the shared slash-command handlers (hermes_cli/write_approval_commands.py).

Covers the boolean write_approval gate (off by default = write freely; on =
require approval) for both subsystems, the foreground-vs-background staging
split, pending store CRUD, and the list/approve/reject/diff/approval
subcommand dispatch.
"""

import asyncio
import json
import os
import shutil
import tempfile
import threading
import time
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

_MISSING = object()


def _pending_path(hermes_home, subsystem, pending_id):
    path = Path(hermes_home) / "pending" / subsystem / f"{pending_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_pending_record(
    hermes_home,
    subsystem,
    pending_id,
    *,
    created_at=_MISSING,
    summary=None,
    origin="foreground",
    action="write_file",
    payload=None,
    mtime=None,
):
    record = {
        "id": pending_id,
        "subsystem": subsystem,
        "action": action,
        "summary": summary or pending_id,
        "origin": origin,
        "payload": payload or {"action": action, "name": f"{pending_id}-skill", "file_path": "notes.txt", "file_content": pending_id},
    }
    if created_at is not _MISSING:
        record["created_at"] = created_at
    path = _pending_path(hermes_home, subsystem, pending_id)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _set_skill_policy(max_pending, ttl_days):
    import hermes_cli.config as cfg

    c = cfg.load_config()
    c.setdefault("skills", {})["write_approval_max_pending"] = max_pending
    c.setdefault("skills", {})["write_approval_ttl_days"] = ttl_days
    cfg.save_config(c)


def _skill_pending_ids(hermes_home):
    pending_dir = Path(hermes_home) / "pending" / "skills"
    if not pending_dir.exists():
        return []
    return sorted(path.stem for path in pending_dir.glob("*.json"))


def _create_skill(hermes_home, name="test-skill", content=_SKILL):
    skill_dir = Path(hermes_home) / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


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


def test_issue_75130_backlog_is_bounded_and_observable(hermes_home, caplog):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa

    caplog.set_level("INFO", logger="tools.write_approval")
    now = time.time()
    for i in range(357):
        created_at = now - ((356 - i) * 2034)
        _write_pending_record(
            hermes_home,
            wa.SKILLS,
            f"s{i:03d}",
            created_at=created_at,
            summary=f"skill backlog {i}",
            payload={"action": "write_file", "name": f"skill-{i}", "file_path": "notes.txt", "file_content": f"entry {i}"},
        )

    out = handle_pending_subcommand(wa.SKILLS, ["pending"])
    records = wa.list_pending(wa.SKILLS)

    assert "Pending skills writes (100):" in out
    assert "Cleanup removed 257 overflow pending skill write(s)." in out
    assert len(records) == 100
    assert wa.pending_count(wa.SKILLS) == 100
    assert len(_skill_pending_ids(hermes_home)) == 100
    assert _skill_pending_ids(hermes_home)[0] == "s257"
    assert _skill_pending_ids(hermes_home)[-1] == "s356"
    assert "retained=100" in caplog.text
    assert "s000" in caplog.text
    assert "s256" in caplog.text


def test_skill_pending_ttl_boundary(hermes_home, monkeypatch):
    from tools import write_approval as wa

    now = 1_800_000_000.0
    monkeypatch.setattr(wa.time, "time", lambda: now)

    _write_pending_record(hermes_home, wa.SKILLS, "exact", created_at=now - (30 * 86400))
    _write_pending_record(hermes_home, wa.SKILLS, "expired", created_at=now - (30 * 86400) - 1)
    _write_pending_record(hermes_home, wa.SKILLS, "future", created_at=now + 60)

    records = wa.list_pending(wa.SKILLS)
    ids = [record["id"] for record in records]

    assert ids == ["exact", "future"]
    assert wa.pending_count(wa.SKILLS) == 2
    assert wa.get_pending(wa.SKILLS, "expired") is None
    assert wa.get_pending(wa.SKILLS, "exact")["id"] == "exact"
    assert wa.get_pending(wa.SKILLS, "future")["id"] == "future"
    assert _skill_pending_ids(hermes_home) == ["exact", "future"]


def test_skill_pending_policy_defaults_overrides_and_invalid_values(hermes_home, monkeypatch):
    from tools import write_approval as wa

    assert wa._resolve_skill_pending_policy() == (100, 30.0)

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"skills": {"write_approval_max_pending": 7, "write_approval_ttl_days": 9}},
    )
    assert wa._resolve_skill_pending_policy() == (7, 9.0)

    for bad_max in (0, -1, False, "bad", 10**1000):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda bad_max=bad_max: {
                "skills": {"write_approval_max_pending": bad_max, "write_approval_ttl_days": 9}
            },
        )
        assert wa._resolve_skill_pending_policy() == (100, 9.0)

    for bad_ttl in (0, -2, False, "bad", 10**1000):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda bad_ttl=bad_ttl: {
                "skills": {"write_approval_max_pending": 7, "write_approval_ttl_days": bad_ttl}
            },
        )
        assert wa._resolve_skill_pending_policy() == (7, 30.0)


def test_skill_pending_approval_preserves_success_and_failure(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa

    skill_dir = _create_skill(hermes_home, "test-skill")

    ok_record = wa.stage_write(
        wa.SKILLS,
        {"action": "write_file", "name": "test-skill", "file_path": "references/notes.txt", "file_content": "approved"},
        summary="approved write",
        origin="foreground",
    )
    fail_record = wa.stage_write(
        wa.SKILLS,
        {"action": "write_file", "name": "missing-skill", "file_path": "references/notes.txt", "file_content": "still pending"},
        summary="missing skill write",
        origin="foreground",
    )

    out = handle_pending_subcommand(wa.SKILLS, ["approve", "all"])

    assert "Approved 1 skills write(s)." in out
    assert f"{fail_record['id']}:" in out
    assert wa.pending_count(wa.SKILLS) == 1
    assert wa.get_pending(wa.SKILLS, ok_record["id"]) is None
    assert wa.get_pending(wa.SKILLS, fail_record["id"])["id"] == fail_record["id"]
    assert (skill_dir / "references" / "notes.txt").read_text(encoding="utf-8") == "approved"


def test_skill_pending_lifecycle_leaves_memory_unchanged(hermes_home):
    from tools import write_approval as wa

    base = time.time() - (45 * 86400)
    for i in range(120):
        _write_pending_record(
            hermes_home,
            wa.MEMORY,
            f"m{i:03d}",
            created_at=base + i,
            action="add",
            payload={"action": "add", "target": "memory", "content": f"memory {i}"},
        )

    before_ids = [record["id"] for record in wa.list_pending(wa.MEMORY)]
    before_count = wa.pending_count(wa.MEMORY)

    skill_record = wa.stage_write(
        wa.SKILLS,
        {"action": "write_file", "name": "test-skill", "file_path": "notes.txt", "file_content": "queued"},
        summary="queued skill",
        origin="foreground",
    )
    assert skill_record["id"]
    assert wa.list_pending(wa.SKILLS)
    assert wa.get_pending(wa.SKILLS, skill_record["id"])["id"] == skill_record["id"]
    assert wa.pending_count(wa.SKILLS) == 1

    after_ids = [record["id"] for record in wa.list_pending(wa.MEMORY)]
    assert after_ids == before_ids
    assert wa.pending_count(wa.MEMORY) == before_count == 120
    assert len(list((Path(hermes_home) / "pending" / "memory").glob("*.json"))) == 120


def test_skill_pending_legacy_timestamp_and_unreadable_record_are_conservative(hermes_home, caplog):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa

    _set_skill_policy(2, 30)
    now = time.time()

    _write_pending_record(hermes_home, wa.SKILLS, "legacy", created_at=_MISSING, mtime=now - 1000)
    _write_pending_record(hermes_home, wa.SKILLS, "invalid", created_at="bad", mtime=now - 100)
    _write_pending_record(hermes_home, wa.SKILLS, "fresh", created_at=now)

    unreadable = _pending_path(hermes_home, wa.SKILLS, "broken")
    unreadable.write_text("{not json", encoding="utf-8")
    wrong_shape = _pending_path(hermes_home, wa.SKILLS, "wrong-shape")
    wrong_shape.write_text("[]", encoding="utf-8")

    out = handle_pending_subcommand(wa.SKILLS, ["pending"])

    assert "Cleanup removed 1 overflow pending skill write(s)." in out
    assert [record["id"] for record in wa.list_pending(wa.SKILLS)] == ["invalid", "fresh"]
    assert wa.get_pending(wa.SKILLS, "legacy") is None
    assert wa.get_pending(wa.SKILLS, "invalid")["id"] == "invalid"
    assert unreadable.exists()
    assert wrong_shape.exists()
    assert "Skipping unreadable pending record" in caplog.text


def test_skill_pending_stage_list_get_and_count_share_snapshot(hermes_home, monkeypatch):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa

    _set_skill_policy(2, 30)
    now = 1_800_000_000.0
    monkeypatch.setattr(wa.time, "time", lambda: now)

    _write_pending_record(hermes_home, wa.SKILLS, "expired", created_at=now - (31 * 86400))
    _write_pending_record(hermes_home, wa.SKILLS, "older", created_at=now - 200)
    stage = wa.stage_write(
        wa.SKILLS,
        {"action": "write_file", "name": "new-skill", "file_path": "notes.txt", "file_content": "fresh"},
        summary="fresh skill",
        origin="foreground",
    )

    ids = [record["id"] for record in wa.list_pending(wa.SKILLS)]

    assert ids == ["older", stage["id"]]
    assert wa.pending_count(wa.SKILLS) == 2
    assert wa.get_pending(wa.SKILLS, "expired") is None
    assert wa.get_pending(wa.SKILLS, "older")["id"] == "older"
    assert wa.get_pending(wa.SKILLS, stage["id"])["id"] == stage["id"]

    out = handle_pending_subcommand(wa.SKILLS, ["pending"])
    assert stage["id"] in out
    assert out.count("Cleanup removed 1 expired pending skill write(s).") == 1
    assert "Cleanup removed 1 expired pending skill write(s)." not in handle_pending_subcommand(
        wa.SKILLS, ["pending"]
    )


def test_skill_pending_stage_and_review_serialize_cleanup_counts(hermes_home, monkeypatch):
    from tools import write_approval as wa

    _set_skill_policy(2, 30)
    now = 1_800_000_000.0
    monkeypatch.setattr(wa.time, "time", lambda: now)
    _write_pending_record(hermes_home, wa.SKILLS, "expired", created_at=now - (31 * 86400))

    barrier = threading.Barrier(2)
    snapshots = []

    def stage():
        barrier.wait()
        wa.stage_write(
            wa.SKILLS,
            {"action": "write_file", "name": "new-skill", "file_path": "notes.txt", "file_content": "fresh"},
            summary="fresh skill",
            origin="foreground",
        )

    def review():
        barrier.wait()
        snapshots.append(wa.pending_review_snapshot(wa.SKILLS))

    threads = [threading.Thread(target=stage), threading.Thread(target=review)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    counts = [
        snapshot.get("expired_count", 0)
        for snapshot in snapshots
    ]
    counts.append(wa.pending_review_snapshot(wa.SKILLS).get("expired_count", 0))
    assert sum(counts) == 1


def test_skill_approve_all_serializes_cleanup(hermes_home, monkeypatch):
    import hermes_cli.write_approval_commands as commands
    from tools import write_approval as wa

    _set_skill_policy(1, 30)
    now = 1_800_000_000.0
    monkeypatch.setattr(wa.time, "time", lambda: now)
    _write_pending_record(hermes_home, wa.SKILLS, "pending", created_at=now)

    apply_started = threading.Event()
    release_apply = threading.Event()

    def blocked_apply(subsystem, record, memory_store):
        apply_started.set()
        assert release_apply.wait(5)
        return True, ""

    monkeypatch.setattr(commands, "_apply_one", blocked_apply)
    approval = threading.Thread(
        target=commands.handle_pending_subcommand,
        args=(wa.SKILLS, ["approve", "all"]),
    )
    approval.start()
    assert apply_started.wait(5)

    staging = threading.Thread(
        target=wa.stage_write,
        args=(
            wa.SKILLS,
            {"action": "write_file", "name": "new", "file_path": "new.txt", "file_content": "fresh"},
        ),
        kwargs={"summary": "new skill", "origin": "foreground"},
    )
    staging.start()
    time.sleep(0.05)
    assert staging.is_alive()

    release_apply.set()
    approval.join(timeout=5)
    staging.join(timeout=5)
    assert not approval.is_alive()
    assert not staging.is_alive()


@pytest.mark.parametrize("command", ["/skills", "/skills pending"])
def test_gateway_pending_list_keeps_cleanup_notice_when_gate_is_off(hermes_home, command):
    from gateway.platforms.base import MessageEvent
    from gateway.slash_commands import GatewaySlashCommandsMixin
    from tools import write_approval as wa

    _write_pending_record(
        hermes_home,
        wa.SKILLS,
        "expired",
        created_at=time.time() - (31 * 86400),
    )
    runner = GatewaySlashCommandsMixin.__new__(GatewaySlashCommandsMixin)
    runner._session_key_for_source = lambda source: "test-session"

    out = asyncio.run(runner._handle_skills_command(MessageEvent(text=command)))

    assert "Cleanup removed 1 expired pending skill write(s)." in out
    assert out.count("Cleanup removed 1 expired pending skill write(s).") == 1


def test_gateway_pending_list_surfaces_stage_cleanup_notice(hermes_home):
    from gateway.platforms.base import MessageEvent
    from gateway.slash_commands import GatewaySlashCommandsMixin
    from tools import write_approval as wa

    _write_pending_record(
        hermes_home,
        wa.SKILLS,
        "expired",
        created_at=time.time() - (31 * 86400),
    )
    wa.stage_write(
        wa.SKILLS,
        {"action": "write_file", "name": "new", "file_path": "new.txt", "file_content": "fresh"},
        summary="new skill",
        origin="foreground",
    )
    runner = GatewaySlashCommandsMixin.__new__(GatewaySlashCommandsMixin)
    runner._session_key_for_source = lambda source: "test-session"

    out = asyncio.run(
        runner._handle_skills_command(MessageEvent(text="/skills pending"))
    )

    assert out.count("Cleanup removed 1 expired pending skill write(s).") == 1


def test_skill_pending_legacy_record_uses_filename_as_id(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa

    _create_skill(hermes_home, "test-skill")
    path = _write_pending_record(
        hermes_home,
        wa.SKILLS,
        "legacy",
        created_at=_MISSING,
        payload={
            "action": "write_file",
            "name": "test-skill",
            "file_path": "references/legacy.txt",
            "file_content": "approved",
        },
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    del record["id"]
    path.write_text(json.dumps(record), encoding="utf-8")

    pending = handle_pending_subcommand(wa.SKILLS, ["pending"])
    approved = handle_pending_subcommand(wa.SKILLS, ["approve", "legacy"])

    assert "legacy" in pending
    assert "Approved 1 skills write(s)." in approved
    assert not path.exists()
