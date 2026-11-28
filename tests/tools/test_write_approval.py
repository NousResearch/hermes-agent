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

def test_list_pending_skips_non_dict_record(hermes_home):
    """A parseable-but-non-object pending file must be skipped, not crash the sort."""
    from tools import write_approval as wa
    wa.stage_write("memory", {"action": "add", "target": "user", "content": "ok"},
                   summary="ok", origin="foreground")
    pending_dir = wa._pending_path("memory", "").parent
    (pending_dir / "bad.json").write_text('"not a record"', encoding="utf-8")
    records = wa.list_pending("memory")
    assert len(records) == 1 and records[0]["payload"]["content"] == "ok"
    assert wa.get_pending("memory", "bad") is None

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
    from tools.memory_tool import MemoryStore, load_on_disk_store

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
    defaults = MemoryStore()
    assert fallback.memory_char_limit == defaults.memory_char_limit
    assert fallback.user_char_limit == defaults.user_char_limit
    assert fallback.memory_enabled is True
    assert fallback.user_profile_enabled is True

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

def test_handle_approve_surfaces_overwritten_entry(hermes_home):
    """#117952: on the /memory approve surface a partial-entry replace must show the
    approver the FULL entry it overwrote — the store's replaced_entries field used to be
    dropped by _apply_one, so the incident path stayed silent."""
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools.memory_tool import MemoryStore
    from tools import write_approval as wa
    store = MemoryStore(); store.load_from_disk()
    entry = "RULE A: gate merges. RULE B: ci per HEAD. RULE C: never squash."
    store.add("memory", entry)
    wa.stage_write("memory", {"action": "batch", "target": "memory", "operations": [
        {"action": "replace", "old_text": "RULE B: ci per HEAD.", "content": "RULE B: CI is per-head.",
         "matched_entry": entry}]},
        summary="batch", origin="background_review")
    out = handle_pending_subcommand(wa.MEMORY, ["approve", "all"], memory_store=store)
    assert "Approved 1" in out and entry in out
    assert store.memory_entries == ["RULE B: CI is per-head."]


def test_handle_approve_lists_a_patch_as_the_entry_before_the_edit(hermes_home):
    """An approved patch keeps the rest of its entry; the approve output called it an
    overwrite and told the user to re-add text they had deliberately changed."""
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools.memory_tool import MemoryStore
    from tools import write_approval as wa
    store = MemoryStore(); store.load_from_disk()
    entry = "RULE A: gate merges. RULE B: ci per HEAD. RULE C: never squash."
    store.add("memory", entry)
    wa.stage_write("memory", {"action": "batch", "target": "memory", "operations": [
        {"action": "patch", "old_text": "ci per HEAD", "content": "CI is per-head", "matched_entry": entry}]},
        summary="batch", origin="background_review")
    out = handle_pending_subcommand(wa.MEMORY, ["approve", "all"], memory_store=store)
    assert "Patched entry (as it was before the edit):" in out and entry in out
    assert "Overwrote" not in out
    assert store.memory_entries == ["RULE A: gate merges. RULE B: CI is per-head. RULE C: never squash."]


_KEPT = "Repo lives in ~/src/app; tests via make test"
_REVIEWED = "Staging DB: pg-staging-2 (old cluster, retiring)"


def _review_stages_remove(shape):
    """Seed memory, then stage a remove the way the unattended background review does."""
    from tools.memory_tool import MemoryStore, memory_tool
    from tools.skill_provenance import (reset_current_write_origin, reset_review_attended,
                                        set_current_write_origin, set_review_attended)
    store = MemoryStore(); store.load_from_disk()
    for entry in (_KEPT, _REVIEWED):
        assert store.add("memory", entry)["success"]
    op = {"action": "remove", "old_text": "Staging DB"}
    kwargs = op if shape == "single" else {"operations": [op, {"action": "add", "content": "Deploys via make ship"}]}
    origin, attended = set_current_write_origin("background_review"), set_review_attended(False)
    try:
        staged = json.loads(memory_tool(target="memory", store=store, **kwargs))
    finally:
        reset_review_attended(attended)
        reset_current_write_origin(origin)
    assert staged["staged"] is True, staged
    return store, staged["pending_id"]


@pytest.mark.parametrize("shape", ["single", "batch"])
def test_approve_refuses_staged_remove_whose_entry_changed(hermes_home, shape):
    """Approval re-ran the staged old_text search against the file as it is THEN, so it
    deleted the newer entry the live agent had written in place, which the approver never saw."""
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools.memory_tool import load_on_disk_store, memory_tool
    from tools import write_approval as wa
    store, pid = _review_stages_remove(shape)
    newer = "Staging DB: pg-staging-3 (migrated 2026-09-20, creds in vault 'stg')"
    assert json.loads(memory_tool(action="patch", old_text=_REVIEWED, content=newer, store=store))["success"]

    out = handle_pending_subcommand(wa.MEMORY, ["approve", pid], memory_store=load_on_disk_store())

    assert load_on_disk_store().memory_entries == [_KEPT, newer], out
    assert "changed since it was staged" in out
    assert wa.get_pending(wa.MEMORY, pid) is not None
    # The pending list shows the whole entry the write targets, not just its search string.
    assert _REVIEWED in handle_pending_subcommand(wa.MEMORY, ["pending"])


@pytest.mark.parametrize("shape", ["single", "batch"])
def test_approve_names_the_entry_a_remove_deleted(hermes_home, shape):
    """Approve listed what a replace overwrote but was silent about what a remove deleted."""
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools.memory_tool import load_on_disk_store
    from tools import write_approval as wa
    _store, pid = _review_stages_remove(shape)
    out = handle_pending_subcommand(wa.MEMORY, ["approve", pid], memory_store=load_on_disk_store())
    assert _REVIEWED not in load_on_disk_store().memory_entries, out
    assert _REVIEWED in out


def test_approve_refuses_unpinned_legacy_remove(hermes_home):
    """A record staged before removes were pinned to their full entry has no verifiable target,
    so approve refuses it (keeping the record) instead of replaying its old_text search."""
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools.memory_tool import load_on_disk_store
    from tools import write_approval as wa
    _store, pid = _review_stages_remove("single")
    path = wa._pending_path(wa.MEMORY, pid)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["payload"].pop("matched_entry", None)
    path.write_text(json.dumps(record), encoding="utf-8")
    assert "unpinned legacy target" in handle_pending_subcommand(wa.MEMORY, ["pending"])

    out = handle_pending_subcommand(wa.MEMORY, ["approve", pid], memory_store=load_on_disk_store())

    assert "Approved 0" in out and "predates entry pinning" in out, out
    assert _REVIEWED in load_on_disk_store().memory_entries
    assert wa.get_pending(wa.MEMORY, pid) is not None

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


@pytest.mark.parametrize("context", ["single_query", "cron", "webhook"])
def test_headless_memory_write_stages_without_inline_prompt(
    hermes_home, approval_callback_cleanup, monkeypatch, context
):
    """`hermes chat -q`, cron and unattended platforms can register the CLI callback but have no human to answer
    it: the write stages at once instead of waiting the approval timeout."""
    from tools.memory_tool import memory_tool, MemoryStore
    from tools.terminal_tool import set_approval_callback
    from tools import write_approval as wa

    _set_approval("memory", True)
    monkeypatch.setenv(*{"single_query": ("HERMES_SINGLE_QUERY_SESSION", "1"), "cron": ("HERMES_CRON_SESSION", "1"),
                         "webhook": ("HERMES_SESSION_PLATFORM", "webhook")}[context])

    calls = []

    def approve_if_prompted(*args, **kwargs):
        calls.append((args, kwargs))
        return "once"

    set_approval_callback(approve_if_prompted)
    store = MemoryStore()
    store.load_from_disk()

    r = json.loads(memory_tool("add", "memory", "headless fact", store=store))

    assert calls == []
    assert r["success"] is True
    assert r["staged"] is True
    assert store.memory_entries == []
    assert wa.pending_count("memory") == 1


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


# ---------------------------------------------------------------------------
# Staged-write review hint is surface-aware (#98330)
# ---------------------------------------------------------------------------

def _stage_one_memory_write():
    from tools.memory_tool import memory_tool, MemoryStore
    store = MemoryStore(); store.load_from_disk()
    r = json.loads(memory_tool("add", "memory", "surface hint fact", store=store))
    assert r.get("staged") is True, r
    return r


def test_staged_hint_names_the_review_command_on_slash_surfaces(hermes_home):
    from tools import write_approval as wa
    _set_approval("memory", True)
    r = _stage_one_memory_write()
    # No headless markers bound (plain foreground turn) → the slash hint stands.
    assert "/memory pending" in r["message"], r["message"]
    assert wa.pending_count("memory") == 1


def test_staged_hint_names_the_pending_dir_on_headless_surfaces(hermes_home, monkeypatch):
    from tools import write_approval as wa
    _set_approval("memory", True)
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    r = _stage_one_memory_write()
    assert "review with /memory pending" not in r["message"], r["message"]
    assert "pending records live in" in r["message"]
    assert str(wa._pending_path(wa.MEMORY, "").parent) in r["message"]
    monkeypatch.delenv("HERMES_CRON_SESSION")

    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-7")
    r = _stage_one_memory_write()
    assert "review with /memory pending" not in r["message"], r["message"]
    monkeypatch.delenv("HERMES_KANBAN_TASK")

    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "api_server")
    r = _stage_one_memory_write()
    assert "review with /memory pending" not in r["message"], r["message"]
    assert "pending records live in" in r["message"]
    monkeypatch.delenv("HERMES_SESSION_PLATFORM")
    assert wa.pending_count("memory") == 3


def test_staged_hint_keeps_command_for_chat_gateway_platform(hermes_home, monkeypatch):
    _set_approval("memory", True)
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    r = _stage_one_memory_write()
    assert "/memory pending" in r["message"], r["message"]


# ---------------------------------------------------------------------------
# Skill write-approval attribution through staging + replay
# ---------------------------------------------------------------------------

_SKILL_DOC = "---\nname: {name}\ndescription: Use when testing staged approval.\n---\n\nInitial rule.\n"


def _stage_skill_in_review_fork(hermes_home, monkeypatch, *, explicit: bool, focus, call_fn):
    """Execute ``call_fn()`` inside a real ``spawn_background_review_thread`` fork turn."""
    from pathlib import Path
    from types import SimpleNamespace
    from typing import Any
    from agent import background_review
    from run_agent import AIAgent

    skills_dir = Path(hermes_home) / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("agent.skill_utils.get_all_skills_dirs", lambda: [skills_dir])
    monkeypatch.setattr("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None)
    out = {}

    def _create(**_kw):
        out["staged"] = call_fn()
        msg = SimpleNamespace(content="Done.", reasoning_content=None, reasoning=None, tool_calls=None)
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")], model="test-model", usage=usage,
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create)),
        close=lambda: None,
    )
    monkeypatch.setattr("agent.process_bootstrap._OPENAI_CLS_CACHE", lambda **_kw: client)

    parent: Any = AIAgent(
        api_key="test-key", base_url="http://127.0.0.1:9/v1", model="test-model",
        quiet_mode=True, session_id="sess-wa", skip_context_files=True, skip_memory=True,
    )
    parent._cached_system_prompt = "stable parent prompt"
    import contextvars
    target, _prompt = background_review.spawn_background_review_thread(
        parent, [], review_memory=False, review_skills=True, focus=focus, task_cfg={}, explicit=explicit,
    )
    contextvars.copy_context().run(target)
    return out["staged"]


@pytest.mark.parametrize(
    ("explicit", "focus", "expected_trigger"),
    [
        (False, None, "automatic"),
        (True, None, "refine"),
        (False, "tighten error handling", "focus_only"),
    ],
)
def test_approved_background_review_skill_write_preserves_curator_and_trigger(
    hermes_home, monkeypatch, explicit, focus, expected_trigger
):
    """With ``skills.write_approval = true``, every background-review skill write stages first;
    approving it from the foreground CLI must still record ``actor == 'curator'`` and the fork's
    ``review_trigger`` in the permanent ledger without re-arming the fork's interactive guards."""
    from pathlib import Path
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import skill_ledger, write_approval as wa
    from tools.skill_manager_guards import mark_background_review_skill_read
    from tools.skill_manager_tool import skill_manage

    skills_dir = Path(hermes_home) / "skills"
    monkeypatch.setattr("agent.skill_utils.get_all_skills_dirs", lambda: [skills_dir])

    # Seed an agent-created skill before turning the approval gate on, then stage a patch from the
    # review fork so replay also proves the fork's skill_view read guard stays off for the approver.
    def _seed():
        return json.loads(skill_manage(
            action="", name="",
            operations=[{"action": "create", "name": "review-skill",
                         "content": _SKILL_DOC.format(name="review-skill")}],
        ))

    assert _stage_skill_in_review_fork(
        hermes_home, monkeypatch, explicit=explicit, focus=focus, call_fn=_seed,
    )["success"] is True

    _set_approval("skills", True)

    def _stage_patch():
        mark_background_review_skill_read(skills_dir / "review-skill" / "SKILL.md")
        return json.loads(skill_manage(
            action="", name="",
            operations=[{"action": "patch", "name": "review-skill",
                         "old_string": "Initial rule.", "new_string": "Curated rule."}],
        ))

    staged = _stage_skill_in_review_fork(
        hermes_home, monkeypatch, explicit=explicit, focus=focus, call_fn=_stage_patch,
    )
    assert staged.get("staged") is True, staged
    pid = staged["pending_id"]

    # Foreground approval (no skill_view in this context) applies the write and preserves provenance.
    out = handle_pending_subcommand(wa.SKILLS, ["approve", pid]) or ""
    assert "Approved 1 skills write(s)." in out, out
    assert "Curated rule." in (skills_dir / "review-skill" / "SKILL.md").read_text(encoding="utf-8")

    patch_rows = [r for r in skill_ledger.list_entries(skill="review-skill") if r["action"] == "patch"]
    assert len(patch_rows) == 1
    assert (patch_rows[0]["actor"], patch_rows[0]["evidence"].get("review_trigger")) == (
        "curator",
        expected_trigger,
    )


def test_approved_foreground_skill_write_keeps_agent_actor_without_review_trigger(
    hermes_home, monkeypatch
):
    """A foreground staged skill write approved alongside (or after) a background-review write
    still records ``actor == 'agent'`` and carries no ``review_trigger``."""
    from pathlib import Path
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import skill_ledger, write_approval as wa
    from tools.skill_manager_tool import skill_manage

    skills_dir = Path(hermes_home) / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("agent.skill_utils.get_all_skills_dirs", lambda: [skills_dir])
    _set_approval("skills", True)

    bg_staged = _stage_skill_in_review_fork(
        hermes_home, monkeypatch, explicit=False, focus=None,
        call_fn=lambda: json.loads(skill_manage(
            action="", name="",
            operations=[{"action": "create", "name": "bg-skill",
                         "content": _SKILL_DOC.format(name="bg-skill")}],
        )),
    )
    fg_staged = json.loads(skill_manage(
        action="", name="",
        operations=[{"action": "create", "name": "fg-skill",
                     "content": _SKILL_DOC.format(name="fg-skill")}],
    ))
    assert bg_staged.get("staged") is True and fg_staged.get("staged") is True
    fg_record = wa.get_pending(wa.SKILLS, fg_staged["pending_id"])
    assert fg_record is not None and "review_trigger" not in fg_record

    out = handle_pending_subcommand(wa.SKILLS, ["approve", "all"]) or ""
    assert "Approved 2 skills write(s)." in out, out

    (fg_entry,) = skill_ledger.list_entries(skill="fg-skill")
    assert fg_entry["actor"] == "agent"
    assert "review_trigger" not in fg_entry["evidence"]
    (bg_entry,) = skill_ledger.list_entries(skill="bg-skill")
    assert (bg_entry["actor"], bg_entry["evidence"].get("review_trigger")) == ("curator", "automatic")
