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
    pending_dir = wa._pending_dir("memory")
    (pending_dir / "bad.json").write_text('"not a record"', encoding="utf-8")
    (pending_dir / "0badf00d.json").write_text('"not a record"', encoding="utf-8")  # a minted-format name
    records = wa.list_pending("memory")
    assert len(records) == 1 and records[0]["payload"]["content"] == "ok"
    assert wa.get_pending("memory", "bad") is None and wa.get_pending("memory", "0badf00d") is None

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
    assert str(wa._pending_dir(wa.MEMORY)) in r["message"]
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
# Pending store integrity: only minted records inside pending/<subsystem>/
# ---------------------------------------------------------------------------

def test_pending_ids_outside_the_minted_format_are_refused(hermes_home):
    """approve|reject|diff take the id as typed (#119997): a path-like id must not read, apply or
    delete any file outside pending/<subsystem>/."""
    from pathlib import Path
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    wa.stage_write(wa.SKILLS, {"action": "create", "name": "probe", "content": "x"},
                   summary="create probe", origin="foreground")  # pending/skills/ exists
    victim = Path(hermes_home, "victim.json")
    victim.write_text('{"id": "victim", "summary": "VICTIM", "payload": {}}', encoding="utf-8")
    for bad in ("../../victim", str(victim.with_suffix("")), "a/b", "../skills/x"):
        for sub in ("approve", "reject", "diff"):
            out = handle_pending_subcommand(wa.SKILLS, [sub, bad])
            assert out.startswith("No pending") and "VICTIM" not in out, (sub, bad, out)
        assert wa.get_pending(wa.SKILLS, bad) is None
        assert wa.discard_pending(wa.SKILLS, bad) is False
    assert victim.exists() and wa.pending_count(wa.SKILLS) == 1
    with pytest.raises(ValueError):
        wa._pending_path("../memory", "0123abcd")


def test_symlinks_planted_in_the_pending_store_are_never_followed(hermes_home, tmp_path):
    """A record file or the subsystem dir linked out of the store must not be read, applied, deleted
    or written through."""
    from pathlib import Path
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.json"
    secret.write_text('{"id": "deadbeef", "subsystem": "skills", "summary": "OUTSIDE", "payload": '
                      '{"action": "create", "name": "x", "content": "outside text"}}', encoding="utf-8")
    pending = Path(hermes_home, "pending", "skills")
    pending.mkdir(parents=True)
    (pending / "deadbeef.json").symlink_to(secret)                       # a linked record
    assert wa.get_pending(wa.SKILLS, "deadbeef") is None and wa.list_pending(wa.SKILLS) == []
    for sub in ("diff", "approve", "reject"):
        assert "OUTSIDE" not in handle_pending_subcommand(wa.SKILLS, [sub, "deadbeef"])
    assert wa.discard_pending(wa.SKILLS, "deadbeef") is False and wa.pending_count(wa.SKILLS) == 0
    assert secret.exists() and wa._find_skill_path("x") is None
    (pending / "deadbeef.json").unlink()
    pending.rmdir()
    victim = outside / "cafebabe.json"
    victim.write_text('{"id": "cafebabe"}', encoding="utf-8")
    pending.symlink_to(outside)                                          # a linked subsystem dir
    assert wa.discard_pending(wa.SKILLS, "cafebabe") is False and victim.exists()
    assert wa.list_pending(wa.SKILLS) == [] and wa.pending_count(wa.SKILLS) == 0
    from tools.memory_tool import MemoryStore, memory_tool
    from tools.skill_manager_tool import skill_manage
    _set_approval("skills", True)
    _set_approval("memory", True)
    Path(hermes_home, "pending", "memory").symlink_to(outside)           # nothing stored: the tools say so
    store = MemoryStore()
    store.load_from_disk()
    for out in (skill_manage(action="create", name="probe", content="---\nname: probe\ndescription: p\n---\nx\n"),
                memory_tool("add", "memory", "a fact", store=store)):
        assert json.loads(out)["success"] is False and "staged" not in json.loads(out), out
    wa.stage_write(wa.SKILLS, {"action": "create", "name": "probe", "content": "x"},
                   summary="create probe", origin="foreground")
    assert sorted(p.name for p in outside.iterdir()) == ["cafebabe.json", "secret.json"]


def test_a_record_is_known_by_its_file_name_not_the_id_written_inside_it(hermes_home):
    """A forged 'id' inside a pending file must not make approve apply one record and delete another,
    nor leave the applied record pending to be replayed."""
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    beta = wa.stage_write(wa.SKILLS, {"action": "create", "name": "beta",
                                      "content": "---\nname: beta\ndescription: b\n---\nb\n"},
                          summary="create beta", origin="foreground")
    gamma = wa.stage_write(wa.SKILLS, {"action": "create", "name": "gamma",
                                       "content": "---\nname: gamma\ndescription: g\n---\ng\n"},
                           summary="create gamma", origin="foreground")
    path = wa._pending_path(wa.SKILLS, beta["id"])
    forged = json.loads(path.read_text(encoding="utf-8"))
    forged["id"] = gamma["id"]
    path.write_text(json.dumps(forged), encoding="utf-8")
    assert sorted(r["id"] for r in wa.list_pending(wa.SKILLS)) == sorted([beta["id"], gamma["id"]])
    assert wa.get_pending(wa.SKILLS, beta["id"])["id"] == beta["id"]
    handle_pending_subcommand(wa.SKILLS, ["approve", beta["id"]])
    assert wa.get_pending(wa.SKILLS, beta["id"]) is None             # applied and gone, not left to replay
    assert wa.get_pending(wa.SKILLS, gamma["id"]) is not None        # the unreviewed record is untouched


def test_skills_diff_never_reads_a_staged_path_outside_the_skill(hermes_home, tmp_path):
    """A staged write_file/patch names its own file_path; /skills diff must not show a file outside
    the skill as the diff's current side, alone or inside a staged batch."""
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    rec = wa.stage_write(wa.SKILLS, {"action": "create", "name": "probe",
                                     "content": "---\nname: probe\ndescription: p\n---\nbody\n"},
                         summary="create probe", origin="foreground")
    handle_pending_subcommand(wa.SKILLS, ["approve", rec["id"]])
    secret = tmp_path / "private.txt"
    secret.write_text("PRIVATE-LINE\n", encoding="utf-8")
    skill_dir = wa._find_skill_path("probe")
    rel = os.path.relpath(secret, skill_dir)
    for bad in (str(secret), rel, "references/../../" + rel):
        staged = wa.stage_write(wa.SKILLS, {"action": "write_file", "name": "probe", "file_path": bad,
                                            "file_content": "new\n"}, summary="write", origin="foreground")
        out = wa.skill_pending_diff(staged)
        assert "PRIVATE-LINE" not in out and out.startswith("(refused"), (bad, out)
        batch = {"payload": {"action": "batch", "operations": [
            {"action": "patch", "name": "probe", "file_path": bad, "old_string": "PRIVATE", "new_string": "X"},
            {"action": "write_file", "name": "probe", "file_path": bad, "file_content": "new\n"}]}}
        assert "PRIVATE-LINE" not in wa.skill_pending_diff(batch), bad
    ok = wa.stage_write(wa.SKILLS, {"action": "write_file", "name": "probe", "file_path": "references/notes.md",
                                    "file_content": "new\n"}, summary="write", origin="foreground")
    assert "+new" in wa.skill_pending_diff(ok)


def test_two_approvals_of_one_pending_write_apply_it_once(hermes_home, monkeypatch):
    """Approve claims the record atomically: a second approval while the first is applying applies
    nothing, and an approval that fails puts the record back in the queue."""
    import threading
    import hermes_cli.write_approval_commands as wac
    from tools import write_approval as wa
    rec = wa.stage_write(wa.SKILLS, {"action": "create", "name": "probe",
                                     "content": "---\nname: probe\ndescription: p\n---\n# body\n"},
                         summary="create probe", origin="foreground")
    wac.handle_pending_subcommand(wa.SKILLS, ["approve", rec["id"]])
    patch = wa.stage_write(wa.SKILLS, {"action": "patch", "name": "probe", "old_string": "# body",
                                       "new_string": "# body\nADDED-LINE"}, summary="patch", origin="foreground")
    real, calls = wac._apply_one, []
    first_in, second_done = threading.Event(), threading.Event()

    def held(*a, **k):  # the first apply stays open until the second approval has returned
        calls.append(1)
        if len(calls) == 1:
            first_in.set()
            second_done.wait(timeout=10)
        return real(*a, **k)
    monkeypatch.setattr(wac, "_apply_one", held)
    outs = []
    first = threading.Thread(target=lambda: outs.append(
        wac.handle_pending_subcommand(wa.SKILLS, ["approve", patch["id"]])))
    first.start()
    assert first_in.wait(timeout=10)
    outs.append(wac.handle_pending_subcommand(wa.SKILLS, ["approve", patch["id"]]))
    second_done.set()
    first.join(timeout=10)
    body = (wa._find_skill_path("probe") / "SKILL.md").read_text(encoding="utf-8")
    assert body.count("ADDED-LINE") == 1 and len(calls) == 1, outs
    assert sorted(o.startswith("Approved 1 ") for o in outs) == [False, True], outs  # the loser applied nothing
    assert wa.get_pending(wa.SKILLS, patch["id"]) is None and not list(wa._pending_dir(wa.SKILLS).glob("*.claimed"))

    failing = wa.stage_write(wa.SKILLS, {"action": "create", "name": "probe2", "content": "x"},
                             summary="create probe2", origin="foreground")
    monkeypatch.setattr(wac, "_apply_one", lambda *a, **k: (False, "boom", {}))
    assert "boom" in wac.handle_pending_subcommand(wa.SKILLS, ["approve", failing["id"]])
    assert wa.get_pending(wa.SKILLS, failing["id"]) is not None


def test_a_claim_left_by_a_crash_goes_back_to_the_queue(hermes_home):
    """A process that dies between claim and finish leaves <id>.claimed: listing puts it back once it is
    older than any real apply, and leaves a live claim alone."""
    import time
    from tools import write_approval as wa
    crashed, live = (wa.stage_write(wa.MEMORY, {"action": "add", "target": "memory", "content": c},
                                    summary=c, origin="foreground") for c in ("one", "two"))
    old, d = time.time() - wa._STALE_CLAIM_SECONDS - 60, wa._pending_dir(wa.MEMORY)
    os.utime(d / f"{live['id']}.json", (old, old))                       # staged long ago, claimed now
    assert wa.claim_pending(wa.MEMORY, crashed["id"]) and wa.claim_pending(wa.MEMORY, live["id"])
    os.utime(d / f"{crashed['id']}.claimed", (old, old))
    assert [r["id"] for r in wa.list_pending(wa.MEMORY)] == [crashed["id"]]
    assert wa.claim_pending(wa.MEMORY, live["id"]) is None             # the live claim is still held


# ---------------------------------------------------------------------------
# Rejections are kept with their reason and reported to on_write_rejected
# ---------------------------------------------------------------------------

def _capture_rejections():
    from hermes_cli.plugins import get_plugin_manager
    mgr, events = get_plugin_manager(), []
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("on_write_rejected", []).append(lambda **kw: events.append(kw))
    return mgr, saved, events


def _set_rejected_writes(subsystem, **conf):
    import hermes_cli.config as cfg
    c = cfg.load_config()
    c.setdefault(subsystem, {})["rejected_writes"] = conf
    cfg.save_config(c)


def _reject_one(hermes_home, reason_words):
    """Stage a skill write, reject it with *reason_words*; return (staged, kept file text, hook events)."""
    from pathlib import Path
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    rec = wa.stage_write(wa.SKILLS, {"action": "create", "name": "probe", "content": "PROPOSED-TEXT"},
                         summary="create probe: PROPOSED-TEXT", origin="foreground")
    mgr, saved, events = _capture_rejections()
    try:
        handle_pending_subcommand(wa.SKILLS, ["reject", rec["id"], *reason_words])
    finally:
        mgr._hooks = saved
    assert wa.get_pending(wa.SKILLS, rec["id"]) is None
    return rec, Path(hermes_home, "pending", "rejected", "skills", f"{rec['id']}.json").read_text(), events


def test_reject_keeps_metadata_with_reason_and_fires_hook(hermes_home):
    """By default the kept rejection and the on_write_rejected record carry the reason, id, target, times and
    a hash of the proposal, never the proposed content; the queue drops the entry."""
    import hashlib
    rec, text, events = _reject_one(hermes_home, ["duplicates", "an", "existing", "skill"])
    kept = json.loads(text)
    assert "PROPOSED-TEXT" not in text and "payload" not in kept and "summary" not in kept
    assert {k: kept[k] for k in ("id", "action", "target", "origin", "reason")} == {
        "id": rec["id"], "action": "create", "target": "probe", "origin": "foreground",
        "reason": "duplicates an existing skill"}
    assert kept["created_at"] == rec["created_at"] and kept["rejected_at"] >= rec["created_at"]
    assert kept["payload_sha256"] == hashlib.sha256(
        json.dumps(rec["payload"], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    assert [(e["subsystem"], e["pending_id"], e["reason"], e["record"]) for e in events] == [
        ("skills", rec["id"], "duplicates an existing skill", kept)]


def test_reject_keeps_the_content_only_when_opted_in(hermes_home):
    _set_rejected_writes("skills", keep_content=True)
    rec, text, events = _reject_one(hermes_home, ["no"])
    kept = json.loads(text)
    assert kept["payload"] == rec["payload"] and kept["reason"] == "no" and "payload_sha256" in kept
    assert events[0]["record"] == kept


def test_kept_rejections_are_pruned_after_retention_days(hermes_home):
    """Listing or staging deletes kept rejections older than retention_days; 0 keeps them all."""
    import time
    from tools import write_approval as wa
    old_rec, _, _ = _reject_one(hermes_home, ["old"])
    new_rec, _, _ = _reject_one(hermes_home, ["new"])
    d = wa._pending_dir(wa.SKILLS, rejected=True)
    old = time.time() - 31 * 86400
    os.utime(d / f"{old_rec['id']}.json", (old, old))
    _set_rejected_writes("skills", retention_days=0)
    wa.list_pending(wa.SKILLS)
    assert sorted(p.stem for p in d.glob("*.json")) == sorted([old_rec["id"], new_rec["id"]])
    _set_rejected_writes("skills", retention_days=30)
    wa.list_pending(wa.SKILLS)
    assert [p.stem for p in d.glob("*.json")] == [new_rec["id"]]
    os.utime(d / f"{new_rec['id']}.json", (old, old))
    wa.stage_write(wa.MEMORY, {"action": "add", "target": "memory", "content": "c"}, summary="c", origin="foreground")
    assert [p.stem for p in d.glob("*.json")] == [new_rec["id"]]              # memory's store is not skills'
    wa.stage_write(wa.SKILLS, {"action": "delete", "name": "probe"}, summary="d", origin="foreground")
    assert list(d.glob("*.json")) == []


def test_reject_never_copies_or_reports_a_file_outside_the_store(hermes_home, tmp_path):
    """A path-like id or a linked record is refused by reject_pending (nothing copied under
    pending/rejected/, no hook), and a linked rejected/ dir is never written through."""
    from pathlib import Path
    from tools import write_approval as wa
    victim = Path(hermes_home, "victim.json")
    victim.write_text('{"id": "victim", "payload": {}}', encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    pending = Path(hermes_home, "pending", "skills")
    pending.mkdir(parents=True)
    (pending / "deadbeef.json").symlink_to(victim)
    mgr, saved, events = _capture_rejections()
    try:
        for bad in ("../../victim", str(victim.with_suffix("")), "a/b", "deadbeef"):
            assert wa.reject_pending(wa.SKILLS, bad, "why") is False, bad
    finally:
        mgr._hooks = saved
    assert victim.exists() and events == [] and not Path(hermes_home, "pending", "rejected").exists()
    (pending / "deadbeef.json").unlink()

    rec = wa.stage_write(wa.SKILLS, {"action": "create", "name": "probe", "content": "x"},
                         summary="create probe", origin="foreground")
    Path(hermes_home, "pending", "rejected").mkdir()
    Path(hermes_home, "pending", "rejected", "skills").symlink_to(outside)
    assert wa.reject_pending(wa.SKILLS, rec["id"], "why") is True
    assert list(outside.iterdir()) == [] and wa.get_pending(wa.SKILLS, rec["id"]) is None
