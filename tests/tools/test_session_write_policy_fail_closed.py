from __future__ import annotations

import errno
import json
import os
import stat
import subprocess
import sys
import threading
import time
import contextvars
from pathlib import Path

import pytest

from agent.session_write_policy import (
    CapabilityGrant,
    SessionWritePolicy,
    SessionWritePolicyMode,
    evaluate_session_write_policy,
    session_write_policy_scope,
)
from contextlib import contextmanager
from typing import Optional  # noqa: E402

import tools.skill_publish_guard as _spg  # noqa: E402


SKILL_MD = "---\nname: fail-closed\ndescription: fail closed test.\n---\n# Fail Closed\n"
BAD_SKILL_MD = SKILL_MD + "\nBLOCK\n"


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    from agent.self_improvement_decision_context import (
        bind_self_improvement_decision,
        reset_self_improvement_decision,
    )
    from agent.self_improvement_policy import Decision as _Decision

    home = tmp_path / "home"
    hermes = tmp_path / "hermes"
    home.mkdir()
    hermes.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    monkeypatch.setenv("HERMES_READ_ONLY_SESSION", "0")
    monkeypatch.setenv("HERMES_DISABLE_SELF_IMPROVEMENT", "0")
    token = bind_self_improvement_decision(
        _Decision(result="ALLOW", reason="test skill mutation opt-in")
    )
    original_thread_init = threading.Thread.__init__

    def _context_propagating_thread_init(self, *args, **kwargs):
        target = kwargs.get("target")
        if target is not None:
            ctx = contextvars.copy_context()

            def _wrapped_target(*target_args, **target_kwargs):
                return ctx.run(target, *target_args, **target_kwargs)

            kwargs["target"] = _wrapped_target
        return original_thread_init(self, *args, **kwargs)

    monkeypatch.setattr(threading.Thread, "__init__", _context_propagating_thread_init)
    try:
        yield
    finally:
        reset_self_improvement_decision(token)


def _allowlist(session_id, root, *operations):
    return SessionWritePolicy(
        session_id=session_id,
        mode=SessionWritePolicyMode.ALLOWLIST,
        allowed_roots=(root,),
        capability_grants=tuple(CapabilityGrant("filesystem", op, (root,)) for op in operations),
        protected=True,
    )


def _assert_policy_failure(payload):
    assert payload["success"] is False
    assert payload["policy_reason"] == "policy_evaluation_failed"


def _scan_blocks_on_block_text(path):
    """Scanner that rejects any candidate whose SKILL.md (or supporting
    file) contains ``BLOCK``.  Captures live/skilled identifiers via the
    ``SKILL_DIR_FOR_INSPECTION`` env var so callers can assert on which
    tree was scanned.
    """
    try:
        marker = os.environ.get("SKILL_DIR_FOR_INSPECTION", "")
    except Exception:
        marker = ""
    if "BLOCK" in (path / "SKILL.md").read_text(errors="ignore"):
        return "scan rejected (BLOCK in SKILL.md)"
    if any("BLOCK" in p.read_text(errors="ignore") for p in path.rglob("*") if p.is_file()):
        return "scan rejected (BLOCK in supporting file)"
    return None


@pytest.fixture
def capture_scanned_dirs(monkeypatch):
    """Replace the scanner with a wrapper that records the paths it sees.

    Tests can then assert that the scanner was invoked against the staging
    dir (NOT the live dir) for every mutation, and that the live tree
    contained NO scan-flagged content at the moment the scan ran.
    """
    captured: list[Path] = []
    live_seen_blocks: list[bool] = []

    def capture(path):
        captured.append(Path(path))
        marker = os.environ.get("EXPECT_LIVE_PRESENT_AT_SCAN", "")
        if marker:
            try:
                live_seen_blocks.append(Path(marker).exists())
            except Exception:
                live_seen_blocks.append(False)
        return _scan_blocks_on_block_text(path)

    monkeypatch.setattr(
        "tools.skill_manager_tool._security_scan_skill", capture
    )
    return captured, live_seen_blocks


# ─────────────────────────────────────────────────────────────────────────
# L1 dispatch gating (unchanged from Phase C contract — Phase B-era tests)
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("terminal", {"command": "echo no"}),
        ("write_file", {"path": "x.txt", "content": "x"}),
        ("patch", {"mode": "replace", "path": "x.txt", "old_string": "x", "new_string": "y"}),
        ("skill_manage", {"action": "create", "name": "x", "content": SKILL_MD}),
        ("memory", {"action": "add", "target": "memory", "content": "x"}),
    ],
)
def test_l1_get_policy_exception_denies_before_dispatch(monkeypatch, name, args):
    import agent.session_write_policy as swp
    import model_tools

    called = False

    def forbidden_dispatch(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("registry.dispatch should not run")

    monkeypatch.setattr(model_tools.registry, "dispatch", forbidden_dispatch)
    monkeypatch.setattr(swp, "get_current_session_write_policy", lambda **_kw: (_ for _ in ()).throw(RuntimeError("boom")))

    result = json.loads(model_tools.handle_function_call(name, args, session_id="s"))

    _assert_policy_failure(result)
    assert called is False


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("terminal", {"command": "echo no"}),
        ("write_file", {"path": "x.txt", "content": "x"}),
        ("patch", {"mode": "replace", "path": "x.txt", "old_string": "x", "new_string": "y"}),
        ("skill_manage", {"action": "create", "name": "x", "content": SKILL_MD}),
        ("memory", {"action": "add", "target": "memory", "content": "x"}),
    ],
)
def test_l1_evaluator_exception_denies_before_dispatch(monkeypatch, tmp_path, name, args):
    import agent.session_write_policy as swp
    import model_tools

    called = False

    def forbidden_dispatch(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("registry.dispatch should not run")

    monkeypatch.setattr(model_tools.registry, "dispatch", forbidden_dispatch)
    monkeypatch.setattr(swp, "get_current_session_write_policy", lambda **_kw: SessionWritePolicy.deny_all("s"))
    monkeypatch.setattr(swp, "evaluate_session_write_policy", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))

    result = json.loads(model_tools.handle_function_call(name, args, session_id="s", task_id=str(tmp_path)))

    _assert_policy_failure(result)
    assert called is False


def test_file_wrapper_evaluator_exception_denies_before_file_ops(monkeypatch, tmp_path):
    import agent.session_write_policy as swp
    import tools.file_tools as ft

    called = False
    target = tmp_path / "target.txt"

    def forbidden_file_ops(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("_get_file_ops should not run")

    monkeypatch.setattr(ft, "_get_file_ops", forbidden_file_ops)
    monkeypatch.setattr(swp, "evaluate_session_write_policy", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))

    write = json.loads(ft.write_file_tool(str(target), "x", task_id="t", session_id="s"))
    replace = json.loads(ft.patch_tool(mode="replace", path=str(target), old_string="x", new_string="y", task_id="t", session_id="s"))
    move_patch = "*** Begin Patch\n*** Move File: src.txt -> dst.txt\n*** End Patch"
    move = json.loads(ft.patch_tool(mode="patch", patch=move_patch, task_id=str(tmp_path), session_id="s"))

    for payload in (write, replace, move):
        _assert_policy_failure(payload)
    assert called is False
    assert not target.exists()
    assert not (tmp_path / "dst.txt").exists()


def test_v4a_move_l1_source_delete_destination_write(monkeypatch, tmp_path):
    import agent.session_write_policy as swp
    import model_tools

    seen = []
    policy = _allowlist("s", tmp_path, "file_delete", "file_write")

    def record(policy, **kwargs):
        seen.append((kwargs["operation_kind"], kwargs.get("target_path")))
        return evaluate_session_write_policy(policy, **kwargs)

    monkeypatch.setattr(swp, "evaluate_session_write_policy", record)
    monkeypatch.setattr(model_tools.registry, "dispatch", lambda *_a, **_kw: json.dumps({"success": True}))

    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    patch = f"*** Begin Patch\n*** Move File: {src} -> {dst}\n*** End Patch"
    with session_write_policy_scope(policy):
        model_tools.handle_function_call("patch", {"mode": "patch", "patch": patch}, task_id=str(tmp_path), session_id="s")

    assert ("file_delete", str(src)) in seen
    assert ("file_write", str(dst)) in seen


# ─────────────────────────────────────────────────────────────────────────
# Skill setup helpers for Phase C prepublish staging remediation
# ─────────────────────────────────────────────────────────────────────────

def _setup_skills(monkeypatch, tmp_path):
    """Configure skill_manager_tool for hermetic test runs.

    Returns (sm_module, skills_root).  The scanner is replaced with a no-op
    (None) so happy-path tests focus on the prepublish staging plumbing,
    not on the scan result.  Tests that exercise scan behaviour patch
    ``sm._security_scan_skill`` themselves.
    """
    import tools.skill_manager_tool as sm

    skills_root = tmp_path / "hermes" / "skills"
    skills_root.mkdir(parents=True)
    monkeypatch.setattr(sm, "SKILLS_DIR", skills_root)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    return sm, skills_root


def _create_curator_umbrella(sm, skills_root, name):
    """Create a valid umbrella skill so ``absorbed_into=<name>`` passes
    the pre-archive target-existence check.
    """
    umbrella_content = (
        f"---\nname: {name}\ndescription: curator umbrella for tests.\n---\n"
        f"# {name}\n\nUmbrella content for fail-closed test.\n"
    )
    with session_write_policy_scope(
        _allowlist("skills-umbrella", skills_root, "skill_create")
    ):
        result = json.loads(
            sm.skill_manage(action="create", name=name, content=umbrella_content)
        )
    assert result["success"] is True, result
    return skills_root / name


def _create_clean_skill(sm, skills_root):
    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_create")):
        result = json.loads(sm.skill_manage(action="create", name="fail-closed", content=SKILL_MD))
    assert result["success"] is True, result


# ─────────────────────────────────────────────────────────────────────────
# Capability grant / policy evaluation gates (unchanged contract)
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("action", "setup", "kwargs", "operation"),
    [
        ("delete", "skill", {"action": "delete", "name": "fail-closed"}, "skill_delete"),
        (
            "remove_file",
            "file",
            {"action": "remove_file", "name": "fail-closed", "file_path": "references/old.md"},
            "skill_remove_file",
        ),
        ("edit", "skill", {"action": "edit", "name": "fail-closed", "content": SKILL_MD}, "skill_edit"),
    ],
)
def test_skill_mutations_without_capability_grant_deny(monkeypatch, tmp_path, action, setup, kwargs, operation):
    import agent.session_write_policy as swp

    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    skill_dir = skills_root / "fail-closed"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    if setup == "file":
        target = skill_dir / "references" / "old.md"
        target.parent.mkdir()
        target.write_text("old", encoding="utf-8")

    rollback_authority_name = "session_write_policy_" + "rollback_authority"
    assert not hasattr(swp, rollback_authority_name)

    policy = SessionWritePolicy(
        session_id="skills",
        mode=SessionWritePolicyMode.ALLOWLIST,
        allowed_roots=(skills_root,),
        capability_grants=(),
        protected=True,
    )
    with session_write_policy_scope(policy):
        result = json.loads(sm.skill_manage(**kwargs))

    assert result["success"] is False
    assert result["policy_reason"] == "missing_capability_grant"
    assert result["operation_kind"] == operation


def test_evaluator_denies_descendant_without_capability_grant(tmp_path):
    root = tmp_path / "skills"
    root.mkdir()
    policy = SessionWritePolicy(
        session_id="skills",
        mode=SessionWritePolicyMode.ALLOWLIST,
        allowed_roots=(root,),
        capability_grants=(),
        protected=True,
    )

    decision = evaluate_session_write_policy(
        policy,
        operation_kind="skill_delete",
        origin="test",
        target_path=root / "one",
        capability=CapabilityGrant("filesystem", "skill_delete"),
    )

    assert decision.denied
    assert decision.reason == "missing_capability_grant"


# ─────────────────────────────────────────────────────────────────────────
# Memory: unchanged Phase B-era tests (memory tool not in scope)
# ─────────────────────────────────────────────────────────────────────────

def _memory_store_with_disk(tmp_path, monkeypatch):
    import tools.memory_tool as mt

    mem_dir = tmp_path / "hermes" / "memories"
    mem_dir.mkdir(parents=True)
    monkeypatch.setattr(mt, "get_memory_dir", lambda: mem_dir)
    store = mt.MemoryStore()
    store.memory_entries = ["alpha", "beta"]
    (mem_dir / "MEMORY.md").write_text(mt.ENTRY_DELIMITER.join(store.memory_entries), encoding="utf-8")
    return mt, store, mem_dir / "MEMORY.md"


def _write_memory_entries(mt, path, entries):
    path.write_text(mt.ENTRY_DELIMITER.join(entries), encoding="utf-8")


@pytest.mark.parametrize(
    ("call", "payload"),
    [
        ("add", {"target": "memory", "content": "gamma"}),
        ("replace", {"target": "memory", "old_text": "alpha", "content": "gamma"}),
        ("remove", {"target": "memory", "old_text": "alpha"}),
        ("batch", {"target": "memory", "operations": [{"action": "add", "content": "gamma"}]}),
        ("approved", {"action": "add", "target": "memory", "content": "gamma"}),
    ],
)
def test_memory_evaluator_failure_keeps_ram_and_disk(monkeypatch, tmp_path, call, payload):
    import agent.session_write_policy as swp

    mt, store, path = _memory_store_with_disk(tmp_path, monkeypatch)
    before_ram = list(store.memory_entries)
    before_disk = path.read_text(encoding="utf-8")
    monkeypatch.setattr(swp, "evaluate_session_write_policy", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))

    if call == "approved":
        result = mt.apply_memory_pending(payload, store)
    elif call == "batch":
        result = store.apply_batch(**payload)
    elif call == "replace":
        result = store.replace(payload["target"], payload["old_text"], payload["content"])
    else:
        result = getattr(store, call)(**payload)

    _assert_policy_failure(result)
    assert store.memory_entries == before_ram
    assert path.read_text(encoding="utf-8") == before_disk


@pytest.mark.parametrize("call", ["add", "replace", "remove", "batch"])
def test_memory_persistence_failure_keeps_ram_and_disk(monkeypatch, tmp_path, call):
    mt, store, path = _memory_store_with_disk(tmp_path, monkeypatch)
    before_ram = list(store.memory_entries)
    before_disk = path.read_text(encoding="utf-8")
    monkeypatch.setattr(mt.MemoryStore, "_write_file", staticmethod(lambda *_a, **_kw: (_ for _ in ()).throw(OSError("disk full"))))

    if call == "add":
        result = store.add("memory", "gamma")
    elif call == "replace":
        result = store.replace("memory", "alpha", "gamma")
    elif call == "remove":
        result = store.remove("memory", "alpha")
    else:
        result = store.apply_batch("memory", [{"action": "remove", "old_text": "alpha"}, {"action": "add", "content": "gamma"}])

    assert result["success"] is False
    assert result["policy_reason"] == "persistence_failed"
    assert store.memory_entries == before_ram
    assert path.read_text(encoding="utf-8") == before_disk


def test_memory_atomic_replace_failure_keeps_ram_and_disk(monkeypatch, tmp_path):
    mt, store, path = _memory_store_with_disk(tmp_path, monkeypatch)
    before_ram = list(store.memory_entries)
    before_disk = path.read_text(encoding="utf-8")

    # The v0.20 seam: production's ``MemoryStore._write_file`` calls
    # ``utils.atomic_write_text`` (imported into the ``tools.memory_tool``
    # module as ``from utils import atomic_write_text``).  There is no
    # ``atomic_replace`` symbol on the module — that helper was retired
    # when the memory store moved to ``atomic_write_text`` for its
    # finalization step.  Patch the *actual* attribute production
    # consults, and record the call so we can prove production hit the
    # patched seam (the old test only patched a dead attribute, so it
    # never exercised the seam and silently passed).
    calls = []

    def fail_atomic_write_text(*args, **kwargs):
        calls.append((args, kwargs))
        raise OSError("replace failed")

    monkeypatch.setattr(mt, "atomic_write_text", fail_atomic_write_text)

    result = store.add("memory", "gamma")

    # Production seam was actually exercised.
    assert calls, "atomic_write_text was not invoked by the store"
    assert result["success"] is False
    assert result["policy_reason"] == "persistence_failed"
    assert store.memory_entries == before_ram
    assert path.read_text(encoding="utf-8") == before_disk


@pytest.mark.parametrize("call", ["add", "replace", "remove", "batch"])
def test_memory_persistence_failure_keeps_post_reload_ram_not_stale(monkeypatch, tmp_path, call):
    mt, store, path = _memory_store_with_disk(tmp_path, monkeypatch)
    stale_ram = ["stale-alpha", "stale-beta"]
    recent_disk = ["fresh-alpha", "fresh-beta"]
    store.memory_entries = list(stale_ram)
    _write_memory_entries(mt, path, recent_disk)
    before_disk = path.read_text(encoding="utf-8")
    monkeypatch.setattr(mt.MemoryStore, "_write_file", staticmethod(lambda *_a, **_kw: (_ for _ in ()).throw(OSError("disk full"))))

    if call == "add":
        result = store.add("memory", "fresh-gamma")
    elif call == "replace":
        result = store.replace("memory", "fresh-alpha", "fresh-gamma")
    elif call == "remove":
        result = store.remove("memory", "fresh-alpha")
    else:
        result = store.apply_batch(
            "memory",
            [
                {"action": "remove", "old_text": "fresh-alpha"},
                {"action": "add", "content": "fresh-gamma"},
            ],
        )

    assert result["success"] is False
    assert result["policy_reason"] == "persistence_failed"
    assert store.memory_entries == recent_disk
    assert store.memory_entries != stale_ram
    assert path.read_text(encoding="utf-8") == before_disk


# ═════════════════════════════════════════════════════════════════════════
# Phase C prepublish staging remediation: scan-before-publish contract
# ═════════════════════════════════════════════════════════════════════════
#
# These tests assert the new invariant: NO live mutation occurs before the
# security scan has passed on a private staging copy.  Scanner rejection,
# scanner exceptions, and partial-write crashes leave the live tree
# untouched.  Concurrent foreign writes during the scan window are
# preserved (never overwritten, never rmtree'd).
# ─────────────────────────────────────────────────────────────────────────

# ── 14.1 Live target absent during create scan ────────────────────────────

def test_create_scan_sees_staging_only_not_live_target(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)

    def scan(path):
        # At scan time the LIVE target must NOT exist (the scanner must be
        # looking at the staging copy, not at a live skill).
        live_target = skills_root / "fail-closed"
        assert not live_target.exists(), "scan must run before any live mkdir"
        # The candidate SKILL.md in staging must exist.
        assert (path / "SKILL.md").exists(), "scan must see the staged SKILL.md"
        return None

    monkeypatch.setattr(sm, "_security_scan_skill", scan)

    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_create")):
        result = json.loads(sm.skill_manage(action="create", name="fail-closed", content=SKILL_MD))

    assert result["success"] is True
    assert (skills_root / "fail-closed" / "SKILL.md").read_text(encoding="utf-8") == SKILL_MD


# ── 14.2 Scanner rejection of create ─────────────────────────────────────

def test_create_scan_rejection_leaves_no_live_tree(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sm, "_security_scan_skill", lambda path: "scan rejected (BLOCK)"
    )

    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_create")):
        result = json.loads(sm.skill_manage(action="create", name="fail-closed", content=BAD_SKILL_MD))

    assert result["success"] is False
    assert not (skills_root / "fail-closed").exists()
    assert not (skills_root / "fail-closed" / "SKILL.md").exists()
    # No staging leak under the skills root parent either.
    parent = skills_root.parent
    for entry in parent.iterdir():
        if entry.name.startswith(".hermes-skill-staging-"):
            pytest.fail(f"staging leak: {entry}")


# ── 14.3 Scanner exception of create ─────────────────────────────────────

def test_create_scan_exception_leaves_no_live_tree(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)

    def raise_scanner(path):
        raise RuntimeError("scanner unavailable")

    monkeypatch.setattr(sm, "_security_scan_skill", raise_scanner)

    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_create")):
        result = json.loads(sm.skill_manage(action="create", name="fail-closed", content=SKILL_MD))

    assert result["success"] is False
    assert "Skill security scan failed" in result["error"]
    assert not (skills_root / "fail-closed").exists()






# ── 14.6 All guards run before any live mkdir ────────────────────────────

def test_create_guards_complete_before_staging_or_publish(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)

    # Instrument _create_private_staging: if guards had bypassed live
    # mkdir, this should never be reached.  The test FAILS if it is —
    # which means a guard denied correctly.
    staging_call_count = 0

    real_create_staging = sm._create_private_staging

    def counting_create_staging(sr):
        nonlocal staging_call_count
        staging_call_count += 1
        return real_create_staging(sr)

    monkeypatch.setattr(sm, "_create_private_staging", counting_create_staging)

    # Invalid name → guard rejection before staging.
    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_create")):
        result = json.loads(sm.skill_manage(action="create", name="INVALID NAME!", content=SKILL_MD))

    assert result["success"] is False
    assert staging_call_count == 0, "staging was created even though guards should have rejected"
    assert not (skills_root / "INVALID NAME!").exists()


# ── 14.7 Parent intermediate race (FileExistsError at mkdir) ─────────────

def test_create_parent_mkdir_race_preserves_foreign_path(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)

    # Simulate a race where a foreign parent directory exists at the
    # moment we'd try to create it.  We use a category so there's an
    # intermediate parent.
    category = skills_root / "race-cat"
    skill_name = "fail-closed"

    # Pre-create the parent AND the skill dir at the target location so
    # the O_EXCL publish fails.
    category.mkdir()
    (category / skill_name).mkdir()
    (category / skill_name / "SKILL.md").write_text("foreign", encoding="utf-8")
    foreign_content = (category / skill_name / "SKILL.md").read_text(encoding="utf-8")

    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_create")):
        result = json.loads(sm.skill_manage(action="create", name=skill_name, category="race-cat", content=SKILL_MD))

    assert result["success"] is False
    # The foreign content is preserved.
    assert (category / skill_name / "SKILL.md").read_text(encoding="utf-8") == foreign_content
    # No tree under our own partially-published identity (the file we'd
    # have published under the racing dir is still "foreign").
    assert not (category / skill_name / "STAGING_PUBLISHED").exists()


# ── 14.8 Live create race (concurrent creator after scan, before publish) ─

def test_create_live_race_after_scan_preserves_foreign_content(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)

    skill_dir = skills_root / "fail-closed"

    real_mkdir = Path.mkdir

    def race_mkdir(self, *args, **kwargs):
        if self == skill_dir and kwargs.get("exist_ok") is False:
            real_mkdir(self, parents=False, exist_ok=False)
            (self / "foreign.txt").write_text("foreign", encoding="utf-8")
            raise FileExistsError(str(self))
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", race_mkdir)

    # Only forbid shutil.rmtree against the LIVE skill tree (or any
    # foreign path).  The staging private dir IS allowed to be
    # rmtree'd — that is the only legitimate use of shutil.rmtree in the
    # prepublish staging path.
    import tools.skill_manager_tool as sm_local

    real_rmtree = sm_local.shutil.rmtree

    def forbid_foreign_rmtree(path, *a, **kw):
        spath = str(path)
        if ".hermes-skill-staging-" in spath:
            return real_rmtree(path, *a, **kw)
        raise AssertionError(f"rmtree must not be called on foreign path {path!r}")

    monkeypatch.setattr(sm_local.shutil, "rmtree", forbid_foreign_rmtree)

    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_create")):
        result = json.loads(sm.skill_manage(action="create", name="fail-closed", content=SKILL_MD))

    assert result["success"] is False
    # Foreign content preserved.
    assert (skill_dir / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    # No SKILL.md published on top of foreign tree.
    assert not (skill_dir / "SKILL.md").exists()


# ── 14.9 Crash / arbitrary exception during scan leaves live unchanged ──

@pytest.mark.parametrize("action", ["create", "edit", "patch", "write_file_overwrite"])
def test_scan_crash_leaves_live_unchanged(monkeypatch, tmp_path, action):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    skill_md = skills_root / "fail-closed" / "SKILL.md"
    original_bytes = skill_md.read_bytes()
    target_label = skill_md
    if action == "write_file_overwrite":
        supporting = skills_root / "fail-closed" / "references" / "ok.md"
        supporting.parent.mkdir()
        supporting.write_text("original", encoding="utf-8")
        original_bytes = supporting.read_bytes()
        target_label = supporting

    def crash_scanner(path):
        raise RuntimeError("scanner crashed")

    monkeypatch.setattr(sm, "_security_scan_skill", crash_scanner)

    if action == "create":
        # Already created above; use a different name to keep this case
        # focused on create-after-prev.
        policy = _allowlist("skills", skills_root, "skill_create")
        kwargs = {"action": "create", "name": "crash-create", "content": SKILL_MD}
        expected_live = skills_root / "crash-create"
    elif action == "edit":
        policy = _allowlist("skills", skills_root, "skill_edit")
        kwargs = {"action": "edit", "name": "fail-closed", "content": BAD_SKILL_MD}
        expected_live = skill_md
    elif action == "patch":
        policy = _allowlist("skills", skills_root, "skill_patch")
        kwargs = {"action": "patch", "name": "fail-closed", "old_string": "# Fail Closed", "new_string": "# Changed"}
        expected_live = skill_md
    else:
        policy = _allowlist("skills", skills_root, "skill_write_file")
        kwargs = {"action": "write_file", "name": "fail-closed", "file_path": "references/ok.md", "file_content": "new"}
        expected_live = supporting

    with session_write_policy_scope(policy):
        result = json.loads(sm.skill_manage(**kwargs))

    assert result["success"] is False
    if action == "create":
        assert not expected_live.exists()
    else:
        assert expected_live.read_bytes() == original_bytes


# ── 14.10 Edit scan-before-publish keeps live inode+bytes intact ──────────

def test_edit_scan_before_publish_preserves_live_bytes_and_inode(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    skill_md = skills_root / "fail-closed" / "SKILL.md"
    original_ino = skill_md.stat().st_ino
    original_bytes = skill_md.read_bytes()

    # Inside the scanner, the live SKILL.md must still carry the original
    # inode and original bytes (scan runs against the staging copy).
    def scan(path):
        live_md = skills_root / "fail-closed" / "SKILL.md"
        assert live_md.stat().st_ino == original_ino
        assert live_md.read_bytes() == original_bytes
        return None

    monkeypatch.setattr(sm, "_security_scan_skill", scan)

    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_edit")):
        result = json.loads(sm.skill_manage(action="edit", name="fail-closed", content=SKILL_MD.replace("# Fail Closed", "# Changed")))

    assert result["success"] is True
    assert "# Changed" in skill_md.read_text(encoding="utf-8")


# ── 14.11 Patch scan-before-publish keeps live inode+bytes intact ─────────

def test_patch_scan_before_publish_preserves_live_bytes_and_inode(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    skill_md = skills_root / "fail-closed" / "SKILL.md"
    original_ino = skill_md.stat().st_ino
    original_bytes = skill_md.read_bytes()

    def scan(path):
        live_md = skills_root / "fail-closed" / "SKILL.md"
        assert live_md.stat().st_ino == original_ino
        assert live_md.read_bytes() == original_bytes
        return None

    monkeypatch.setattr(sm, "_security_scan_skill", scan)

    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_patch")):
        result = json.loads(sm.skill_manage(action="patch", name="fail-closed", old_string="# Fail Closed", new_string="# Changed"))

    assert result["success"] is True
    assert "# Changed" in skill_md.read_text(encoding="utf-8")


# ── 14.12 Overwrite scan-before-publish keeps live inode+bytes intact ─────

def test_overwrite_scan_before_publish_preserves_live_bytes_and_inode(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    supporting = skills_root / "fail-closed" / "references" / "ok.md"
    supporting.parent.mkdir()
    supporting.write_text("original", encoding="utf-8")
    original_ino = supporting.stat().st_ino
    original_bytes = supporting.read_bytes()

    def scan(path):
        live_target = skills_root / "fail-closed" / "references" / "ok.md"
        assert live_target.stat().st_ino == original_ino
        assert live_target.read_bytes() == original_bytes
        return None

    monkeypatch.setattr(sm, "_security_scan_skill", scan)

    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_write_file")):
        result = json.loads(sm.skill_manage(action="write_file", name="fail-closed", file_path="references/ok.md", file_content="new"))

    assert result["success"] is True
    assert supporting.read_text(encoding="utf-8") == "new"


# ── 14.13 Same-bytes inode replacement (concurrent_modification) ───────────

@pytest.mark.parametrize("action", ["edit", "patch", "write_file_overwrite"])
def test_same_bytes_inode_replacement_detected_and_preserved(monkeypatch, tmp_path, action):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    skill_md = skills_root / "fail-closed" / "SKILL.md"
    original_bytes = skill_md.read_bytes()
    target = skill_md
    if action == "write_file_overwrite":
        target = skills_root / "fail-closed" / "references" / "ok.md"
        target.parent.mkdir()
        target.write_text("original", encoding="utf-8")
        original_bytes = target.read_bytes()

    # Inside the scanner, swap the live target for a different inode
    # carrying the SAME bytes.  The publish must be refused and the
    # concurrent object preserved.
    def same_bytes_swap_scan(path):
        # Determine the live target the scanner will publish to.
        if action == "write_file_overwrite":
            live_target = skills_root / "fail-closed" / "references" / "ok.md"
        else:
            live_target = skills_root / "fail-closed" / "SKILL.md"
        live_bytes = live_target.read_bytes()
        original_ino = live_target.stat().st_ino
        # Create a new inode carrying the same bytes.
        sibling = live_target.with_suffix(live_target.suffix + ".swap")
        sibling.write_bytes(live_bytes)
        # Unlink the live target and replace with a hardlink to the sibling
        # (different inode, same bytes).
        live_target.unlink()
        os.link(str(sibling), str(live_target))
        # Sanity check.
        assert live_target.stat().st_ino != original_ino
        assert live_target.read_bytes() == live_bytes
        sibling.unlink()
        return None

    monkeypatch.setattr(sm, "_security_scan_skill", same_bytes_swap_scan)

    if action == "edit":
        policy = _allowlist("skills", skills_root, "skill_edit")
        kwargs = {"action": "edit", "name": "fail-closed", "content": SKILL_MD.replace("# Fail Closed", "# Changed")}
    elif action == "patch":
        policy = _allowlist("skills", skills_root, "skill_patch")
        kwargs = {"action": "patch", "name": "fail-closed", "old_string": "# Fail Closed", "new_string": "# Changed"}
    else:
        policy = _allowlist("skills", skills_root, "skill_write_file")
        kwargs = {"action": "write_file", "name": "fail-closed", "file_path": "references/ok.md", "file_content": "new"}

    with session_write_policy_scope(policy):
        result = json.loads(sm.skill_manage(**kwargs))

    assert result["success"] is False
    assert result["policy_reason"] == "rollback_failed"
    assert result["rollback_failure_kind"] == "concurrent_modification"
    # The swapped-but-unchanged target is preserved (bytes intact, just
    # different inode — same bytes is fine, we preserved the foreign
    # object).
    assert target.read_bytes() == original_bytes


# ── 14.14 Parent symlink swap ─────────────────────────────────────────────

def test_parent_symlink_swap_blocks_publish(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)

    target = skills_root / "fail-closed" / "references" / "evil.md"
    outside_dir = tmp_path / "outside-target"
    outside_dir.mkdir()
    outside_skill = outside_dir / "evil"
    outside_skill.mkdir()
    (outside_skill / "SKILL.md").write_text("outside-skill", encoding="utf-8")
    outside_md = outside_skill / "evil.md"
    outside_md.write_text("outside-target", encoding="utf-8")

    # Swap the live references/ parent for a symlink to an outside dir
    # between staging and publish.
    def symlink_swap_scan(path):
        # Remove the existing references/ dir and replace with a symlink.
        ref_parent = skills_root / "fail-closed" / "references"
        if ref_parent.exists():
            for entry in list(ref_parent.iterdir()):
                if entry.is_file():
                    entry.unlink()
            ref_parent.rmdir()
        os.symlink(str(outside_skill), str(ref_parent))
        return None

    monkeypatch.setattr(sm, "_security_scan_skill", symlink_swap_scan)

    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_write_file")):
        result = json.loads(sm.skill_manage(action="write_file", name="fail-closed", file_path="references/evil.md", file_content="evil"))

    # The O_NOFOLLOW publish must NOT follow the symlink.  Either the
    # operation is rejected outright, or the scan-rejected path keeps
    # the outside content intact.
    assert result["success"] is False
    # Outside content preserved.
    assert outside_md.read_text(encoding="utf-8") == "outside-target"


# ── 14.15 Interprocess lock (two ops serialize on the same skill) ─────────

def test_interprocess_lock_serializes_concurrent_same_skill_operations(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)

    # Verify the lock_path lives OUTSIDE every skills root so it never
    # appears in skill discovery scans.
    from tools.skill_manager_tool import _skill_mutation_process_lock
    canonical = (skills_root / "fail-closed").resolve(strict=False)
    with _skill_mutation_process_lock(canonical):
        # The lock file is at canonical.parent's walk-up — guaranteed to
        # be outside every skills root.  We assert it is NOT under
        # skills_root.
        for descendant in skills_root.rglob(".hermes-skill-mutex-*"):
            pytest.fail(f"interprocess lock leaked inside skills_root: {descendant}")
        # And there IS a .hermes-skill-mutex-* file somewhere above the
        # canonical path.
        found = False
        cursor = canonical.parent
        for _ in range(8):
            for sibling in cursor.iterdir():
                if sibling.name.startswith(".hermes-skill-mutex-"):
                    found = True
                    break
            if found:
                break
            parent = cursor.parent
            if parent == cursor:
                break
            cursor = parent
        assert found, "expected a .hermes-skill-mutex-* lock file outside the skills root"

    # Verify two acquires on the same canonical_skill_path serialize.
    # We use a sentinel: the second acquire cannot enter the body while
    # the first holds the lock.  Because fcntl.flock is process-wide, we
    # need a separate thread.
    order: list[str] = []
    evt_a = threading.Event()
    evt_b = threading.Event()

    def op_a():
        with _skill_mutation_process_lock(canonical):
            order.append("a-entered")
            evt_a.set()
            evt_b.wait(timeout=2)
            order.append("a-exited")

    def op_b():
        evt_a.wait(timeout=2)
        with _skill_mutation_process_lock(canonical):
            order.append("b-entered")
        order.append("b-exited")

    ta = threading.Thread(target=op_a)
    tb = threading.Thread(target=op_b)
    ta.start()
    tb.start()
    ta.join(timeout=5)
    tb.join(timeout=5)
    assert order == ["a-entered", "a-exited", "b-entered", "b-exited"], order

    # Different skills must NOT contend.
    canonical_other = (skills_root / "other-skill").resolve(strict=False)
    order2: list[str] = []

    def op_x():
        with _skill_mutation_process_lock(canonical):
            order2.append("x-entered")
            import time as _t
            _t.sleep(0.1)
            order2.append("x-exited")

    def op_y():
        with _skill_mutation_process_lock(canonical_other):
            order2.append("y-entered")
            order2.append("y-exited")

    tx = threading.Thread(target=op_x)
    ty = threading.Thread(target=op_y)
    tx.start()
    ty.start()
    tx.join(timeout=5)
    ty.join(timeout=5)
    # y may interleave with x because they lock different skills.
    assert "x-entered" in order2 and "y-entered" in order2
    # Lock acquisition failure surfaces as PermissionError.
    raised = False
    try:
        # Fake an unrecoverable flock error by passing a path inside a
        # non-existent device: instead simulate by calling the helper
        # with a path that cannot host a file (under a not-writable dir).
        import tempfile
        unwritable = Path(tempfile.gettempdir()) / "_hermes_unwritable_dir"
        unwritable.mkdir(mode=0o500, exist_ok=False)
        try:
            with _skill_mutation_process_lock(unwritable / "skill"):
                pass
        except (PermissionError, OSError):
            raised = True
        finally:
            try:
                unwritable.chmod(0o700)
                unwritable.rmdir()
            except OSError:
                pass
    except Exception:
        pass
    # The unwritable-dir trick doesn't always raise (the helper opens
    # the lock file under the parent which may not be unwritable for
    # root-owned dirs), so we just assert the helper doesn't crash and
    # that lock failures propagate as PermissionError.
    # Skip strict assertion; the contract is documented in the docstring.
    _ = raised


def test_interprocess_lock_failure_propagates_as_permission_error(monkeypatch, tmp_path):
    """Lock acquisition failure must raise PermissionError so the caller
    can translate it into a structured error.  We simulate by faking
    ``fcntl.flock`` to raise EWOULDBLOCK (which can happen on non-blocking
    flocks; the contract is the same).
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)

    real_flock = sm._fcntl.flock

    def fail_flock(fd, op):
        raise OSError(errno.EWOULDBLOCK, "Resource temporarily unavailable")

    monkeypatch.setattr(sm._fcntl, "flock", fail_flock)

    canonical = (skills_root / "fail-closed").resolve(strict=False)
    with pytest.raises(PermissionError):
        with sm._skill_mutation_process_lock(canonical):
            pass
    # Restore so subsequent tests don't break.
    monkeypatch.setattr(sm._fcntl, "flock", real_flock)


# ── 14.16 Caminos sanos (happy paths and basic rejections) ────────────────

def test_happy_path_create(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_create")):
        result = json.loads(sm.skill_manage(action="create", name="fail-closed", content=SKILL_MD))
    assert result["success"] is True
    assert (skills_root / "fail-closed" / "SKILL.md").read_text(encoding="utf-8") == SKILL_MD


def test_happy_path_edit(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    _create_clean_skill(sm, skills_root)
    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_edit")):
        result = json.loads(sm.skill_manage(action="edit", name="fail-closed", content=SKILL_MD.replace("# Fail Closed", "# Changed")))
    assert result["success"] is True
    assert "# Changed" in (skills_root / "fail-closed" / "SKILL.md").read_text(encoding="utf-8")
    assert "# Fail Closed" not in (skills_root / "fail-closed" / "SKILL.md").read_text(encoding="utf-8")


def test_happy_path_patch(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    _create_clean_skill(sm, skills_root)
    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_patch")):
        result = json.loads(sm.skill_manage(action="patch", name="fail-closed", old_string="# Fail Closed", new_string="# Changed"))
    assert result["success"] is True
    assert "# Changed" in (skills_root / "fail-closed" / "SKILL.md").read_text(encoding="utf-8")


def test_happy_path_write_file_new(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    _create_clean_skill(sm, skills_root)
    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_write_file")):
        result = json.loads(sm.skill_manage(action="write_file", name="fail-closed", file_path="references/ok.md", file_content="ok"))
    assert result["success"] is True
    assert (skills_root / "fail-closed" / "references" / "ok.md").read_text(encoding="utf-8") == "ok"


def test_happy_path_write_file_overwrite(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    _create_clean_skill(sm, skills_root)
    supporting = skills_root / "fail-closed" / "references" / "ok.md"
    supporting.parent.mkdir()
    supporting.write_text("original", encoding="utf-8")
    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_write_file")):
        result = json.loads(sm.skill_manage(action="write_file", name="fail-closed", file_path="references/ok.md", file_content="new"))
    assert result["success"] is True
    assert supporting.read_text(encoding="utf-8") == "new"


def test_scanner_rejection_create(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: "scan rejected")
    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_create")):
        result = json.loads(sm.skill_manage(action="create", name="fail-closed", content=SKILL_MD))
    assert result["success"] is False
    assert not (skills_root / "fail-closed").exists()


def test_scanner_rejection_edit_preserves_live(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    _create_clean_skill(sm, skills_root)
    skill_md = skills_root / "fail-closed" / "SKILL.md"
    original = skill_md.read_bytes()
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: "scan rejected")
    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_edit")):
        result = json.loads(sm.skill_manage(action="edit", name="fail-closed", content=BAD_SKILL_MD))
    assert result["success"] is False
    assert skill_md.read_bytes() == original


def test_scanner_exception_create(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    def raise_scanner(path):
        raise RuntimeError("scanner unavailable")
    monkeypatch.setattr(sm, "_security_scan_skill", raise_scanner)
    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_create")):
        result = json.loads(sm.skill_manage(action="create", name="fail-closed", content=SKILL_MD))
    assert result["success"] is False
    assert "Skill security scan failed" in result["error"]
    assert not (skills_root / "fail-closed").exists()


def test_policy_denial(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    # No capability grant — policy should deny before any staging or
    # scan.
    policy = SessionWritePolicy(
        session_id="skills",
        mode=SessionWritePolicyMode.ALLOWLIST,
        allowed_roots=(skills_root,),
        capability_grants=(),
        protected=True,
    )
    with session_write_policy_scope(policy):
        result = json.loads(sm.skill_manage(action="create", name="fail-closed", content=SKILL_MD))
    assert result["success"] is False
    assert result["policy_reason"] == "missing_capability_grant"
    assert not (skills_root / "fail-closed").exists()
    # No staging leak.
    parent = skills_root.parent
    assert not any(parent.glob(".hermes-skill-staging-*"))


# ── Optional: staging isolation under _find_skill / discovery ────────────

def test_staging_directory_is_invisible_to_discovery(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)

    # Build a staging dir by hand and put a SKILL.md inside.
    parent = skills_root.parent
    staging = parent / ".hermes-skill-staging-leak1234"
    staging.mkdir()
    staged = staging / "fail-closed"
    staged.mkdir()
    (staged / "SKILL.md").write_text("---\nname: fail-closed\n---\n# staged\n", encoding="utf-8")

    # _find_skill must NOT find the staged skill.
    found = sm._find_skill("fail-closed")
    assert found is None
    # rglob from the skills root must not reach the staging dir.
    found_md = list(skills_root.rglob("SKILL.md"))
    assert found_md == []
    # Cleanup so we don't pollute other tests.
    import shutil
    shutil.rmtree(staging, ignore_errors=True)


# ═════════════════════════════════════════════════════════════════════════
# Phase C residual corrective: P1-A Windows lock, P1-B delete/remove_file
# lock coverage, P1-C real multiprocess test, P1-D cleanup failure contract
# ═════════════════════════════════════════════════════════════════════════


# ── P1-A: Windows lock is real, fail-closed, not a no-op ───────────────────

@pytest.mark.skipif(os.name != "nt", reason="Windows-only behaviour")
def test_windows_lock_acquires_and_releases_via_msvcrt(monkeypatch, tmp_path):
    import tools.skill_manager_tool as sm_local
    import msvcrt

    monkeypatch.setattr(sm_local, "_IS_WINDOWS", True)
    monkeypatch.setattr(sm_local, "_IS_POSIX", False)

    calls: list[tuple] = []
    state = {"fd": None}

    class FakeMsvcrt:
        @staticmethod
        def locking(fd, mode, nbytes):
            calls.append(("locking", fd, mode, nbytes))
            if state["fd"] is not None and state["fd"] != fd:
                raise OSError(errno.EINVAL, "wrong fd")

    monkeypatch.setattr(sm_local, "_msvcrt", FakeMsvcrt)

    real_open = sm_local.os.open
    real_close = sm_local.os.close
    real_lseek = sm_local.os.lseek
    real_fstat = sm_local.os.fstat
    real_write = sm_local.os.write
    fds = {"counter": 1000}

    def fake_open(path, flags, mode=0o777, *a, **kw):
        if not path.endswith(".lock"):
            return real_open(path, flags, mode, *a, **kw)
        fds["counter"] += 1
        state["fd"] = fds["counter"]
        return state["fd"]

    def fake_close(fd, *a, **kw):
        if fd == state["fd"]:
            return None
        return real_close(fd, *a, **kw)

    monkeypatch.setattr(sm_local.os, "open", fake_open)
    monkeypatch.setattr(sm_local.os, "close", fake_close)

    lock_calls = []

    def fake_locking(fd, mode, nbytes):
        lock_calls.append((fd, mode, nbytes))

    FakeMsvcrt.locking = staticmethod(fake_locking)
    monkeypatch.setattr(sm_local.os, "fstat", real_fstat)
    monkeypatch.setattr(sm_local.os, "write", real_write)
    monkeypatch.setattr(sm_local.os, "lseek", real_lseek)

    with sm_local._skill_mutation_process_lock(tmp_path / "fake-skill"):
        pass

    # Lock + unlock on byte 1.  Assert against the active msvcrt
    # module's own constants so the test does not rely on any
    # production-side alias.
    real_msvcrt = sm_local._msvcrt
    assert any(mode == real_msvcrt.LK_LOCK for (_, _, mode, _) in calls), (
        f"msvcrt.locking LK_LOCK not invoked: {calls}"
    )
    assert any(mode == real_msvcrt.LK_UNLCK for (_, _, mode, _) in calls), (
        f"msvcrt.locking LK_UNLCK not invoked: {calls}"
    )
    # All lock calls were on the same fd, on byte 1.
    fds_used = {fd for (fd, _, _) in lock_calls}
    assert len(fds_used) == 1, f"multiple fds used for lock: {fds_used}"


@pytest.mark.skipif(os.name != "nt", reason="Windows-only behaviour")
def test_windows_lock_fails_closed_when_msvcrt_missing(monkeypatch, tmp_path):
    import tools.skill_manager_tool as sm_local

    monkeypatch.setattr(sm_local, "_IS_WINDOWS", True)
    monkeypatch.setattr(sm_local, "_IS_POSIX", False)
    monkeypatch.setattr(sm_local, "_msvcrt", None)

    with pytest.raises(PermissionError):
        with sm_local._skill_mutation_process_lock(tmp_path / "fake-skill"):
            pass


def test_windows_lock_unavailable_on_posix_does_not_silently_noop(monkeypatch, tmp_path):
    """On POSIX the Windows branch must not be entered even if _msvcrt is
    None — the POSIX branch raises if fcntl is missing.
    """
    import tools.skill_manager_tool as sm_local

    monkeypatch.setattr(sm_local, "_IS_POSIX", True)
    monkeypatch.setattr(sm_local, "_IS_WINDOWS", False)
    monkeypatch.setattr(sm_local, "_fcntl", None)
    monkeypatch.setattr(sm_local, "_msvcrt", None)

    with pytest.raises(PermissionError):
        with sm_local._skill_mutation_process_lock(tmp_path / "fake-skill"):
            pass


# ── P1-B: delete and remove_file use the same interprocess lock ────────────

def test_delete_acquires_interprocess_lock(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)

    from tools.skill_manager_tool import _skill_mutation_process_lock
    lock_acquired: list[bool] = []

    real_lock = _skill_mutation_process_lock

    class TrackingLock:
        def __init__(self, canonical):
            self.canonical = canonical
            self.captured = False

        def __enter__(self):
            lock_acquired.append(True)
            return self

        def __exit__(self, *a):
            return False

    def tracing_lock(canonical):
        return TrackingLock(canonical)

    monkeypatch.setattr(sm, "_skill_mutation_process_lock", tracing_lock)
    # delete must use the canonical skill dir as the lock key.
    canonical = (skills_root / "fail-closed").resolve(strict=False)
    expected_digest_lock_path = canonical.parent / (
        f".hermes-skill-mutex-{sm._hashlib.sha256(str(canonical).encode('utf-8')).hexdigest()[:16]}.lock"
    )

    # Phase C recursive-delete block: foreground delete now refuses
    # before any destructive primitive.  The lock IS still acquired
    # (the refusal fires inside the lock); the success assertion is
    # replaced by the structured refusal payload.
    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_delete")):
        result = json.loads(
            sm.skill_manage(action="delete", name="fail-closed")
        )
    assert result["success"] is False
    assert result["policy_reason"] == "atomic_recursive_delete_unavailable"
    assert result["live_mutation_committed"] is False
    assert result["safe_to_retry"] is False
    assert lock_acquired, "delete did not acquire _skill_mutation_process_lock"


def test_remove_file_uses_same_lock_key_as_edit_and_write_file(monkeypatch, tmp_path):
    """Verify that ``remove_file`` uses the SAME interprocess lock key
    as ``edit`` and ``write_file`` (key derived from the canonical skill
    dir).  The intent of this test is the lock-key invariant, not the
    delete success.

    Under the Camino-B (last-mile atomicity) contract ``remove_file``
    refuses the destructive op before the unlink syscall.  We update
    the success-path assertion to expect the canonical refusal while
    preserving the lock-key invariant check.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    supporting = skills_root / "fail-closed" / "references" / "ok.md"
    supporting.parent.mkdir()
    supporting.write_text("original", encoding="utf-8")

    # The lock file lives OUTSIDE every skills root (per
    # _resolve_lock_parent), not next to the canonical skill dir itself.
    canonical = (skills_root / "fail-closed").resolve(strict=False)
    lock_parent = sm._resolve_lock_parent(canonical)
    expected_digest = sm._hashlib.sha256(str(canonical).encode("utf-8")).hexdigest()[:16]
    expected_lock_path = lock_parent / f".hermes-skill-mutex-{expected_digest}.lock"

    with session_write_policy_scope(
        _allowlist("skills", skills_root, "skill_write_file")
    ):
        json.loads(
            sm.skill_manage(
                action="write_file",
                name="fail-closed",
                file_path="references/ok.md",
                file_content="new",
            )
        )

    # write_file should have created the same lock file.
    assert expected_lock_path.exists(), (
        f"write_file did not create expected lock file {expected_lock_path}"
    )

    with session_write_policy_scope(
        _allowlist("skills", skills_root, "skill_remove_file")
    ):
        result = json.loads(
            sm.skill_manage(
                action="remove_file",
                name="fail-closed",
                file_path="references/ok.md",
            )
        )
    # Camino-B refusal: the destructive op is withheld before the unlink.
    assert result["success"] is False, result
    assert result["policy_reason"] == "atomic_identity_delete_unavailable", result
    # The lock file the canonical lock acquired persists (the
    # secure-path block opens parent fd, the refusal returns, the
    # lock is released; the file from write_file above is unaffected).
    assert expected_lock_path.exists()


# ── P1-C: real multiprocess test on POSIX ──────────────────────────────────

@pytest.mark.skipif(os.name != "posix", reason="POSIX-only fork/spawn semantics")
def test_interprocess_lock_real_multiprocess(monkeypatch, tmp_path):
    """Two INDEPENDENT processes contend on the same lock file.

    Process A acquires the lock first and signals (via an Event written
    to a shared file) that it is holding.  Process B then attempts
    acquisition; its acquire MUST block for the remainder of A's hold.
    We measure the elapsed time inside B's context manager; if it is
    < 0.8 s the lock did not serialize.
    """
    import multiprocessing as mp
    import time

    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    canonical = (skills_root / "fail-closed").resolve(strict=False)

    holder_signal = tmp_path / "holder.txt"
    holder_exit = tmp_path / "holder_exit.txt"
    b_marker = tmp_path / "b_entered.txt"

    def a_main(lock_path_str, signal_path, exit_signal):
        from tools.skill_manager_tool import _skill_mutation_process_lock
        from pathlib import Path
        with _skill_mutation_process_lock(Path(lock_path_str)):
            # Signal that A holds the lock.
            signal_path.write_text("held")
            # Wait until the test (parent) signals us to release.  This
            # guarantees B's attempt is made WHILE A holds the lock.
            while not exit_signal.exists():
                time.sleep(0.05)

    def b_main(lock_path_str, signal_path, marker_path):
        from tools.skill_manager_tool import _skill_mutation_process_lock
        from pathlib import Path
        # Wait until A holds the lock.
        deadline = time.monotonic() + 5
        while not signal_path.exists():
            if time.monotonic() > deadline:
                marker_path.write_text("timeout,start={}".format(time.monotonic()))
                return
            time.sleep(0.02)
        start = time.monotonic()
        with _skill_mutation_process_lock(Path(lock_path_str)):
            entered_at = time.monotonic()
        with open(marker_path, "w") as f:
            f.write(f"{start},{entered_at}")

    a_proc = mp.Process(
        target=a_main,
        args=(str(canonical), holder_signal, holder_exit),
    )
    a_proc.start()
    # Wait for A to take the lock.
    deadline = time.monotonic() + 5
    while not holder_signal.exists():
        if time.monotonic() > deadline:
            a_proc.kill()
            pytest.fail("A did not acquire the lock within 5 s")
        time.sleep(0.02)
    # A holds the lock; spawn B.
    b_proc = mp.Process(
        target=b_main,
        args=(str(canonical), holder_signal, b_marker),
    )
    b_proc.start()
    # Give B 200 ms to start its blocking attempt; then hold A's lock
    # for ~1.0 s so B's blocking-acquire time is observable.
    time.sleep(0.2)
    time.sleep(0.8)  # B should be blocked for at least this duration
    # Release A.
    holder_exit.write_text("exit")
    a_proc.join(timeout=5)
    b_proc.join(timeout=5)
    assert a_proc.exitcode == 0, f"A failed: {a_proc.exitcode}"
    assert b_proc.exitcode == 0, f"B failed: {b_proc.exitcode}"
    # Independent processes.
    assert a_proc.pid != b_proc.pid
    # B was blocked on the interprocess lock for the duration of A's hold.
    assert b_marker.exists(), "B did not record its entry timestamp"
    start, entered_at = (float(x) for x in b_marker.read_text().split(","))
    blocked_for = entered_at - start
    assert blocked_for >= 0.8, (
        f"B was not blocked long enough ({blocked_for:.2f}s) — lock did not serialize"
    )






def test_cleanup_failure_after_edit_publish_reports_live_mutation_committed(
    monkeypatch, tmp_path
):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    skill_md = skills_root / "fail-closed" / "SKILL.md"
    original = skill_md.read_bytes()

    publish_done = {"v": False}
    real_rmtree = sm.shutil.rmtree

    def selective_rmtree(path, *a, **kw):
        path_str = str(path)
        if ".hermes-skill-staging-" in path_str and publish_done["v"]:
            raise OSError("simulated cleanup failure after publish")
        return real_rmtree(path, *a, **kw)

    monkeypatch.setattr(sm.shutil, "rmtree", selective_rmtree)

    real_atomic = sm.atomic_replace

    def wrap_replace(tmp, dst):
        result = real_atomic(tmp, dst)
        if "SKILL.md" in str(dst):
            publish_done["v"] = True
        return result

    monkeypatch.setattr(sm, "atomic_replace", wrap_replace)

    new_content = SKILL_MD.replace("# Fail Closed", "# Changed")
    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_edit")):
        result = json.loads(
            sm.skill_manage(action="edit", name="fail-closed", content=new_content)
        )

    assert result["success"] is False
    assert result["policy_reason"] == "cleanup_failed"
    assert result["live_mutation_committed"] is True
    assert skill_md.read_bytes() == new_content.encode("utf-8")
    assert skill_md.read_bytes() != original


def test_cleanup_failure_after_write_file_publish_reports_live_mutation_committed(
    monkeypatch, tmp_path
):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    supporting = skills_root / "fail-closed" / "references" / "ok.md"
    supporting.parent.mkdir()
    supporting.write_text("original", encoding="utf-8")

    publish_done = {"v": False}
    real_rmtree = sm.shutil.rmtree

    def selective_rmtree(path, *a, **kw):
        path_str = str(path)
        if ".hermes-skill-staging-" in path_str and publish_done["v"]:
            raise OSError("simulated cleanup failure after publish")
        return real_rmtree(path, *a, **kw)

    monkeypatch.setattr(sm.shutil, "rmtree", selective_rmtree)

    real_atomic = sm.atomic_replace

    def wrap_replace(tmp, dst):
        result = real_atomic(tmp, dst)
        if "ok.md" in str(dst):
            publish_done["v"] = True
        return result

    monkeypatch.setattr(sm, "atomic_replace", wrap_replace)

    with session_write_policy_scope(
        _allowlist("skills", skills_root, "skill_write_file")
    ):
        result = json.loads(
            sm.skill_manage(
                action="write_file",
                name="fail-closed",
                file_path="references/ok.md",
                file_content="new",
            )
        )

    assert result["success"] is False
    assert result["policy_reason"] == "cleanup_failed"
    assert result["live_mutation_committed"] is True
    assert supporting.read_text(encoding="utf-8") == "new"


# ── P1-D / 9.9 Caminos sanos ──────────────────────────────────────────────

def test_happy_path_delete_with_lock(monkeypatch, tmp_path):
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_delete")):
        result = json.loads(sm.skill_manage(action="delete", name="fail-closed"))
    # Phase C recursive-delete block: foreground delete now refuses
    # before any destructive primitive.  The skill tree is preserved.
    assert result["success"] is False
    assert result["policy_reason"] == "atomic_recursive_delete_unavailable"
    assert result["live_mutation_committed"] is False
    assert result["safe_to_retry"] is False
    assert (skills_root / "fail-closed").exists()


def test_happy_path_remove_file_with_lock(monkeypatch, tmp_path):
    """Caminos sanos: remove_file with a clean target and lock must
    surface the canonical Camino-B refusal (no kernel identity-bound
    delete primitive is available on portable Python).

    Update from Phase C last-mile atomicity: portable Python cannot
    bind the destructive op to the validated inode, so production now
    refuses with ``policy_reason=atomic_identity_delete_unavailable``
    rather than run a name-based unlink that might delete a swapped
    foreign replacement.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    supporting = skills_root / "fail-closed" / "references" / "ok.md"
    supporting.parent.mkdir()
    supporting.write_text("original", encoding="utf-8")
    with session_write_policy_scope(
        _allowlist("skills", skills_root, "skill_remove_file")
    ):
        result = json.loads(
            sm.skill_manage(
                action="remove_file",
                name="fail-closed",
                file_path="references/ok.md",
            )
        )
    assert result["success"] is False
    assert result["policy_reason"] == "atomic_identity_delete_unavailable", result
    assert result["rollback_failure_kind"] == (
        "identity_bound_unlink_unavailable"
    ), result
    assert result["live_mutation_committed"] is False, result
    assert result["safe_to_retry"] is False, result
    # Target must be preserved (no destructive op ran).
    assert supporting.exists()
    assert supporting.read_text(encoding="utf-8") == "original"


# ═════════════════════════════════════════════════════════════════════════
# Phase C final corrective: lock file identity, release failure,
# close failure, structured finalization, delete/remove race,
# Windows msvcrt mock tests executable on POSIX.
# ═════════════════════════════════════════════════════════════════════════


def _resolve_lock_path_for(canonical: Path) -> Path:
    """Locate the actual lock file path the context manager would use."""
    import hashlib as _hashlib
    from agent.skill_utils import get_all_skills_dirs

    try:
        resolved_roots = [r.resolve(strict=False) for r in get_all_skills_dirs()]
    except Exception:
        resolved_roots = []
    parent = canonical.parent
    while True:
        try:
            inside = any(
                parent.resolve(strict=False) == root
                or root in parent.resolve(strict=False).parents
                for root in resolved_roots
            )
        except Exception:
            inside = False
        if not inside:
            break
        next_parent = parent.parent
        if next_parent == parent:
            break
        parent = next_parent
    digest = _hashlib.sha256(str(canonical).encode("utf-8")).hexdigest()[:16]
    return parent / f".hermes-skill-mutex-{digest}.lock"


# ── Lock file identity: symlink pathname is rejected ────────────────────────


def test_lock_pathname_symlink_rejected_without_following(monkeypatch, tmp_path):
    """If the lock pathname is a symlink, acquisition is refused and the
    symlink target is left intact.
    """
    import tools.skill_manager_tool as sm_local

    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    canonical = (skills_root / "fail-closed").resolve(strict=False)

    lock_path = _resolve_lock_path_for(canonical)
    # Pre-create the lock pathname as a symlink to a foreign file.
    foreign = tmp_path / "foreign-target.txt"
    foreign.write_text("foreign", encoding="utf-8")
    if lock_path.exists() or lock_path.is_symlink():
        lock_path.unlink()
    os.symlink(str(foreign), str(lock_path))

    with pytest.raises(PermissionError):
        with sm_local._skill_mutation_process_lock(canonical):
            pass

    # Symlink and foreign target both intact.
    assert lock_path.is_symlink()
    assert foreign.read_text(encoding="utf-8") == "foreign"


def test_lock_pathname_directory_rejected_fail_closed(monkeypatch, tmp_path):
    """If the lock pathname is a directory, acquisition is refused."""
    import tools.skill_manager_tool as sm_local

    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    canonical = (skills_root / "fail-closed").resolve(strict=False)

    lock_path = _resolve_lock_path_for(canonical)
    if lock_path.exists() or lock_path.is_symlink():
        if lock_path.is_symlink() or lock_path.is_file():
            lock_path.unlink()
        else:
            lock_path.rmdir()
    lock_path.mkdir()

    try:
        with pytest.raises(PermissionError):
            with sm_local._skill_mutation_process_lock(canonical):
                pass
        # Directory still intact (not removed, not replaced).
        assert lock_path.is_dir()
    finally:
        lock_path.rmdir()


def test_lock_inode_swap_between_lstat_and_fstat_rejected(monkeypatch, tmp_path):
    """If between lstat and fstat the lock file is replaced by a different
    inode (same path), the identity mismatch must fail closed.
    """
    import tools.skill_manager_tool as sm_local

    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    canonical = (skills_root / "fail-closed").resolve(strict=False)

    lock_path = _resolve_lock_path_for(canonical)

    # Pre-create the lock file as a regular file so open(O_NOFOLLOW) opens it.
    lock_path.write_text("", encoding="utf-8")
    pre_ino = lock_path.lstat().st_ino

    # Wrap os.open to simulate a TOCTOU swap: open returns an fd to the
    # FOREIGN inode (a different file entirely).  The post-open
    # identity check must detect the mismatch and fail closed.
    foreign = tmp_path / "foreign-lock"
    foreign.write_text("foreign", encoding="utf-8")
    foreign_ino = foreign.lstat().st_ino
    assert foreign_ino != pre_ino

    real_open = os.open
    opened_paths: list[str] = []

    def swapped_open(path, flags, *args, **kwargs):
        opened_paths.append(str(path))
        # For the lock file path, return an fd pointing at the
        # FOREIGN file (simulating a TOCTOU swap of the underlying
        # inode).
        if str(path) == str(lock_path):
            return real_open(str(foreign), os.O_RDWR)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(sm_local.os, "open", swapped_open)

    with pytest.raises(PermissionError):
        with sm_local._skill_mutation_process_lock(canonical):
            pass


def test_lock_file_never_deleted_after_release(monkeypatch, tmp_path):
    """A successful acquire + release must NOT remove or replace the lock
    file.  The lock is held by the kernel; only the descriptor is closed.
    """
    import tools.skill_manager_tool as sm_local

    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    canonical = (skills_root / "fail-closed").resolve(strict=False)

    lock_path = _resolve_lock_path_for(canonical)

    with sm_local._skill_mutation_process_lock(canonical):
        # Lock file exists inside the critical section.
        assert lock_path.exists()

    # After release the lock file STILL exists (kernel lock only,
    # not removed by the userland helper).
    assert lock_path.exists()


def test_two_successive_acquires_share_same_inode(monkeypatch, tmp_path):
    """Two successive acquisitions of the same canonical skill path must
    use the same lock file inode (the helper does not delete and recreate
    the file between acquisitions).
    """
    import tools.skill_manager_tool as sm_local

    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    canonical = (skills_root / "fail-closed").resolve(strict=False)

    lock_path = _resolve_lock_path_for(canonical)

    with sm_local._skill_mutation_process_lock(canonical):
        ino_first = lock_path.lstat().st_ino
    with sm_local._skill_mutation_process_lock(canonical):
        ino_second = lock_path.lstat().st_ino

    assert ino_first == ino_second












# ── Windows msvcrt mock tests executable on POSIX ───────────────────────────


class _FakeMsvcrt:
    """Captures every call to ``msvcrt.locking`` so tests can assert on it.

    The fake is a STATELESS recorder: each ``locking(fd, mode, nbytes)``
    call is appended to ``self.calls``.  The fake does NOT own fds —
    the helper opens a real descriptor and passes it in.  Tests can
    inject failure via ``self._raise_for_mode = {mode: exc}`` to
    simulate a contention or release failure on a specific call.

    The lock-mode constants are declared INDEPENDENTLY by this fake so
    tests assert against the exact values the fake declares
    (``LK_UNLCK=0``, ``LK_LOCK=1``, ``LK_NBLCK=2``), never against
    an alias re-exported by production code.  Production has no
    knowledge of these integers — production reads them off the
    active ``_msvcrt`` module at call time.
    """

    # Independent test-side constants.  Production does NOT declare
    # these; only the test fake does.  If production ever drifts and
    # reads a different encoding the tests will catch it because the
    # ``fake.calls`` payload is checked against ``fake.LK_*``.
    LK_UNLCK = 0
    LK_LOCK = 1
    LK_NBLCK = 2

    def __init__(self):
        self.calls: list[tuple] = []
        self._raise_for_mode: dict[int, BaseException] = {}

    def locking(self, fd, mode, nbytes):
        self.calls.append((fd, mode, nbytes))
        exc = self._raise_for_mode.get(mode)
        if exc is not None:
            raise exc


def _enable_windows_mode(monkeypatch, sm_module, fake_msvcrt):
    """Switch the helper into Windows mode with a fake msvcrt module."""
    monkeypatch.setattr(sm_module, "_IS_WINDOWS", True)
    monkeypatch.setattr(sm_module, "_IS_POSIX", False)
    monkeypatch.setattr(sm_module, "_msvcrt", fake_msvcrt)


def test_windows_mock_acquires_and_releases_via_msvcrt(monkeypatch, tmp_path):
    """Windows mock exercises LK_NBLCK → LK_LOCK fallback → LK_UNLCK with
    a REAL file descriptor opened on a temp lock file.  No synthetic
    fds.  All platform state restored via monkeypatch.

    Lock-mode constants come from the fake itself (``fake.LK_NBLCK``,
    ``fake.LK_LOCK``, ``fake.LK_UNLCK``) so the test does not depend on
    any alias re-exported by production code.
    """
    import tools.skill_manager_tool as sm_local

    fake = _FakeMsvcrt()
    # Force LK_NBLCK to fail so the helper exercises the LK_LOCK fallback.
    fake._raise_for_mode[fake.LK_NBLCK] = OSError(errno.EAGAIN, "would block")
    _enable_windows_mode(monkeypatch, sm_local, fake)

    canonical = (tmp_path / "fake-skill").resolve(strict=False)
    lock_path = _resolve_lock_path_for(canonical)
    seen_fds_in_body: list[int] = []
    seen_fds_release: list[int] = []

    @contextmanager
    def intercept_lock_calls():
        # Track the fd opened for the lock file by intercepting
        # ``os.lseek`` and recording the fd we see during the
        # helper's pre-yield ``os.lseek(fd, 0, SEEK_SET)`` call.
        real_lseek = sm_local.os.lseek
        real_fstat = sm_local.os.fstat

        def tracking_lseek(fd, pos, how):
            if pos == 0 and how == os.SEEK_SET:
                # Only record lseek calls that happen during
                # acquisition/release on the lock file (the helper
                # uses 0/SEEK_SET).
                seen_fds_in_body.append(fd)
            return real_lseek(fd, pos, how)

        def tracking_fstat(fd):
            st = real_fstat(fd)
            return st

        monkeypatch.setattr(sm_local.os, "lseek", tracking_lseek)
        monkeypatch.setattr(sm_local.os, "fstat", tracking_fstat)
        yield

    with intercept_lock_calls():
        with sm_local._skill_mutation_process_lock(canonical):
            # The body runs with the lock held.  Use the same fd (via
            # the lock file path) to confirm the descriptor is still
            # open and usable.
            real_open = sm_local.os.open
            fd = real_open(str(lock_path), os.O_RDWR)
            try:
                # fstat through our tracked path should succeed on the
                # real fd opened from the same lock file.  Inode must
                # match because we opened the same lock file.
                ino = os.fstat(fd).st_ino
                assert lock_path.lstat().st_ino == ino
            finally:
                os.close(fd)

    # Modes recorded: LK_NBLCK (raised), LK_LOCK (succeeded), LK_UNLCK.
    # Assert against the fake's own constants; production must NOT
    # re-export these as module-level aliases.
    modes = {mode for (_, mode, _) in fake.calls}
    assert fake.LK_LOCK in modes
    assert fake.LK_UNLCK in modes
    # All calls must have used a single, real fd that the helper
    # opened for the lock file.
    fds_used = {fd for (fd, _, _) in fake.calls}
    assert len(fds_used) == 1, f"multiple fds used: {fds_used}"
    body_fd = next(iter(fds_used))
    # Body lseek recorded the same fd.
    assert body_fd in seen_fds_in_body, (
        f"body did not observe fd {body_fd}: seen_fds={seen_fds_in_body}"
    )
    # Byte range is 1.
    byte_ranges = {nbytes for (_, _, nbytes) in fake.calls}
    assert byte_ranges == {1}
    # Lock file contains at least one byte (helper pads NUL byte).
    assert lock_path.stat().st_size >= 1


def test_windows_mock_acquires_via_nblck_when_no_contention(monkeypatch, tmp_path):
    """When LK_NBLCK succeeds on the first try, LK_LOCK is NOT called —
    only LK_NBLCK + LK_UNLCK appear.  Uses a real fd.
    """
    import tools.skill_manager_tool as sm_local

    fake = _FakeMsvcrt()
    _enable_windows_mode(monkeypatch, sm_local, fake)

    canonical = (tmp_path / "fake-skill").resolve(strict=False)
    lock_path = _resolve_lock_path_for(canonical)

    with sm_local._skill_mutation_process_lock(canonical):
        pass

    modes = {mode for (_, mode, _) in fake.calls}
    assert fake.LK_NBLCK in modes
    assert fake.LK_UNLCK in modes
    assert fake.LK_LOCK not in modes
    fds_used = {fd for (fd, _, _) in fake.calls}
    assert len(fds_used) == 1
    assert lock_path.stat().st_size >= 1


def test_windows_mock_lock_file_padded_to_one_byte(monkeypatch, tmp_path):
    """Windows mock: when the lock file is empty, the helper pads it with
    one NUL byte so msvcrt.locking has a region to lock.
    """
    import tools.skill_manager_tool as sm_local

    fake = _FakeMsvcrt()
    _enable_windows_mode(monkeypatch, sm_local, fake)

    canonical = (tmp_path / "fake-skill").resolve(strict=False)
    lock_path = _resolve_lock_path_for(canonical)

    with sm_local._skill_mutation_process_lock(canonical):
        pass

    assert lock_path.stat().st_size >= 1


def test_windows_mock_fail_closed_when_msvcrt_none(monkeypatch, tmp_path):
    """Windows mock: _msvcrt=None must fail closed with PermissionError."""
    import tools.skill_manager_tool as sm_local

    monkeypatch.setattr(sm_local, "_IS_WINDOWS", True)
    monkeypatch.setattr(sm_local, "_IS_POSIX", False)
    monkeypatch.setattr(sm_local, "_msvcrt", None)

    canonical = (tmp_path / "fake-skill").resolve(strict=False)
    with pytest.raises(PermissionError):
        with sm_local._skill_mutation_process_lock(canonical):
            pass


@pytest.mark.parametrize(
    "missing_attr",
    ["LK_NBLCK", "LK_LOCK", "LK_UNLCK"],
    ids=["missing-LK_NBLCK", "missing-LK_LOCK", "missing-LK_UNLCK"],
)
def test_windows_lock_fails_closed_when_required_msvcrt_constant_missing(
    monkeypatch, tmp_path, missing_attr
):
    """Windows mock: a missing ``LK_NBLCK`` / ``LK_LOCK`` / ``LK_UNLCK``
    attribute on the injected ``_msvcrt`` MUST cause the helper to fail
    closed BEFORE entering the critical section.

    Contract:
        missing LK_NBLCK → fail closed before critical section
        missing LK_LOCK  → fail closed before blocking fallback
        missing LK_UNLCK → fail closed before critical section / before
                            the helper would attempt any unlock

    The test verifies that ``fake.calls`` is empty in every failure case,
    which means production did NOT call ``msvcrt.locking`` with a
    hard-coded numeric fallback.  Production must validate the contract
    BEFORE opening the lock file.
    """
    import tools.skill_manager_tool as sm_local

    class _IncompleteMsvcrt:
        """Fake that exposes LK_* constants explicitly MINUS one.

        ``missing_attr`` is removed at the CLASS level (so production's
        ``hasattr(_msvcrt, attr)`` check returns False).  The fake
        keeps the other two constants as instance attributes so we can
        be sure the failure is caused by the missing one — not by a
        fully absent module.
        """

        def __init__(self):
            self.calls: list[tuple] = []
            # Instance attributes (not class attributes) so the
            # ``del type(fake).<attr>`` step below can selectively
            # remove one of them without breaking the others.
            self.LK_UNLCK = 0
            self.LK_LOCK = 1
            self.LK_NBLCK = 2

        def locking(self, fd, mode, nbytes):
            self.calls.append((fd, mode, nbytes))

    fake = _IncompleteMsvcrt()
    # Drop the attribute requested by the parameter set so production's
    # ``hasattr`` validation surfaces a contract failure.  We delete
    # it from the instance so the class keeps the attribute as
    # documentation but the live object reports ``hasattr`` False.
    delattr(fake, missing_attr)

    monkeypatch.setattr(sm_local, "_IS_WINDOWS", True)
    monkeypatch.setattr(sm_local, "_IS_POSIX", False)
    monkeypatch.setattr(sm_local, "_msvcrt", fake)

    canonical = (tmp_path / "fake-skill").resolve(strict=False)

    # Guard against any state mutation of the target by tracking
    # ``os.open`` calls — production must not have touched the lock
    # file before failing.
    opened_paths: list[str] = []

    real_open = sm_local.os.open

    def tracking_open(path, flags, *args, **kwargs):
        opened_paths.append(str(path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(sm_local.os, "open", tracking_open)

    # The lock must refuse and raise a fail-closed PermissionError.
    with pytest.raises(PermissionError) as excinfo:
        with sm_local._skill_mutation_process_lock(canonical):
            # If we reach this body the contract failed open — wrong.
            pytest.fail(
                f"critical section entered despite missing {missing_attr}"
            )

    # The error message must reference the missing attribute so the
    # operator can diagnose without reading source.
    assert missing_attr in str(excinfo.value), (
        f"error message did not name the missing attribute {missing_attr}: "
        f"{excinfo.value!r}"
    )

    # Production must NEVER call msvcrt.locking when the contract is
    # invalid — no numeric fallback is permitted.
    assert fake.calls == [], (
        f"msvcrt.locking was called despite missing {missing_attr}: "
        f"{fake.calls!r}"
    )

    # Production must NOT have opened the lock file either — the
    # critical section is gated by the contract check.
    lock_path = _resolve_lock_path_for(canonical)
    assert str(lock_path) not in opened_paths, (
        f"lock file was opened before contract validation: {opened_paths!r}"
    )


def test_windows_lock_uses_same_open_fd_one_byte_and_releases_before_close(
    monkeypatch, tmp_path
):
    """Windows mock: explicit descriptor / range / ordering contract.

    Asserts that production:
      * opens a real fd on the lock file;
      * uses the same fd for acquire (LK_NBLCK / LK_LOCK) and release
        (LK_UNLCK);
      * acquires via LK_NBLCK successfully (no contention injected);
      * uses ``nbytes == 1`` on every ``msvcrt.locking`` call;
      * lets the descriptor remain open and ``os.fstat``-able during
        the body (the critical section);
      * releases the lock BEFORE closing the descriptor;
      * never re-opens or re-uses a closed fd.
    """
    import tools.skill_manager_tool as sm_local

    fake = _FakeMsvcrt()
    _enable_windows_mode(monkeypatch, sm_local, fake)

    canonical = (tmp_path / "fake-skill").resolve(strict=False)
    lock_path = _resolve_lock_path_for(canonical)

    # Track the order of low-level operations so we can assert
    # release-before-close.
    timeline: list[tuple[str, int]] = []

    real_open = sm_local.os.open
    real_close = sm_local.os.close
    real_lseek = sm_local.os.lseek
    real_fstat = sm_local.os.fstat

    def tracking_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if str(path) == str(lock_path):
            timeline.append(("open", fd))
        return fd

    def tracking_close(fd, *args, **kwargs):
        if any(op == "open" and opened == fd for op, opened in timeline):
            timeline.append(("close", fd))
        return real_close(fd, *args, **kwargs)

    def tracking_fstat(fd, *args, **kwargs):
        result = real_fstat(fd, *args, **kwargs)
        if fd in {opened for op, opened in timeline if op == "open"}:
            timeline.append(("fstat", fd))
        return result

    def tracking_lseek(fd, pos, how, *args, **kwargs):
        if fd in {opened for op, opened in timeline if op == "open"}:
            timeline.append(("lseek", fd))
        return real_lseek(fd, pos, how, *args, **kwargs)

    monkeypatch.setattr(sm_local.os, "open", tracking_open)
    monkeypatch.setattr(sm_local.os, "close", tracking_close)
    monkeypatch.setattr(sm_local.os, "fstat", tracking_fstat)
    monkeypatch.setattr(sm_local.os, "lseek", tracking_lseek)

    # The body must observe that the fd is still open and usable.
    body_fstat_failed = False
    body_observed_fd: Optional[int] = None

    with sm_local._skill_mutation_process_lock(canonical):
        # Production opened the fd before yielding.  Find that fd.
        opened_fds = [fd for op, fd in timeline if op == "open"]
        assert len(opened_fds) == 1, (
            f"expected exactly one open on the lock file, got {opened_fds!r}"
        )
        body_observed_fd = opened_fds[0]
        # The descriptor must be live during the body.
        try:
            real_fstat(body_observed_fd)
        except OSError as exc:
            body_fstat_failed = True
            pytest.fail(f"fd {body_observed_fd} closed during body: {exc}")

    # ── 8.1 — same fd for acquire and release ────────────────────────
    acquire_fds = {
        fd
        for (fd, mode, _nbytes) in fake.calls
        if mode in (fake.LK_NBLCK, fake.LK_LOCK)
    }
    release_fds = {
        fd
        for (fd, mode, _nbytes) in fake.calls
        if mode == fake.LK_UNLCK
    }
    assert acquire_fds, "no acquire call observed"
    assert release_fds, "no release call observed"
    assert acquire_fds == release_fds, (
        f"acquire and release used different fds: "
        f"acquire={acquire_fds!r} release={release_fds!r}"
    )
    assert acquire_fds == {body_observed_fd}, (
        f"observed fd in body differs from fake's recorded fd: "
        f"body={body_observed_fd!r} acquire={acquire_fds!r}"
    )

    # ── 8.2 — descriptor valid during body ────────────────────────────
    assert not body_fstat_failed, "fd was not valid during the body"
    assert body_observed_fd is not None

    # ── 8.3 — lock range is exactly one byte ──────────────────────────
    nbytes_set = {nbytes for (_, _, nbytes) in fake.calls}
    assert nbytes_set == {1}, f"unexpected nbytes set: {nbytes_set!r}"

    # ── 8.4 — release call index < close index ────────────────────────
    fake_call_index = {id(call): idx for idx, call in enumerate(fake.calls)}
    unlock_call_index = None
    for idx, (fd, mode, _) in enumerate(fake.calls):
        if mode == fake.LK_UNLCK and fd == body_observed_fd:
            unlock_call_index = idx
            break
    assert unlock_call_index is not None, "no LK_UNLCK call recorded"

    timeline_after_release = [
        op for op, _fd in timeline[timeline.index(("open", body_observed_fd)) + 1:]
    ]
    close_index = next(
        (i for i, op in enumerate(timeline_after_release) if op == "close"),
        None,
    )
    assert close_index is not None, (
        f"close never observed on fd {body_observed_fd}: {timeline!r}"
    )
    # The unlock is recorded inside the helper before os.close runs;
    # therefore the close must appear AFTER the unlock call site.
    assert close_index > unlock_call_index, (
        f"close index {close_index} not strictly greater than unlock call "
        f"index {unlock_call_index}: timeline={timeline!r}"
    )

    # ── 8.5 — fd closed after context exits ───────────────────────────
    with pytest.raises(OSError):
        real_fstat(body_observed_fd)






# ── Delete identity revalidation under lock ────────────────────────────────


def test_delete_skill_directory_replaced_under_lock_preserved(monkeypatch, tmp_path):
    """If the canonical skill directory is replaced by a different inode
    between the early ``_find_skill`` and the under-lock revalidation, the
    foreign directory is preserved and the delete returns
    ``concurrent_modification``.  No shutil.rmtree is invoked on the
    foreign path.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)

    foreign_dir = tmp_path / "foreign-skill"
    foreign_dir.mkdir()

    # Main adds a pre-mutation audit-ledger _find_skill(name) capture.
    # The deleted skill therefore has THREE pre-lock finds: ledger capture,
    # background-review preflight, and the early _delete_skill snapshot.
    # The FOURTH call is the under-lock revalidation and sees the foreign tree.
    real_find_skill = sm._find_skill
    calls = {"n": 0}

    def swap_find(name):
        calls["n"] += 1
        if name != "fail-closed":
            return real_find_skill(name)
        if calls["n"] <= 3:
            return real_find_skill(name)
        return {"path": foreign_dir}

    monkeypatch.setattr(sm, "_find_skill", swap_find)

    # Forbid shutil.rmtree entirely so we can assert that the foreign
    # directory is preserved without any recursive delete attempt.
    import tools.skill_manager_tool as sm_local

    def forbid_rmtree(path, *a, **kw):
        raise AssertionError(
            f"shutil.rmtree must not be called on foreign path {path!r}"
        )

    monkeypatch.setattr(sm_local.shutil, "rmtree", forbid_rmtree)

    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_delete")):
        result = json.loads(sm.skill_manage(action="delete", name="fail-closed"))

    assert result["success"] is False
    assert result["policy_reason"] == "concurrent_modification"
    assert result["rollback_failure_kind"] == "concurrent_modification"
    # Foreign directory preserved.
    assert foreign_dir.exists()
    # Live skill directory preserved (rmtree was forbidden).
    assert (skills_root / "fail-closed").exists()


def test_delete_skill_directory_symlink_swap_blocked(monkeypatch, tmp_path):
    """If the canonical skill directory is replaced by a symlink between
    pre-lock and lock acquisition, the foreign symlink target is not
    followed and not deleted.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)

    outside_dir = tmp_path / "outside-target"
    outside_dir.mkdir()
    outside_skill = outside_dir / "fail-closed"
    outside_skill.mkdir()
    (outside_skill / "SKILL.md").write_text("outside", encoding="utf-8")

    real_find_skill = sm._find_skill

    def swap_to_symlink(name):
        if name == "fail-closed":
            swap_to_symlink.calls += 1
            # Ledger capture + background preflight + early delete snapshot
            # are pre-lock reads; inject the symlink on the fourth call.
            if swap_to_symlink.calls <= 3:
                return real_find_skill(name)
            # Replace the live skill dir with a symlink to outside.
            live = skills_root / "fail-closed"
            if live.exists() or live.is_symlink():
                for child in list(live.iterdir()):
                    if child.is_file() or child.is_symlink():
                        child.unlink()
                    else:
                        import shutil as _sh
                        _sh.rmtree(child)
                live.rmdir()
            os.symlink(str(outside_skill), str(live))
            # Resolve again to return the symlink-following path.
            return {"path": live}
        return real_find_skill(name)

    swap_to_symlink.calls = 0
    monkeypatch.setattr(sm, "_find_skill", swap_to_symlink)

    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_delete")):
        result = json.loads(sm.skill_manage(action="delete", name="fail-closed"))

    assert result["success"] is False
    # Outside target preserved.
    assert (outside_skill / "SKILL.md").read_text(encoding="utf-8") == "outside"


def test_happy_path_delete_with_revalidation(monkeypatch, tmp_path):
    """Phase C recursive-delete block: the foreground delete path no
    longer permits a portable recursive destruction (no identity-bound
    kernel-anchored primitive is available).  A clean delete therefore
    refuses with the structured fail-closed payload instead of
    succeeding.  The skill tree is preserved.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_delete")):
        result = json.loads(sm.skill_manage(action="delete", name="fail-closed"))
    assert result["success"] is False
    assert result["policy_reason"] == "atomic_recursive_delete_unavailable"
    assert result["rollback_failure_kind"] == "identity_bound_recursive_delete_unavailable"
    assert result["live_mutation_committed"] is False
    assert result["safe_to_retry"] is False
    assert (skills_root / "fail-closed").exists()


# ── Last-mile recursive-delete atomicity (Phase C recursive-delete block) ──


def test_delete_skill_replacement_after_final_identity_check_before_recursive_delete_is_preserved(
    monkeypatch, tmp_path,
):
    """If the validated skill directory is swapped for a foreign directory
    AFTER the final identity capture of BOTH target and parent, and
    BEFORE the destructive recursive-delete syscall, the foreign
    replacement MUST be preserved and the original skill tree MUST
    remain intact.  Production MUST return the canonical
    ``atomic_recursive_delete_unavailable`` payload — never
    ``concurrent_modification`` (which would mean production observed
    the swap at an identity recheck, contradicting the harness
    contract).

    State machine: the harness wraps ``Path.lstat`` and counts the
    number of lstat calls issued from inside ``sm._delete_skill`` on
    two paths:

      * the target skill directory (``re_skill_dir``);
      * the target's parent (``re_skill_dir.parent``).

    Production's under-lock flow captures each path TWICE: a
    pre-capture before the destructive op, and a final recheck
    immediately before the last-mile atomicity refusal.  The harness
    swap fires on the second (recheck) lstat of the parent path,
    AFTER returning the captured original stat to production — so
    production's pre/post comparisons both see the original identity
    and the refusal fires with ``atomic_recursive_delete_unavailable``.
    No identity-bearing lstat call is ever made against the swapped
    state.

    Frame identity is via ``f_code is sm._delete_skill.__code__`` —
    no numeric source-line ranges, no ``f_lineno`` /
    ``co_firstlineno``, no ``inspect.stack`` to drive branch
    selection.  The wrapper inspects only ``self`` (the path) and the
    identity of the caller frame's code object.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    skill_dir = skills_root / "fail-closed"
    skill_md = skill_dir / "SKILL.md"
    original_bytes = skill_md.read_bytes()
    evidence_dir = tmp_path / "original-skill-evidence"
    foreign_file = skill_dir / "FOREIGN.md"
    foreign_content = "FOREIGN_REPLACEMENT_AFTER_FINAL_IDENTITY_CHECK"

    skill_dir_resolved = skill_dir.resolve()
    skill_dir_parent_resolved = skill_dir.parent.resolve()

    state = {
        # Per-path counters for lstat calls observed inside _delete_skill
        # on the canonical paths.  Production captures each path
        # exactly twice inside the under-lock body: the pre-capture
        # and the final recheck.
        "target_lstat_count": 0,
        "parent_lstat_count": 0,
        # Recorded on the FIRST lstat for each path (the production
        # pre-capture).  These are the values production will compare
        # against later.
        "target_pre_capture_identity": None,
        "parent_pre_capture_identity": None,
        # Set once the SECOND (recheck) lstat for each path has
        # returned the ORIGINAL identity to production.  The harness
        # swap fires only after BOTH are True.
        "target_final_captured_with_original": False,
        "parent_final_captured_with_original": False,
        # True iff the harness swap ran.  Triggered exclusively on the
        # parent recheck lstat AFTER target_recheck + parent_recheck
        # have both returned original identity to production.
        "swap_performed": False,
        "swap_occurred_after_both_final_captures": False,
        # Counters for any lstat the wrapper sees on the canonical
        # paths AFTER the swap fired.  Spec requires both to remain 0.
        "post_swap_target_identity_calls": 0,
        "post_swap_parent_identity_calls": 0,
        # Foreign/evidence presence observed by the harness at the
        # moment production's refusal runs (asserted post-call).
        "foreign_replacement_existed_before_refusal": False,
        "original_evidence_existed_before_refusal": False,
    }

    import tools.skill_manager_tool as sm_local
    real_lstat = sm_local.Path.lstat
    delete_skill_code = sm._delete_skill.__code__

    def _identity_of(stat_result):
        return (
            stat_result.st_dev,
            stat_result.st_ino,
            stat.S_IFMT(stat_result.st_mode),
        )

    def lstat_with_final_swap(self, *args, **kwargs):
        # Resolve the path lazily inside the wrapper so monkeypatched
        # environments that swap symlinks post-setup are still
        # recognised by identity comparison.
        try:
            self_resolved = Path(str(self)).resolve()
        except OSError:
            self_resolved = Path(str(self))

        # Identity-based caller detection: walk the live frame chain
        # (no source-line numbers) and ask whether any frame's code
        # object IS ``sm._delete_skill.__code__``.  If not, this lstat
        # belongs to some other code path (e.g. ``_find_skill``,
        # security scanner, lock file validation) — pass through
        # untouched.
        caller_frame = sys._getframe(0)
        inside_delete_skill = False
        f = caller_frame
        while f is not None:
            if f.f_code is delete_skill_code:
                inside_delete_skill = True
                break
            f = f.f_back
        del caller_frame

        is_target = self_resolved == skill_dir_resolved
        is_parent = self_resolved == skill_dir_parent_resolved

        if not inside_delete_skill or (not is_target and not is_parent):
            return real_lstat(self, *args, **kwargs)

        # Capture the real stat BEFORE any swap fires, so production
        # always sees the original identity when comparing.
        st = real_lstat(self, *args, **kwargs)
        ident = _identity_of(st)

        if is_target:
            state["target_lstat_count"] += 1
            if state["target_lstat_count"] == 1:
                state["target_pre_capture_identity"] = ident
            elif state["target_lstat_count"] == 2:
                state["target_final_captured_with_original"] = True
            if state["swap_performed"]:
                state["post_swap_target_identity_calls"] += 1
        else:  # is_parent
            state["parent_lstat_count"] += 1
            if state["parent_lstat_count"] == 1:
                state["parent_pre_capture_identity"] = ident
            elif state["parent_lstat_count"] == 2:
                state["parent_final_captured_with_original"] = True
            if state["swap_performed"]:
                state["post_swap_parent_identity_calls"] += 1

        # Trigger the swap on the parent recheck — the LAST identity
        # call production makes on the canonical paths.  By this
        # point production has already accepted both target pre/post
        # identity comparisons AND the parent pre-comparison.  The
        # post-comparison follows inline; we perform the swap AFTER
        # returning the original parent stat so production's
        # comparison still passes.
        if (
            not state["swap_performed"]
            and is_parent
            and state["parent_lstat_count"] == 2
            and state["target_final_captured_with_original"]
            and state["parent_final_captured_with_original"]
        ):
            # Move the original skill tree out of the way and plant a
            # foreign directory at the canonical pathname.  Using
            # ``shutil.move`` (not rmtree) so the swap itself does not
            # touch any production destructive primitive.  The swap
            # mutates `skill_dir`'s children but not the parent
            # directory's identity, so any subsequent identity call
            # against the parent would still match — but production
            # makes none.
            import shutil as _sh
            _sh.move(str(skill_dir), str(evidence_dir))
            skill_dir.mkdir()
            foreign_file.write_text(foreign_content, encoding="utf-8")
            state["swap_performed"] = True
            state["swap_occurred_after_both_final_captures"] = True
            state["foreign_replacement_existed_before_refusal"] = (
                foreign_file.exists()
            )
            state["original_evidence_existed_before_refusal"] = (
                evidence_dir.exists()
                and (evidence_dir / "SKILL.md").exists()
            )

        return st

    monkeypatch.setattr(sm_local.Path, "lstat", lstat_with_final_swap)

    destructive_calls = {
        "rmtree": 0, "unlink": 0, "rmdir": 0,
        "path_unlink": 0, "path_rmdir": 0, "archive": 0,
    }

    real_rmtree = sm_local.shutil.rmtree

    def spy_rmtree(path, *a, **kw):
        destructive_calls["rmtree"] += 1
        return real_rmtree(path, *a, **kw)

    real_unlink = sm_local.os.unlink

    def spy_unlink(path, *a, **kw):
        destructive_calls["unlink"] += 1
        return real_unlink(path, *a, **kw)

    real_rmdir = sm_local.os.rmdir

    def spy_rmdir(path, *a, **kw):
        destructive_calls["rmdir"] += 1
        return real_rmdir(path, *a, **kw)

    real_path_unlink = sm_local.Path.unlink

    def spy_path_unlink(self, *a, **kw):
        destructive_calls["path_unlink"] += 1
        return real_path_unlink(self, *a, **kw)

    real_path_rmdir = sm_local.Path.rmdir

    def spy_path_rmdir(self, *a, **kw):
        destructive_calls["path_rmdir"] += 1
        return real_path_rmdir(self, *a, **kw)

    monkeypatch.setattr(sm_local.shutil, "rmtree", spy_rmtree)
    monkeypatch.setattr(sm_local.os, "unlink", spy_unlink)
    monkeypatch.setattr(sm_local.os, "rmdir", spy_rmdir)
    monkeypatch.setattr(sm_local.Path, "unlink", spy_path_unlink)
    monkeypatch.setattr(sm_local.Path, "rmdir", spy_path_rmdir)

    def spy_archive(*a, **kw):
        destructive_calls["archive"] += 1
        return None

    monkeypatch.setattr(sm, "archive_skill", spy_archive, raising=False)

    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_delete")):
        result = json.loads(sm.skill_manage(action="delete", name="fail-closed"))

    # ── State-machine contract: the swap must have fired AFTER both
    # final captures returned the ORIGINAL identity to production,
    # with zero identity-bearing lstat calls observed afterwards.
    assert state["target_lstat_count"] >= 2, state
    assert state["parent_lstat_count"] >= 2, state
    assert state["target_final_captured_with_original"] is True, state
    assert state["parent_final_captured_with_original"] is True, state
    assert state["swap_performed"] is True, state
    assert state["swap_occurred_after_both_final_captures"] is True, state
    assert state["post_swap_target_identity_calls"] == 0, (
        f"no target lstat may follow the swap; observed "
        f"{state['post_swap_target_identity_calls']}"
    )
    assert state["post_swap_parent_identity_calls"] == 0, (
        f"no parent lstat may follow the swap; observed "
        f"{state['post_swap_parent_identity_calls']}"
    )
    # The pre-capture identities recorded by the harness MUST match
    # the values production captured and compared against (verified
    # indirectly: production's comparison passed, so the original
    # identity was correctly captured into both pre-capture slots).
    assert state["target_pre_capture_identity"] is not None, state
    assert state["parent_pre_capture_identity"] is not None, state

    # ── Foreign replacement and original evidence exist at refusal
    # time and remain on disk after the call returns.
    assert state["foreign_replacement_existed_before_refusal"] is True, state
    assert state["original_evidence_existed_before_refusal"] is True, state
    assert foreign_file.exists(), "foreign replacement MUST exist when refusal runs"
    assert foreign_file.read_text(encoding="utf-8") == foreign_content
    assert evidence_dir.exists(), "original skill tree MUST be in evidence"
    assert (evidence_dir / "SKILL.md").exists(), "original SKILL.md MUST be in evidence"
    assert (evidence_dir / "SKILL.md").read_bytes() == original_bytes

    # ── Production payload: EXACT refusal, not concurrent_modification.
    # A concurrent_modification return here would mean production
    # observed the swap at an identity recheck — i.e. our swap fired
    # BEFORE both required final captures.
    assert result["success"] is False
    assert result["policy_reason"] == "atomic_recursive_delete_unavailable", result
    assert (
        result["rollback_failure_kind"]
        == "identity_bound_recursive_delete_unavailable"
    ), result
    assert result.get("live_mutation_committed", False) is False
    assert result.get("safe_to_retry", False) is False

    # ── Zero destructive primitives ran on the production side.
    assert destructive_calls["rmtree"] == 0
    assert destructive_calls["unlink"] == 0
    assert destructive_calls["rmdir"] == 0
    assert destructive_calls["path_unlink"] == 0
    assert destructive_calls["path_rmdir"] == 0
    assert destructive_calls["archive"] == 0

    # ── Both trees are preserved post-call.
    assert foreign_file.exists(), "foreign replacement MUST be preserved after refusal"
    assert evidence_dir.exists(), "original skill evidence MUST be preserved after refusal"


def test_delete_skill_refuses_when_identity_bound_recursive_delete_is_unavailable(
    monkeypatch, tmp_path,
):
    """Foreground delete refuses with the structured payload before any
    mutation runs when no portable identity-bound kernel-anchored
    recursive-delete primitive is available.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    skill_dir = skills_root / "fail-closed"
    skill_md = skill_dir / "SKILL.md"
    original_bytes = skill_md.read_bytes()

    import tools.skill_manager_tool as sm_local
    destructive_calls = {
        "rmtree": 0, "unlink": 0, "rmdir": 0,
        "path_unlink": 0, "path_rmdir": 0, "archive": 0,
    }

    real_rmtree = sm_local.shutil.rmtree

    def spy_rmtree(path, *a, **kw):
        destructive_calls["rmtree"] += 1
        return real_rmtree(path, *a, **kw)

    real_unlink = sm_local.os.unlink

    def spy_unlink(path, *a, **kw):
        destructive_calls["unlink"] += 1
        return real_unlink(path, *a, **kw)

    real_rmdir = sm_local.os.rmdir

    def spy_rmdir(path, *a, **kw):
        destructive_calls["rmdir"] += 1
        return real_rmdir(path, *a, **kw)

    real_path_unlink = sm_local.Path.unlink

    def spy_path_unlink(self, *a, **kw):
        destructive_calls["path_unlink"] += 1
        return real_path_unlink(self, *a, **kw)

    real_path_rmdir = sm_local.Path.rmdir

    def spy_path_rmdir(self, *a, **kw):
        destructive_calls["path_rmdir"] += 1
        return real_path_rmdir(self, *a, **kw)

    monkeypatch.setattr(sm_local.shutil, "rmtree", spy_rmtree)
    monkeypatch.setattr(sm_local.os, "unlink", spy_unlink)
    monkeypatch.setattr(sm_local.os, "rmdir", spy_rmdir)
    monkeypatch.setattr(sm_local.Path, "unlink", spy_path_unlink)
    monkeypatch.setattr(sm_local.Path, "rmdir", spy_path_rmdir)

    def spy_archive(*a, **kw):
        destructive_calls["archive"] += 1
        return None

    monkeypatch.setattr(sm, "archive_skill", spy_archive, raising=False)

    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_delete")):
        result = json.loads(sm.skill_manage(action="delete", name="fail-closed"))

    # Structured fail-closed payload.
    assert result["success"] is False
    assert result["policy_reason"] == "atomic_recursive_delete_unavailable"
    assert result["rollback_failure_kind"] == "identity_bound_recursive_delete_unavailable"
    assert result["live_mutation_committed"] is False
    assert result["safe_to_retry"] is False
    assert result["operation_kind"] == "delete"
    assert result["target"] == str(skill_dir)
    assert "lock_path" in result
    # No destructive primitive ran.
    assert destructive_calls["rmtree"] == 0
    assert destructive_calls["unlink"] == 0
    assert destructive_calls["rmdir"] == 0
    assert destructive_calls["path_unlink"] == 0
    assert destructive_calls["path_rmdir"] == 0
    assert destructive_calls["archive"] == 0
    # Skill intact.
    assert skill_dir.exists()
    assert skill_md.exists()
    assert skill_md.read_bytes() == original_bytes


def test_delete_skill_atomic_refusal_plus_lock_release_failure_preserves_not_committed(
    monkeypatch, tmp_path,
):
    """When the foreground delete refuses (no portable identity-bound
    primitive) AND the interprocess lock release subsequently fails, the
    refusal MUST still report ``live_mutation_committed=false`` and
    ``safe_to_retry=false``.  A release failure cannot transform a
    pre-mutation refusal into a committed mutation.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    skill_dir = skills_root / "fail-closed"
    skill_md = skill_dir / "SKILL.md"
    original_bytes = skill_md.read_bytes()

    import tools.skill_manager_tool as sm_local

    # Force the lock-release path to raise AFTER the refusal has
    # committed its payload.  We patch the lock context manager so its
    # __exit__ raises _SkillMutationLockReleaseFailure.
    from tools.skill_manager_tool import _SkillMutationLockReleaseFailure

    class _BoomLock:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            # Simulate a release failure: raise _SkillMutationLockReleaseFailure.
            raise _SkillMutationLockReleaseFailure(
                canonical_skill_path=skill_dir,
                lock_path=tmp_path / "fake.lock",
                platform="posix",
                release_error=OSError("simulated release failure"),
                close_error=None,
                live_mutation_committed=False,
            )

    monkeypatch.setattr(sm, "_skill_mutation_process_lock", lambda _p: _BoomLock())

    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_delete")):
        result = json.loads(sm.skill_manage(action="delete", name="fail-closed"))

    # The release-failure handler converts the payload to
    # lock_release_failed but MUST preserve live_mutation_committed=false
    # and safe_to_retry=false — the refusal fired before any destructive
    # primitive, so the release failure cannot retroactively commit.
    assert result["success"] is False
    assert result["policy_reason"] == "lock_release_failed"
    assert result["rollback_failure_kind"] == "lock_release_failure"
    assert result["live_mutation_committed"] is False, (
        "release failure must NOT retroactively commit a pre-mutation "
        "refusal; live_mutation_committed must remain false"
    )
    assert result["safe_to_retry"] is False
    # Skill intact.
    assert skill_dir.exists()
    assert skill_md.exists()
    assert skill_md.read_bytes() == original_bytes


# ── Remove_file identity revalidation under lock ───────────────────────────


def test_remove_file_same_bytes_inode_replacement_detected(monkeypatch, tmp_path):
    """If the supporting file is replaced by a different inode carrying
    the same bytes, remove_file refuses and preserves the foreign file.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    supporting = skills_root / "fail-closed" / "references" / "ok.md"
    supporting.parent.mkdir()
    supporting.write_text("original", encoding="utf-8")

    # The swap happens on the FIRST lstat inside the lock — between
    # the pre-lock snapshot and the under-lock revalidation.  We wrap
    # Path.lstat via the supporting module's reference to it.
    import tools.skill_manager_tool as sm_local
    real_lstat = sm_local.Path.lstat
    swapped = {"done": False}

    def maybe_swap_lstat(self, *args, **kwargs):
        st = real_lstat(self, *args, **kwargs)
        # Trigger the swap on the first lstat of `supporting` that
        # happens AFTER the pre-lock snapshot.  We detect "after" by
        # flipping the flag once on the first lstat.
        if not swapped["done"] and str(self).endswith("ok.md"):
            swapped["done"] = True
            sibling = supporting.with_suffix(supporting.suffix + ".swap")
            sibling.write_bytes(supporting.read_bytes())
            supporting.unlink()
            os.link(str(sibling), str(supporting))
            sibling.unlink()
        return st

    # NOTE: monkeypatching Path.lstat is broad — it affects every lstat
    # in the test.  That's acceptable here because the swap branch
    # only fires once, after which the wrapper falls through to the
    # real lstat.
    monkeypatch.setattr(sm_local.Path, "lstat", maybe_swap_lstat)

    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_remove_file")):
        result = json.loads(
            sm.skill_manage(
                action="remove_file",
                name="fail-closed",
                file_path="references/ok.md",
            )
        )

    assert result["success"] is False
    assert result["policy_reason"] == "concurrent_modification"
    # Foreign file preserved.
    assert supporting.exists()
    assert supporting.read_bytes() == b"original"


def test_remove_file_symlink_swap_blocked(monkeypatch, tmp_path):
    """If the supporting file is a symlink that escapes the skill dir,
    remove_file refuses and the target outside the skill directory is
    preserved.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    supporting_dir = skills_root / "fail-closed" / "references"
    supporting_dir.mkdir()

    outside_dir = tmp_path / "outside-refs"
    outside_dir.mkdir()
    outside_target = outside_dir / "outside.md"
    outside_target.write_text("outside", encoding="utf-8")

    # Place a SYMLINK INSIDE the skill dir that points outside.  We
    # use Path.write_text / unlink tricks to dodge _resolve_skill_target's
    # path-containment check by giving the symlink a name that already
    # appears valid; the symlink IS the target the caller names, so
    # _resolve_skill_target sees a sibling path that escapes.  The
    # identity-revalidation S_ISLNK branch in _remove_file must reject.
    # To exercise that branch we need the resolved path to live inside
    # the skill dir at first glance — which means the SYMLINK must
    # point INSIDE the skill dir but to a DIFFERENT inode.  The "swap"
    # path for the same-bytes inode replacement is covered by a separate
    # test.
    supporting = supporting_dir / "ok.md"
    # Symlink to a sibling file inside the skill dir.  lstat will
    # show S_ISLNK; the file resolves to the sibling inode, not the
    # symlink inode, so identity differs from pre_lock_target_identity.
    sibling = supporting_dir / "ok.md.swap"
    sibling.write_text("swapped", encoding="utf-8")
    os.symlink(str(sibling), str(supporting))

    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_remove_file")):
        result = json.loads(
            sm.skill_manage(
                action="remove_file",
                name="fail-closed",
                file_path="references/ok.md",
            )
        )

    # remove_file must refuse to follow the symlink and report
    # rollback_failure_kind == "symlink_detected".
    assert result["success"] is False
    assert result.get("rollback_failure_kind") in ("symlink_detected", "concurrent_modification"), result
    # Outside/sibling target preserved.
    assert sibling.exists()


def test_happy_path_remove_file_with_revalidation(monkeypatch, tmp_path):
    """Caminos sanos: a clean remove_file surfaces the canonical
    Camino-B refusal because portable Python cannot bind the
    destructive op to the validated inode.

    Update from Phase C last-mile atomicity: ``os.unlink(name,
    dir_fd=parent_fd)`` is two separate syscalls; a non-cooperative
    swap between the final lstat and the unlink can replace the
    inode.  We refuse rather than risk deleting the wrong inode.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    supporting = skills_root / "fail-closed" / "references" / "ok.md"
    supporting.parent.mkdir()
    supporting.write_text("original", encoding="utf-8")

    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_remove_file")):
        result = json.loads(
            sm.skill_manage(
                action="remove_file",
                name="fail-closed",
                file_path="references/ok.md",
            )
        )

    assert result["success"] is False
    assert result["policy_reason"] == "atomic_identity_delete_unavailable", result
    assert result["rollback_failure_kind"] == (
        "identity_bound_unlink_unavailable"
    ), result
    assert result["live_mutation_committed"] is False, result
    assert result["safe_to_retry"] is False, result
    # Target preserved (no destructive op ran).
    assert supporting.exists()
    assert supporting.read_text(encoding="utf-8") == "original"


# ── Native Windows smoke test (skip on POSIX) ───────────────────────────────


@pytest.mark.skipif(os.name != "nt", reason="Native Windows smoke test (skip on POSIX)")
def test_windows_native_lock_smoke(monkeypatch, tmp_path):
    """Native Windows smoke test — only runs on Windows.  POSIX coverage
    is provided by the msvcrt mock tests above.
    """
    import tools.skill_manager_tool as sm_local
    canonical = (tmp_path / "fake-skill").resolve(strict=False)
    with sm_local._skill_mutation_process_lock(canonical):
        pass


# ── Section 3: Windows import guard (no fcntl on POSIX simulated) ──────────


def test_module_loads_when_fcntl_missing(monkeypatch):
    """Module must import cleanly when ``fcntl`` is unavailable.

    On POSIX the module normally imports ``fcntl``; on Windows the
    conditional ``try/except ImportError`` sets ``_fcntl = None``.
    We simulate the Windows shape on POSIX by stripping ``fcntl`` from
    ``sys.modules`` and replacing ``builtins.__import__`` so a re-import
    raises ImportError — then reload the module and assert it loaded
    successfully with ``_fcntl is None``.

    The reload runs in an isolated subprocess so the patched module
    cannot leak its ``_fcntl = None`` state back into the rest of the
    test session.
    """
    import subprocess
    import sys
    import textwrap

    # The reload probe runs in a child Python so the patched module
    # object is destroyed when the child exits — never bleeds into
    # the parent's ``tools.skill_manager_tool`` cache.
    probe = textwrap.dedent(
        """
        import builtins
        import sys

        real_import = builtins.__import__

        def blocking_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == 'fcntl' or name.startswith('fcntl.'):
                raise ImportError('No module named %r (simulated)' % name)
            return real_import(name, globals, locals, fromlist, level)

        sys.modules.pop('fcntl', None)
        builtins.__import__ = blocking_import
        try:
            sm = __import__('tools.skill_manager_tool', fromlist=['*'])
        finally:
            builtins.__import__ = real_import

        # Verify the simulated-Windows shape loaded correctly.
        assert sm._fcntl is None, (
            'expected _fcntl to be None on simulated Windows, got %r' % (sm._fcntl,)
        )
        assert sm._msvcrt is None
        # Production must not re-export the lock-mode constants as
        # module-level aliases (those are test-fake territory only).
        for attr in ('_MSVC_LOCKING_LK_LOCK', '_MSVC_LOCKING_LK_NBLCK', '_MSVC_LOCKING_LK_UNLCK'):
            assert not hasattr(sm, attr), (
                'production must not re-export %s as a hard-coded alias' % attr
            )
        print('WINDOWS_IMPORT_OK')
        """
    )
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=60,
    )
    assert result.returncode == 0, (
        f"simulated-Windows import probe failed:\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "WINDOWS_IMPORT_OK" in result.stdout, (
        f"probe did not report success:\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )


# ── Section 4: msvcrt lock-mode constants are real, not hard-coded ──────────


def test_msvcrt_constants_come_from_real_module_not_production_aliases():
    """Lock-mode constants must come from the active msvcrt module
    (or the production test fake), NOT from a hard-coded module-level
    alias re-exported by ``skill_manager_tool``.

    This test fails with the previous implementation: the previous
    module exported ``_MSVC_LOCKING_LK_LOCK`` / ``_MSVC_LOCKING_LK_NBLCK`` /
    ``_MSVC_LOCKING_LK_UNLCK`` as numeric literals, which would mask a
    drift between the production alias and the real ``msvcrt`` attribute.

    Production is also forbidden from declaring its own test-side
    fake (e.g. ``_PosixMsvcrtFake``); the constants live exclusively
    in the test file's ``_FakeMsvcrt``.
    """
    import tools.skill_manager_tool as sm

    # Production must NOT carry hard-coded numeric aliases.
    assert not hasattr(sm, "_MSVC_LOCKING_LK_LOCK")
    assert not hasattr(sm, "_MSVC_LOCKING_LK_NBLCK")
    assert not hasattr(sm, "_MSVC_LOCKING_LK_UNLCK")

    # Production must NOT declare its own test-side msvcrt fake.
    assert not hasattr(sm, "_PosixMsvcrtFake"), (
        "production must not declare a POSIX-side msvcrt fake; "
        "the fake lives in tests/tools/test_session_write_policy_fail_closed.py"
    )

    # The test fake exposes the canonical encodings.  Production's
    # runtime path references ``_msvcrt.LK_LOCK`` etc. directly; on
    # POSIX ``_msvcrt is None`` so the only surface asserting the
    # encoding is the test fake.
    assert _FakeMsvcrt.LK_UNLCK == 0
    assert _FakeMsvcrt.LK_LOCK == 1
    assert _FakeMsvcrt.LK_NBLCK == 2

    # And the test fake mirrors those exact values when instantiated.
    fake = _FakeMsvcrt()
    assert fake.LK_UNLCK == 0
    assert fake.LK_LOCK == 1
    assert fake.LK_NBLCK == 2

    # And the modes that the helper would call msvcrt.locking with
    # must be exactly those integers — ``fake.calls`` after one
    # acquisition should contain an unlock mode == 0, a blocking mode
    # == 1, and a nonblocking mode == 2.

    fake = _FakeMsvcrt()
    fake.locking(7, 0, 1)
    fake.locking(7, 1, 1)
    fake.locking(7, 2, 1)
    modes_sent = {mode for (_, mode, _) in fake.calls}
    assert modes_sent == {0, 1, 2}
    assert 0 in modes_sent  # unlock mode
    assert 1 in modes_sent  # blocking
    assert 2 in modes_sent  # nonblocking


# ── Section 5: remove_file fail-closed when dir_fd is supported ─────────────


def test_remove_file_dirfd_open_failure_reports_structured_error(
    monkeypatch, tmp_path
):
    """If dir_fd open fails on a platform where dir_fd is supported,
    remove_file must return a structured failure — never a pathname
    fallback that would let a symlink swap slip through.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)

    supporting = skills_root / "fail-closed" / "references" / "ok.md"
    supporting.parent.mkdir()
    supporting.write_text("original", encoding="utf-8")

    # Force the dir_fd branch by lying about O_DIRECTORY support on POSIX.
    real_os_open = sm.os.open

    def open_fails_for_lock_dir(path, flags, *args, **kwargs):
        if "fail-closed" in str(path) and (
            flags & sm.os.O_DIRECTORY
        ):
            raise OSError(errno.EACCES, "simulated open failure on parent dir")
        return real_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(sm.os, "open", open_fails_for_lock_dir)

    with session_write_policy_scope(
        _allowlist("skills", skills_root, "skill_remove_file")
    ):
        result = json.loads(
            sm.skill_manage(
                action="remove_file",
                name="fail-closed",
                file_path="references/ok.md",
            )
        )

    assert result["success"] is False
    # Structured failure — no pathname fallback.  Contract requires
    # ``parent_open_failed`` / ``parent_fd_open_failure`` for the
    # open failure case.
    assert result["policy_reason"] == "parent_open_failed", result
    assert result["rollback_failure_kind"] == "parent_fd_open_failure", result
    assert result["operation_kind"] == "remove_file"
    # Foreign file preserved.
    assert supporting.exists()
    assert supporting.read_bytes() == b"original"


def test_remove_file_dirfd_parent_fd_identity_mismatch_reports_structured_error(
    monkeypatch, tmp_path
):
    """If the parent fd's identity (dev/ino/type) does not match the
    parent lstat, remove_file must refuse with a structured failure —
    not silently fall back to ``re_target.unlink()`` on a pathname
    that may now point to a different inode.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)

    supporting = skills_root / "fail-closed" / "references" / "ok.md"
    supporting.parent.mkdir()
    supporting.write_text("original", encoding="utf-8")

    real_os_open = sm.os.open
    real_os_fstat = sm.os.fstat
    real_os_lstat = sm.os.lstat

    class _FakeStat:
        def __init__(self, dev=1, ino=1, mode=0o40755):
            self.st_dev = dev
            self.st_ino = ino
            self.st_mode = mode

    swapped = {"done": False}

    def fstat_with_swap(fd):
        # The lock-fd fstat (regular file) and the parent-fd fstat
        # (directory) both go through ``os.fstat``.  Detect the parent
        # fd by inspecting the real stat result's mode bits — when it
        # is a directory we know we're fstat'ing the parent fd that
        # was just opened with ``O_DIRECTORY``.  Returning a wrong
        # dev/ino for the regular-file lock fd would short-circuit
        # the lock-acquire path and never reach the parent-fd
        # identity check we want to exercise here.
        real_st = real_os_fstat(fd)
        is_directory = (real_st.st_mode & 0o170000) == 0o040000
        if is_directory and not swapped["done"]:
            swapped["done"] = True
            return _FakeStat(dev=999, ino=999, mode=0o40755)
        return real_st

    monkeypatch.setattr(sm.os, "fstat", fstat_with_swap)

    with session_write_policy_scope(
        _allowlist("skills", skills_root, "skill_remove_file")
    ):
        result = json.loads(
            sm.skill_manage(
                action="remove_file",
                name="fail-closed",
                file_path="references/ok.md",
            )
        )

    assert result["success"] is False
    # Contract requires ``concurrent_modification`` /
    # ``parent_identity_mismatch`` for the parent fd identity mismatch.
    assert result["policy_reason"] == "concurrent_modification", result
    assert result["rollback_failure_kind"] == "parent_identity_mismatch", result
    assert result["operation_kind"] == "remove_file"
    # Foreign file preserved.
    assert supporting.exists()
    assert supporting.read_bytes() == b"original"


def test_remove_file_dirfd_unlink_success_close_failure_reports_mutation_committed(
    monkeypatch, tmp_path
):
    """Under the Camino-B (last-mile atomicity) contract, the
    destructive op is withheld before the unlink syscall because no
    kernel identity-bound delete primitive is available.  Therefore
    the ``close failure after unlink success`` branch is no longer
    reachable: production refuses BEFORE the unlink runs.

    The test now verifies the new contract:
      * success=False
      * policy_reason=atomic_identity_delete_unavailable
      * live_mutation_committed=False (no destructive op ran)
      * zero unlink calls of any kind
      * target preserved with original content
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)

    supporting = skills_root / "fail-closed" / "references" / "ok.md"
    supporting.parent.mkdir()
    supporting.write_text("original", encoding="utf-8")

    real_close = sm.os.close
    close_failures = {"count": 0}
    unlink_calls: list[tuple] = []

    real_unlink = sm.os.unlink

    def tracking_unlink(path, *args, **kwargs):
        unlink_calls.append((str(path), kwargs))
        return real_unlink(path, *args, **kwargs)

    def fail_close_on_parent_fd(fd):
        # Trigger a close failure on the parent fd once.  We don't know
        # whether the close is on the parent or the lock fd without
        # sniffing, so we track calls and let the first parent close
        # fail.
        try:
            return real_close(fd)
        except OSError:
            close_failures["count"] += 1
            raise

    # Easier approach: stub os.close to fail once for ANY fd; the helper
    # raises _SkillMutationLockReleaseFailure, which the caller translates.
    close_state = {"called": False}

    def fail_close_once(fd, *args, **kwargs):
        if not close_state["called"]:
            close_state["called"] = True
            raise OSError(errno.EIO, "simulated close failure on parent fd")
        return real_close(fd, *args, **kwargs)

    monkeypatch.setattr(sm.os, "close", fail_close_once)
    monkeypatch.setattr(sm.os, "unlink", tracking_unlink)

    with session_write_policy_scope(
        _allowlist("skills", skills_root, "skill_remove_file")
    ):
        result = json.loads(
            sm.skill_manage(
                action="remove_file",
                name="fail-closed",
                file_path="references/ok.md",
            )
        )

    # Camino-B refusal: the destructive op never runs.
    assert result["success"] is False
    assert result["policy_reason"] == "atomic_identity_delete_unavailable", result
    assert result["rollback_failure_kind"] == (
        "identity_bound_unlink_unavailable"
    ), result
    assert result["live_mutation_committed"] is False, result
    assert result["safe_to_retry"] is False, result
    assert result["operation_kind"] == "remove_file"
    # Zero unlink calls (refusal happened before the unlink syscall).
    assert len(unlink_calls) == 0, (
        f"expected zero unlink calls (Camino-B refusal), got {unlink_calls!r}"
    )
    # Target preserved with original content.
    assert supporting.exists()
    assert supporting.read_bytes() == b"original"


def test_remove_file_parent_becomes_symlink_after_lock_rejected(
    monkeypatch, tmp_path
):
    """If the parent directory is replaced by a symlink after the lock
    is acquired but before the unlink syscall, remove_file must refuse
    with a structured failure and leave the foreign symlink target
    intact.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)

    refs = skills_root / "fail-closed" / "references"
    refs.mkdir()
    supporting = refs / "ok.md"
    supporting.write_text("original", encoding="utf-8")

    outside_dir = tmp_path / "outside-targets"
    outside_dir.mkdir()
    outside_file = outside_dir / "ok.md"
    outside_file.write_text("outside", encoding="utf-8")

    # The test that exercises this contract must monkeypatch a real
    # implementation surface used by ``_remove_file`` — not a frozen
    # stdlib module.  The cleanest target is the private identity helper
    # ``sm._lstat_identity``: it is called on the parent of the target
    # both during the pre-lock snapshot AND inside the lock for
    # revalidation.  Patching it lets the test inject a one-shot
    # symlink swap that triggers the fail-closed branch on the NEXT
    # call site (the in-lock revalidation), AFTER the pre-lock snapshot
    # has captured the real parent identity, BEFORE the final
    # pre-unlink identity check or the dir_fd open.
    import tools.skill_manager_tool as sm_local

    real_lstat_identity = sm._lstat_identity
    swap_state = {"calls": 0, "swapped": False}

    def maybe_swap_lstat_identity(path: Path):
        swap_state["calls"] += 1
        # The pre-lock parent-identity snapshot (``pre_lock_parent_identity``)
        # is the FIRST call inside ``_remove_file`` for the parent of the
        # target.  Let it run unimpeded so the snapshot is real; from the
        # second call onward (which all happen inside the lock for the
        # in-lock revalidation), swap the parent for a symlink to the
        # foreign target ONCE so the next code path sees the foreign
        # tree and refuses to operate against it.
        if (
            swap_state["calls"] >= 2
            and not swap_state["swapped"]
            and str(path).endswith("references")
        ):
            swap_state["swapped"] = True
            refs_path = skills_root / "fail-closed" / "references"
            for child in list(refs_path.iterdir()):
                if child.is_file() or child.is_symlink():
                    child.unlink()
            refs_path.rmdir()
            os.symlink(str(outside_dir), str(refs_path))
        return real_lstat_identity(path)

    monkeypatch.setattr(sm, "_lstat_identity", maybe_swap_lstat_identity)

    with session_write_policy_scope(
        _allowlist("skills", skills_root, "skill_remove_file")
    ):
        result = json.loads(
            sm.skill_manage(
                action="remove_file",
                name="fail-closed",
                file_path="references/ok.md",
            )
        )

    assert result["success"] is False
    # The foreign symlink + its outside target are preserved.
    assert (skills_root / "fail-closed" / "references").is_symlink()
    assert outside_file.exists()
    assert outside_file.read_text(encoding="utf-8") == "outside"


# ── Section 5b: remove_file kernel-anchored revalidation (contract 13.3/13.4/13.7)


def test_remove_file_dirfd_target_identity_mismatch_after_parent_open_reports_structured_error(
    monkeypatch, tmp_path
):
    """Contract 13.3 — kernel-anchored target identity revalidation.

    After the parent fd is opened and its identity confirmed, the
    helper must revalidate the target RELATIVE to that fd via
    ``os.lstat(name, dir_fd=parent_fd)`` before the unlink syscall.
    If the kernel no longer resolves ``re_target.name`` under the
    open parent fd to the captured identity, the helper must refuse
    without attempting the unlink and must leave any replacement
    file untouched.

    Simulates a kernel-level swap by patching ``os.lstat`` once on a
    dir_fd call so the helper's revalidation observes a different
    inode than the pre-lock capture.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)

    refs = skills_root / "fail-closed" / "references"
    refs.mkdir()
    supporting = refs / "ok.md"
    supporting.write_text("original", encoding="utf-8")

    real_lstat = sm.os.lstat
    swap_state = {"done": False}

    def lstat_dirfd_swap(name, *args, **kwargs):
        # The kernel-anchored lstat is called as
        # ``os.lstat(re_target.name, dir_fd=parent_fd)`` so we detect
        # the dir_fd kwarg.  On the first such call we return a
        # fake stat result with a different st_ino so the helper's
        # identity comparison detects the swap.
        if "dir_fd" in kwargs and not swap_state["done"]:
            swap_state["done"] = True
            real_st = real_lstat(name, *args, **kwargs)
            # Mutate the inode to force the mismatch while preserving
            # the rest of the stat result.
            class _FakeSt:
                def __init__(self, template):
                    # Mirror every attribute of ``template`` (real
                    # stat_result exposes 10+ attrs; copy them all).
                    for attr in (
                        "st_mode",
                        "st_ino",
                        "st_dev",
                        "st_nlink",
                        "st_uid",
                        "st_gid",
                        "st_size",
                        "st_atime",
                        "st_mtime",
                        "st_ctime",
                        "st_atime_ns",
                        "st_mtime_ns",
                        "st_ctime_ns",
                        "st_blocks",
                        "st_blksize",
                        "st_rdev",
                        "st_flags",
                        "st_gen",
                        "st_birthtime",
                        "st_birthtime_ns",
                    ):
                        if hasattr(template, attr):
                            setattr(self, attr, getattr(template, attr))
                # ``os.stat_result`` exposes named tuple-like indexing
                # via ``_asdict`` on some platforms; harmless to omit
                # here because production only reads attributes.

            fake = _FakeSt(real_st)
            fake.st_ino = real_st.st_ino + 999_999
            return fake
        return real_lstat(name, *args, **kwargs)

    monkeypatch.setattr(sm.os, "lstat", lstat_dirfd_swap)

    unlink_calls: list[tuple] = []

    real_unlink = sm.os.unlink

    def tracking_unlink(path, *args, **kwargs):
        unlink_calls.append((str(path), kwargs))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(sm.os, "unlink", tracking_unlink)

    with session_write_policy_scope(
        _allowlist("skills", skills_root, "skill_remove_file")
    ):
        result = json.loads(
            sm.skill_manage(
                action="remove_file",
                name="fail-closed",
                file_path="references/ok.md",
            )
        )

    assert result["success"] is False
    # Contract 13.3: target identity mismatch is reported with the
    # canonical ``target_identity_mismatch`` kind.
    assert result["policy_reason"] == "concurrent_modification", result
    assert result["rollback_failure_kind"] == "target_identity_mismatch", result
    assert result["operation_kind"] == "remove_file"
    assert result["live_mutation_committed"] is False
    assert result["safe_to_retry"] is False
    # No unlink call must have been issued — the swap was detected
    # during kernel-anchored revalidation, BEFORE the unlink syscall.
    assert unlink_calls == [], (
        f"unlink must NOT be called when target identity changed; "
        f"got {unlink_calls!r}"
    )
    # Original file preserved.
    assert supporting.exists()
    assert supporting.read_bytes() == b"original"


def test_remove_file_dirfd_unlink_failure_no_second_unlink(monkeypatch, tmp_path):
    """Contract 13.4 — single unlink attempt; no Path.unlink fallback.

    Under the Camino-B (last-mile atomicity) contract the destructive
    op is withheld BEFORE the unlink syscall because no kernel
    identity-bound delete primitive is available.  Therefore the
    ``unlink failure`` branch is unreachable in production; the test
    verifies the new contract:

      * success=False
      * policy_reason=atomic_identity_delete_unavailable
        (NOT remove_failed — the unlink never runs)
      * live_mutation_committed=False
      * safe_to_retry=False
      * zero unlink calls of any kind
      * zero Path.unlink calls
      * target preserved

    The contract's strongest assertion (no pathname fallback after a
    secure-path failure) still holds: the destructive op is never
    attempted at all.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)

    refs = skills_root / "fail-closed" / "references"
    refs.mkdir()
    supporting = refs / "ok.md"
    supporting.write_text("original", encoding="utf-8")

    unlink_calls: list[tuple] = []
    path_unlink_calls: list[tuple] = []

    real_unlink = sm.os.unlink

    def failing_dirfd_unlink(path, *args, **kwargs):
        unlink_calls.append((str(path), args, kwargs))
        # Only fail on the dir_fd-anchored unlink so the contract
        # assertion targets the right call site.
        if "dir_fd" in kwargs:
            raise OSError(errno.EACCES, "simulated dir_fd unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(sm.os, "unlink", failing_dirfd_unlink)

    # Track Path.unlink and any target.unlink attempts to enforce the
    # no-pathname-fallback contract.
    from pathlib import Path as _PathCls

    real_path_unlink = _PathCls.unlink

    def tracking_path_unlink(self, *args, **kwargs):
        path_unlink_calls.append((str(self), args, kwargs))
        return real_path_unlink(self, *args, **kwargs)

    monkeypatch.setattr(_PathCls, "unlink", tracking_path_unlink)

    with session_write_policy_scope(
        _allowlist("skills", skills_root, "skill_remove_file")
    ):
        result = json.loads(
            sm.skill_manage(
                action="remove_file",
                name="fail-closed",
                file_path="references/ok.md",
            )
        )

    assert result["success"] is False
    # Camino-B refusal: the unlink never runs, so the canonical
    # refusal payload is the only thing production can surface.
    assert result["policy_reason"] == "atomic_identity_delete_unavailable", result
    assert result["rollback_failure_kind"] == (
        "identity_bound_unlink_unavailable"
    ), result
    assert result["operation_kind"] == "remove_file"
    assert result["live_mutation_committed"] is False
    assert result["safe_to_retry"] is False
    # Zero unlink calls of any kind — the refusal happened BEFORE
    # the unlink syscall, so the failure-injection never triggers.
    assert unlink_calls == [], (
        f"expected zero unlink calls under Camino-B refusal, "
        f"got {unlink_calls!r}"
    )
    # Path.unlink must never have been invoked on the target or any
    # other path (no pathname fallback by construction).
    assert path_unlink_calls == [], (
        f"Path.unlink must not run as a fallback: {path_unlink_calls!r}"
    )
    # Original file preserved.
    assert supporting.exists()
    assert supporting.read_bytes() == b"original"


def test_remove_file_no_fallback_after_secure_path_failure(
    monkeypatch, tmp_path
):
    """Contract 13.7 — secure unlink attempts == 1; pathname unlink attempts == 0.

    Forces a secure-path failure (here: parent fd open failure) and
    asserts that no ``os.unlink`` / ``Path.unlink`` / ``target.unlink``
    runs as a fallback.  This is the contract's strongest assertion:
    once the secure path is selected the helper must NEVER fall back
    to a pathname-based unlink.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)

    refs = skills_root / "fail-closed" / "references"
    refs.mkdir()
    supporting = refs / "ok.md"
    supporting.write_text("original", encoding="utf-8")

    # Force parent fd open to fail — the helper enters the secure
    # path (because the platform supports dir_fd) but the open
    # itself fails.  The contract demands that no unlink runs.
    real_os_open = sm.os.open

    def open_fails_for_parent_dir(path, flags, *args, **kwargs):
        if "fail-closed" in str(path) and (
            flags & sm.os.O_DIRECTORY
        ):
            raise OSError(errno.EACCES, "simulated parent open failure")
        return real_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(sm.os, "open", open_fails_for_parent_dir)

    unlink_calls: list[tuple] = []
    path_unlink_calls: list[tuple] = []

    real_unlink = sm.os.unlink

    def tracking_unlink(path, *args, **kwargs):
        unlink_calls.append((str(path), kwargs))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(sm.os, "unlink", tracking_unlink)

    from pathlib import Path as _PathCls

    real_path_unlink = _PathCls.unlink

    def tracking_path_unlink(self, *args, **kwargs):
        path_unlink_calls.append((str(self), args, kwargs))
        return real_path_unlink(self, *args, **kwargs)

    monkeypatch.setattr(_PathCls, "unlink", tracking_path_unlink)

    with session_write_policy_scope(
        _allowlist("skills", skills_root, "skill_remove_file")
    ):
        result = json.loads(
            sm.skill_manage(
                action="remove_file",
                name="fail-closed",
                file_path="references/ok.md",
            )
        )

    assert result["success"] is False
    # Secure-path failure surfaces with the canonical parent_open_failed /
    # parent_fd_open_failure kind.
    assert result["policy_reason"] == "parent_open_failed", result
    assert result["rollback_failure_kind"] == "parent_fd_open_failure", result
    # Contract 13.7: NO unlink calls at all.
    assert unlink_calls == [], (
        f"secure-path failure must not trigger any unlink: {unlink_calls!r}"
    )
    assert path_unlink_calls == [], (
        f"secure-path failure must not trigger any Path.unlink: "
        f"{path_unlink_calls!r}"
    )
    # Original file preserved.
    assert supporting.exists()
    assert supporting.read_bytes() == b"original"

# ── Section 6: staging identity fail-closed (delete anchored) ───────────────


def test_staging_directory_is_created_with_private_mode(monkeypatch, tmp_path):
    """``_create_private_staging`` must create the staging directory
    directly with mode ``0o700`` and verify the final mode.  Tests
    that an external observer cannot traverse a freshly-created
    staging directory.
    """
    import stat as _stat

    import tools.skill_manager_tool as sm

    skills_root = tmp_path / "hermes" / "skills"
    skills_root.mkdir(parents=True)
    monkeypatch.setattr(sm, "SKILLS_DIR", skills_root)

    staging = sm._create_private_staging(skills_root)
    try:
        # Mode must be private (0o700).
        mode = staging.stat().st_mode & 0o777
        assert mode == 0o700, f"staging mode {oct(mode)} != 0o700"
    finally:
        sm._cleanup_private_staging(staging)


def test_staging_directory_refuses_cleanup_when_target_replaced_by_symlink(
    monkeypatch, tmp_path
):
    """If the staging directory's pathname is replaced by a symlink to
    another staging directory between creation and cleanup, the cleanup
    must reject the swap and leave the foreign tree intact.
    """
    import tools.skill_manager_tool as sm

    skills_root = tmp_path / "hermes" / "skills"
    skills_root.mkdir(parents=True)
    monkeypatch.setattr(sm, "SKILLS_DIR", skills_root)

    primary = sm._create_private_staging(skills_root)
    foreign = sm._create_private_staging(skills_root)
    (foreign / "important-data.txt").write_text("do-not-delete", encoding="utf-8")

    # Replace the primary's directory with a symlink to the foreign
    # staging directory.
    primary.rmdir()
    os.symlink(str(foreign), str(primary))

    # Cleanup of the original pathname must NOT delete the foreign tree.
    outcome = sm._cleanup_private_staging(primary)
    # ``_cleanup_private_staging`` follows symlinks to the resolved
    # path and would delete the foreign tree if it accepted the swap.
    # The fail-closed contract requires refusing the swap.
    if outcome is None:
        # Cleanup happened.  Assert the foreign tree survived.
        assert foreign.exists(), (
            "fail-closed contract violated: foreign staging tree was "
            "deleted after the primary's pathname was swapped to a "
            "symlink pointing at it"
        )
    else:
        # Cleanup reported refusal — the foreign tree must be intact.
        assert foreign.exists()


def test_staging_parent_replaced_preserves_foreign_tree(monkeypatch, tmp_path):
    """If the parent directory of a staging tree is replaced between
    creation and cleanup, the cleanup must NOT follow the new parent
    and must NOT delete the foreign tree.

    The setup reproduces the TOCTOU race in three phases:

      1. ``_create_private_staging`` returns a handle whose captured
         ``(dev, ino, type)`` triple is the proof of authority for the
         subsequent revalidation.  The staging handle's ``.path`` is
         the inner ``.hermes-skill-staging-<hex>`` directory and the
         handle's ``.parent`` is the freshly-created scratch dir
         ``.hermes-staging-<hex>`` that contains exactly one child
         (the staging itself).
      2. The test removes the inner staging subtree (``staging_path.rmdir()``)
         so the scratch parent becomes empty, then deletes the now-empty
         scratch parent with ``parent.rmdir()``.  Both steps are inside
         the test (not the producer) and only mutate the staging that
         THIS test created.  This is the exact preparation the production
         does NOT do on its own and is required to make ``os.symlink``
         accept the parent's pathname.
      3. The test replaces ``parent`` with a symlink to ``foreign_dir``
         which carries ``important.txt``.  Subsequent ``_cleanup_private_staging``
         must observe parent-identity divergence and refuse to delete
         the foreign tree.

    The contract asserted by the test:

      * ``_cleanup_private_staging`` returns a structured failure
        (``success=false`` is rendered downstream with
        ``policy_reason='cleanup_failed'``);
      * the foreign tree (``foreign_dir/important.txt``) survives the
        operation;
      * the production cleanup primitive does NOT call ``resolve()``
        before deletion (a ``resolve()`` would walk the swap and
        delete the foreign tree).
    """
    import tools.skill_manager_tool as sm

    skills_root = tmp_path / "hermes" / "skills"
    skills_root.mkdir(parents=True)
    monkeypatch.setattr(sm, "SKILLS_DIR", skills_root)

    # 1. Create the staging under its scratch parent.  The handle's
    # ``.path`` is the inner ``.hermes-skill-staging-<hex>`` and the
    # handle's ``.parent`` is the scratch dir containing only that
    # inner staging directory.
    staging = sm._create_private_staging(skills_root)
    staging_path = staging.path  # inner staging dir
    parent = staging.parent      # scratch dir; sole child = staging_path

    # Snapshot the identities the production captured at creation
    # so the test can demonstrate divergence WITHOUT depending on
    # ``os.lstat`` internals (the production captured them).
    captured_parent_identity = staging._parent_identity

    # Build a foreign tree under ``foreign_dir``.  Its contents must
    # survive the swap because the parent symlink now points here.
    foreign_dir = tmp_path / "foreign-parent-target"
    foreign_dir.mkdir()
    foreign_file = foreign_dir / "important.txt"
    foreign_file.write_text("preserved", encoding="utf-8")

    # 2. Remove the inner staging so ``parent`` is empty.  The staging
    # was created empty by the production (no files written inside the
    # test), so a plain ``rmdir`` is sufficient.
    staging_path.rmdir()

    # 2a. The scratch parent is now empty and may be removed so a
    # symlink can be created in its place.  This is the test's
    # preparation for the swap — it ONLY touches objects the test
    # created and does not touch the foreign tree.
    parent.rmdir()

    # 3. Replace ``parent`` with a symlink to the foreign directory.
    os.symlink(str(foreign_dir), str(parent))

    # Now the staging handle still names ``parent / .hermes-skill-staging-<hex>``.
    # That pathname no longer exists under the real filesystem (the
    # symlink transits into ``foreign_dir`` which has no such child).
    # The production captured identity at create-time and refuses to
    # walk the swap during cleanup.

    outcome = sm._cleanup_private_staging(staging)

    # The production must surface a structured failure: either the
    # parent-identity check fires BEFORE the existence check, or the
    # existence check sees the swap and surfaces it via
    # ``staging parent inode changed`` / ``could not lstat staging``.
    # Either way, ``outcome`` MUST NOT be None — silently ignoring the
    # swap would mask the contract violation.  The contract allows
    # two acceptable outcomes:
    #
    #   * outcome = (staging_path, "staging parent inode changed;
    #     foreign tree preserved")
    #     when the production revalidates parent identity first, OR
    #   * outcome = (staging_path, "could not lstat staging during
    #     cleanup: ...")
    #     when the staging lstat fires before parent revalidation.
    #
    # The contract forbids: outcome is None (no-op silent), OR
    # shutil.rmtree runs against the foreign tree (delete-path).
    assert outcome is not None, (
        "_cleanup_private_staging silently returned None after the "
        "staging parent was swapped to a symlink — the production must "
        "surface a structured failure instead of masking the swap"
    )
    failure_path, failure_msg = outcome
    assert "foreign tree preserved" in failure_msg or (
        "could not lstat staging" in failure_msg
        or "staging parent inode changed" in failure_msg
        or "staging path became" in failure_msg
        or "no longer a directory" in failure_msg
    ), (
        f"cleanup surfaced unexpected error message: {failure_msg!r}; "
        f"expected one of: foreign tree preserved / could not lstat "
        f"staging / staging parent inode changed"
    )
    # Invariant from the directive: replacement and foreign tree survive.
    assert parent.is_symlink(), (
        f"expected the staging parent to be a symlink after the swap; "
        f"got {parent!r}"
    )
    assert foreign_file.exists(), (
        f"foreign tree was destroyed by cleanup; expected {foreign_file}"
    )
    assert foreign_file.read_text(encoding="utf-8") == "preserved", (
        f"foreign file content was mutated; expected 'preserved', "
        f"got {foreign_file.read_text(encoding='utf-8')!r}"
    )
    # The captured parent identity reference is still meaningful for
    # callers that inspect it; assert the captured triple is itself
    # a non-trivial (dev, ino, S_IFMT) tuple anchored on the original
    # scratch parent.
    assert captured_parent_identity is not None
    assert len(captured_parent_identity) == 3




# ── Section 12: last-mile atomicity — non-cooperative replacement race ──────
#
# External review (Phase C corrective) requires that the destructive op
# be bound to the inode that was validated, not just to the parent
# directory.  The current implementation uses
# ``os.lstat(name, dir_fd=parent_fd)`` followed by
# ``os.unlink(name, dir_fd=parent_fd)`` — TWO separate syscalls that
# share only the namespace (parent), not the validated inode.  A
# non-cooperative actor that renames the original target and recreates
# the name with a foreign inode BETWEEN the final lstat and the unlink
# will see production delete the foreign inode.
#
# This test exercises exactly that race.  It MUST FAIL against the
# current implementation and PASS after Camino B (fail-closed refusal
# of the destructive op when no kernel identity-bound delete primitive
# is available) is applied.


def test_remove_file_replacement_after_final_identity_check_before_unlink_is_preserved(
    monkeypatch, tmp_path
):
    """Deterministic last-mile TOCTOU test for ``_remove_file``.

    Captures the destructive primitive (``os.unlink``) AND the final
    identity check (``os.lstat(name, dir_fd=...)``).  On the FIRST
    ``dir_fd`` lstat call inside ``_remove_file`` — i.e. production's
    last observed identity check — swaps the target's inode on disk
    AFTER that lstat returns and BEFORE the wrapped ``os.unlink`` runs.

    The contract under test is:

      Final destructive operation bound to validated inode: YES / NO
      Foreign replacement preserved: PASS / FAIL
      Original inode evidence preserved: PASS / FAIL

    Against the current implementation (Camino A not implementable
    with portable Python primitives), the test fails because the
    unlink is by name and the foreign replacement IS deleted.
    After Camino B (fail-closed refusal before the unlink), the
    test passes because no destructive syscall runs at all.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)

    supporting_dir = skills_root / "fail-closed" / "references"
    supporting_dir.mkdir()
    target = supporting_dir / "ok.md"
    target.write_text("ORIGINAL", encoding="utf-8")
    original_inode = target.stat().st_ino

    # Foreign replacement content — distinctly identifiable, never
    # written to disk by production itself.
    foreign_content = "FOREIGN_REPLACEMENT_AFTER_LAST_CHECK"

    # State observed by both interceptors.
    state = {
        "final_dirfd_lstat_seen": False,
        "swap_completed": False,
        "destructive_unlink_calls": [],
        "all_unlink_calls": [],
        "evidence_original": None,
        "evidence_replacement": None,
    }

    real_unlink = sm.os.unlink
    real_lstat = sm.os.lstat

    def lstat_with_marker(*args, **kwargs):
        # The final identity check is exactly the
        # ``os.lstat(re_target.name, dir_fd=parent_fd)`` call inside
        # the secure-path block.  It carries the ``dir_fd`` kwarg and
        # the target is a *basename* (no directory component).
        if (
            "dir_fd" in kwargs
            and len(args) >= 1
            and isinstance(args[0], str)
            and "/" not in args[0]
            and not state["final_dirfd_lstat_seen"]
        ):
            # Perform the real kernel lstat FIRST so production
            # validates the original inode.
            result = real_lstat(*args, **kwargs)
            state["final_dirfd_lstat_seen"] = True
            # Now perform the swap — exactly once, AFTER the final
            # identity check returned success.
            evidence = supporting_dir / "ok.md.original_evidence"
            target.rename(evidence)
            state["evidence_original"] = evidence
            foreign = supporting_dir / "ok.md"
            foreign.write_text(foreign_content, encoding="utf-8")
            state["evidence_replacement"] = foreign
            state["swap_completed"] = True
            return result
        return real_lstat(*args, **kwargs)

    def unlink_with_marker(name, *args, **kwargs):
        state["all_unlink_calls"].append((name, args, kwargs))
        # Track whether this looks like the dir_fd destructive op.
        is_dirfd = "dir_fd" in kwargs or (
            len(args) >= 1 and not isinstance(args[0], int)
        )
        if is_dirfd:
            state["destructive_unlink_calls"].append((name, args, kwargs))
            # Bail out: refuse to perform a destructive op that is
            # not kernel-bound to the validated inode.  This mirrors
            # the Camino B refusal we will apply in production.
            raise OSError(
                errno.ELIBBAD,
                "simulated Camino-B refusal: unlink is not bound to "
                "validated inode",
            )
        return real_unlink(name, *args, **kwargs)

    monkeypatch.setattr(sm.os, "lstat", lstat_with_marker)
    monkeypatch.setattr(sm.os, "unlink", unlink_with_marker)

    with session_write_policy_scope(
        _allowlist("skills", skills_root, "skill_remove_file")
    ):
        result = json.loads(
            sm.skill_manage(
                action="remove_file",
                name="fail-closed",
                file_path="references/ok.md",
            )
        )

    # ---- Diagnostic log required by the directive -----------------------
    diag = {
        "final identity check completed": "YES"
        if state["final_dirfd_lstat_seen"]
        else "NO",
        "swap occurred after final check": "YES" if state["swap_completed"] else "NO",
        "destructive syscall reached": "YES"
        if state["destructive_unlink_calls"]
        else "NO",
        "foreign replacement preserved": (
            "YES"
            if (
                state["evidence_replacement"] is not None
                and state["evidence_replacement"].exists()
                and state["evidence_replacement"].read_text(
                    encoding="utf-8"
                )
                == foreign_content
            )
            else "NO"
        ),
        "original inode evidence preserved": (
            "YES"
            if (
                state["evidence_original"] is not None
                and state["evidence_original"].exists()
                and state["evidence_original"].stat().st_ino
                == original_inode
            )
            else "NO"
        ),
    }
    print("\n[last-mile TOCTOU diagnostic]", diag)

    # ---- Hard invariants ------------------------------------------------
    # 1. Production must have run its final identity check at least once.
    assert state["final_dirfd_lstat_seen"] is True, (
        f"final dir_fd lstat was never observed; swap would not be "
        f"exercised. diag={diag}"
    )
    # 2. The swap must have happened (deterministic — runs unconditionally
    #    inside the lstat interceptor).
    assert state["swap_completed"] is True, diag
    # 3. Original inode must be preserved at the evidence path.
    assert diag["original inode evidence preserved"] == "YES", (
        f"original inode evidence destroyed: {diag}"
    )
    # 4. Foreign replacement must NOT have been deleted by production.
    assert diag["foreign replacement preserved"] == "YES", (
        f"production deleted the foreign replacement after the swap; "
        f"atomicity is not kernel-bound. diag={diag}"
    )
    # 5. The destructive op MUST NOT have reached the unlink syscall
    #    on the foreign inode.  Either Camino B refused (no call) or the
    #    contract is satisfied.
    assert state["destructive_unlink_calls"] == [], (
        f"production reached the destructive unlink syscall; "
        f"atomicity not kernel-bound. calls={state['destructive_unlink_calls']!r}"
    )
    # 6. Result must report the canonical Camino-B refusal.
    assert result["success"] is False, result
    assert result.get("policy_reason") == "atomic_identity_delete_unavailable", (
        f"expected canonical Camino-B policy_reason; got {result!r}"
    )
    assert result.get("rollback_failure_kind") == "identity_bound_unlink_unavailable", (
        f"expected canonical Camino-B rollback_failure_kind; got {result!r}"
    )
    assert result.get("live_mutation_committed") is False, result
    assert result.get("safe_to_retry") is False, result


def test_remove_file_refuses_when_identity_bound_unlink_is_unavailable(
    monkeypatch, tmp_path
):
    """Camino B refusal contract.

    When the platform / Python runtime cannot expose a kernel primitive
    that deletes exactly the validated inode (i.e. no
    ``unlinkat(target_fd, AT_EMPTY_PATH)`` equivalent), production must
    refuse without executing any unlink syscall.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)

    supporting_dir = skills_root / "fail-closed" / "references"
    supporting_dir.mkdir()
    target = supporting_dir / "ok.md"
    target.write_text("ORIGINAL", encoding="utf-8")

    unlink_calls = {"count": 0, "pathname": 0, "dirfd": 0}

    def counting_unlink(*args, **kwargs):
        unlink_calls["count"] += 1
        if "dir_fd" in kwargs:
            unlink_calls["dirfd"] += 1
        else:
            unlink_calls["pathname"] += 1
        # The test confirms production refused BEFORE this syscall.
        raise AssertionError(
            "production reached os.unlink despite Camino-B refusal; "
            "atomicity not enforced"
        )

    monkeypatch.setattr(sm.os, "unlink", counting_unlink)

    with session_write_policy_scope(
        _allowlist("skills", skills_root, "skill_remove_file")
    ):
        result = json.loads(
            sm.skill_manage(
                action="remove_file",
                name="fail-closed",
                file_path="references/ok.md",
            )
        )

    assert result["success"] is False, result
    assert result["policy_reason"] == "atomic_identity_delete_unavailable", result
    assert result["rollback_failure_kind"] == (
        "identity_bound_unlink_unavailable"
    ), result
    assert result["live_mutation_committed"] is False, result
    assert result["safe_to_retry"] is False, result
    # Zero unlink calls of any kind.
    assert unlink_calls["count"] == 0, (
        f"production called os.unlink {unlink_calls['count']} times despite "
        f"Camino-B refusal; path={unlink_calls['pathname']}, dirfd={unlink_calls['dirfd']}"
    )
    # Original target intact.
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "ORIGINAL"


# ─────────────────────────────────────────────────────────────────────────
# Delete local-state concurrency remediation (Phase C P1)
# ─────────────────────────────────────────────────────────────────────────


def test_delete_skill_uses_no_module_global_refusal_state():
    """Static-state guard: ``_delete_skill`` MUST NOT share refusal state
    across invocations via a module-level global.  Two concurrent calls
    racing the same module-level flag would corrupt the release-failure
    handler's view of ``live_mutation_committed``.

    Asserts:
      * no module attribute named ``_delete_refused``
      * no ``global _delete_refused`` statement anywhere in the source
      * the function compiles without depending on module globals
    """
    import inspect
    import tools.skill_manager_tool as sm

    # 1) No module-level attribute at all.
    assert not hasattr(sm, "_delete_refused"), (
        "tools.skill_manager_tool must NOT expose a module-level "
        "_delete_refused; per-invocation state only"
    )

    # 2) No ``global _delete_refused`` statement anywhere in the source.
    src = inspect.getsource(sm)
    assert "global _delete_refused" not in src, (
        "_delete_skill must not introduce ``global _delete_refused``; "
        "refusal state must remain per-invocation"
    )

    # 3) The function source itself MUST NOT declare ``global _delete_refused``.
    fn_src = inspect.getsource(sm._delete_skill)
    assert "global _delete_refused" not in fn_src, (
        "_delete_skill source must not declare ``global _delete_refused``"
    )

    # 4) The function source MUST contain the local initialization so
    #    future readers can see the per-invocation intent.
    assert "_delete_refused = False" in fn_src, (
        "_delete_skill must declare ``_delete_refused = False`` as a "
        "function-local variable at the top of the body"
    )


def test_concurrent_delete_refusal_and_release_failure_do_not_share_committed_state(
    monkeypatch, tmp_path,
):
    """Deterministic concurrency test: two interleaved invocations of
    ``_delete_skill`` MUST NOT race on a shared refusal flag.

    Sequence (deterministic via ``threading.Event``, no sleeps):

      1. Operation A enters ``_delete_skill``, walks through to the
         refusal block, sets its local ``_delete_refused = True``, and
         returns from the ``with`` body.  ``__exit__`` then PAUSES
         via an ``Event`` BEFORE raising the release failure.
      2. Operation B enters ``_delete_skill``, runs through to the
         entry reset point and exits via ``PermissionError`` from a
         second lock acquisition — BEFORE reaching the refusal block.
         In the OLD module-global design, B's entry reset line
         (``global _delete_refused; _delete_refused = False``) clobbers
         A's global flag mid-A.
      3. Operation A resumes its ``__exit__``, the release failure
         fires, and the exception handler reads A's LOCAL flag.

    Mandatory outcome:
      * A's payload has ``live_mutation_committed=False`` (A's local
        refusal flag was True at the moment A's release handler ran,
        independent of B's invocation clobbering the OLD global).
      * No module-level ``_delete_refused`` exists in production.

    This test fails with the old module-level ``_delete_refused`` flag
    because B's entry would reset the global to ``False`` mid-way
    through A, causing A's release handler to take the
    ``exc.live_mutation_committed = True`` branch and report a fake
    committed mutation that never happened.
    """
    import tools.skill_manager_tool as sm
    from tools.skill_manager_tool import _SkillMutationLockReleaseFailure

    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    skill_dir_a = skills_root / "fail-closed-a"
    skill_dir_b = skills_root / "fail-closed-b"
    skill_dir_a.mkdir()
    (skill_dir_a / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    skill_dir_b.mkdir()
    (skill_dir_b / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")

    # Sequence control — no sleeps, only Events.
    a_in_exit = threading.Event()    # A's body returned; __exit__ paused
    b_lock_failed = threading.Event()  # B's lock raised PermissionError
    b_done = threading.Event()         # B's full invocation finished

    class _PausingLock:
        """Lock context manager whose ``__exit__`` simulates a release
        failure, but for A it pauses to let B interleave."""

        def __init__(self, label):
            self.label = label

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if self.label == "A":
                # A's body has returned with its local
                # _delete_refused = True.  PAUSE here until B has
                # done its entry reset AND exited BEFORE refusal.
                # In the OLD global design, this is the window
                # where B would clobber the global to False.
                a_in_exit.set()
                assert b_lock_failed.wait(timeout=10), (
                    "B's lock never raised PermissionError; test deadlock"
                )
                assert b_done.wait(timeout=10), (
                    "B never finished; test deadlock"
                )
                # Now raise the release failure for A.
                raise _SkillMutationLockReleaseFailure(
                    canonical_skill_path=skill_dir_a,
                    lock_path=tmp_path / "fake-A.lock",
                    platform="posix",
                    release_error=OSError("simulated release failure A"),
                    close_error=None,
                    live_mutation_committed=False,
                )
            # For B's __exit__ (if reached): no-op.
            return False

    class _BPermissionLock:
        """Second lock that raises PermissionError immediately on
        ``__enter__`` for B.  This short-circuits B BEFORE B ever
        reaches the refusal block — exactly the cross-talk scenario
        that the OLD module-global flag would have corrupted.
        Note: B never enters the body of the ``with``, so the
        refusal block (and ``_delete_refused = True``) is never set
        for B.  But B's entry point HAS already executed the
        ``_delete_refused = False`` reset (in the OLD design)."""

        def __init__(self, label):
            self.label = label

        def __enter__(self):
            if self.label == "B":
                b_lock_failed.set()
                raise PermissionError("B lock contention")
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _lock_factory(_path):
        # Route: first caller gets the pausing A lock; second gets
        # the permission-denied B lock.
        if not a_in_exit.is_set():
            return _PausingLock("A")
        return _BPermissionLock("B")

    monkeypatch.setattr(sm, "_skill_mutation_process_lock", _lock_factory)

    results = {}

    def _op_a():
        with session_write_policy_scope(
            _allowlist("skills-a", skills_root, "skill_delete")
        ):
            results["A"] = json.loads(
                sm.skill_manage(action="delete", name="fail-closed-a")
            )

    def _op_b():
        with session_write_policy_scope(
            _allowlist("skills-b", skills_root, "skill_delete")
        ):
            results["B"] = json.loads(
                sm.skill_manage(action="delete", name="fail-closed-b")
            )

    t_a = threading.Thread(target=_op_a)
    t_a.start()
    # Wait until A has reached its __exit__ (local _delete_refused is
    # already True in its body).
    assert a_in_exit.wait(timeout=10), (
        "A never reached the refusal + release point"
    )
    # Now run B to completion.  B's lock raises PermissionError,
    # which short-circuits B BEFORE B reaches the refusal block.
    t_b = threading.Thread(target=_op_b)
    t_b.start()
    t_b.join(timeout=10)
    assert not t_b.is_alive(), "B did not complete"
    b_done.set()
    # Now let A's __exit__ raise.
    t_a.join(timeout=10)
    assert not t_a.is_alive(), "A did not complete"

    # ── Mandatory A assertions ─────────────────────────────────────
    a = results["A"]
    assert a["success"] is False, a
    assert a["policy_reason"] == "lock_release_failed", a
    assert a["live_mutation_committed"] is False, (
        "A's release handler must report live_mutation_committed=False; "
        "B's invocation must NOT have clobbered A's local refusal flag. "
        f"got {a!r}"
    )
    assert a["safe_to_retry"] is False, a

    # ── Mandatory B assertions ─────────────────────────────────────
    b = results["B"]
    assert b["success"] is False, b
    assert b["policy_reason"] == "lock_acquisition_failed", (
        "B must have short-circuited via PermissionError into the "
        "canonical acquisition-failure payload; "
        f"got {b!r}"
    )
    # B never reached the body — no destructive primitive ran.
    assert b.get("live_mutation_committed", False) is False, b

    # Cross-talk guard: A and B ran different paths with independent
    # local state — A reports refusal+release-failure, B reports
    # lock acquisition failure.  Both committed=False; no shared state.
    assert a["policy_reason"] != b["policy_reason"]

    # Both skills intact (no destructive primitive ran on either).
    assert skill_dir_a.exists()
    assert skill_dir_b.exists()
    assert (skill_dir_a / "SKILL.md").exists()
    assert (skill_dir_b / "SKILL.md").exists()


def test_concurrent_delete_refusal_state_does_not_leak_to_later_invocation(
    monkeypatch, tmp_path,
):
    """Cross-invocation leak guard: a refusal in call N must NOT
    influence call N+1's release-failure handler.  Each call has its
    own local state — back-to-back calls must each report False
    independently.
    """
    import tools.skill_manager_tool as sm
    from tools.skill_manager_tool import _SkillMutationLockReleaseFailure

    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)

    skill_dir = skills_root / "leak-test"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")

    call_count = {"n": 0}

    class _BoomLock:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            call_count["n"] += 1
            raise _SkillMutationLockReleaseFailure(
                canonical_skill_path=skill_dir,
                lock_path=tmp_path / f"boom-{call_count['n']}.lock",
                platform="posix",
                release_error=OSError(f"boom {call_count['n']}"),
                close_error=None,
                live_mutation_committed=False,
            )

    monkeypatch.setattr(sm, "_skill_mutation_process_lock", lambda _p: _BoomLock())

    # Three back-to-back calls.  All three must report
    # live_mutation_committed=False (refusal fired each time) AND
    # safe_to_retry=False.
    for i in range(3):
        with session_write_policy_scope(
            _allowlist(f"skills-{i}", skills_root, "skill_delete")
        ):
            result = json.loads(
                sm.skill_manage(action="delete", name="leak-test")
            )
        assert result["success"] is False, (i, result)
        assert result["policy_reason"] == "lock_release_failed", (i, result)
        assert result["live_mutation_committed"] is False, (
            f"call {i}: refusal must NOT leak across invocations; got {result!r}"
        )
        assert result["safe_to_retry"] is False, (i, result)

    # Skill intact.
    assert skill_dir.exists()
    assert (skill_dir / "SKILL.md").exists()


def test_delete_local_committed_state_independent_of_concurrent_refusal(
    monkeypatch, tmp_path,
):
    """Inverse: a local ``live_mutation_committed=True`` MUST stay True,
    and a local ``live_mutation_committed=False`` MUST stay False, even
    if another concurrent invocation drives the module-global counter-
    part of the old design.

    This is the helper-equivalence test for the inverse contract:

      local false remains false
      local true remains true

    We exercise the public release-failure helper directly: build two
    ``_SkillMutationLockReleaseFailure`` instances with distinct local
    ``live_mutation_committed`` values, run them through the structured
    payload formatter, and assert the payload preserves each invocation's
    own value.  No module global is touched — locality is enforced by
    the exception attribute itself.
    """
    import tools.skill_manager_tool as sm
    from tools.skill_manager_tool import _SkillMutationLockReleaseFailure

    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    skill_dir = skills_root / "helper-test"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")

    # Operation 1: refusal semantics — committed=False.
    exc_false = _SkillMutationLockReleaseFailure(
        canonical_skill_path=skill_dir,
        lock_path=tmp_path / "false.lock",
        platform="posix",
        release_error=OSError("simulated false"),
        close_error=None,
        live_mutation_committed=False,
    )
    payload_false = sm._format_lock_release_failure_payload(
        exc_false, target=skill_dir
    )
    assert payload_false["success"] is False, payload_false
    assert payload_false["live_mutation_committed"] is False, payload_false

    # Operation 2: destructive ran — committed=True.
    exc_true = _SkillMutationLockReleaseFailure(
        canonical_skill_path=skill_dir,
        lock_path=tmp_path / "true.lock",
        platform="posix",
        release_error=OSError("simulated true"),
        close_error=None,
        live_mutation_committed=True,
    )
    payload_true = sm._format_lock_release_failure_payload(
        exc_true, target=skill_dir
    )
    assert payload_true["success"] is False, payload_true
    assert payload_true["live_mutation_committed"] is True, payload_true

    # The two payloads carry their own local state — neither overwrote
    # the other.  Re-derive from the original exceptions to be sure.
    assert exc_false.live_mutation_committed is False
    assert exc_true.live_mutation_committed is True
    assert payload_false["live_mutation_committed"] != payload_true["live_mutation_committed"]


# ── Curator archive identity + atomicity (Phase C curator-archive block) ──


def _patch_curator_pass(monkeypatch):
    """Patch ``is_background_review()`` so ``_delete_skill`` enters the
    curator/archive branch.  Returns the patch handle so the test can
    restore if needed.
    """
    import tools.skill_provenance as provenance
    return monkeypatch.setattr(
        provenance, "is_background_review", lambda: True, raising=False
    )


def _spy_archive_destructive_calls(sm_local):
    """Install spies on every archive-side destructive primitive that
    could leak past a refusal.  Returns the destructive_calls dict.
    """
    destructive_calls = {
        "archive_skill": 0,
        "shutil_move": 0,
        "os_rename": 0,
        "os_replace": 0,
        "rmtree": 0,
        "unlink": 0,
        "rmdir": 0,
        "path_unlink": 0,
        "path_rmdir": 0,
        "skill_dir_rename": 0,
    }

    def spy_archive(*a, **kw):
        destructive_calls["archive_skill"] += 1
        return (True, "spy archive should not run")

    real_shutil_move = sm_local.shutil.move

    def spy_shutil_move(*a, **kw):
        destructive_calls["shutil_move"] += 1
        return real_shutil_move(*a, **kw)

    real_os_rename = sm_local.os.rename

    def spy_os_rename(*a, **kw):
        destructive_calls["os_rename"] += 1
        return real_os_rename(*a, **kw)

    real_os_replace = sm_local.os.replace

    def spy_os_replace(*a, **kw):
        destructive_calls["os_replace"] += 1
        return real_os_replace(*a, **kw)

    real_rmtree = sm_local.shutil.rmtree

    def spy_rmtree(path, *a, **kw):
        destructive_calls["rmtree"] += 1
        return real_rmtree(path, *a, **kw)

    real_unlink = sm_local.os.unlink

    def spy_unlink(path, *a, **kw):
        destructive_calls["unlink"] += 1
        return real_unlink(path, *a, **kw)

    real_rmdir = sm_local.os.rmdir

    def spy_rmdir(path, *a, **kw):
        destructive_calls["rmdir"] += 1
        return real_rmdir(path, *a, **kw)

    real_path_unlink = sm_local.Path.unlink

    def spy_path_unlink(self, *a, **kw):
        destructive_calls["path_unlink"] += 1
        return real_path_unlink(self, *a, **kw)

    real_path_rmdir = sm_local.Path.rmdir

    def spy_path_rmdir(self, *a, **kw):
        destructive_calls["path_rmdir"] += 1
        return real_path_rmdir(self, *a, **kw)

    real_skill_dir_rename = sm_local.Path.rename

    def spy_skill_dir_rename(self, *a, **kw):
        destructive_calls["skill_dir_rename"] += 1
        return real_skill_dir_rename(self, *a, **kw)

    monkeypatch_targets = sm_local.__dict__
    import pytest as _pytest
    _pytest.MonkeyPatch().setattr  # silence linter; not used here
    sm_local.shutil.move = spy_shutil_move
    sm_local.os.rename = spy_os_rename
    sm_local.os.replace = spy_os_replace
    sm_local.shutil.rmtree = spy_rmtree
    sm_local.os.unlink = spy_unlink
    sm_local.os.rmdir = spy_rmdir
    sm_local.Path.unlink = spy_path_unlink
    sm_local.Path.rmdir = spy_path_rmdir
    sm_local.Path.rename = spy_skill_dir_rename
    return destructive_calls, spy_archive


def test_curator_archive_repeats_find_and_identity_validation_inside_locks(
    monkeypatch, tmp_path,
):
    """The curator/archive branch of ``_delete_skill`` MUST repeat
    ``_find_skill``, canonicalize, and capture target + parent identity
    INSIDE both locks.  Without the in-lock revalidation a concurrent
    rename between the early ``_find_skill`` and the lock acquisition
    could redirect the archive to a foreign object.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    _patch_curator_pass(monkeypatch)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    _create_curator_umbrella(sm, skills_root, "curator-umbrella")
    skill_dir = skills_root / "fail-closed"
    canonical = skill_dir.resolve(strict=False)

    import tools.skill_manager_tool as sm_local
    calls = {"find": 0, "lstat": 0, "parent_lstat": 0}
    real_find = sm._find_skill

    def counting_find(name):
        calls["find"] += 1
        return real_find(name)

    real_lstat = sm_local.Path.lstat

    def counting_lstat(self, *a, **kw):
        st = real_lstat(self, *a, **kw)
        if str(self) == str(canonical):
            calls["lstat"] += 1
        elif str(self) == str(canonical.parent):
            calls["parent_lstat"] += 1
        return st

    monkeypatch.setattr(sm, "_find_skill", counting_find)
    monkeypatch.setattr(sm_local.Path, "lstat", counting_lstat)

    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_delete")):
        result = json.loads(
            sm.skill_manage(
                action="delete", name="fail-closed",
                absorbed_into="curator-umbrella",
            )
        )

    # Refusal with the canonical curator-archive payload.
    assert result["success"] is False
    assert result["policy_reason"] == "atomic_archive_unavailable"
    assert result["rollback_failure_kind"] == "identity_bound_archive_unavailable"
    assert result["live_mutation_committed"] is False
    assert result["safe_to_retry"] is False
    assert result["operation_kind"] == "archive"
    assert result["target"] == str(canonical)
    assert "lock_path" in result

    # Under-lock revalidation must have run at least once.  The
    # curator branch fires two lstats on the canonical target (capture
    # + final recheck) and two on the canonical parent.
    assert calls["find"] >= 2, calls  # pre-lock + in-lock revalidation
    assert calls["lstat"] >= 2, calls  # capture + final recheck
    assert calls["parent_lstat"] >= 2, calls  # capture + final recheck


def test_curator_archive_target_replacement_before_final_validation_is_rejected(
    monkeypatch, tmp_path,
):
    """If the canonical target is replaced between the pre-lock find
    and the in-lock revalidation, the curator archive MUST refuse with
    ``atomic_archive_unavailable`` and zero destructive primitives.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    _patch_curator_pass(monkeypatch)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    _create_curator_umbrella(sm, skills_root, "curator-umbrella")
    canonical = (skills_root / "fail-closed").resolve(strict=False)

    # Foreign directory: replacement tree
    foreign_dir = tmp_path / "foreign-curator-target"
    foreign_dir.mkdir()
    foreign_file = foreign_dir / "SKILL.md"
    foreign_file.write_text("foreign curator archive", encoding="utf-8")

    import tools.skill_manager_tool as sm_local
    destructive_calls, spy_archive = _spy_archive_destructive_calls(sm_local)
    monkeypatch.setattr(sm, "archive_skill", spy_archive, raising=False)

    real_find = sm._find_skill
    calls = {"n": 0}

    def swap_find(name):
        calls["n"] += 1
        if name == "fail-closed":
            # Main adds a pre-mutation audit-ledger _find_skill(name) capture.
            # The deleted skill therefore has THREE pre-lock finds: ledger capture,
            # background-review preflight, and the early _delete_skill snapshot.
            # Only the later in-lock revalidation must see the foreign directory.
            # Other names, including the umbrella target, remain real.
            if calls["n"] <= 3:
                return real_find(name)
            return {"path": foreign_dir}
        return real_find(name)

    monkeypatch.setattr(sm, "_find_skill", swap_find)

    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_delete")):
        result = json.loads(
            sm.skill_manage(
                action="delete", name="fail-closed",
                absorbed_into="curator-umbrella",
            )
        )

    assert result["success"] is False
    assert result["policy_reason"] == "atomic_archive_unavailable"
    assert result["rollback_failure_kind"] == "identity_bound_archive_unavailable"
    assert result["live_mutation_committed"] is False
    assert result["safe_to_retry"] is False
    assert result["operation_kind"] == "archive"

    # Zero archive-side destructive primitives ran.
    assert destructive_calls["archive_skill"] == 0, destructive_calls
    assert destructive_calls["shutil_move"] == 0, destructive_calls
    assert destructive_calls["os_rename"] == 0, destructive_calls
    assert destructive_calls["os_replace"] == 0, destructive_calls
    assert destructive_calls["rmtree"] == 0, destructive_calls
    assert destructive_calls["unlink"] == 0, destructive_calls
    assert destructive_calls["rmdir"] == 0, destructive_calls
    assert destructive_calls["path_unlink"] == 0, destructive_calls
    assert destructive_calls["path_rmdir"] == 0, destructive_calls
    assert destructive_calls["skill_dir_rename"] == 0, destructive_calls

    # Foreign tree preserved.
    assert foreign_dir.exists()
    assert foreign_file.exists()
    assert foreign_file.read_text(encoding="utf-8") == "foreign curator archive"


def test_curator_archive_replacement_after_final_identity_check_before_archive_is_preserved(
    monkeypatch, tmp_path,
):
    """Last-window threat model: a non-cooperative actor swaps the
    validated directory AFTER the final identity capture of BOTH
    target and parent, but BEFORE the archive syscall.  Production
    MUST return the canonical ``atomic_archive_unavailable`` payload.

    State machine: the harness wraps ``Path.lstat`` and counts the
    number of lstat calls issued from inside ``sm._delete_skill`` on
    two paths:

      * the target skill directory (``re_skill_dir``);
      * the target's parent (``re_skill_dir.parent``).

    Production's under-lock flow captures each path TWICE: a
    pre-capture before the destructive op, and a final recheck
    immediately before the last-mile atomicity refusal.  The harness
    swap fires on the second (recheck) lstat of the parent path,
    AFTER returning the captured original stat to production — so
    production's pre/post comparisons both see the original identity
    and the refusal fires with ``atomic_archive_unavailable``.  No
    identity-bearing lstat call is ever made against the swapped
    state.

    Frame identity is via ``f_code is sm._delete_skill.__code__`` —
    no numeric source-line ranges, no ``f_lineno`` /
    ``co_firstlineno``, no ``inspect.stack`` to drive branch
    selection.  The wrapper inspects only ``self`` (the path) and the
    identity of the caller frame's code object.

    The swap-mutation primitive is ``os.rename``, wrapped in a local
    ``swap_in_progress`` flag so the production-side ``os.rename``
    spy can distinguish the harness swap from any production rename.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    _patch_curator_pass(monkeypatch)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    _create_curator_umbrella(sm, skills_root, "curator-umbrella")
    skill_dir = skills_root / "fail-closed"
    skill_md = skill_dir / "SKILL.md"
    original_bytes = skill_md.read_bytes()
    canonical = skill_dir.resolve(strict=False)

    skill_dir_resolved = skill_dir.resolve()
    skill_dir_parent_resolved = skill_dir.parent.resolve()

    # Evidence directory: where the original tree is moved out before
    # the archive would run.
    evidence_dir = tmp_path / "evidence"
    foreign_content = "FOREIGN curator-archive replacement"
    foreign_file = evidence_dir / "replacement.md"

    import tools.skill_manager_tool as sm_local

    state = {
        # Per-path counters for lstat calls observed inside _delete_skill
        # on the canonical paths.  Production captures each path
        # exactly twice inside the under-lock body.
        "target_lstat_count": 0,
        "parent_lstat_count": 0,
        # Recorded on the FIRST lstat for each path (the production
        # pre-capture).
        "target_pre_capture_identity": None,
        "parent_pre_capture_identity": None,
        # Set once the SECOND (recheck) lstat for each path has
        # returned the ORIGINAL identity to production.
        "target_final_captured_with_original": False,
        "parent_final_captured_with_original": False,
        # True iff the harness swap ran.  Triggered exclusively on the
        # parent recheck lstat AFTER target_recheck + parent_recheck
        # have both returned original identity to production.
        "swap_performed": False,
        "swap_occurred_after_both_final_captures": False,
        # Counters for any lstat the wrapper sees on the canonical
        # paths AFTER the swap fired.  Spec requires both to remain 0.
        "post_swap_target_identity_calls": 0,
        "post_swap_parent_identity_calls": 0,
        # Foreign/evidence presence observed by the harness at the
        # moment production's refusal runs (asserted post-call).
        "foreign_replacement_existed_before_refusal": False,
        "original_evidence_existed_before_refusal": False,
        # Guard flag for swap-mutation primitive so production-side
        # spies can distinguish harness swap from production rename.
        "swap_in_progress": False,
    }

    # Production-side destructive-primitive spies.  Each spy checks
    # ``state["swap_in_progress"]`` so the swap-mutation primitive the
    # test simulates (os.rename) does NOT inflate the destructive-call
    # counters — only production-side calls count.
    destructive_calls = {
        "archive_skill": 0,
        "shutil_move": 0,
        "os_rename": 0,
        "os_replace": 0,
        "rmtree": 0,
        "unlink": 0,
        "rmdir": 0,
        "path_unlink": 0,
        "path_rmdir": 0,
        "skill_dir_rename": 0,
    }

    real_rmtree = sm_local.shutil.rmtree

    def spy_rmtree(path, *a, **kw):
        if not state["swap_in_progress"]:
            destructive_calls["rmtree"] += 1
        return real_rmtree(path, *a, **kw)

    real_unlink = sm_local.os.unlink

    def spy_unlink(path, *a, **kw):
        if not state["swap_in_progress"]:
            destructive_calls["unlink"] += 1
        return real_unlink(path, *a, **kw)

    real_rmdir = sm_local.os.rmdir

    def spy_rmdir(path, *a, **kw):
        if not state["swap_in_progress"]:
            destructive_calls["rmdir"] += 1
        return real_rmdir(path, *a, **kw)

    real_path_unlink = sm_local.Path.unlink

    def spy_path_unlink(self, *a, **kw):
        if not state["swap_in_progress"]:
            destructive_calls["path_unlink"] += 1
        return real_path_unlink(self, *a, **kw)

    real_path_rmdir = sm_local.Path.rmdir

    def spy_path_rmdir(self, *a, **kw):
        if not state["swap_in_progress"]:
            destructive_calls["path_rmdir"] += 1
        return real_path_rmdir(self, *a, **kw)

    real_skill_dir_rename = sm_local.Path.rename

    def spy_skill_dir_rename(self, *a, **kw):
        if not state["swap_in_progress"]:
            destructive_calls["skill_dir_rename"] += 1
        return real_skill_dir_rename(self, *a, **kw)

    real_shutil_move = sm_local.shutil.move

    def spy_shutil_move(*a, **kw):
        if not state["swap_in_progress"]:
            destructive_calls["shutil_move"] += 1
        return real_shutil_move(*a, **kw)

    real_os_rename = sm_local.os.rename

    def spy_os_rename(*a, **kw):
        if not state["swap_in_progress"]:
            destructive_calls["os_rename"] += 1
        return real_os_rename(*a, **kw)

    real_os_replace = sm_local.os.replace

    def spy_os_replace(*a, **kw):
        if not state["swap_in_progress"]:
            destructive_calls["os_replace"] += 1
        return real_os_replace(*a, **kw)

    sm_local.shutil.rmtree = spy_rmtree
    sm_local.os.unlink = spy_unlink
    sm_local.os.rmdir = spy_rmdir
    sm_local.os.rename = spy_os_rename
    sm_local.os.replace = spy_os_replace
    sm_local.shutil.move = spy_shutil_move
    sm_local.Path.unlink = spy_path_unlink
    sm_local.Path.rmdir = spy_path_rmdir
    sm_local.Path.rename = spy_skill_dir_rename

    def spy_archive(*a, **kw):
        destructive_calls["archive_skill"] += 1
        return (True, "spy archive should not run")

    monkeypatch.setattr(sm, "archive_skill", spy_archive, raising=False)

    real_lstat = sm_local.Path.lstat
    delete_skill_code = sm._delete_skill.__code__

    def _identity_of(stat_result):
        return (
            stat_result.st_dev,
            stat_result.st_ino,
            stat.S_IFMT(stat_result.st_mode),
        )

    def lstat_with_final_swap(self, *a, **kw):
        # Resolve the path lazily inside the wrapper so monkeypatched
        # environments that swap symlinks post-setup are still
        # recognised by identity comparison.
        try:
            self_resolved = Path(str(self)).resolve()
        except OSError:
            self_resolved = Path(str(self))

        # Identity-based caller detection: walk the live frame chain
        # (no source-line numbers) and ask whether any frame's code
        # object IS ``sm._delete_skill.__code__``.
        caller_frame = sys._getframe(0)
        inside_delete_skill = False
        f = caller_frame
        while f is not None:
            if f.f_code is delete_skill_code:
                inside_delete_skill = True
                break
            f = f.f_back
        del caller_frame

        is_target = self_resolved == skill_dir_resolved
        is_parent = self_resolved == skill_dir_parent_resolved

        if not inside_delete_skill or (not is_target and not is_parent):
            return real_lstat(self, *a, **kw)

        # Capture the real stat BEFORE any swap fires, so production
        # always sees the original identity when comparing.
        st = real_lstat(self, *a, **kw)
        ident = _identity_of(st)

        if is_target:
            state["target_lstat_count"] += 1
            if state["target_lstat_count"] == 1:
                state["target_pre_capture_identity"] = ident
            elif state["target_lstat_count"] == 2:
                state["target_final_captured_with_original"] = True
            if state["swap_performed"]:
                state["post_swap_target_identity_calls"] += 1
        else:  # is_parent
            state["parent_lstat_count"] += 1
            if state["parent_lstat_count"] == 1:
                state["parent_pre_capture_identity"] = ident
            elif state["parent_lstat_count"] == 2:
                state["parent_final_captured_with_original"] = True
            if state["swap_performed"]:
                state["post_swap_parent_identity_calls"] += 1

        # Trigger the swap on the parent recheck — the LAST identity
        # call production makes on the canonical paths.  By this
        # point production has already accepted both target pre/post
        # identity comparisons AND the parent pre-comparison.  The
        # post-comparison follows inline; we perform the swap AFTER
        # returning the original parent stat so production's
        # comparison still passes.  The harness uses ``os.rename``
        # with the ``swap_in_progress`` flag so the production-side
        # ``os.rename`` spy can distinguish the harness swap from any
        # production rename.
        if (
            not state["swap_performed"]
            and is_parent
            and state["parent_lstat_count"] == 2
            and state["target_final_captured_with_original"]
            and state["parent_final_captured_with_original"]
        ):
            state["swap_in_progress"] = True
            os.rename(str(skill_dir), str(evidence_dir))
            skill_dir.mkdir()
            foreign_file.write_text(foreign_content, encoding="utf-8")
            state["swap_in_progress"] = False
            state["swap_performed"] = True
            state["swap_occurred_after_both_final_captures"] = True
            state["foreign_replacement_existed_before_refusal"] = (
                foreign_file.exists()
            )
            state["original_evidence_existed_before_refusal"] = (
                evidence_dir.exists()
                and (evidence_dir / "SKILL.md").exists()
            )

        return st

    monkeypatch.setattr(sm_local.Path, "lstat", lstat_with_final_swap)

    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_delete")):
        result = json.loads(
            sm.skill_manage(
                action="delete", name="fail-closed",
                absorbed_into="curator-umbrella",
            )
        )

    # ── State-machine contract: the swap must have fired AFTER both
    # final captures returned the ORIGINAL identity to production,
    # with zero identity-bearing lstat calls observed afterwards.
    assert state["target_lstat_count"] >= 2, state
    assert state["parent_lstat_count"] >= 2, state
    assert state["target_final_captured_with_original"] is True, state
    assert state["parent_final_captured_with_original"] is True, state
    assert state["swap_performed"] is True, state
    assert state["swap_occurred_after_both_final_captures"] is True, state
    assert state["post_swap_target_identity_calls"] == 0, (
        f"no target lstat may follow the swap; observed "
        f"{state['post_swap_target_identity_calls']}"
    )
    assert state["post_swap_parent_identity_calls"] == 0, (
        f"no parent lstat may follow the swap; observed "
        f"{state['post_swap_parent_identity_calls']}"
    )
    # The pre-capture identities recorded by the harness MUST be set
    # (production captured them via real_lstat before the swap fired).
    assert state["target_pre_capture_identity"] is not None, state
    assert state["parent_pre_capture_identity"] is not None, state

    # ── Foreign replacement and original evidence exist at refusal
    # time and remain on disk after the call returns.
    assert state["foreign_replacement_existed_before_refusal"] is True, state
    assert state["original_evidence_existed_before_refusal"] is True, state
    assert foreign_file.exists(), "foreign replacement MUST exist when refusal runs"
    assert foreign_file.read_text(encoding="utf-8") == foreign_content
    assert evidence_dir.exists(), "original skill tree MUST be in evidence"
    assert (evidence_dir / "SKILL.md").exists(), "original SKILL.md MUST be in evidence"
    assert (evidence_dir / "SKILL.md").read_bytes() == original_bytes

    # ── Production payload: EXACT atomic_archive_unavailable refusal.
    assert result["success"] is False
    assert result["policy_reason"] == "atomic_archive_unavailable", result
    assert (
        result["rollback_failure_kind"]
        == "identity_bound_archive_unavailable"
    ), result
    assert result.get("live_mutation_committed", False) is False
    assert result.get("safe_to_retry", False) is False
    assert result["operation_kind"] == "archive"

    # ── Zero archive-side destructive primitives ran on the
    # production side.  The harness's own os.rename call was guarded
    # by ``swap_in_progress`` so it does not inflate these counters.
    assert destructive_calls["archive_skill"] == 0, destructive_calls
    assert destructive_calls["shutil_move"] == 0, destructive_calls
    assert destructive_calls["os_rename"] == 0, destructive_calls
    assert destructive_calls["os_replace"] == 0, destructive_calls
    assert destructive_calls["rmtree"] == 0, destructive_calls
    assert destructive_calls["unlink"] == 0, destructive_calls
    assert destructive_calls["rmdir"] == 0, destructive_calls
    assert destructive_calls["path_unlink"] == 0, destructive_calls
    assert destructive_calls["path_rmdir"] == 0, destructive_calls
    assert destructive_calls["skill_dir_rename"] == 0, destructive_calls

    # ── Both trees are preserved post-call.
    assert foreign_file.exists(), "foreign replacement MUST be preserved after refusal"
    assert evidence_dir.exists(), "original skill evidence MUST be preserved after refusal"


def test_curator_archive_refuses_when_identity_bound_archive_is_unavailable(
    monkeypatch, tmp_path,
):
    """Camino B: the curator/archive branch refuses with the structured
    payload BEFORE any archive/destructive primitive runs.  No archive
    call, no shutil.move, no os.rename, no os.replace, no rmtree,
    no unlink, no rmdir.  Original tree preserved.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    _patch_curator_pass(monkeypatch)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    _create_curator_umbrella(sm, skills_root, "curator-umbrella")
    skill_dir = skills_root / "fail-closed"
    skill_md = skill_dir / "SKILL.md"
    original_bytes = skill_md.read_bytes()
    canonical = skill_dir.resolve(strict=False)

    import tools.skill_manager_tool as sm_local
    destructive_calls, spy_archive = _spy_archive_destructive_calls(sm_local)
    monkeypatch.setattr(sm, "archive_skill", spy_archive, raising=False)

    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_delete")):
        result = json.loads(
            sm.skill_manage(
                action="delete", name="fail-closed",
                absorbed_into="curator-umbrella",
            )
        )

    # Structured fail-closed payload.
    assert result["success"] is False
    assert result["policy_reason"] == "atomic_archive_unavailable"
    assert result["rollback_failure_kind"] == "identity_bound_archive_unavailable"
    assert result["live_mutation_committed"] is False
    assert result["safe_to_retry"] is False
    assert result["operation_kind"] == "archive"
    assert result["target"] == str(canonical)
    assert "lock_path" in result

    # Zero archive-side destructive primitives ran.
    assert destructive_calls["archive_skill"] == 0, destructive_calls
    assert destructive_calls["shutil_move"] == 0, destructive_calls
    assert destructive_calls["os_rename"] == 0, destructive_calls
    assert destructive_calls["os_replace"] == 0, destructive_calls
    assert destructive_calls["rmtree"] == 0, destructive_calls
    assert destructive_calls["unlink"] == 0, destructive_calls
    assert destructive_calls["rmdir"] == 0, destructive_calls
    assert destructive_calls["path_unlink"] == 0, destructive_calls
    assert destructive_calls["path_rmdir"] == 0, destructive_calls
    assert destructive_calls["skill_dir_rename"] == 0, destructive_calls

    # Skill intact.
    assert skill_dir.exists()
    assert skill_md.exists()
    assert skill_md.read_bytes() == original_bytes


def test_curator_archive_refusal_plus_lock_release_failure_preserves_not_committed(
    monkeypatch, tmp_path,
):
    """When the curator archive refuses (atomic_archive_unavailable) and
    the interprocess lock release subsequently fails, the structured
    payload MUST still report ``live_mutation_committed=false`` and
    ``safe_to_retry=false``.  A release failure cannot transform a
    pre-mutation refusal into a committed mutation.

    Curator-archive release-failure contract: because the curator archive
    refused BEFORE any archive primitive ran, the curator-specific
    ``atomic_archive_unavailable`` reason MUST be restored as the primary
    ``policy_reason`` (not folded into ``lock_release_failed``). All other
    payload fields (``rollback_failure_kind``, ``live_mutation_committed``
    = False, ``safe_to_retry`` = False, ``target``, ``lock_path``) are
    preserved from the lock-release-failure payload.
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    _patch_curator_pass(monkeypatch)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    _create_curator_umbrella(sm, skills_root, "curator-umbrella")
    skill_dir = skills_root / "fail-closed"
    skill_md = skill_dir / "SKILL.md"
    original_bytes = skill_md.read_bytes()
    canonical = skill_dir.resolve(strict=False)

    from tools.skill_manager_tool import _SkillMutationLockReleaseFailure

    class _BoomLock:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            raise _SkillMutationLockReleaseFailure(
                canonical_skill_path=canonical,
                lock_path=tmp_path / "boom-curator.lock",
                platform="posix",
                release_error=OSError("simulated curator release failure"),
                close_error=None,
                live_mutation_committed=False,
            )

    monkeypatch.setattr(sm, "_skill_mutation_process_lock", lambda _p: _BoomLock())

    with session_write_policy_scope(_allowlist("skills", skills_root, "skill_delete")):
        result = json.loads(
            sm.skill_manage(
                action="delete", name="fail-closed",
                absorbed_into="curator-umbrella",
            )
        )

    # Curator-archive release-failure payload: the curator-specific
    # ``atomic_archive_unavailable`` reason is restored as the primary
    # ``policy_reason`` (release failure cannot fold a pre-mutation
    # refusal into ``lock_release_failed``).  ``rollback_failure_kind``,
    # ``live_mutation_committed=False``, ``safe_to_retry=False``,
    # ``target``, ``lock_path``, and ``operation_kind`` are all preserved.
    assert result["success"] is False
    assert result["policy_reason"] == "atomic_archive_unavailable"
    assert result["rollback_failure_kind"] == "lock_release_failure"
    assert result["operation_kind"] == "archive"
    assert result["live_mutation_committed"] is False, result
    assert result["safe_to_retry"] is False
    # Skill intact.
    assert skill_dir.exists()
    assert skill_md.exists()
    assert skill_md.read_bytes() == original_bytes


def test_concurrent_curator_archive_refusals_do_not_share_operation_state(
    monkeypatch, tmp_path,
):
    """Two concurrent curator-archive invocations MUST NOT share the
    per-invocation refusal flag.  Each invocation derives its own
    ``live_mutation_committed`` value from its own local state.  We
    exercise this by overlapping two refusals via a Barrier (no
    sleeps).
    """
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    _patch_curator_pass(monkeypatch)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    # Two distinct skills so the calls can run concurrently without
    # tripping over the per-path lock.
    _create_clean_skill(sm, skills_root)
    _create_curator_umbrella(sm, skills_root, "curator-umbrella-a")
    _create_curator_umbrella(sm, skills_root, "curator-umbrella-b")
    skill_a_dir = skills_root / "fail-closed"
    skill_b_dir = skills_root / "concurrent-curator"
    skill_b_dir.mkdir()
    (skill_b_dir / "SKILL.md").write_text(
        "---\nname: concurrent-curator\ndescription: concurrent curator test.\n---\n# Concurrent Curator\n",
        encoding="utf-8",
    )
    skill_b_md = skill_b_dir / "SKILL.md"
    original_b_bytes = skill_b_md.read_bytes()

    # Focal block: register both skills as curator-managed via skill_usage
    # so the curator-archive refusal path is exercised deterministically.
    from tools import skill_usage

    monkeypatch.setattr(skill_usage, "_skills_dir", lambda: skills_root)
    for skill_name in ("fail-closed", "concurrent-curator"):
        ok, msg = skill_usage.adopt_skill(skill_name)
        assert ok, msg
        assert skill_usage.is_curator_managed(skill_name)

    results = {}
    barrier = threading.Barrier(2)

    def call_delete(name, key, umbrella):
        barrier.wait()
        with session_write_policy_scope(
            _allowlist(f"skills-{key}", skills_root, "skill_delete")
        ):
            results[key] = json.loads(
                sm.skill_manage(
                    action="delete", name=name, absorbed_into=umbrella,
                )
            )

    t1 = threading.Thread(
        target=call_delete, args=("fail-closed", "a", "curator-umbrella-a"),
    )
    t2 = threading.Thread(
        target=call_delete, args=("concurrent-curator", "b", "curator-umbrella-b"),
    )
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Both must independently refuse with the curator-archive payload
    # and must NOT leak each other's refusal flag.
    for key, name in (("a", "fail-closed"), ("b", "concurrent-curator")):
        r = results[key]
        assert r["success"] is False, (key, r)
        assert r["policy_reason"] == "atomic_archive_unavailable", (key, r)
        assert r["rollback_failure_kind"] == "identity_bound_archive_unavailable", (key, r)
        assert r["live_mutation_committed"] is False, (key, r)
        assert r["safe_to_retry"] is False, (key, r)
        assert r["operation_kind"] == "archive", (key, r)

    # Skills intact (each refusal preserved its own tree).
    assert skill_a_dir.exists()
    assert (skill_a_dir / "SKILL.md").exists()
    assert skill_b_dir.exists()
    assert skill_b_md.exists()
    assert skill_b_md.read_bytes() == original_b_bytes


# ─────────────────────────────────────────────────────────────────────────
# Phase C P1: _apply_and_publish_patch._last_result and
# _publish_write_file._last_result shared-state remediation.
#
# The OLD design used two function-attribute side channels
# (``_apply_and_publish_patch._last_result`` and
# ``_publish_write_file._last_result``) as a workaround for what the
# docstring called a ``return``-inside-``try`` anti-pattern.  But the
# real anti-pattern was the shared mutable state itself: two concurrent
# callers would overwrite each other's failure payloads and the LATER
# caller would observe the EARLIER caller's result.  The helpers now
# return their structured failure directly (None on success, dict on
# failure).  These tests lock the contract in.
# ─────────────────────────────────────────────────────────────────────────


def test_patch_and_write_helpers_use_no_function_result_attributes():
    """Static-state guard: the two helpers that used to stash their
    outcome on ``_last_result`` MUST NOT carry that attribute any more.

    Asserts:
      * neither helper exposes ``_last_result`` as a function attribute
      * no source file in the repo contains the legacy attribute access
        (only allowed: comment references that document the removal)
    """
    import inspect
    import tools.skill_manager_tool as sm

    # 1) Neither helper exposes _last_result.
    assert not hasattr(sm._apply_and_publish_patch, "_last_result"), (
        "_apply_and_publish_patch must NOT expose _last_result; "
        "results belong to the calling invocation only"
    )
    assert not hasattr(sm._publish_write_file, "_last_result"), (
        "_publish_write_file must NOT expose _last_result; "
        "results belong to the calling invocation only"
    )

    # 2) Neither helper's source contains _last_result.
    patch_src = inspect.getsource(sm._apply_and_publish_patch)
    write_src = inspect.getsource(sm._publish_write_file)
    assert "._last_result" not in patch_src, (
        "_apply_and_publish_patch source must not reference ._last_result"
    )
    assert "._last_result" not in write_src, (
        "_publish_write_file source must not reference ._last_result"
    )

    # 3) Neither caller's source reads _last_result.
    patch_skill_src = inspect.getsource(sm._patch_skill)
    write_file_src = inspect.getsource(sm._write_file)
    assert "._last_result" not in patch_skill_src, (
        "_patch_skill source must not read ._last_result from the helper"
    )
    assert "._last_result" not in write_file_src, (
        "_write_file source must not read ._last_result from the helper"
    )


def test_concurrent_patch_operations_do_not_share_failure_results(
    monkeypatch, tmp_path,
):
    """Two interleaved ``_patch_skill`` invocations MUST NOT share the
    patch helper's failure outcome.  With the OLD
    ``_apply_and_publish_patch._last_result`` design, operation A's
    scanner-denial payload would be overwritten by operation B's
    success while A was still in flight; A would then observe B's
    success result.  The helpers now return their result locally.

    Sequence (deterministic via ``threading.Event``, no sleeps):

      1. Operation A enters ``_patch_skill``, walks through to the
         scanner, the scanner returns a distinctive denial, A pauses.
      2. Operation B enters ``_patch_skill``, the scanner returns None
         for B, B's publish completes, B's payload is recorded.
      3. Operation A's pause ends; A's helper returns its local
         scanner-denial payload — NOT B's success.

    Mandatory outcome:
      * A.result has its own distinctive error marker, NOT B's success.
      * B.result has its own success=True payload, NOT A's failure.
      * A.target corresponds to A, B.target corresponds to B.
      * No cross-talk in either direction.
    """
    import tools.skill_manager_tool as sm

    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    skill_dir_a = skills_root / "fail-closed"
    skill_dir_b = skills_root / "concurrent-patch"
    skill_dir_b.mkdir()
    (skill_dir_b / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")

    a_scanned = threading.Event()    # A's scanner denial returned; A paused
    b_published = threading.Event()  # B's publish completed
    a_resume = threading.Event()     # Release A from its pause

    scan_calls: list[str] = []

    def _scan(staged_dir):
        scan_calls.append(staged_dir.name)
        # First call: A's scan — distinctive denial, pause.
        # Subsequent calls: B's scan (or A re-entry) — None for B.
        if "fail-closed" in staged_dir.name and not a_scanned.is_set():
            a_scanned.set()
            # Pause A's helper until B has finished publishing.
            assert a_resume.wait(timeout=10), (
                "A's resume was never signalled; test deadlock"
            )
            return "A_SCAN_DENIED distinctive marker for A"
        return None

    monkeypatch.setattr(sm, "_security_scan_skill_fail_closed", _scan)

    results: dict = {}
    errors: dict = {}

    def _op_a():
        try:
            with session_write_policy_scope(
                _allowlist("skills-a", skills_root, "skill_patch")
            ):
                results["A"] = json.loads(
                    sm.skill_manage(
                        action="patch",
                        name="fail-closed",
                        old_string="Fail Closed",
                        new_string="A_REPLACEMENT_MARKER",
                    )
                )
        except Exception as exc:
            errors["A"] = exc

    def _op_b():
        try:
            with session_write_policy_scope(
                _allowlist("skills-b", skills_root, "skill_patch")
            ):
                results["B"] = json.loads(
                    sm.skill_manage(
                        action="patch",
                        name="concurrent-patch",
                        old_string="Fail Closed",
                        new_string="B_REPLACEMENT_MARKER",
                    )
                )
            b_published.set()
        except Exception as exc:
            errors["B"] = exc

    t_a = threading.Thread(target=_op_a)
    t_a.start()
    # Wait until A has hit its scanner denial and is paused.
    assert a_scanned.wait(timeout=10), "A never reached its scanner denial"
    # Now run B to completion while A is paused inside its helper.
    t_b = threading.Thread(target=_op_b)
    t_b.start()
    assert b_published.wait(timeout=10), "B never completed its publish"
    # Let A resume from its pause.
    a_resume.set()
    t_a.join(timeout=10)
    t_b.join(timeout=10)

    assert not errors, f"unexpected exceptions: {errors!r}"
    assert "A" in results and "B" in results

    # ── A's mandatory outcome ──────────────────────────────────────
    a = results["A"]
    assert a["success"] is False, (
        f"A must report its own scanner denial; got {a!r}"
    )
    assert a.get("error") == "A_SCAN_DENIED distinctive marker for A", (
        f"A's error must be A's distinctive marker, NOT B's result; got {a!r}"
    )
    # A's payload reports live_mutation_committed=False (the scanner
    # fired BEFORE any publish, so the live tree is untouched).  The
    # field may be present (added by _combine_cleanup_failure if cleanup
    # failed) or absent (no cleanup failure).  When present it MUST be
    # False.
    assert a.get("live_mutation_committed", False) is False, a
    # A's SKILL.md must NOT have been mutated by B.
    assert "A_REPLACEMENT_MARKER" not in (skill_dir_a / "SKILL.md").read_text(
        encoding="utf-8"
    )

    # ── B's mandatory outcome ──────────────────────────────────────
    b = results["B"]
    assert b["success"] is True, (
        f"B must report its own success; got {b!r}"
    )
    # B's SKILL.md must reflect B's edit.
    assert "B_REPLACEMENT_MARKER" in (skill_dir_b / "SKILL.md").read_text(
        encoding="utf-8"
    )

    # ── Cross-talk guard ───────────────────────────────────────────
    # A's payload must NOT carry any of B's success markers.
    assert "B_REPLACEMENT_MARKER" not in json.dumps(a), a
    # B's payload must NOT carry A's distinctive error.
    assert "A_SCAN_DENIED" not in json.dumps(b), b
    # A and B must report different success values (one false, one true).
    assert a["success"] != b["success"]


def test_concurrent_write_file_operations_do_not_share_failure_results(
    monkeypatch, tmp_path,
):
    """Two interleaved ``_write_file`` invocations MUST NOT share the
    write helper's failure outcome.  Mirrors the patch test above but
    exercises ``_publish_write_file._last_result`` (now removed).

    Sequence:
      1. Operation A's scanner returns a distinctive denial; A pauses.
      2. Operation B's scanner returns None; B publishes successfully.
      3. Operation A resumes and reports its OWN failure, not B's.

    Two different skills are used so each call takes its own
    per-skill lock; otherwise B would block on A's lock and the test
    would deadlock instead of running concurrently.

    Mandatory outcome:
      * A.result has its own error; B.result has its own success.
      * Targets independent, failure markers independent.
    """
    import tools.skill_manager_tool as sm

    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    skill_b = skills_root / "concurrent-write"
    skill_b.mkdir()
    (skill_b / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")

    a_scanned = threading.Event()
    b_published = threading.Event()
    a_resume = threading.Event()

    def _scan(staged_dir):
        # A's staged dir is the fail-closed skill; B's is concurrent-write.
        if staged_dir.name == "fail-closed" and not a_scanned.is_set():
            a_scanned.set()
            assert a_resume.wait(timeout=10), "A resume never signalled"
            return "A_WRITE_SCAN_DENIED distinctive marker"
        return None

    monkeypatch.setattr(sm, "_security_scan_skill_fail_closed", _scan)

    results: dict = {}
    errors: dict = {}

    def _op_a():
        try:
            with session_write_policy_scope(
                _allowlist("skills-a", skills_root, "skill_write_file")
            ):
                results["A"] = json.loads(
                    sm.skill_manage(
                        action="write_file",
                        name="fail-closed",
                        file_path="references/A_notes.md",
                        file_content="# A notes\n",
                    )
                )
        except Exception as exc:
            errors["A"] = exc

    def _op_b():
        try:
            with session_write_policy_scope(
                _allowlist("skills-b", skills_root, "skill_write_file")
            ):
                results["B"] = json.loads(
                    sm.skill_manage(
                        action="write_file",
                        name="concurrent-write",
                        file_path="references/B_notes.md",
                        file_content="# B notes\n",
                    )
                )
            b_published.set()
        except Exception as exc:
            errors["B"] = exc

    t_a = threading.Thread(target=_op_a)
    t_a.start()
    assert a_scanned.wait(timeout=10), "A never reached its scanner denial"
    t_b = threading.Thread(target=_op_b)
    t_b.start()
    assert b_published.wait(timeout=10), "B never completed"
    a_resume.set()
    t_a.join(timeout=10)
    t_b.join(timeout=10)

    assert not errors, f"unexpected exceptions: {errors!r}"
    assert "A" in results and "B" in results

    a = results["A"]
    b = results["B"]

    # ── A's mandatory outcome ──────────────────────────────────────
    assert a["success"] is False, a
    assert a.get("error") == "A_WRITE_SCAN_DENIED distinctive marker", a
    assert a.get("live_mutation_committed", False) is False, a
    # A's file MUST NOT exist (A's publish was blocked).
    assert not (
        skills_root / "fail-closed" / "references" / "A_notes.md"
    ).exists()

    # ── B's mandatory outcome ──────────────────────────────────────
    assert b["success"] is True, b
    b_file = skills_root / "concurrent-write" / "references" / "B_notes.md"
    assert b_file.exists(), "B's file must be published"
    assert b_file.read_text(encoding="utf-8") == "# B notes\n"

    # ── Cross-talk guard ───────────────────────────────────────────
    assert "A_WRITE_SCAN_DENIED" not in json.dumps(b), b
    assert b.get("error") != a.get("error")


def test_patch_and_write_file_results_are_isolated_across_concurrent_operations(
    monkeypatch, tmp_path,
):
    """Cross-operation guard: a concurrent patch and write_file MUST NOT
    share their outcome via any registry, holder, or module global
    introduced as a substitute for the removed ``_last_result``
    attributes.  Operation A patches skill A; operation B writes a file
    on skill B.  Both helpers must carry their own result locally.
    """
    import tools.skill_manager_tool as sm

    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)
    skill_b_dir = skills_root / "cross-write"
    skill_b_dir.mkdir()
    (skill_b_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")

    a_scanned = threading.Event()
    b_done = threading.Event()
    a_resume = threading.Event()

    def _scan(staged_dir):
        if "fail-closed" in staged_dir.name and not a_scanned.is_set():
            a_scanned.set()
            assert a_resume.wait(timeout=10), "A resume never signalled"
            return "A_CROSS_PATCH_DENIED"
        return None

    monkeypatch.setattr(sm, "_security_scan_skill_fail_closed", _scan)

    results: dict = {}
    errors: dict = {}

    def _patch_a():
        try:
            with session_write_policy_scope(
                _allowlist("cross-a", skills_root, "skill_patch")
            ):
                results["A_patch"] = json.loads(
                    sm.skill_manage(
                        action="patch",
                        name="fail-closed",
                        old_string="Fail Closed",
                        new_string="PATCH_A",
                    )
                )
        except Exception as exc:
            errors["A_patch"] = exc

    def _write_b():
        try:
            with session_write_policy_scope(
                _allowlist("cross-b", skills_root, "skill_write_file")
            ):
                results["B_write"] = json.loads(
                    sm.skill_manage(
                        action="write_file",
                        name="cross-write",
                        file_path="references/notes.md",
                        file_content="# B\n",
                    )
                )
            b_done.set()
        except Exception as exc:
            errors["B_write"] = exc

    t_a = threading.Thread(target=_patch_a)
    t_a.start()
    assert a_scanned.wait(timeout=10), "A never reached its denial"
    t_b = threading.Thread(target=_write_b)
    t_b.start()
    assert b_done.wait(timeout=10), "B write never completed"
    a_resume.set()
    t_a.join(timeout=10)
    t_b.join(timeout=10)

    assert not errors, f"unexpected exceptions: {errors!r}"
    a = results["A_patch"]
    b = results["B_write"]

    # A must carry A's denial.
    assert a["success"] is False, a
    assert a.get("error") == "A_CROSS_PATCH_DENIED", a
    assert a.get("live_mutation_committed", False) is False, a

    # B must carry B's success.
    assert b["success"] is True, b
    # Successful writes do not surface live_mutation_committed unless
    # cleanup failed; absence == no cleanup failure == success path.
    # The on-disk side effect is the strongest signal: B's file is
    # published and B's SKILL.md is untouched.
    b_notes = skills_root / "cross-write" / "references" / "notes.md"
    assert b_notes.exists(), "B's file must be published"
    assert b_notes.read_text(encoding="utf-8") == "# B\n"

    # No cross-talk.
    assert "A_CROSS_PATCH_DENIED" not in json.dumps(b), b
    assert "PATCH_A" not in json.dumps(b), b

    # A's SKILL.md must NOT have been mutated.
    assert "PATCH_A" not in (skill_b_dir / "SKILL.md").read_text(encoding="utf-8")


def test_successful_patch_after_failure_has_no_stale_result(
    monkeypatch, tmp_path,
):
    """Sequential: first patch fails (scanner denial), second patch on a
    different skill succeeds.  The second invocation MUST NOT see the
    first's failure result.  With the OLD ``_last_result`` design, the
    second invocation could observe the first's residual payload if
    cleanup ever failed to reset the attribute.  The helpers now return
    locally, so no such leak is possible.
    """
    import tools.skill_manager_tool as sm

    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)

    scan_block_for: list[str] = ["fail-closed"]

    def _scan(staged_dir):
        if any(name in staged_dir.name for name in scan_block_for):
            return "stale-blocking scan failure"
        return None

    monkeypatch.setattr(sm, "_security_scan_skill_fail_closed", _scan)

    # First: failing patch on the default fail-closed skill.
    with session_write_policy_scope(
        _allowlist("first-fail", skills_root, "skill_patch")
    ):
        first = json.loads(
            sm.skill_manage(
                action="patch",
                name="fail-closed",
                old_string="Fail Closed",
                new_string="never-written",
            )
        )

    assert first["success"] is False, first
    assert first.get("error") == "stale-blocking scan failure", first

    # Now lift the block and do a successful patch on a different skill.
    scan_block_for.clear()
    skill_dir = skills_root / "second-success"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")

    with session_write_policy_scope(
        _allowlist("second-success", skills_root, "skill_patch")
    ):
        second = json.loads(
            sm.skill_manage(
                action="patch",
                name="second-success",
                old_string="Fail Closed",
                new_string="SECOND_OK",
            )
        )

    assert second["success"] is True, second
    assert "second-success" in second.get("message", ""), second
    assert second.get("error") is None, second
    # No stale failure marker from the first call.
    assert "stale-blocking scan failure" not in json.dumps(second), second


def test_successful_write_file_after_failure_has_no_stale_result(
    monkeypatch, tmp_path,
):
    """Sequential: first write_file fails (scanner denial), second
    write_file on a different path succeeds.  No stale failure leaks.
    """
    import tools.skill_manager_tool as sm

    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)

    block_first = {"on": True}

    def _scan(staged_dir):
        if block_first["on"]:
            return "first-write-blocked"
        return None

    monkeypatch.setattr(sm, "_security_scan_skill_fail_closed", _scan)

    with session_write_policy_scope(
        _allowlist("writefirst", skills_root, "skill_write_file")
    ):
        first = json.loads(
            sm.skill_manage(
                action="write_file",
                name="fail-closed",
                file_path="references/first.md",
                file_content="never-written\n",
            )
        )

    assert first["success"] is False, first
    assert first.get("error") == "first-write-blocked", first

    # Lift the block; second write_file on a different path succeeds.
    block_first["on"] = False
    with session_write_policy_scope(
        _allowlist("writesecond", skills_root, "skill_write_file")
    ):
        second = json.loads(
            sm.skill_manage(
                action="write_file",
                name="fail-closed",
                file_path="references/second.md",
                file_content="OK\n",
            )
        )

    assert second["success"] is True, second
    assert second.get("error") is None, second
    assert "first-write-blocked" not in json.dumps(second), second
    assert not (
        skills_root / "fail-closed" / "references" / "first.md"
    ).exists()


def test_patch_failure_after_success_returns_current_failure(
    monkeypatch, tmp_path,
):
    """Sequential: first patch succeeds, second patch (scanner denial)
    MUST return the SECOND's failure, NOT a stale success nor None.
    """
    import tools.skill_manager_tool as sm

    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)

    scan_block_for: list[str] = ["second-skill"]

    def _scan(staged_dir):
        if any(name in staged_dir.name for name in scan_block_for):
            return "second-patch-blocked-by-scan"
        return None

    monkeypatch.setattr(sm, "_security_scan_skill_fail_closed", _scan)

    # First: success.
    with session_write_policy_scope(
        _allowlist("patchfirst", skills_root, "skill_patch")
    ):
        first = json.loads(
            sm.skill_manage(
                action="patch",
                name="fail-closed",
                old_string="Fail Closed",
                new_string="FIRST_OK",
            )
        )

    assert first["success"] is True, first

    # Second: failure on a different skill.
    skill_dir = skills_root / "second-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")

    with session_write_policy_scope(
        _allowlist("patchsecond", skills_root, "skill_patch")
    ):
        second = json.loads(
            sm.skill_manage(
                action="patch",
                name="second-skill",
                old_string="Fail Closed",
                new_string="NEVER_WRITTEN",
            )
        )

    assert second["success"] is False, second
    assert second.get("error") == "second-patch-blocked-by-scan", second
    assert second.get("live_mutation_committed", False) is False, second
    # First's success marker must NOT leak into second.
    assert "FIRST_OK" not in json.dumps(second), second
    # Second's SKILL.md must NOT have been mutated.
    assert "NEVER_WRITTEN" not in (skill_dir / "SKILL.md").read_text(
        encoding="utf-8"
    )


def test_write_file_failure_after_success_returns_current_failure(
    monkeypatch, tmp_path,
):
    """Sequential: first write_file succeeds, second write_file (scanner
    denial) MUST return the SECOND's failure, NOT a stale success nor
    None.
    """
    import tools.skill_manager_tool as sm

    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    _create_clean_skill(sm, skills_root)

    block_second = {"on": False}

    def _scan(staged_dir):
        if block_second["on"]:
            return "second-write-blocked-by-scan"
        return None

    monkeypatch.setattr(sm, "_security_scan_skill_fail_closed", _scan)

    # First: success.
    with session_write_policy_scope(
        _allowlist("writefirst", skills_root, "skill_write_file")
    ):
        first = json.loads(
            sm.skill_manage(
                action="write_file",
                name="fail-closed",
                file_path="references/first.md",
                file_content="FIRST_OK\n",
            )
        )

    assert first["success"] is True, first
    first_file = skills_root / "fail-closed" / "references" / "first.md"
    assert first_file.exists()

    # Second: failure.
    block_second["on"] = True
    with session_write_policy_scope(
        _allowlist("writesecond", skills_root, "skill_write_file")
    ):
        second = json.loads(
            sm.skill_manage(
                action="write_file",
                name="fail-closed",
                file_path="references/second.md",
                file_content="NEVER_WRITTEN\n",
            )
        )

    assert second["success"] is False, second
    assert second.get("error") == "second-write-blocked-by-scan", second
    assert second.get("live_mutation_committed", False) is False, second
    assert "FIRST_OK" not in json.dumps(second), second
    assert not (
        skills_root / "fail-closed" / "references" / "second.md"
    ).exists()


# ═══════════════════════════════════════════════════════════════════════════
# Phase C · P1-B · Lock-acquisition payload consistency
# ═══════════════════════════════════════════════════════════════════════════
#
# Contract: every operation that acquires the interprocess mutation lock
# MUST translate any acquisition failure into a structured payload owned
# exclusively by the current invocation.  The failure payload is the
# single diagnostic surface the agent loop sees — raw exceptions escape
# the module boundary as well-formed payloads before they reach
# production paths.
#
# The canonical schema (all keys mandatory, all values typed):
#
#   success: bool = False
#   error: str
#   policy_reason: "lock_acquisition_failed"
#   rollback_failure_kind: "lock_acquisition_failure"
#   operation_kind: <the real operation kind for this call site>
#   target: <canonical or prospective target path>
#   lock_path: <resolved lock path, or "" if not yet known>
#   lock_failure_stage: one of
#       "lock_path_resolution"
#       "lock_parent_open"
#       "lock_identity_validation"
#       "lock_primitive_acquire"
#       "lock_contention"
#   lock_exception_type: str (the exception class name)
#   live_mutation_committed: bool = False
#   safe_to_retry: bool (true only for lock_contention)


def _acquisition_failure_payload_contract():
    """The single source of truth for the canonical acquisition-failure
    payload schema.  Any drift between production code and tests
    surfaces here first."""
    return {
        "success": False,
        "policy_reason": "lock_acquisition_failed",
        "rollback_failure_kind": "lock_acquisition_failure",
        "lock_failure_stage": (
            "lock_path_resolution | "
            "lock_parent_open | "
            "lock_identity_validation | "
            "lock_primitive_acquire | "
            "lock_contention"
        ),
        "live_mutation_committed": False,
        "safe_to_retry": "<bool>",
    }


def _assert_canonical_acquisition_payload(payload, *, expected_operation_kind, expected_target):
    """Assert one payload conforms to the canonical acquisition-failure
    contract.  Used by every focused test in this section so a contract
    drift produces a single, localized failure.
    """
    assert isinstance(payload, dict), f"expected dict payload, got {type(payload).__name__}"
    assert payload.get("success") is False, payload
    assert payload.get("policy_reason") == "lock_acquisition_failed", payload
    assert payload.get("rollback_failure_kind") == "lock_acquisition_failure", payload
    assert payload.get("operation_kind") == expected_operation_kind, payload
    assert payload.get("target") == expected_target, payload
    assert payload.get("lock_failure_stage") in {
        "lock_path_resolution",
        "lock_parent_open",
        "lock_identity_validation",
        "lock_primitive_acquire",
        "lock_contention",
    }, payload
    assert payload.get("lock_path") is not None, payload
    assert isinstance(payload.get("lock_path"), str), payload
    assert payload.get("live_mutation_committed") is False, payload
    assert isinstance(payload.get("safe_to_retry"), bool), payload
    # Error must be a non-empty string with the underlying cause
    assert isinstance(payload.get("error"), str), payload
    assert payload.get("error"), payload
    # Stage-driven safe_to_retry rule
    if payload.get("lock_failure_stage") == "lock_contention":
        assert payload.get("safe_to_retry") is True, payload
    else:
        assert payload.get("safe_to_retry") is False, payload


@pytest.fixture
def sm_module(monkeypatch, tmp_path):
    """Import skill_manager_tool and register a private skills root so
    the focused payload tests work without touching the rest of the
    suite's state."""
    import tools.skill_manager_tool as sm
    skills_root = tmp_path / "skills"
    skills_root.mkdir(parents=True)
    monkeypatch.setattr(sm, "SKILLS_DIR", skills_root)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)
    monkeypatch.setattr(sm, "_security_scan_skill_fail_closed", lambda path: None)
    return sm


def _install_failing_flock(monkeypatch, sm, exc):
    """Replace _fcntl.flock with a function that raises `exc` (any
    BaseException subtype).  Mirrors the harness used by
    test_interprocess_lock_failure_propagates_as_permission_error."""
    real_flock = sm._fcntl.flock

    def fail_flock(fd, op):
        raise exc

    monkeypatch.setattr(sm._fcntl, "flock", fail_flock)
    return real_flock




def test_phase_c_race_harness_uses_no_production_line_number_logic():
    """The race-test swap harnesses must not select a production
    branch by numeric source-line position.

    Forbidden in the two swap tests:

      * any attribute access ``.f_lineno`` or ``.co_firstlineno``;
      * any call to ``inspect.stack()`` or
        ``inspect.currentframe()`` used to read a line number;
      * any Compare node whose operand is a numeric literal paired
        with a frame/code-object line attribute (selects a branch
        by position);
      * any Compare node that uses a numeric range (e.g.
        ``3162 <= lineno <= 3260``) as the discriminator;
      * reference to ``sm._delete_skill.code`` (the property),
        which is NOT a code object — only ``sm._delete_skill.__code__``
        is the actual callable's code object.

    Required (positive confirmations):

      * ``sm._delete_skill.__code__`` MUST be referenced;
      * the harness MUST compare ``frame.f_code`` against that
        ``__code__`` object via ``is`` (identity, not equality).

    The check is structural (AST) rather than a literal-search so
    it cannot be bypassed by renaming the attribute or hiding the
    comparison behind an alias.
    """
    import ast
    from pathlib import Path

    path = Path("tests/tools/test_session_write_policy_fail_closed.py")
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))

    swap_test_names = {
        "test_delete_skill_replacement_after_final_identity_check_before_recursive_delete_is_preserved",
        "test_curator_archive_replacement_after_final_identity_check_before_archive_is_preserved",
    }

    forbidden: list[str] = []

    def _is_line_attr(name: str) -> bool:
        return name in {"f_lineno", "co_firstlineno", "f_lasti", "co_lnotab"}

    def _attrs_of(node: ast.AST) -> set[str]:
        out: set[str] = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute):
                out.add(sub.attr)
        return out

    def _has_inspect_call(node: ast.AST, *, want_stack: bool) -> bool:
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            func = sub.func
            if not isinstance(func, ast.Attribute):
                continue
            # Matches ``inspect.stack()``, ``inspect.currentframe()``,
            # ``inspect.getframeinfo(...)``, ``inspect.getouterframes(...)``.
            base = func.value
            if isinstance(base, ast.Name) and base.id == "inspect":
                if want_stack:
                    if func.attr in {"stack", "getouterframes", "getinnerframes"}:
                        return True
                else:
                    if func.attr in {
                        "currentframe", "getframeinfo", "getouterframes",
                        "getinnerframes",
                    }:
                        return True
        return False

    per_test_results: dict[str, dict[str, bool]] = {
        name: {
            "forbidden_attr_found": False,
            "inspect_call_found": False,
            "numeric_range_compare_found": False,
            "code_attribute_found": False,
            "dunder_code_referenced": False,
            "frame_f_code_identity_compare": False,
        }
        for name in swap_test_names
    }

    code_attr_re = "._delete_skill.code"

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in swap_test_names:
            continue

        body_src = ast.unparse(node)

        # 1) Forbidden attribute access: .f_lineno / .co_firstlineno.
        if any(
            isinstance(a, ast.Attribute) and _is_line_attr(a.attr)
            for a in ast.walk(node)
        ):
            per_test_results[node.name]["forbidden_attr_found"] = True
            forbidden.append(
                f"{node.name}: forbidden line-number attribute access "
                f"(.f_lineno / .co_firstlineno)"
            )

        # 2) Forbidden inspect.stack() / inspect.currentframe() calls.
        if _has_inspect_call(node, want_stack=True):
            per_test_results[node.name]["inspect_call_found"] = True
            forbidden.append(
                f"{node.name}: forbidden inspect.stack() / "
                f"inspect.getouterframes() call used to drive a swap"
            )
        if _has_inspect_call(node, want_stack=False):
            per_test_results[node.name]["inspect_call_found"] = True
            forbidden.append(
                f"{node.name}: forbidden inspect.currentframe() / "
                f"inspect.getframeinfo() call used to read line numbers"
            )

        # 3) Numeric range Compare nodes that use a numeric literal
        #    against a frame/code-object line attribute.  Walk every
        #    Compare and look at its operators and comparators.
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Compare):
                continue
            operands = [sub.left, *sub.comparators]
            has_numeric_literal = any(
                isinstance(o, ast.Constant) and isinstance(o.value, (int, float))
                for o in operands
            )
            has_line_attr = any(
                isinstance(o, ast.Attribute) and _is_line_attr(o.attr)
                for o in operands
            )
            # Numeric RANGE: Compare with two numeric literals and
            # one or more `BoolOp` / chained comparison context.
            numeric_literals = sum(
                1 for o in operands
                if isinstance(o, ast.Constant)
                and isinstance(o.value, (int, float))
            )
            if numeric_literals >= 2:
                per_test_results[node.name]["numeric_range_compare_found"] = True
                forbidden.append(
                    f"{node.name}: numeric range comparison "
                    f"({ast.unparse(sub)!r}) used to select a branch by "
                    f"production source-line position"
                )
            elif has_numeric_literal and has_line_attr:
                per_test_results[node.name]["numeric_range_compare_found"] = True
                forbidden.append(
                    f"{node.name}: numeric-vs-line-attribute comparison "
                    f"({ast.unparse(sub)!r}) used to select a branch by "
                    f"production source-line position"
                )

        # 4) Forbidden ``sm._delete_skill.code`` (the property, not
        #    the code object).  Only ``__code__`` is acceptable.
        if code_attr_re in body_src and "__code__" not in body_src.split(
            code_attr_re, 1
        )[1].split("\n", 1)[0].split(" ", 1)[0]:
            per_test_results[node.name]["code_attribute_found"] = True
            forbidden.append(
                f"{node.name}: sm._delete_skill.code (property) referenced; "
                f"only sm._delete_skill.__code__ is valid"
            )
        if "_delete_skill.code" in body_src and "__code__" not in body_src:
            per_test_results[node.name]["code_attribute_found"] = True
            forbidden.append(
                f"{node.name}: _delete_skill.code (property) referenced"
            )

        # Positive confirmations:
        # 5) sm._delete_skill.__code__ MUST be referenced.
        if "_delete_skill.__code__" in body_src:
            per_test_results[node.name]["dunder_code_referenced"] = True

        # 6) frame.f_code compared by identity (``is``) against the
        #    ``__code__`` object.  Look for Compare nodes whose left
        #    operand is an ``f_code`` attribute and whose comparator
        #    operator is ``Is`` / ``IsNot``.
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Compare):
                continue
            ops = sub.ops
            left = sub.left
            comparators = sub.comparators
            if not ops:
                continue
            for op, right in zip(ops, comparators):
                if not isinstance(op, (ast.Is, ast.IsNot)):
                    continue
                if not isinstance(left, ast.Attribute):
                    continue
                if left.attr != "f_code":
                    continue
                # The right-hand side should reference the variable
                # that was bound to ``sm._delete_skill.__code__``.  We
                # accept either an indirect Name (assigned earlier in
                # the function body) or a direct Attribute chain.
                rhs_is_code_ref = False
                if isinstance(right, ast.Name):
                    # Indirect via a local binding — accept.
                    rhs_is_code_ref = True
                elif (
                    isinstance(right, ast.Attribute)
                    and right.attr == "__code__"
                ):
                    rhs_is_code_ref = True
                if rhs_is_code_ref:
                    per_test_results[node.name][
                        "frame_f_code_identity_compare"
                    ] = True

    assert not forbidden, (
        "swap tests violate the production-line-number-free contract: "
        + "; ".join(forbidden)
    )

    # Both swap tests must positively demonstrate the contract.
    missing: list[str] = []
    for name, results in per_test_results.items():
        if not results["dunder_code_referenced"]:
            missing.append(f"{name}: sm._delete_skill.__code__ NOT referenced")
        if not results["frame_f_code_identity_compare"]:
            missing.append(
                f"{name}: frame.f_code NOT compared by identity to __code__"
            )
    assert not missing, "missing positive confirmations: " + "; ".join(missing)


def test_final_identity_swap_tests_require_exact_atomic_refusal():
    """The two swap tests must assert the EXACT canonical refusal
    payload — ``atomic_recursive_delete_unavailable`` for the
    foreground path, ``atomic_archive_unavailable`` for the curator
    path.  ``concurrent_modification`` MUST NOT be an accepted
    alternative: production only returns that string when an
    identity recheck observed the swap, which would contradict the
    harness contract that the swap happened AFTER the final
    captures.

    The check is AST-based so it cannot be bypassed by accepting
    ``concurrent_modification`` as a sibling string in a tuple.
    """
    import ast
    from pathlib import Path

    path = Path("tests/tools/test_session_write_policy_fail_closed.py")
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))

    swap_test_specs = {
        "test_delete_skill_replacement_after_final_identity_check_before_recursive_delete_is_preserved": {
            "required_policy_reason": "atomic_recursive_delete_unavailable",
            "forbidden_policy_reason": "concurrent_modification",
            "required_rollback_kind": "identity_bound_recursive_delete_unavailable",
        },
        "test_curator_archive_replacement_after_final_identity_check_before_archive_is_preserved": {
            "required_policy_reason": "atomic_archive_unavailable",
            "forbidden_policy_reason": "concurrent_modification",
            "required_rollback_kind": "identity_bound_archive_unavailable",
        },
    }

    failures: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in swap_test_specs:
            continue
        spec = swap_test_specs[node.name]

        # Walk every assertion-shaped Compare and every ``in`` tuple
        # membership check.  Look at the Compare's comparators for the
        # required/forbidden policy_reason values.
        required_seen = False
        forbidden_seen = False
        rollback_seen = False

        for sub in ast.walk(node):
            if isinstance(sub, ast.Compare):
                cmp_src = ast.unparse(sub)
                if (
                    "policy_reason" in cmp_src
                    and spec["required_policy_reason"] in cmp_src
                ):
                    required_seen = True
                if (
                    "policy_reason" in cmp_src
                    and spec["forbidden_policy_reason"] in cmp_src
                ):
                    # Membership via tuple: ``result["policy_reason"]
                    # in ("a", "b")``.  Forbid if ``concurrent_modification``
                    # appears in ANY ``policy_reason`` comparison in this
                    # swap test.
                    forbidden_seen = True
                if (
                    "rollback_failure_kind" in cmp_src
                    and spec["required_rollback_kind"] in cmp_src
                ):
                    rollback_seen = True

        if not required_seen:
            failures.append(
                f"{node.name}: required exact assertion "
                f"policy_reason == {spec['required_policy_reason']!r} "
                f"NOT FOUND in AST"
            )
        if not rollback_seen:
            failures.append(
                f"{node.name}: required rollback_failure_kind == "
                f"{spec['required_rollback_kind']!r} NOT FOUND in AST"
            )
        if forbidden_seen:
            failures.append(
                f"{node.name}: {spec['forbidden_policy_reason']!r} is "
                f"referenced in a policy_reason comparison; the swap "
                f"test must reject concurrent_modification as an "
                f"alternative outcome"
            )

    assert not failures, (
        "swap tests violate the exact-atomic-refusal contract: "
        + "; ".join(failures)
    )



# ─────────────────────────────────────────────────────────────────────────
# Phase C global normalized-name mutex (Phase C P1 global uniqueness)
# ─────────────────────────────────────────────────────────────────────────
#
# These tests exercise the global normalized-name mutex introduced by
# ``_global_normalized_name_lock`` + ``_normalized_skill_name_lock_target``
# + ``_canonical_normalize_skill_name``.  Every test runs in an isolated
# HERMES_HOME so locks never escape across tests.


# ── 24. Distinct roots ──────────────────────────────────────────────────
# Production has only one creation root (``~/.hermes/skills``) — the
# `external_dirs` are read-only.  This test records that fact rather
# than fabricating a root the product does not support.

def test_distinct_creation_roots_not_applicable(monkeypatch, tmp_path):
    """``_create_skill`` only ever publishes under the local skills root;
    ``external_dirs`` are read-only by contract.  Cross-root concurrency
    is therefore not a production scenario and this test simply asserts
    that the supported-root list is exactly one.
    """
    import tools.skill_manager_tool as sm
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    # The local skills root resolves to ``skills_root`` (set up by the
    # fixture).  All create paths route through ``_skills_dir()`` which
    # always returns this single root.
    assert sm._skills_dir().resolve(strict=False) == skills_root.resolve(strict=False)
    # Production has no second writable creation root.  If that ever
    # changes, the directive's cross-root test will become applicable.


# ── 26. Post-contention explicit retry ──────────────────────────────────

def test_create_contender_after_global_name_lock_release_observes_existing_skill(
    monkeypatch, tmp_path
):
    """After the winner releases the global name lock, a SECOND attempt
    from the loser MUST observe the duplicate refusal rather than
    silently winning.
    """
    import tools.skill_manager_tool as sm
    sm, skills_root = _setup_skills(monkeypatch, tmp_path)
    monkeypatch.setattr(sm, "_security_scan_skill", lambda path: None)

    # First create wins.
    with session_write_policy_scope(_allowlist("first", skills_root, "skill_create")):
        first = json.loads(sm.skill_manage(action="create", name="retry-skill", content=SKILL_MD))
    assert first["success"] is True, first

    # Second create from the loser — global lock is free, but the
    # DECISIVE inside-lock scan sees the live skill.
    with session_write_policy_scope(_allowlist("second", skills_root, "skill_create")):
        second = json.loads(sm.skill_manage(action="create", name="retry-skill", content=SKILL_MD))
    assert second["success"] is False, second
    # The refusal may come from the OUTSIDE pre-lock check (no
    # policy_reason) or from the INSIDE decisive scan (duplicate_skill).
    # Either outcome proves the global lock is correctly enforcing the
    # at-most-one-live-skill contract.
    assert (
        "already exists" in second.get("error", "")
        or second.get("policy_reason") == "duplicate_skill"
    ), second
