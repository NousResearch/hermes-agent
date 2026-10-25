"""Hardening regression tests: fail-closed containment for delegated children.

Two goals, per the hardening instruction:

GOAL A — when a delegated child has an active isolated worktree, a mutating
operation resolving outside that worktree must fail closed BEFORE mutation:
  * terminal with an explicit outside ``workdir`` (parent checkout or a
    sibling worktree) is refused before the command runs
  * write_file / patch (and V4A multi-file headers) resolving outside the
    worktree are refused before the write
  * a child's in-shell ``cd`` escape is not PERSISTED as its session cwd
  * valid in-worktree operations still succeed
  * non-delegated / unregistered callers keep historical behavior

GOAL B — a repo-isolated delegation whose worktree could NOT be established
is never silently downgraded: the dispatch records the failure, the child's
result entry carries it, and repo-mutating tool calls are denied.

The tests cross the REAL tool seams (``terminal_tool`` / ``write_file`` /
``patch`` / ``seed_workspace`` / ``_create_isolated_worktree``) against real
git worktrees — no mocked-away mechanism. Negative controls (the
``_control_`` functions) assert the SAME tests fail against genuine
pre-hardening behavior: with the containment registry bypassed (worktree
approved but task_id never registered — exactly what the pre-hardening code
did), the parent write WOULD have succeeded.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import tools.terminal_tool as terminal_tool  # noqa: E402
import tools.file_tools as file_tools  # noqa: E402
from tools import child_containment  # noqa: E402
from tools.delegate_tool_child_run import _ChildRun  # noqa: E402


def _git(args, cwd, check=True):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=check)


def _make_repo(root: Path, name: str = "repo") -> Path:
    repo = root / name
    repo.mkdir(parents=True)
    _git(["init", "-q"], repo)
    _git(["config", "user.email", "test@test"], repo)
    _git(["config", "user.name", "Test"], repo)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "seed"], repo)
    return repo


class _FakeChild:
    provider = "openrouter"
    model = "some/model"


class _FakeParent:
    _current_task_id = None
    cwd = ""


@pytest.fixture(autouse=True)
def _clean_environment():
    """Isolate every test: empty registries, empty cwd store, and no ambient
    ``HERMES_SESSION_KEY`` (a hosting gateway's subprocess would otherwise
    hijack session-key resolution for workdir-less commands)."""
    child_containment._reset_for_tests()
    _clear_cwd_store()
    saved_key = os.environ.pop("HERMES_SESSION_KEY", None)
    try:
        yield
    finally:
        if saved_key is not None:
            os.environ["HERMES_SESSION_KEY"] = saved_key
        child_containment._reset_for_tests()
        _clear_cwd_store()


def _clear_cwd_store():
    with terminal_tool._session_cwd_lock:
        terminal_tool._session_cwd.clear()
    with terminal_tool._container_alias_lock:
        terminal_tool._container_aliases.clear()


class _RecordingEnv:
    """Env double: refuses to run when cwd is outside the approved worktree is
    enforced BEFORE env.execute; records the cwd each command was executed in."""

    env = {}

    def __init__(self):
        self.executed_cwds = []

    def execute(self, command, **kwargs):
        self.executed_cwds.append(kwargs.get("cwd"))
        return {"output": "ok", "returncode": 0,
                "cwd_observed": True, "cwd": kwargs.get("cwd")}


def _drive_terminal(task_id: str, command: str = "pwd", workdir: str = None):
    """Call the REAL terminal_tool for *task_id*; return (result, env)."""
    env = _RecordingEnv()
    from unittest import mock

    with mock.patch.object(terminal_tool, "_get_env_config") as cfg, \
            mock.patch.object(terminal_tool, "_check_all_guards",
                              return_value={"approved": True}), \
            mock.patch.object(terminal_tool, "_run_approval_guards",
                              return_value=mock.Mock(note="", approved_run=True)), \
            mock.patch.object(terminal_tool, "_acquire_env",
                              side_effect=lambda plan, tid: env):
        cfg.return_value = {"env_type": "local", "cwd": os.getcwd(), "timeout": 60, "lifetime_seconds": 3600}
        result = terminal_tool.terminal_tool(command=command, task_id=task_id, workdir=workdir)
    return result, env


# ═══════════════════════════ GOAL A ═════════════════════════════════════

def test_isolated_child_terminal_workdir_parent_denied(tmp_path):
    """Explicit terminal workdir pointing at the parent checkout must be
    refused BEFORE the command runs (nothing executed)."""
    repo = _make_repo(tmp_path)
    from unittest import mock
    from tools import subagent_worktree as sw

    info = sw.create_subagent_worktree(str(repo), subagent_id="sa-0-hardened")
    assert info is not None
    child_containment.register_child_worktree("sa-0-hardened", info)

    result, env = _drive_terminal("sa-0-hardened", workdir=str(repo))
    payload = json.loads(result)
    assert payload.get("status") == "blocked", payload
    assert "delegated-child containment" in payload.get("error", ""), payload
    assert env.executed_cwds == [], "the command must NOT have executed"


def test_isolated_child_terminal_workdir_sibling_denied(tmp_path):
    """Explicit terminal workdir pointing at a SIBLING worktree is refused."""
    repo = _make_repo(tmp_path)
    from tools import subagent_worktree as sw

    own = sw.create_subagent_worktree(str(repo), subagent_id="sa-0-own")
    sibling = sw.create_subagent_worktree(str(repo), subagent_id="sa-1-sib")
    child_containment.register_child_worktree("sa-0-own", own)

    result, env = _drive_terminal("sa-0-own", workdir=sibling["path"])
    payload = json.loads(result)
    assert payload.get("status") == "blocked", payload
    assert env.executed_cwds == []


def test_isolated_child_terminal_workdir_inside_own_worktree_succeeds(tmp_path):
    """Positive: an explicit workdir INSIDE the child's own worktree runs."""
    repo = _make_repo(tmp_path)
    from tools import subagent_worktree as sw

    own = sw.create_subagent_worktree(str(repo), subagent_id="sa-0-ok")
    child_containment.register_child_worktree("sa-0-ok", own)
    _clear_cwd_store()
    terminal_tool.record_session_cwd("sa-0-ok", own["path"])

    result, env = _drive_terminal("sa-0-ok", workdir=own["path"])
    payload = json.loads(result)
    assert payload.get("exit_code") == 0, payload
    assert env.executed_cwds == [own["path"]]


def test_isolated_child_terminal_command_without_workdir_runs_in_worktree(tmp_path):
    """Positive: a workdir-less command resolves to the recorded worktree cwd."""
    repo = _make_repo(tmp_path)
    from tools import subagent_worktree as sw

    own = sw.create_subagent_worktree(str(repo), subagent_id="sa-0-plain")
    child_containment.register_child_worktree("sa-0-plain", own)
    _clear_cwd_store()
    terminal_tool.record_session_cwd("sa-0-plain", own["path"])

    result, env = _drive_terminal("sa-0-plain")
    payload = json.loads(result)
    assert payload.get("exit_code") == 0, payload
    assert env.executed_cwds == [own["path"]], "child must run in its own worktree"


def test_isolated_child_cwd_escape_not_persisted(tmp_path):
    """A child command whose observed cwd escaped the worktree (in-shell cd)
    must NOT have the escape persisted as its session cwd."""
    repo = _make_repo(tmp_path)
    from tools import subagent_worktree as sw

    own = sw.create_subagent_worktree(str(repo), subagent_id="sa-0-esc")
    child_containment.register_child_worktree("sa-0-esc", own)
    _clear_cwd_store()
    terminal_tool.record_session_cwd("sa-0-esc", own["path"])

    from unittest import mock
    env = _RecordingEnv()
    # The env reports it ended up in the PARENT checkout (a cd escape).
    env.execute = lambda command, **kwargs: (
        env.executed_cwds.append(kwargs.get("cwd")),
        {"output": "ok", "returncode": 0, "cwd_observed": True, "cwd": str(repo)},
    )[1]
    with mock.patch.object(terminal_tool, "_get_env_config") as cfg, \
            mock.patch.object(terminal_tool, "_check_all_guards",
                              return_value={"approved": True}), \
            mock.patch.object(terminal_tool, "_run_approval_guards",
                              return_value=mock.Mock(note="", approved_run=True)), \
            mock.patch.object(terminal_tool, "_acquire_env",
                              side_effect=lambda plan, tid: env):
        cfg.return_value = {"env_type": "local", "cwd": os.getcwd(), "timeout": 60, "lifetime_seconds": 3600}
        result = terminal_tool.terminal_tool(command=f"cd {repo} && pwd", task_id="sa-0-esc")
    payload = json.loads(result)
    assert payload.get("exit_code") == 0, payload
    # The escape was not persisted: the child's record still holds its worktree.
    assert terminal_tool.get_session_cwd("sa-0-esc") == own["path"], (
        "the in-shell cd escape was persisted as the child's session cwd"
    )
    assert "containment_note" in payload, "the result must explain the non-persistence"


def test_isolated_child_write_file_parent_denied(tmp_path):
    """write_file resolving to the parent checkout is refused before mutation."""
    repo = _make_repo(tmp_path)
    from tools import subagent_worktree as sw

    own = sw.create_subagent_worktree(str(repo), subagent_id="sa-0-wf")
    child_containment.register_child_worktree("sa-0-wf", own)
    _clear_cwd_store()
    terminal_tool.record_session_cwd("sa-0-wf", own["path"])

    target = repo / "evil.txt"
    result = file_tools.write_file_tool(str(target), "MUST NOT LAND", task_id="sa-0-wf")
    payload = json.loads(result)
    assert payload.get("error") and "delegated-child containment" in payload["error"], payload
    assert not target.exists(), "the parent checkout was mutated — containment failed"
    # In-worktree write still succeeds:
    ok_target = Path(own["path"]) / "good.txt"
    result = file_tools.write_file_tool(str(ok_target), "written by child", task_id="sa-0-wf")
    payload = json.loads(result)
    assert not payload.get("error"), payload
    assert ok_target.exists()


def test_isolated_child_write_file_sibling_denied(tmp_path):
    """write_file resolving to a sibling worktree is refused before mutation."""
    repo = _make_repo(tmp_path)
    from tools import subagent_worktree as sw

    own = sw.create_subagent_worktree(str(repo), subagent_id="sa-0-ws")
    sibling = sw.create_subagent_worktree(str(repo), subagent_id="sa-1-victim")
    child_containment.register_child_worktree("sa-0-ws", own)
    _clear_cwd_store()
    terminal_tool.record_session_cwd("sa-0-ws", own["path"])

    target = Path(sibling["path"]) / "evil.txt"
    result = file_tools.write_file_tool(str(target), "MUST NOT LAND", task_id="sa-0-ws")
    payload = json.loads(result)
    assert payload.get("error") and "delegated-child containment" in payload["error"], payload
    assert not target.exists(), "the sibling worktree was mutated — containment failed"


def test_isolated_child_patch_parent_denied(tmp_path):
    """patch (replace mode) targeting the parent checkout is refused; an
    in-worktree patch still succeeds."""
    repo = _make_repo(tmp_path)
    from tools import subagent_worktree as sw

    own = sw.create_subagent_worktree(str(repo), subagent_id="sa-0-pt")
    child_containment.register_child_worktree("sa-0-pt", own)
    _clear_cwd_store()
    terminal_tool.record_session_cwd("sa-0-pt", own["path"])

    parent_readme = repo / "README.md"
    result = file_tools.patch_tool(
        mode="replace", path=str(parent_readme), old_string="hello",
        new_string="HACKED", task_id="sa-0-pt")
    payload = json.loads(result)
    assert payload.get("error") and "delegated-child containment" in payload["error"], payload
    assert parent_readme.read_text(encoding="utf-8") == "hello\n", (
        "the parent checkout was mutated — containment failed")

    own_readme = Path(own["path"]) / "README.md"
    result = file_tools.patch_tool(
        mode="replace", path=str(own_readme), old_string="hello",
        new_string="child edit", task_id="sa-0-pt")
    payload = json.loads(result)
    assert not payload.get("error"), payload
    assert "child edit" in own_readme.read_text(encoding="utf-8")


def test_isolated_child_patch_v4a_header_outside_denied(tmp_path):
    """A V4A multi-file patch whose header names a parent-checkout file is
    refused in full (all-or-nothing) before mutation."""
    repo = _make_repo(tmp_path)
    from tools import subagent_worktree as sw

    own = sw.create_subagent_worktree(str(repo), subagent_id="sa-0-v4a")
    child_containment.register_child_worktree("sa-0-v4a", own)
    _clear_cwd_store()
    terminal_tool.record_session_cwd("sa-0-v4a", own["path"])

    parent_readme = repo / "README.md"
    patch = (
        "*** Begin Patch\n"
        f"*** Update File: {parent_readme}\n"
        "@@\n"
        "-hello\n"
        "+HACKED\n"
        "*** End Patch\n"
    )
    result = file_tools.patch_tool(mode="patch", patch=patch, task_id="sa-0-v4a")
    payload = json.loads(result)
    assert payload.get("error") and "delegated-child containment" in payload["error"], payload
    assert parent_readme.read_text(encoding="utf-8") == "hello\n"


# ── Scope guards: non-children keep historical behavior ──────────────────

def test_unregistered_task_terminal_workdir_unchanged(tmp_path):
    """A NON-isolated caller (registry miss) keeps historical workdir behavior."""
    repo = _make_repo(tmp_path)
    result, env = _drive_terminal("unrelated-task", workdir=str(repo))
    payload = json.loads(result)
    assert payload.get("exit_code") == 0, payload
    assert env.executed_cwds == [str(repo)], "non-child workdir must run as before"


def test_unregistered_task_write_file_unchanged(tmp_path):
    """A NON-isolated caller writes wherever it always could."""
    repo = _make_repo(tmp_path)
    target = repo / "normal.txt"
    result = file_tools.write_file_tool(str(target), "parent's own write", task_id="unrelated-task")
    payload = json.loads(result)
    assert not payload.get("error"), payload
    assert target.exists()


# ═══════════════════════════ GOAL B ═════════════════════════════════════

def test_non_repo_parent_cwd_isolation_failure_recorded_not_silent(tmp_path):
    """Repo-isolated delegation from a non-repo parent cwd: the failure is
    recorded explicitly (registry downgrade), the result entry carries it, and
    repo-mutating tool calls are denied — never a silent downgrade."""
    from unittest import mock

    non_repo = tmp_path / "plain-dir"
    non_repo.mkdir()
    _clear_cwd_store()

    run = _ChildRun(_FakeChild(), _FakeParent(), 0, "Do repo work", "sa-0-downgrade", None)

    # _create_isolated_worktree imports both helpers from tools.delegate_tool
    # at CALL time, so the mocks must target that namespace to take effect.
    with mock.patch("tools.delegate_tool._get_worktree_isolation",
                    return_value=True), \
            mock.patch("tools.delegate_tool._resolve_workspace_hint",
                       return_value=str(non_repo)):
        run.seed_workspace()

    # The dispatch recorded the downgrade, keyed by the child's real task id.
    assert run.worktree_info is None
    assert run.isolation_downgrade_reason and "not inside a git repository" in run.isolation_downgrade_reason
    assert child_containment.is_isolation_downgraded("sa-0-downgrade")

    # The result entry surfaces the downgrade explicitly.
    entry = run.attach_worktree({})
    assert "worktree_isolation_downgraded" in entry, entry
    assert entry["worktree_isolation_downgraded"]["reason"] == run.isolation_downgrade_reason

    # Repo-mutating tool calls are denied for the downgraded child.
    target = non_repo / "x.txt"
    result = file_tools.write_file_tool(str(target), "MUST NOT LAND", task_id="sa-0-downgrade")
    payload = json.loads(result)
    assert payload.get("error") and "delegated-child containment" in payload["error"], payload
    assert not target.exists()

    # Terminal workdir also denied for the downgraded child.
    result, env = _drive_terminal("sa-0-downgrade", workdir=str(non_repo))
    payload = json.loads(result)
    assert payload.get("status") == "blocked", payload
    assert env.executed_cwds == []

    # Cleanup drops the downgrade record (idempotent teardown).
    from tools.delegate_tool_registry import _unregister_subagent  # noqa: F401
    run.cleanup(heartbeat=mock.Mock(stop=mock.Mock()), child_pool=None,
                leased_cred_id=None, close_deferred=False)
    assert not child_containment.is_isolation_downgraded("sa-0-downgrade")


def test_successful_isolation_registers_containment(tmp_path):
    """Positive Goal A wiring: a successful seed_workspace registers the
    child's worktree in the containment registry under its real task id."""
    from unittest import mock

    repo = _make_repo(tmp_path)
    _clear_cwd_store()

    run = _ChildRun(_FakeChild(), _FakeParent(), 0, "Do the thing", "sa-0-good", None)
    with mock.patch("tools.delegate_tool._get_worktree_isolation",
                    return_value=True), \
            mock.patch("tools.delegate_tool._resolve_workspace_hint",
                       return_value=str(repo)):
        run.seed_workspace()

    assert run.worktree_info is not None
    assert child_containment.approved_worktree_for("sa-0-good") == os.path.realpath(run.worktree_info["path"])
    assert run.isolation_downgrade_reason is None

    run.cleanup(heartbeat=mock.Mock(stop=mock.Mock()), child_pool=None,
                leased_cred_id=None, close_deferred=False)
    assert child_containment.approved_worktree_for("sa-0-good") is None


def test_isolation_not_requested_no_containment_no_downgrade(tmp_path):
    """isolation=False: no registry entry, no downgrade — historical behavior."""
    from unittest import mock

    _clear_cwd_store()
    run = _ChildRun(_FakeChild(), _FakeParent(), 0, "plain task", "sa-0-plain", None)
    with mock.patch("tools.delegate_tool._get_worktree_isolation",
                    return_value=False):
        run.seed_workspace()

    assert run.worktree_info is None
    assert run.isolation_downgrade_reason is None
    assert child_containment.approved_worktree_for("sa-0-plain") is None
    assert not child_containment.is_isolation_downgraded("sa-0-plain")


def test_setup_exception_is_recorded_not_silent(tmp_path):
    """Goal B fail-closed on the exception path: an unexpected error inside
    isolation setup must produce an explicit downgrade record — never the
    pre-hardening behavior where ``_quiet`` swallowed the exception and fell
    through to a silent ``return None`` (no worktree, NO downgrade record,
    NO mutation guard for the child)."""
    from unittest import mock

    _clear_cwd_store()
    run = _ChildRun(_FakeChild(), _FakeParent(), 0, "repo task", "sa-0-raise", None)

    def _boom(parent_agent):
        raise RuntimeError("simulated setup crash (registry corruption)")

    with mock.patch("tools.delegate_tool._get_worktree_isolation",
                    return_value=True), \
            mock.patch("tools.delegate_tool._resolve_workspace_hint",
                       side_effect=_boom):
        run.seed_workspace()

    # The dispatch recorded the downgrade, keyed by the child's real task id.
    assert run.worktree_info is None
    assert run.isolation_downgrade_reason and "setup raised RuntimeError" in run.isolation_downgrade_reason
    assert child_containment.is_isolation_downgraded("sa-0-raise"), (
        "an isolation-setup exception must NOT silently downgrade — the child "
        "must be registered as downgraded so mutation guards apply")

    # Downgraded children are denied repo file writes + explicit workdirs.
    target = tmp_path / "evil.txt"
    result = file_tools.write_file_tool(str(target), "MUST NOT LAND", task_id="sa-0-raise")
    payload = json.loads(result)
    assert payload.get("error") and "delegated-child containment" in payload["error"], payload
    assert not target.exists()

    result, env = _drive_terminal("sa-0-raise", workdir=str(tmp_path))
    payload = json.loads(result)
    assert payload.get("status") == "blocked", payload
    assert env.executed_cwds == []

    # Teardown drops the downgrade record (idempotent).
    run.cleanup(heartbeat=mock.Mock(stop=mock.Mock()), child_pool=None,
                leased_cred_id=None, close_deferred=False)
    assert not child_containment.is_isolation_downgraded("sa-0-raise")


# ═════════════════════ NEGATIVE CONTROLS ═════════════════════════════════
# The same scenarios run against genuine PRE-hardening behavior: the worktree
# exists (isolation engaged) but the task_id was never registered with the
# containment registry — exactly the state before this hardening. Each control
# must show the mutation WOULD have gone through, proving the tests above are
# discriminating, not vacuous.

def test_control_pre_hardening_terminal_workdir_parent_would_run(tmp_path):
    repo = _make_repo(tmp_path)
    from tools import subagent_worktree as sw

    info = sw.create_subagent_worktree(str(repo), subagent_id="sa-0-ctl")
    assert info is not None  # isolation engaged
    # Pre-hardening state: no containment registration. Worktree EXISTS, but
    # the guards have no registry entry for the task id.
    result, env = _drive_terminal("sa-0-ctl", workdir=str(repo))
    payload = json.loads(result)
    assert payload.get("exit_code") == 0, payload
    assert env.executed_cwds == [str(repo)], (
        "pre-hardening: the parent-checkout workdir WOULD have run — "
        "this control must demonstrate the escape the hardening closes"
    )


def test_control_pre_hardening_write_file_parent_would_land(tmp_path):
    repo = _make_repo(tmp_path)
    from tools import subagent_worktree as sw

    info = sw.create_subagent_worktree(str(repo), subagent_id="sa-1-ctl")
    assert info is not None
    _clear_cwd_store()
    terminal_tool.record_session_cwd("sa-1-ctl", info["path"])

    target = repo / "would-have-landed.txt"
    result = file_tools.write_file_tool(str(target), "pre-hardening write", task_id="sa-1-ctl")
    payload = json.loads(result)
    assert not payload.get("error"), payload
    assert target.exists(), (
        "pre-hardening: the parent-checkout write WOULD have landed — "
        "this control must demonstrate the escape the hardening closes"
    )
    # Prove the write bypassed a live worktree the child believed was isolated.
    assert child_containment.approved_worktree_for("sa-1-ctl") is None


def test_control_pre_hardening_cwd_escape_would_persist(tmp_path):
    """Pre-hardening: a child command that escaped its worktree had the
    escaped directory PERSISTED as its session cwd, so every later command
    and file-tool anchor followed it. This control must show that persistence."""
    repo = _make_repo(tmp_path)
    from tools import subagent_worktree as sw

    own = sw.create_subagent_worktree(str(repo), subagent_id="sa-2-ctl")
    assert own is not None
    # Pre-hardening state: no containment registration for this task id.
    terminal_tool.record_session_cwd("sa-2-ctl", own["path"])

    from unittest import mock
    env = _RecordingEnv()
    env.execute = lambda command, **kwargs: (
        env.executed_cwds.append(kwargs.get("cwd")),
        {"output": "ok", "returncode": 0, "cwd_observed": True, "cwd": str(repo)},
    )[1]
    with mock.patch.object(terminal_tool, "_get_env_config") as cfg, \
            mock.patch.object(terminal_tool, "_check_all_guards",
                              return_value={"approved": True}), \
            mock.patch.object(terminal_tool, "_run_approval_guards",
                              return_value=mock.Mock(note="", approved_run=True)), \
            mock.patch.object(terminal_tool, "_acquire_env",
                              side_effect=lambda plan, tid: env):
        cfg.return_value = {"env_type": "local", "cwd": os.getcwd(), "timeout": 60, "lifetime_seconds": 3600}
        result = terminal_tool.terminal_tool(command=f"cd {repo} && pwd", task_id="sa-2-ctl")
    payload = json.loads(result)
    assert payload.get("exit_code") == 0, payload
    assert terminal_tool.get_session_cwd("sa-2-ctl") == str(repo), (
        "pre-hardening: the cd escape WAS persisted as the child's session cwd — "
        "this control must demonstrate the persistence the hardening stops"
    )
