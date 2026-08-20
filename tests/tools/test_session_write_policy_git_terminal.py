from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent.session_write_policy as swp
from agent.session_write_policy import (
    CallerType,
    CapabilityGrant,
    PolicyDenied,
    SessionWritePolicy,
    SessionWritePolicyDecision,
    SessionWritePolicyDecisionResult,
    SessionWritePolicyMode,
    session_write_policy_scope,
)
import tools.terminal_tool as tt


def _allowlist(session_id: str, root: Path, *operations: str) -> SessionWritePolicy:
    return SessionWritePolicy(
        session_id=session_id,
        mode=SessionWritePolicyMode.ALLOWLIST,
        allowed_roots=(str(root),),
        capability_grants=tuple(
            CapabilityGrant("filesystem", op, (str(root),)) for op in operations
        ),
        protected=True,
    )


class FakeEnv:
    env = {"SECRET_TOKEN": "must_not_be_forwarded_to_policy"}

    def __init__(self, cwd: str, calls: list[tuple[str, dict]]):
        self.cwd = cwd
        self.calls = calls

    def execute(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return {"output": f"ran:{command}", "returncode": 0}


@pytest.fixture(autouse=True)
def isolated_terminal(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "home").mkdir()
    (tmp_path / "hermes").mkdir()
    monkeypatch.delenv("HERMES_READ_ONLY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_DISABLE_SELF_IMPROVEMENT", raising=False)
    monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(tt, "_check_all_guards", lambda *a, **k: {"approved": True})
    monkeypatch.setattr(tt, "_resolve_container_task_id", lambda task_id: task_id or "default")
    monkeypatch.setattr(tt, "resolve_task_overrides", lambda task_id: {})
    monkeypatch.setattr(tt, "_last_activity", {})
    monkeypatch.setattr(tt, "_creation_locks", {})
    monkeypatch.setattr(tt, "_get_env_config", lambda: {
        "env_type": "local",
        "cwd": str(tmp_path),
        "timeout": 5,
        "lifetime_seconds": 300,
        "local_persistent": False,
    })

    def no_subprocess(*_args, **_kwargs):
        raise AssertionError("test must not spawn a real subprocess")

    monkeypatch.setattr(subprocess, "Popen", no_subprocess)
    monkeypatch.setattr(tt.subprocess, "Popen", no_subprocess)

    # Snapshot of tt._session_cwd BEFORE the test runs so leftover entries from a
    # previous test (e.g. the stale tmp_path cwd recorded by an earlier fixture)
    # cannot leak into the next test in this file. We deliberately snapshot the
    # live dict object rather than monkeypatching the binding, because session-cwd
    # state is module-level mutable state that survives across fixtures.
    session_cwd_snapshot = dict(tt._session_cwd)
    preexisting_keys = set(session_cwd_snapshot)

    # Clear only the pre-existing keys (the snapshot above already captured them,
    # so this does not lose state). This prevents stale cwds (e.g. a leaked
    # '/tmp' or another fixture's tmp_path) from contaminating the test body
    # before the test even runs. New keys introduced by the test are cleaned up
    # in the finally block below.
    for preexisting_key in preexisting_keys:
        tt.clear_session_cwd(preexisting_key)

    try:
        yield
    finally:
        # New keys introduced by the test must be removed; pre-existing keys must
        # be restored to their original cwd value. We use the module's public
        # helpers (clear_session_cwd / record_session_cwd) rather than mutating
        # tt._session_cwd directly, to stay in lock-step with the rest of the
        # module and to respect _session_cwd_lock.
        current_keys = set(tt._session_cwd)
        for new_key in current_keys - preexisting_keys:
            tt.clear_session_cwd(new_key)
        for key, value in session_cwd_snapshot.items():
            tt.record_session_cwd(key, value)


def _install_fake_env(monkeypatch, tmp_path, *, task_id="t", cwd: str | None = None):
    calls: list[tuple[str, dict]] = []
    env = FakeEnv(cwd or str(tmp_path), calls)
    monkeypatch.setattr(tt, "_active_environments", {task_id: env})
    return calls, env


def _allow_pre_spawn(monkeypatch, calls: list[dict] | None = None):
    def fake_pre_spawn(caller_type, **kwargs):
        if calls is not None:
            calls.append({"caller_type": caller_type, **kwargs})
        return SessionWritePolicyDecision(
            result=SessionWritePolicyDecisionResult.ALLOW,
            reason="policy_allow",
            operation_kind=kwargs["operation_kind"],
            origin=str(caller_type.value if isinstance(caller_type, CallerType) else caller_type),
            session_id="normal",
        )

    monkeypatch.setattr(swp, "pre_spawn_consult", fake_pre_spawn)


def _run(command: str, *, task_id="t", session_id="normal", workdir=None):
    with session_write_policy_scope(SessionWritePolicy.normal(session_id)):
        return json.loads(
            tt.terminal_tool(
                command,
                task_id=task_id,
                session_id=session_id,
                workdir=workdir,
            )
        )


def test_protected_deny_all_blocks_before_config_and_pre_spawn(monkeypatch):
    config_calls = {"count": 0}

    def forbidden_config():
        config_calls["count"] += 1
        raise AssertionError("_get_env_config must not be called after early denial")

    def forbidden_pre_spawn(*_args, **_kwargs):
        raise AssertionError("Git pre_spawn must not run after early denial")

    monkeypatch.setattr(tt, "_get_env_config", forbidden_config)
    monkeypatch.setattr(swp, "pre_spawn_consult", forbidden_pre_spawn)
    monkeypatch.setattr(tt, "_active_environments", {})
    with session_write_policy_scope(SessionWritePolicy.deny_all("deny")):
        result = json.loads(tt.terminal_tool("echo blocked", task_id="t", session_id="deny"))

    assert result["status"] == "blocked"
    assert result["success"] is False
    assert result["operation_kind"] == "terminal_exec"
    assert config_calls["count"] == 0


def test_protected_allowlist_blocks_before_subprocess(monkeypatch, tmp_path):
    def forbidden_config():
        raise AssertionError("_get_env_config must not be called after allowlist terminal denial")

    def forbidden_create(*_args, **_kwargs):
        raise AssertionError("executor/environment must not be created after denial")

    monkeypatch.setattr(tt, "_get_env_config", forbidden_config)
    monkeypatch.setattr(tt, "_create_environment", forbidden_create)
    with session_write_policy_scope(_allowlist("allowlist", tmp_path, "file_write")):
        result = json.loads(tt.terminal_tool("git status", task_id="t", session_id="allowlist"))

    assert result["status"] == "blocked"
    assert result["success"] is False
    assert result["policy_reason"] == "terminal_exec_denied_protected_mode"


def test_normal_non_git_executes_and_consults_once(monkeypatch, tmp_path):
    calls, _env = _install_fake_env(monkeypatch, tmp_path)
    consult_calls: list[dict] = []
    _allow_pre_spawn(monkeypatch, consult_calls)

    result = _run("echo hello")

    assert result["exit_code"] == 0
    assert result["output"] == "ran:echo hello"
    assert calls == [("echo hello", {"timeout": 5, "cwd": str(tmp_path), "bounded_capture": True})]
    assert len(consult_calls) == 1


@pytest.mark.parametrize("command", ["git status", "git diff", "git log"])
def test_normal_read_only_git_executes(monkeypatch, tmp_path, command):
    calls, _env = _install_fake_env(monkeypatch, tmp_path)

    result = _run(command)

    assert result["exit_code"] == 0
    assert calls == [(command, {"timeout": 5, "cwd": str(tmp_path), "bounded_capture": True})]


@pytest.mark.parametrize("command", ["git reset --hard", "git clean -fd", "git commit"])
def test_normal_known_mutating_git_semantics_preserved_when_allowed(monkeypatch, tmp_path, command):
    calls, _env = _install_fake_env(monkeypatch, tmp_path)
    _allow_pre_spawn(monkeypatch)

    result = _run(command)

    assert result["exit_code"] == 0
    assert calls == [(command, {"timeout": 5, "cwd": str(tmp_path), "bounded_capture": True})]


@pytest.mark.parametrize("command", ["git frobnicate", "git", "cd repo && git status", "git 'status"])
def test_unknown_or_ambiguous_git_blocks_before_execution(monkeypatch, tmp_path, command):
    calls, _env = _install_fake_env(monkeypatch, tmp_path)
    monkeypatch.setattr(tt, "_create_environment", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not create")))

    result = _run(command)

    assert result["status"] == "blocked"
    assert result["success"] is False
    assert result["exit_code"] == -1
    assert result["policy_reason"].startswith("git_mutation_unknown")
    assert calls == []


def test_returned_deny_decision_is_blocked(monkeypatch, tmp_path):
    calls, _env = _install_fake_env(monkeypatch, tmp_path)

    def deny_without_exception(caller_type, **kwargs):
        return SessionWritePolicyDecision(
            result=SessionWritePolicyDecisionResult.DENY,
            reason="controlled_returned_deny",
            operation_kind="terminal_exec",
            origin=str(caller_type),
            session_id="normal",
        )

    monkeypatch.setattr(swp, "pre_spawn_consult", deny_without_exception)

    result = _run("echo denied")

    assert result["status"] == "blocked"
    assert result["success"] is False
    assert result["policy_reason"] == "controlled_returned_deny"
    assert calls == []


def test_policy_denied_exception_is_blocked_safely(monkeypatch, tmp_path):
    calls, _env = _install_fake_env(monkeypatch, tmp_path)

    def raise_policy_denied(*_args, **_kwargs):
        raise PolicyDenied(
            disposition=PolicyDenied.DISPOSITION_POLICY_DENY,
            caller_type=CallerType.TERMINAL_TOOL,
            operation_kind="terminal_exec",
            reason="resolver_failed_without_env_dump",
            detail={"env": {"SECRET_TOKEN": "do-not-leak"}},
        )

    monkeypatch.setattr(swp, "pre_spawn_consult", raise_policy_denied)

    result = _run("git status")

    assert result["status"] == "blocked"
    assert result["success"] is False
    assert result["disposition"] == PolicyDenied.DISPOSITION_POLICY_DENY
    assert result["policy_reason"] == "resolver_failed_without_env_dump"
    assert "SECRET_TOKEN" not in json.dumps(result)
    assert calls == []


def test_pre_spawn_arguments_use_raw_command_effective_cwd_and_no_env(monkeypatch, tmp_path):
    from tools.approval import reset_current_session_key, set_current_session_key

    session_key = "normal"
    live_cwd = tmp_path / "live-cwd"
    live_cwd.mkdir()

    calls, _env = _install_fake_env(monkeypatch, tmp_path, cwd=str(live_cwd))
    consult_calls: list[dict] = []
    _allow_pre_spawn(monkeypatch, consult_calls)

    previous_cwd = tt.get_session_cwd(session_key)
    token = set_current_session_key(session_key)
    tt.record_session_cwd(session_key, str(live_cwd))
    try:
        result = _run("git status")
    finally:
        reset_current_session_key(token)
        if previous_cwd is None:
            tt.clear_session_cwd(session_key)
        else:
            tt.record_session_cwd(session_key, previous_cwd)

    assert result["exit_code"] == 0
    assert calls == [("git status", {"timeout": 5, "cwd": str(tmp_path / "live-cwd"), "bounded_capture": True})]
    assert len(consult_calls) == 1
    call = consult_calls[0]
    assert call["caller_type"] is CallerType.TERMINAL_TOOL
    assert call["operation_kind"] == "terminal_exec"
    assert call["raw_command"] == "git status"
    assert call["cwd"] == str(tmp_path / "live-cwd")
    assert call["env_subset"] is None
    assert "target_path" not in call
    assert "argv" not in call
    assert "command_argv" not in call


def test_git_dash_c_resolution_is_preserved_without_forcing_target_to_cwd(monkeypatch, tmp_path):
    calls, _env = _install_fake_env(monkeypatch, tmp_path)
    real_pre_spawn = swp.pre_spawn_consult
    consult_calls: list[dict] = []

    def capture_real_pre_spawn(caller_type, **kwargs):
        consult_calls.append({"caller_type": caller_type, **kwargs})
        decision = real_pre_spawn(caller_type, **kwargs)
        consult_calls[-1]["decision_target_path"] = decision.target_path
        return decision

    monkeypatch.setattr(swp, "pre_spawn_consult", capture_real_pre_spawn)

    result = _run("git -C repo reset --hard")

    assert result["exit_code"] == 0
    assert calls == [("git -C repo reset --hard", {"timeout": 5, "cwd": str(tmp_path), "bounded_capture": True})]
    assert len(consult_calls) == 1
    call = consult_calls[0]
    assert call["raw_command"] == "git -C repo reset --hard"
    assert call["cwd"] == str(tmp_path)
    assert "target_path" not in call
    assert call["decision_target_path"] == str((tmp_path / "repo").resolve(strict=False))
    assert call["decision_target_path"] != str(tmp_path.resolve(strict=False))


def test_pre_spawn_deny_happens_before_environment_creation(monkeypatch, tmp_path):
    monkeypatch.setattr(tt, "_active_environments", {})
    monkeypatch.setattr(tt, "_create_environment", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not create environment")))
    monkeypatch.setattr(
        swp,
        "pre_spawn_consult",
        lambda *a, **k: SessionWritePolicyDecision(
            result=SessionWritePolicyDecisionResult.DENY,
            reason="deny_before_env_creation",
            operation_kind="terminal_exec",
            origin="test",
            session_id="normal",
        ),
    )

    result = _run("echo blocked")

    assert result["status"] == "blocked"
    assert result["policy_reason"] == "deny_before_env_creation"
