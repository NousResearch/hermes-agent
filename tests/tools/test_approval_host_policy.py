"""Managed approval contracts through real guards and isolated on-disk policy."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hermes_cli import config
from tools import approval
import tools.approval_context as context
import tools.approval_host_policy as policy
import tools.host_policy as host
from agent.delegation_context import delegated_child_context
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

pytestmark = pytest.mark.platforms("posix")


@pytest.fixture
def environment(tmp_path, monkeypatch):
    root = tmp_path / "host-policy"
    root.mkdir(mode=0o700)
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setattr(host, "policy_root", lambda: root)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    for name in (
        "HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK",
        "HERMES_SINGLE_QUERY_SESSION", "HERMES_CRON_SESSION",
        "HERMES_SESSION_PLATFORM", "HERMES_SESSION_KEY", "HERMES_DELEGATED_CHILD_CONTEXT",
    ):
        monkeypatch.delenv(name, raising=False)
    token = context.set_current_session_key("session-one")
    config._LOAD_CONFIG_CACHE.clear()
    yield home, root
    context.reset_current_session_key(token)
    config._LOAD_CONFIG_CACHE.clear()
    approval._session_approved.clear()
    approval._session_yolo.clear()


def write_config(home, *, mode="manual", **fields):
    approvals = {"mode": mode, **fields}
    (home / "config.yaml").write_text(json.dumps({
        "model": {"default": "test-model"}, "approvals": approvals,
        "command_allowlist": ["rm -r allowed-directory", "recursive delete"],
        "delegation": {"subagent_auto_approve": True},
    }))
    config._LOAD_CONFIG_CACHE.clear()


def write_policy(root, *, revision="reviewed-one", **rules):
    path = root / "policy.json"
    path.write_text(json.dumps({
        "version": 1, "revision": revision,
        "admission": {"mode": "disabled", "max_workers": 2},
        "approvals": rules,
    }))
    path.chmod(0o600)


def exception(home, *, allow, session="session-one", expires=None):
    now = time.time()
    return {
        "id": "owner-reviewed-once", "profile": str(home.resolve()),
        "session_key": session, "reason": "Owner-authorized maintenance window",
        "issued_at": now - 60, "expires_at": expires or now + 60, "allow": allow,
    }


@pytest.mark.parametrize("gate", [
    lambda: approval.check_all_command_guards("rm -r disposable-directory", "local"),
    lambda: approval.check_dangerous_command("rm -r disposable-directory", "local"),
    lambda: approval.check_execute_code_guard("print('test')", "local"),
    lambda: approval.request_tool_approval("example", "Owner approval is required"),
])
def test_installed_default_blocks_headless_flagged_work(environment, gate):
    home, root = environment
    write_config(home, mode="off", unattended_mode="approve")
    write_policy(root)
    approval._YOLO_MODE_FROZEN = True
    result = gate()
    assert result["approved"] is False
    assert result["approval_policy"]["revision"] == "reviewed-one"
    assert result["approval_policy"]["mode"] == "manual"
    assert "profile_mode" in result["approval_policy"]["rejected_overrides"]


@pytest.mark.parametrize("backend", ["local", "docker"])
def test_malformed_installed_policy_blocks_even_isolation_fast_path(environment, backend):
    home, root = environment
    write_config(home, mode="off")
    path = root / "policy.json"
    path.write_text("{broken")
    path.chmod(0o600)
    result = approval.check_all_command_guards("ls", backend)
    assert result["approved"] is False
    assert result["outcome"] == "policy_error"
    assert result["approval_policy"]["error"] is True


def test_absent_policy_keeps_unmanaged_behavior(environment):
    home, _ = environment
    write_config(home, mode="off")
    result = approval.check_all_command_guards("rm -r disposable-directory", "local")
    assert result == {"approved": True, "message": None}


def test_managed_safe_commands_and_existing_isolation_exceptions_remain(environment):
    home, root = environment
    write_config(home)
    write_policy(root)
    assert approval.check_all_command_guards("ls", "local")["approved"] is True
    assert approval.check_all_command_guards("rm -r disposable-directory", "docker")["approved"] is True
    assert approval.check_execute_code_guard("print('test')", "docker")["approved"] is True
    assert approval.check_all_command_guards(
        "rm -r disposable-directory", "docker", has_host_access=True,
    )["approved"] is False


def test_scoped_yolo_exception_expires_and_cannot_cross_sessions(environment, monkeypatch):
    home, root = environment
    write_config(home)
    rule = exception(home, allow=["yolo"])
    write_policy(root, overrides=[rule])
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
    assert approval.check_all_command_guards(
        "rm -r disposable-directory", "local",
    )["approved"] is True
    assert approval.is_approval_bypass_active_for_session("different-session") is False
    monkeypatch.setattr(policy.time, "time", lambda: rule["expires_at"])
    result = approval.check_all_command_guards("rm -r disposable-directory", "local")
    assert result["approved"] is False
    assert result["approval_policy"]["override_ids"] == []


def test_exception_cannot_override_hardline_or_operator_deny(environment, monkeypatch):
    home, root = environment
    write_config(home, deny=["git push --force*"])
    write_policy(root, overrides=[exception(home, allow=["yolo"])])
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
    assert approval.check_all_command_guards("rm -rf /", "local")["hardline"] is True
    for backend in ("local", "docker"):
        assert approval.check_all_command_guards(
            "git push --force origin main", backend,
        )["approved"] is False


def test_profile_mode_relaxation_requires_explicit_exception(environment):
    home, root = environment
    write_config(home, mode="off")
    write_policy(root)
    assert context._get_approval_mode() == "manual"
    write_policy(root, overrides=[exception(home, allow=["profile_mode"])])
    result = approval.check_all_command_guards("rm -r disposable-directory", "local")
    assert result["approved"] is True
    assert result["approval_policy"]["override_ids"] == ["owner-reviewed-once"]


def test_standing_grants_never_override_managed_unattended_denial(environment):
    home, root = environment
    write_config(home, cron_mode="deny")
    write_policy(root, allow_standing_grants=True)
    approval.approve_session("session-one", "recursive delete")
    result = approval.check_all_command_guards("rm -r allowed-directory", "local")
    assert result["approved"] is False


def test_manual_human_approval_does_not_create_unauthorized_standing_grant(environment, monkeypatch):
    home, root = environment
    write_config(home)
    write_policy(root)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    answers = []

    def answer(*args, **kwargs):
        answers.append(kwargs)
        return "always"

    for _ in range(2):
        result = approval.check_all_command_guards(
            "rm -r disposable-directory", "local", approval_callback=answer,
        )
        assert result["approved"] is True
    assert len(answers) == 2
    assert all(item["allow_permanent"] is False for item in answers)
    assert "session-one" not in approval._session_approved


@pytest.mark.parametrize("local", ["cron_mode", "single_query_mode", "unattended_mode"])
def test_profile_unattended_approve_cannot_relax_host_deny(environment, local):
    home, root = environment
    write_config(home, **{local: "approve"})
    write_policy(root)
    assert context._binary_approval_mode(local) == "deny"
    write_policy(root, overrides=[exception(home, allow=["unattended_approve"])])
    assert context._binary_approval_mode(local) == "approve"


def test_delegated_callback_and_guard_follow_current_host_policy(environment):
    home, root = environment
    write_config(home)
    write_policy(root, subagent_auto_approve=True)
    with delegated_child_context("test-child"):
        assert policy.subagent_auto_approve_allowed(True) is True
        assert approval.check_all_command_guards(
            "rm -r disposable-directory", "local",
        )["approved"] is True
        write_policy(root, revision="stricter", subagent_auto_approve=False)
        assert policy.subagent_auto_approve_allowed(True) is False
        result = approval.check_all_command_guards("rm -r disposable-directory", "local")
    assert result["approved"] is False
    assert result["approval_policy"]["revision"] == "stricter"


def test_hosted_room_can_only_relax_with_authorized_exception(environment):
    from gateway.hosted_room_execution_policy import (
        RoomExecutionPolicy, bind_room_execution_policy, reset_room_execution_policy,
    )
    home, root = environment
    write_config(home)
    write_policy(root)
    room = RoomExecutionPolicy(1, "default", ("bot_room",), "off", 10, "test-digest")
    token = bind_room_execution_policy(room)
    try:
        assert context._get_approval_mode() == "manual"
        write_policy(root, overrides=[exception(home, allow=["hosted_room"])])
        assert context._get_approval_mode() == "off"
    finally:
        reset_room_execution_policy(token)


def test_concurrent_profile_scopes_cannot_share_exception(environment, tmp_path, monkeypatch):
    home, root = environment
    other = tmp_path / "other"
    other.mkdir()
    write_config(home)
    write_config(other)
    write_policy(root, overrides=[exception(home, allow=["yolo"])])
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)

    def run(profile):
        profile_token = set_hermes_home_override(profile)
        token = context.set_current_session_key("session-one")
        try:
            return approval.check_all_command_guards("rm -r disposable-directory", "local")
        finally:
            context.reset_current_session_key(token)
            reset_hermes_home_override(profile_token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        permitted, rejected = list(pool.map(run, (home, other)))
    assert permitted["approved"] is True
    assert rejected["approved"] is False
    assert rejected["approval_policy"]["override_ids"] == []


@pytest.mark.parametrize("rules", [
    {"mode": False}, {"mode": "typo"}, {"unknown": True},
    {"subagent_auto_approve": "true"}, {"allow_standing_grants": 1},
    {"cron_mode": "allow"}, {"overrides": {}},
])
def test_host_approval_schema_is_strict(environment, rules):
    _, root = environment
    write_policy(root, **rules)
    with pytest.raises(host.HostPolicyError):
        host.load_host_policy()


@pytest.mark.parametrize("change", [
    {"profile": "relative"}, {"session_key": ""}, {"reason": ""},
    {"expires_at": float("nan")}, {"issued_at": True},
    {"issued_at": 10 ** 1000}, {"expires_at": 10 ** 1000},
    {"allow": ["yolo", "yolo"]}, {"allow": ["everything"]},
    {"expires_at": time.time() + 90000},
])
def test_override_schema_rejects_ambiguous_or_unbounded_authority(environment, change):
    home, root = environment
    override = {**exception(home, allow=["yolo"]), **change}
    write_policy(root, overrides=[override])
    with pytest.raises(host.HostPolicyError):
        host.load_host_policy()


def test_provenance_is_allowlisted_and_does_not_mutate_config(environment):
    home, root = environment
    write_config(home, mode="off")
    write_policy(root, overrides=[exception(home, allow=["profile_mode"])])
    before = (home / "config.yaml").read_bytes()
    result = approval.check_all_command_guards("rm -r disposable-directory", "local")
    metadata = result["approval_policy"]
    assert set(metadata) == {
        "source", "revision", "policy_digest", "profile", "session_key", "mode",
        "override_ids", "rejected_overrides", "error",
    }
    assert policy.approval_policy_status()["revision"] == "reviewed-one"
    assert (home / "config.yaml").read_bytes() == before


def test_prepared_consent_is_rechecked_after_policy_tightens(environment, monkeypatch):
    import agent.terminal_approval_batch as batch
    home, root = environment
    write_config(home, mode="off")
    write_policy(root, overrides=[exception(home, allow=["profile_mode"])])
    prepared = approval.check_all_command_guards("rm -r disposable-directory", "local")
    assert prepared["approved"] is True
    write_policy(root, revision="stricter")
    monkeypatch.setattr(batch, "consume_prepared_guard", lambda *args: prepared)
    result = approval.check_all_command_guards("rm -r disposable-directory", "local")
    assert result["approved"] is False
    assert result["approval_policy"]["revision"] == "stricter"


def test_prepared_consent_fingerprint_detects_same_revision_edits(environment):
    home, root = environment
    write_config(home, mode="off")
    write_policy(root, overrides=[exception(home, allow=["profile_mode"])])
    prepared = approval.check_all_command_guards("rm -r disposable-directory", "local")
    write_policy(root)
    assert policy.prepared_guard_allowed(prepared) is False
    assert policy.prepared_guard_allowed({"approved": False}) is True


def test_managed_session_grant_does_not_survive_policy_revision(environment, monkeypatch):
    home, root = environment
    write_config(home)
    write_policy(root, allow_standing_grants=True)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    answers = []

    def answer(*args, **kwargs):
        answers.append(kwargs)
        return "session"

    for _ in range(2):
        assert approval.check_all_command_guards(
            "rm -r disposable-directory", "local", approval_callback=answer,
        )["approved"] is True
    assert len(answers) == 1
    write_policy(root, revision="reviewed-two", allow_standing_grants=True)
    assert approval.check_all_command_guards(
        "rm -r disposable-directory", "local", approval_callback=answer,
    )["approved"] is True
    assert len(answers) == 2


def test_managed_permanent_grant_requires_a_bounded_exception(environment, monkeypatch):
    home, root = environment
    write_config(home)
    write_policy(root, allow_standing_grants=True)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    answer = lambda *args, **kwargs: "deny"
    assert approval.check_all_command_guards(
        "rm -r allowed-directory", "local", approval_callback=answer,
    )["approved"] is False
    write_policy(root, overrides=[exception(home, allow=["standing_grants"])])
    assert approval.check_all_command_guards(
        "rm -r allowed-directory", "local", approval_callback=answer,
    )["approved"] is True
