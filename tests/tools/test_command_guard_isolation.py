"""Long command guard classification is isolated and fail-closed."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tools.approval as approval
from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_short_command_stays_in_process(monkeypatch):
    monkeypatch.setattr(
        approval,
        "_run_command_guard_worker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("worker used")),
        raising=False,
    )
    assert approval.check_all_command_guards("printf ok", "local")["approved"] is True


def test_long_unconditional_classification_uses_worker(monkeypatch):
    seen = []

    def fake_worker(mode, command):
        seen.append((mode, command))
        return {"kind": "allow"}

    monkeypatch.setattr(approval, "_run_command_guard_worker", fake_worker, raising=False)
    command = "printf x; " + ("printf y; " * 500)
    with patch.object(approval, "_get_approval_mode", return_value="off"):
        result = approval.check_all_command_guards(command, "local")
    assert result["approved"] is True
    assert seen == [("unconditional", command)]


def test_long_worker_timeout_fails_closed_before_yolo(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("guard-worker", 2.0)

    monkeypatch.setattr(subprocess, "run", timeout)
    command = "printf x; " + ("printf y; " * 500)
    with patch.object(approval, "_get_approval_mode", return_value="off"):
        result = approval.check_all_command_guards(command, "local")
    assert result["approved"] is False
    assert result["hardline"] is True
    assert result["guard_timeout"] is True
    assert "Do NOT retry" in result["message"]


def test_long_worker_malformed_output_fails_closed(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="not-json", stderr=""),
    )
    command = "printf x; " + ("printf y; " * 500)
    result = approval.check_all_command_guards(command, "local")
    assert result["approved"] is False
    assert result["hardline"] is True
    assert result["guard_error"] is True


def test_worker_rejects_incomplete_dangerous_response(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout='{"kind":"dangerous"}',
            stderr="",
        ),
    )
    with pytest.raises(ValueError, match="invalid command guard worker response"):
        approval._run_command_guard_worker("unconditional", "printf ok")


def test_real_worker_protocol_round_trip():
    command = "printf x; " + ("printf y; " * 500)
    assert approval._run_command_guard_worker("unconditional", command) == {
        "kind": "allow"
    }


def test_real_worker_uses_context_local_hermes_home(tmp_path, monkeypatch):
    default_home = tmp_path / "default"
    active_home = tmp_path / "finance"
    default_home.mkdir()
    active_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    token = set_hermes_home_override(active_home)
    try:
        command = f"printf x > {active_home / 'config.yaml'}; # " + ("x" * 4_200)
        matched, pattern_key, description = approval.detect_dangerous_command(command)
        assert matched is True
        assert approval._run_command_guard_worker("unconditional", command) == {
            "kind": "dangerous",
            "pattern_key": pattern_key,
            "description": description,
        }
    finally:
        reset_hermes_home_override(token)


def test_worker_preserves_hardline_semantics(monkeypatch):
    command = "printf x; " + ("printf y; " * 500) + " rm -rf /"
    completed = SimpleNamespace(
        returncode=0,
        stdout=json.dumps({"kind": "hardline", "description": "recursive deletion of root filesystem"}),
        stderr="",
    )
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: completed)
    with patch.object(approval, "_get_approval_mode", return_value="off"):
        result = approval.check_all_command_guards(command, "local")
    assert result["approved"] is False
    assert result["hardline"] is True
    assert "recursive deletion" in result["message"]


def test_worker_launch_is_independent_of_caller_cwd(monkeypatch):
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout='{"kind":"allow"}', stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    approval._run_command_guard_worker("unconditional", "printf ok")
    assert captured["cwd"] == str(Path(approval.__file__).resolve().parent.parent)
    assert captured["creationflags"] == approval.windows_hide_flags()


def test_worker_snapshots_sudo_password_from_active_profile_scope(monkeypatch):
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout='{"kind":"allow"}', stderr="")

    monkeypatch.setenv("SUDO_PASSWORD", "other-profile-password")
    monkeypatch.setattr(subprocess, "run", fake_run)
    set_multiplex_active(True)
    token = set_secret_scope({})
    try:
        approval._run_command_guard_worker("unconditional", "printf ok")
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)

    request = json.loads(captured["input"])
    assert request["sudo_password_configured"] is False


def test_long_dangerous_verdict_is_reused_in_parent_approval(monkeypatch):
    command = "printf x; " + ("printf y; " * 500)
    monkeypatch.setattr(
        approval,
        "_run_command_guard_worker",
        lambda *_args: {
            "kind": "dangerous",
            "pattern_key": "worker-key",
            "description": "worker dangerous result",
        },
    )
    monkeypatch.setattr(
        approval,
        "detect_dangerous_command",
        lambda _command: (_ for _ in ()).throw(AssertionError("parent reparsed")),
    )
    monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
    monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval, "_get_approval_mode", lambda: "ask")
    monkeypatch.setattr("tools.tirith_security.check_command_security", lambda _c: {"action": "allow"})
    descriptions = []

    def deny(_command, description, **_kwargs):
        descriptions.append(description)
        return "deny"

    result = approval.check_all_command_guards(
        command,
        "local",
        approval_callback=deny,
    )
    assert result["approved"] is False
    assert descriptions == ["worker dangerous result"]
