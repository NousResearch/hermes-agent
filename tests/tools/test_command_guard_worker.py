import subprocess

from tools import approval


def test_long_commands_are_classified_once_outside_parent(monkeypatch) -> None:
    safe = "; ".join(["printf λ"] * 500)
    monkeypatch.setenv("LC_ALL", "C")
    monkeypatch.setenv("PYTHONUTF8", "0")
    monkeypatch.setattr(
        approval, "detect_dangerous_command",
        lambda _command: (_ for _ in ()).throw(AssertionError("parent reparsed long command")),
    )

    assert approval.check_dangerous_command(safe, "local")["approved"] is True

    hardline = safe + "; rm -rf /"
    result = approval.check_all_command_guards(hardline, "local")
    assert result["approved"] is False
    assert "hardline" in result["message"].lower()


def test_long_command_worker_timeout_fails_closed(monkeypatch) -> None:
    from tools import approval_guard_worker

    completed = subprocess.CompletedProcess(
        [], 0, stdout='{"kind":"dangerous","pattern_key":"x","description":"x"}', stderr="",
    )
    monkeypatch.setattr(approval_guard_worker.subprocess, "run", lambda *_args, **_kwargs: completed)
    try:
        approval_guard_worker.classify_long_command("x" * 5000, full=False)
    except ValueError:
        pass
    else:
        raise AssertionError("user-deny worker accepted a full-mode verdict")
    for check in (approval.check_all_command_guards, approval.check_dangerous_command):
        result = check("x" * 5000, "docker")
        assert result["approved"] is False
        assert result["guard_error"] is True

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("guard", 10)

    monkeypatch.setattr(approval_guard_worker, "classify_long_command", timeout)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)

    result = approval.check_all_command_guards("x" * 5000, "local")
    assert result["approved"] is False
    assert result["guard_timeout"] is True
    assert "Do NOT retry" in result["message"]
