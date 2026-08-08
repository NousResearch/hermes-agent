"""Compute-host spawn must not undo hermes_subprocess_env scrubbing."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from tui_gateway.host_supervisor import HostSupervisor


def test_spawn_does_not_remerge_full_os_environ(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak-via-update")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tg-should-not-leak-via-update")

    captured = {}

    def fake_popen(*args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        proc = MagicMock()
        proc.pid = 4242
        proc.poll.return_value = None
        proc.stdin = MagicMock()
        proc.stdout = MagicMock()
        proc.stderr = MagicMock()
        return proc

    registry = tmp_path / "registry.json"
    supervisor = HostSupervisor(
        registry_path=registry,
        argv=["python", "-c", "pass"],
        cwd=Path(tmp_path),
        autostart=False,
    )

    def _fake_wait(timeout=None):
        supervisor._hello = {
            "hermes_home": supervisor.expected_hermes_home,
            "build_sha": "unknown",
        }
        return True

    with patch("tui_gateway.host_supervisor.subprocess.Popen", side_effect=fake_popen), patch(
        "tui_gateway.host_supervisor._Thread"
    ), patch.object(supervisor._hello_event, "wait", side_effect=_fake_wait), patch.object(
        supervisor, "_persist_registry"
    ):
        with supervisor._lock:
            supervisor._spawn_locked(reason="test")

    env = captured["env"]
    # Tier-1 always-strip keys must stay absent even when inherit_credentials=True.
    assert "TELEGRAM_BOT_TOKEN" not in env
