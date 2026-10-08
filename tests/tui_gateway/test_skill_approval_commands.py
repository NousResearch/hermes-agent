"""TUI fallback and real slash-worker command execution for skill scope settings."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import hermes_yaml as yaml

from tools import write_approval as wa
from tui_gateway import server, slash_worker


@pytest.fixture
def session(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    home = tmp_path / "profiles" / "beta"
    launch.mkdir()
    home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_hermes_home", launch)
    (launch / "config.yaml").write_text("model:\n  default: launch\n")
    (home / "config.yaml").write_text(
        "# policy\nskills:\n  write_approval: false # gate\n  write_approval_mode: create # scope\n"
        "model:\n  default: 'kept' # keep\n")
    record = {"profile_home": str(home), "session_key": "skill-approval-session",
              "agent": SimpleNamespace(model="test/model"), "running": False,
              "history": [], "slash_worker": None}
    monkeypatch.setitem(server._sessions, "skill-approval-session", record)
    return launch, home, record


def dispatch(arg, name="skills"):
    result = server._methods["command.dispatch"](
        "approval", {"name": name, "arg": arg, "session_id": "skill-approval-session"})
    assert "result" in result, result
    return result["result"]["output"]


def test_fallback_roundtrips_preserve_profile_queue_and_comments(session):
    launch, home, record = session
    before = (launch / "config.yaml").read_bytes()
    with server._session_home_scope(record):
        pending = wa.stage_write("skills", {"action": "create", "name": "queued"},
                                 summary="old queued request", origin="foreground")
    path = home / "pending" / "skills" / (pending["id"] + ".json")
    pending_before = path.read_bytes()
    for arg, enabled, scope in [("approval all", True, "all"), ("approval off", False, "all"),
                                ("mode on", True, "all"), ("mode create", True, "create")]:
        assert "set to" in dispatch(arg)
        config = yaml.safe_load((home / "config.yaml").read_text())
        assert config["skills"] == {"write_approval": enabled, "write_approval_mode": scope}
        assert "scope: " + scope in dispatch("approval status")
        assert pending["id"] in dispatch("pending")
        assert path.read_bytes() == pending_before
    assert (launch / "config.yaml").read_bytes() == before
    assert not (home / "skills" / "queued").exists()
    for fragment in ["# policy", "# gate", "# scope", "'kept' # keep"]:
        assert fragment in (home / "config.yaml").read_text()


def test_fallback_invalid_memory_and_replace_failure_leave_config_unchanged(session, monkeypatch):
    import utils
    _launch, home, _record = session
    for arg, subsystem in [("approval typo", "skills"), ("approval all extra", "skills"),
                           ("approval create", "memory"), ("mode all", "memory")]:
        before = (home / "config.yaml").read_bytes()
        assert "Invalid value" in dispatch(arg, subsystem)
        assert (home / "config.yaml").read_bytes() == before
    for mode, enabled in [("on", True), ("off", False)]:
        assert "set to" in dispatch("approval " + mode, "memory")
        assert yaml.safe_load((home / "config.yaml").read_text())["memory"] == {"write_approval": enabled}
    before = (home / "config.yaml").read_bytes()
    def fail_replace(*args, **kwargs):
        raise OSError("simulated replace failure")
    monkeypatch.setattr(utils.os, "replace", fail_replace)
    assert "Failed to set" in dispatch("approval all")
    assert (home / "config.yaml").read_bytes() == before


def test_slash_exec_worker_uses_real_cli_dispatch_and_persistence(session):
    """Replace only worker boot/transport: execute the real worker _run + HermesCLI dispatcher.

    No model initialization, MCP discovery or network is needed for these local commands.
    """
    from cli import HermesCLI
    launch, home, record = session
    before = (launch / "config.yaml").read_bytes()
    cli = object.__new__(HermesCLI)
    cli._slash_metrics_surface = None
    cli.agent = None
    class Worker:
        def run(self, command):
            with server._session_home_scope(record):
                return slash_worker._run(cli, command)
        def pop_seed(self):
            return ""
    record["slash_worker"] = Worker()
    for command, enabled, scope in [("approval all", True, "all"), ("approval off", False, "all"),
                                    ("approval on", True, "all"), ("mode create", True, "create")]:
        result = server._methods["slash.exec"](
            "worker", {"command": "/skills " + command, "session_id": "skill-approval-session"})
        assert "result" in result, result
        assert "set to" in result["result"]["output"]
        assert yaml.safe_load((home / "config.yaml").read_text())["skills"] == {
            "write_approval": enabled, "write_approval_mode": scope}
    assert (launch / "config.yaml").read_bytes() == before
