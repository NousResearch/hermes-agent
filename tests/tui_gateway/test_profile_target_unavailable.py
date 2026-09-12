"""An unavailable explicit target must never become the launch profile."""
from pathlib import Path

import pytest


def test_explicit_profile_target_never_falls_back(tmp_path, monkeypatch):
    from tui_gateway import server
    from hermes_state import SessionDB

    home = tmp_path / ".hermes"
    worker = home / "profiles" / "worker"
    worker.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", home)
    for path, marker in ((home, "launch"), (worker, "worker")):
        (path / "config.yaml").write_text(f"terminal:\n  cwd: /{marker}\n")
        with SessionDB(db_path=path / "state.db") as db:
            db.create_session(marker, "tui")
    for name, marker in ((None, "launch"), ("default", "launch"), ("DEFAULT", "launch"), ("worker", "worker")):
        with server._profile_db({"profile": name}) as db:
            assert db.get_session(marker)
        response = server._methods["config.get"](1, {"profile": name, "key": "full"})
        assert response["result"]["config"]["terminal"]["cwd"] == f"/{marker}"
    before = (home / "config.yaml").read_bytes()
    worker.rename(worker.with_name("gone"))
    for name in ("worker", "unknown"):
        with pytest.raises(FileNotFoundError):
            with server._profile_db({"profile": name}):
                pytest.fail("unavailable profile reached a database")
        with pytest.raises(FileNotFoundError):
            server._methods["config.set"](2, {"profile": name, "key": "busy", "value": "steer"})
        assert (home / "config.yaml").read_bytes() == before
    # A real resolution I/O failure must propagate, too (no predicate patch).
    profiles = home / "profiles"
    profiles.rename(home / "saved-profiles")
    profiles.symlink_to("profiles")
    with pytest.raises((OSError, RuntimeError)):
        server._profile_home("worker")


def test_custom_root_basename_target_fails_closed_when_unavailable(tmp_path, monkeypatch):
    """An explicit profile request matching a custom root basename fails closed."""
    from tui_gateway import server

    custom_home = tmp_path / "customer-data"
    custom_home.mkdir()
    (custom_home / "config.yaml").write_text("terminal:\n  cwd: /custom\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(custom_home))
    monkeypatch.setattr(server, "_hermes_home", custom_home)

    with pytest.raises(FileNotFoundError):
        server._profile_home("customer-data")

    with pytest.raises(FileNotFoundError):
        with server._profile_db({"profile": "customer-data"}):
            pass
