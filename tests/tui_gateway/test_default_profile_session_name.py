"""Default-profile session names must be derived from the home path, not its basename."""

from __future__ import annotations

import contextlib
from pathlib import Path


def _profile_layout(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / ".hermes"
    default_home = root
    launch_home = root / "profiles" / "worker"
    launch_home.mkdir(parents=True)
    return default_home, launch_home


def test_default_home_aliases_are_reported_as_default(tmp_path, monkeypatch):
    """Legacy basename values must not be resolved as missing named profiles."""
    from tui_gateway import server

    default_home, launch_home = _profile_layout(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setattr(server, "_hermes_home", launch_home)

    for alias in (default_home.name, "hermes"):
        assert server._response_profile_name(alias) == "default"
    assert server._session_info(None, {"profile_home": str(default_home)})["profile_name"] == "default"


def test_profile_home_resolution_stamps_default_rows(tmp_path, monkeypatch):
    """Default and named homes resolve canonically, including lazy row creation."""
    from hermes_constants import profile_name_for_home
    from tui_gateway import server

    default_home, launch_home = _profile_layout(tmp_path)
    named_home = default_home / "profiles" / "writer"
    named_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setattr(server, "_hermes_home", launch_home)

    assert profile_name_for_home(default_home) == "default"
    assert profile_name_for_home(named_home) == "writer"
    assert server._session_info(None, {"profile_home": str(named_home)})["profile_name"] == "writer"

    class CaptureDB:
        profile_name = None

        def create_session(self, _key, **kwargs):
            self.profile_name = kwargs["profile_name"]

        def append_messages_batch(self, _key, _messages, *, chunk_rows):
            assert chunk_rows == 500

        def get_session_title(self, _key):
            return "branch"

        def set_session_title(self, _key, _title):
            return None

    captured = CaptureDB()

    @contextlib.contextmanager
    def owner_db(_session, _failure_message=None):
        yield captured

    monkeypatch.setattr(server, "_workdir_owner_db", owner_db)
    monkeypatch.setattr(server, "_workdir_row_model_config", lambda _session: ("test-model", {}))
    monkeypatch.setattr(server, "_session_source", lambda _session: "desktop")
    monkeypatch.setattr(server, "_persisted_session_cwd", lambda _session: None)

    session = {"session_key": "default-row", "profile_home": str(default_home)}
    assert server._ensure_session_db_row(session) is True
    assert captured.profile_name == "default"

    monkeypatch.setattr(server, "_session_db", owner_db)
    record = {"cwd": str(tmp_path), "pending_title": "branch"}
    server._seed_branch_row(record, "seeded-row", "parent-row", [{"role": "user", "content": "hi"}], "desktop",
                            str(default_home))
    assert captured.profile_name == "default"
    assert record["pending_title"] is None
