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
    # "hermes" is a legal profile id: a REAL named profile of that name is never swallowed by the alias.
    (default_home / "profiles" / "hermes").mkdir(parents=True)
    assert server._response_profile_name("hermes") == "hermes"


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


def test_custom_default_root_real_session_db_owner_stamping(tmp_path, monkeypatch):
    """Custom default roots stamp real SessionDB rows as default without leaking to siblings."""
    from hermes_constants import profile_name_for_home
    from hermes_state import SessionDB
    from tui_gateway import server

    custom_root = tmp_path / "custom-root"
    launch_home = custom_root / "profiles" / "worker"
    default_home = custom_root
    default_home.mkdir(parents=True)
    launch_home.mkdir(parents=True)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setattr(server, "_hermes_home", launch_home)

    # 1. Custom default root path resolution
    assert profile_name_for_home(default_home) == "default"
    assert profile_name_for_home(launch_home) == "worker"
    assert server._session_info(None, {"profile_home": str(default_home)})["profile_name"] == "default"
    assert server._session_info(None, {"profile_home": str(launch_home)})["profile_name"] == "worker"

    # 2. Real SessionDB owner stamping
    with SessionDB(default_home / "state.db") as db:
        db.create_session("parent-default", "desktop", profile_name="default")
    with SessionDB(launch_home / "state.db") as db:
        pass

    session = {
        "session_key": "lazy-default",
        "profile_home": str(default_home),
        "cwd": str(tmp_path),
    }
    assert server._ensure_session_db_row(session) is True

    record = {
        "cwd": str(tmp_path),
        "pending_title": "branch",
        "profile_home": str(default_home),
    }
    server._seed_branch_row(
        record,
        "seeded-default",
        "parent-default",
        [{"role": "user", "content": "hello"}],
        "desktop",
        str(default_home),
    )
    assert record["pending_title"] is None

    # Normal branch persistence path
    with SessionDB(default_home / "state.db") as db:
        server._persist_branch(
            db,
            "branched-default",
            "parent-default",
            "Branch Test",
            [{"role": "user", "content": "hello"}],
            source="desktop",
            cwd=str(tmp_path),
            profile_name=profile_name_for_home(str(default_home)) or server._current_profile_name(),
        )

    # Verify rows in default_home / state.db
    with SessionDB(default_home / "state.db") as db:
        lazy_row = db.get_session("lazy-default")
        assert lazy_row is not None
        assert lazy_row["profile_name"] == "default"

        seeded_row = db.get_session("seeded-default")
        assert seeded_row is not None
        assert seeded_row["profile_name"] == "default"

        branched_row = db.get_session("branched-default")
        assert branched_row is not None
        assert branched_row["profile_name"] == "default"

    # 3. Negative assertions: default session/branch rows do not land in sibling store
    with SessionDB(launch_home / "state.db") as launch_db:
        assert launch_db.get_session("parent-default") is None
        assert launch_db.get_session("lazy-default") is None
        assert launch_db.get_session("seeded-default") is None
        assert launch_db.get_session("branched-default") is None
