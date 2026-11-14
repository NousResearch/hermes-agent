"""An ssh session's launch directory is a host path: kept in memory, never persisted as its remote workspace.

A resume adopts a stored ssh cwd as the remote workspace, so a persisted launch directory (in the Docker image
``/opt/hermes``, ``/opt/data`` or ``/opt/data/home``) made every remote terminal and file call ``cd`` into a path that
only exists on the Hermes host.
"""

from __future__ import annotations

import contextlib

import pytest

from tui_gateway import server
from hermes_state import SessionDB


class _ImmediateThread:
    def __init__(self, *, target, **_kwargs):
        self._target = target

    def start(self):
        self._target()


@pytest.fixture
def backend(monkeypatch):
    def use(name: str) -> None:
        monkeypatch.setattr(server, "_effective_terminal_backend", lambda: name)

    use("ssh")
    return use


@pytest.fixture
def db(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server.git_probe, "branch", lambda _cwd: None)
    monkeypatch.setattr(server.git_probe, "common_repo_root", lambda _cwd: None)
    yield db
    db.close()


@pytest.fixture
def resume_db(db, monkeypatch, tmp_path):
    """``session.resume`` against the real SessionDB, agent build and side schedulers off."""
    for name, value in {
        "_resolve_model": lambda: "test-model",
        "_enable_gateway_prompts": lambda: None,
        "_find_live_session_by_key": lambda _key, _home=None: None,
        "_schedule_agent_build": lambda *a, **k: None,
        "_schedule_session_cap_enforcement": lambda *a, **k: None,
        "_maybe_schedule_auto_continue": lambda *a, **k: None,
        "_default_session_cwd": lambda *a, **k: str(tmp_path),
        "_child_run_active": lambda _key, _home=None: False,
    }.items():
        monkeypatch.setattr(server, name, value)
    known = set(server._sessions)
    yield db
    with server._sessions_lock:
        for sid in [s for s in server._sessions if s not in known]:
            server._sessions.pop(sid, None)


def _resume_live(key: str) -> dict:
    resp = server.handle_request({"id": "1", "method": "session.resume", "params": {"session_id": key}})
    assert "error" not in resp, resp
    return server._sessions[resp["result"]["session_id"]]


def _hydrate(db, key: str, session: dict) -> dict:
    sid = f"sid-{key}"
    server._sessions[sid] = session
    try:
        server._hydrate_session_cwd(sid, key, db, None)
    finally:
        server._sessions.pop(sid, None)
    return session


def _hermes_home_subdir(name: str) -> str:
    path = server.get_hermes_home() / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def test_ssh_launch_dir_is_not_persisted(backend, tmp_path):
    launch = str(tmp_path)
    assert server._persisted_session_cwd({"source": "tui", "cwd": launch}) is None


def test_ssh_explicit_cwd_is_persisted(backend):
    session = {"source": "tui", "cwd": "/home/me/proj", "explicit_cwd": True}
    assert server._persisted_session_cwd(session) == "/home/me/proj"


@pytest.mark.parametrize("name", ["local", "docker"])
def test_host_backends_still_persist_the_launch_dir(backend, tmp_path, name):
    backend(name)
    assert server._persisted_session_cwd({"source": "tui", "cwd": str(tmp_path)}) == str(tmp_path)


def test_hermes_owned_cwd(tmp_path):
    home = server.get_hermes_home()
    assert server._is_hermes_owned_cwd(str(home), None)
    assert server._is_hermes_owned_cwd(_hermes_home_subdir("home"), None)
    assert server._is_hermes_owned_cwd(str(server.Path(server.__file__).resolve().parent.parent), None)
    assert not server._is_hermes_owned_cwd("/home/me/proj", None)
    assert not server._is_hermes_owned_cwd(str(tmp_path), None)


def test_hermes_owned_cwd_includes_the_root_for_a_named_profile():
    """A named profile's home sits under the Hermes root; the root's launch dirs (``/opt/data``, ``/opt/data/home``)
    are still Hermes's own."""
    root = server.get_hermes_home()
    profile = str(root / "profiles" / "work")
    assert server._is_hermes_owned_cwd(_hermes_home_subdir("home"), profile)
    assert server._is_hermes_owned_cwd(str(root), profile)
    assert server._is_hermes_owned_cwd(profile, profile)
    assert not server._is_hermes_owned_cwd("/home/me/proj", profile)


def test_named_profile_resume_does_not_adopt_a_root_launch_dir(db):
    root = server.get_hermes_home()
    profile = root / "profiles" / "work"
    profile.mkdir(parents=True, exist_ok=True)
    (profile / "config.yaml").write_text("terminal:\n  backend: ssh\n")
    stale = _hermes_home_subdir("home")
    db.create_session("named", source="tui", model="m", cwd=stale)
    session = {"session_key": "named", "source": "tui", "cwd": "~", "profile_home": str(profile)}
    server._sessions["sid-named"] = session
    try:
        server._hydrate_session_cwd("sid-named", "named", db, str(profile))
    finally:
        server._sessions.pop("sid-named", None)
    assert not session.get("explicit_cwd")
    assert server._terminal_task_cwd(session) == "~"


def test_resume_leaves_a_set_aside_row_unchanged(backend, db):
    """With ``terminal.cwd`` set the session falls back to it as explicit; the stored row is still not rewritten."""
    stale = _hermes_home_subdir("home")
    db.create_session("aside", source="tui", model="m", cwd=stale)
    session = _hydrate(db, "aside", {"session_key": "aside", "source": "tui", "cwd": "/remote/default", "explicit_cwd": True})
    assert session["cwd"] == "/remote/default"
    assert db.get_session("aside")["cwd"] == stale


@pytest.mark.parametrize("stored", ["~", "~/proj"])
def test_resume_keeps_a_remote_tilde_cwd_when_home_is_inside_hermes_home(backend, db, monkeypatch, tmp_path, stored):
    """The Docker image's HOME (``/opt/data/home``) sits inside HERMES_HOME; ``~`` still names the REMOTE home."""
    monkeypatch.setenv("HOME", _hermes_home_subdir("home"))
    db.create_session("tilde", source="tui", model="m", cwd=stored)
    session = _hydrate(db, "tilde", {"session_key": "tilde", "source": "tui", "cwd": str(tmp_path)})

    assert session["cwd"] == stored
    assert session["explicit_cwd"] is True


def test_resume_does_not_adopt_a_stored_hermes_home_cwd(backend, db, tmp_path):
    """The ticket: a row stamped with ``/opt/data/home`` before this rule existed."""
    stale = _hermes_home_subdir("home")
    db.create_session("legacy", source="tui", model="m", cwd=stale)
    session = _hydrate(db, "legacy", {"session_key": "legacy", "source": "tui", "cwd": str(tmp_path)})

    assert session["cwd"] == str(tmp_path)
    assert not session.get("explicit_cwd")
    assert db.get_session("legacy")["cwd"] == stale  # left as-is: every resume heals it again


def test_resume_keeps_a_stored_remote_workspace(backend, db, tmp_path):
    db.create_session("picked", source="tui", model="m", cwd="/home/me/proj")
    session = _hydrate(db, "picked", {"session_key": "picked", "source": "tui", "cwd": str(tmp_path)})

    assert session["cwd"] == "/home/me/proj"
    assert session["explicit_cwd"] is True


def test_resume_without_a_stored_cwd_does_not_persist_the_launch_dir(backend, db, tmp_path):
    db.create_session("fresh", source="tui", model="m")
    session = _hydrate(db, "fresh", {"session_key": "fresh", "source": "tui", "cwd": str(tmp_path)})

    assert session["cwd"] == str(tmp_path)
    assert not session.get("explicit_cwd")
    assert db.get_session("fresh")["cwd"] is None


@pytest.mark.parametrize("name", ["local", "docker"])
def test_host_backends_resume_unchanged(backend, db, tmp_path, name):
    """Docker installs on the local or docker backend keep adopting a stored cwd under HERMES_HOME."""
    backend(name)
    stored = _hermes_home_subdir("home")
    db.create_session("host", source="tui", model="m", cwd=stored)
    session = _hydrate(db, "host", {"session_key": "host", "source": "tui", "cwd": str(tmp_path)})

    assert session["cwd"] == stored
    assert not session.get("explicit_cwd")

    db.create_session("host-fresh", source="tui", model="m")
    _hydrate(db, "host-fresh", {"session_key": "host-fresh", "source": "tui", "cwd": str(tmp_path)})
    assert db.get_session("host-fresh")["cwd"] == str(tmp_path)


@pytest.mark.parametrize(("explicit", "expected"), [(False, None), (True, "/home/me/proj")])
def test_ssh_branch_seed_skips_the_launch_dir(backend, monkeypatch, explicit, expected):
    seen = {}
    monkeypatch.setattr(server, "_session_db", lambda _record: contextlib.nullcontext(object()))
    monkeypatch.setattr(server, "_branch_title", lambda *_a: "t")
    monkeypatch.setattr(server, "_persist_branch", lambda *_a, cwd, **_k: seen.setdefault("cwd", cwd))
    record = {"cwd": "/home/me/proj" if explicit else "/opt/hermes", "explicit_cwd": explicit}

    server._seed_branch_row(record, "child", "parent", [], "desktop", None)

    assert seen["cwd"] == expected


KEY = "20261003_120000_c0ffee"


def test_session_resume_does_not_adopt_a_stored_hermes_home_cwd(backend, resume_db, tmp_path):
    resume_db.create_session(KEY, source="tui", model="test-model", cwd=_hermes_home_subdir("home"))
    live = _resume_live(KEY)

    assert live["cwd"] == str(tmp_path)
    assert not live.get("explicit_cwd")


def test_session_resume_keeps_a_stored_remote_workspace(backend, resume_db):
    resume_db.create_session(KEY, source="tui", model="test-model", cwd="/home/me/proj")
    live = _resume_live(KEY)

    assert live["cwd"] == "/home/me/proj"
    assert live["explicit_cwd"] is True


@pytest.mark.parametrize("name", ["local", "docker"])
def test_session_resume_on_host_backends_unchanged(backend, resume_db, name):
    backend(name)
    stored = _hermes_home_subdir("home")
    resume_db.create_session(KEY, source="tui", model="test-model", cwd=stored)
    live = _resume_live(KEY)

    assert live["cwd"] == stored
    assert live["explicit_cwd"] is True
