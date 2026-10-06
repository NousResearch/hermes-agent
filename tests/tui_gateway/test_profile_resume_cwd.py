"""A resumed session for a NAMED profile lands in THAT profile's workspace (#87584).

A Bot Chat is one forever-session, so fixing where new sessions start never reaches it: its row keeps whatever cwd
was stamped when it was created. Under the Docker image a named profile's chat was stamped with the launch profile's
``/opt/data``; resuming adopted that row as the session's explicit workspace, so ``pwd`` and the Files pane showed the
default profile's files. A resume with no stored cwd fell to ``_default_session_cwd()`` — the launch profile's — too.
"""

from __future__ import annotations

import pytest

import tui_gateway.server as server
from hermes_state import SessionDB
from tests.tui_gateway.test_tui_gateway_server import _write_profile_cfg

KEY = "20261005_120000_b07c4a7"


class _ImmediateThread:
    def __init__(self, *, target, **_kwargs):
        self._target = target

    def start(self):
        self._target()


@pytest.fixture
def profile(monkeypatch):
    """A named local profile under the Hermes root (``/opt/data/profiles/austin``), placeholder ``terminal.cwd``."""
    root = server.get_hermes_home()
    home = _write_profile_cfg(root / "profiles" / "austin", ".")
    monkeypatch.setattr(server, "_profile_home", lambda name: home if name else None)
    monkeypatch.setenv("TERMINAL_CWD", str(root))  # the launch profile's workspace
    return home


@pytest.fixture
def resume(profile, monkeypatch):
    """``session.resume`` for ``profile`` against its real state.db; agent build and side schedulers off."""
    for name, value in {
        "_resolve_model": lambda: "test-model",
        "_enable_gateway_prompts": lambda: None,
        "_find_live_session_by_key": lambda _key, _home=None: None,
        "_schedule_agent_build": lambda *a, **k: None,
        "_schedule_session_cap_enforcement": lambda *a, **k: None,
        "_maybe_schedule_auto_continue": lambda *a, **k: None,
        "_child_run_active": lambda _key, _home=None: False,
    }.items():
        monkeypatch.setattr(server, name, value)
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server.git_probe, "branch", lambda _cwd: None)
    monkeypatch.setattr(server.git_probe, "common_repo_root", lambda _cwd: None)
    db = SessionDB(db_path=profile / "state.db")
    known = set(server._sessions)

    def run(stored: str | None) -> dict:
        db.create_session(KEY, source="desktop", model="test-model", cwd=stored)
        resp = server.handle_request(
            {"id": "1", "method": "session.resume", "params": {"session_id": KEY, "profile": "austin"}}
        )
        assert "error" not in resp, resp
        return server._sessions[resp["result"]["session_id"]]

    run.db = db
    yield run
    with server._sessions_lock:
        for sid in [s for s in server._sessions if s not in known]:
            server._sessions.pop(sid, None)
    db.close()


@pytest.mark.parametrize("stored", ["root", "sibling", "install", None])
def test_resume_lands_in_the_profiles_own_home(resume, profile, stored):
    root = server.get_hermes_home()
    sibling = root / "profiles" / "jordan"
    sibling.mkdir(parents=True, exist_ok=True)
    install = server.Path(server.__file__).resolve().parent.parent
    path = {"root": str(root), "sibling": str(sibling), "install": str(install), None: None}[stored]

    live = resume(path)

    assert live["cwd"] == str(profile)
    assert live["explicit_cwd"] is True
    assert server._terminal_task_cwd(live) == str(profile)

    # The agent build's hydration repairs the row, so the sidebar and later resumes agree with the live session.
    sid = next(s for s, rec in server._sessions.items() if rec is live)
    server._hydrate_session_cwd(sid, KEY, resume.db, str(profile))
    assert resume.db.get_session(KEY)["cwd"] == str(profile)


def test_resume_keeps_a_folder_chosen_for_the_profile(resume, profile, tmp_path):
    inside = profile / "workspace"
    inside.mkdir()
    outside = tmp_path / "proj"
    outside.mkdir()

    live = resume(str(inside))

    assert live["cwd"] == str(inside)
    assert server._resumable_stored_cwd(str(outside), profile) == str(outside)
    assert server._resumable_stored_cwd(str(profile), profile) == str(profile)
    # A workspace a user keeps under the Hermes root (not Hermes's own dirs) is not another profile's.
    under_root = server.get_hermes_home() / "workspace"
    under_root.mkdir()
    assert server._resumable_stored_cwd(str(under_root), profile) == str(under_root)


def test_launch_profile_resume_is_unchanged():
    """The launch profile (``profile_home`` None) keeps main's rule: its own root is a valid stored workspace."""
    root = str(server.get_hermes_home())
    assert server._resumable_stored_cwd(root, None) == root
