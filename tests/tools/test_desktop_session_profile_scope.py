"""Regression: desktop/serve sessions must scope persistent Docker by PROFILE.

On the desktop (``hermes serve --port 9119``) path a turn is bound to a
profile by ``tui_gateway.prompt_turn`` -> ``server._set_session_context``.
Unlike the gateway path (``gateway/run.py::_set_session_env``), which passes
``profile=...`` to ``set_session_vars``, the desktop path historically omitted
it. With ``HERMES_SESSION_PROFILE`` left empty, ``_current_session_profile()``
returns ``""`` and ``_resolve_container_task_id`` collapses a non-default
profile's session to the shared ``"default"`` env slot — so an ``eda`` session
silently reuses whichever profile seeded ``default`` first (here: ml-research),
landing its ``/workspace`` writes in the WRONG profile's data dir.

These tests reproduce that collapse through the REAL desktop seam
(``server._set_session_context``) and assert the env-slot key follows the
session's profile. RED before the fix (key == "default"), GREEN after.
"""

import pytest

from tools import terminal_tool


@pytest.fixture
def _docker_persistent(monkeypatch):
    """Persistent Docker => docker_profile_scoped is True, so the profile gate
    in _resolve_container_task_id is active."""
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")


def _register_fake_session(monkeypatch, profile_home, key="agent:desk:1"):
    """Point server._session_for_key at a synthetic session for *profile_home*."""
    import tui_gateway.server as server
    sess = {"profile_home": profile_home, "cwd": "/workspace"}
    monkeypatch.setattr(server, "_session_for_key", lambda k=key: sess)
    return key


def test_desktop_session_scopes_persistent_docker_to_its_profile(
    monkeypatch, tmp_path, _docker_persistent
):
    """An eda desktop session must key its env slot 'profile:eda', NOT 'default'."""
    import tui_gateway.server as server
    from gateway.session_context import clear_session_vars, get_session_env

    profile_home = str(tmp_path / "profiles" / "eda")
    import os
    os.makedirs(profile_home, exist_ok=True)
    key = _register_fake_session(monkeypatch, profile_home)

    tokens = server._set_session_context(key)
    try:
        # The session's profile must be visible to the terminal layer.
        assert get_session_env("HERMES_SESSION_PROFILE") == "eda"
        # ...and consequently the env-slot key is profile-scoped, not "default".
        assert terminal_tool._resolve_container_task_id(None) == "profile:eda"
    finally:
        clear_session_vars(tokens)


def test_desktop_sessions_of_distinct_profiles_do_not_share_a_slot(
    monkeypatch, tmp_path, _docker_persistent
):
    """Two profiles served by one process must not collapse onto one env slot —
    the exact cross-profile write leak from the incident."""
    import tui_gateway.server as server
    from gateway.session_context import clear_session_vars

    homes = {p: str(tmp_path / "profiles" / p) for p in ("eda", "ml-research")}
    import os
    for h in homes.values():
        os.makedirs(h, exist_ok=True)

    def run(profile):
        key = _register_fake_session(monkeypatch, homes[profile])
        tokens = server._set_session_context(key)
        try:
            return terminal_tool._resolve_container_task_id(None)
        finally:
            clear_session_vars(tokens)

    a = run("eda")
    b = run("ml-research")
    assert a == "profile:eda"
    assert b == "profile:ml-research"
    assert a != b


def test_desktop_default_profile_session_stays_default(
    monkeypatch, tmp_path, _docker_persistent
):
    """Control: a session with no profile (the default profile) must keep the
    shared 'default' slot — the fix must not over-correct."""
    import tui_gateway.server as server
    from gateway.session_context import clear_session_vars

    # A session whose profile_home is None (default profile, launched from the
    # top-level ~/.hermes) must still collapse to "default".
    import tui_gateway.server as srv
    monkeypatch.setattr(srv, "_session_for_key",
                        lambda k: {"profile_home": None, "cwd": "/workspace"})
    key = "agent:desk:default"
    tokens = server._set_session_context(key)
    try:
        assert terminal_tool._resolve_container_task_id(None) == "default"
    finally:
        clear_session_vars(tokens)