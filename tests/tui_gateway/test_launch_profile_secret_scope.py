"""Launch/default-profile turns must re-read .env via secret scope (#91147)."""

from pathlib import Path

from agent.secret_scope import get_secret, reset_secret_scope
from tui_gateway.server import (
    _install_session_secret_scope,
    _profile_home,
    _session_home,
)


def test_profile_home_is_none_for_launch_profile():
    """Unnamed/launch sessions store profile_home=None and skip the override."""
    assert _profile_home(None) is None
    assert _profile_home("") is None


def test_session_home_falls_back_to_launch_home(monkeypatch, tmp_path):
    monkeypatch.setattr("tui_gateway.server._hermes_home", tmp_path)
    assert _session_home({"profile_home": None}) == Path(tmp_path)
    assert _session_home({}) == Path(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    assert _session_home({"profile_home": str(other)}) == other


def test_install_session_secret_scope_rereads_env_without_process_reload(
    monkeypatch, tmp_path
):
    """Edits to .env are visible on the next turn even if os.environ is stale."""
    monkeypatch.setattr("tui_gateway.server._hermes_home", tmp_path)
    monkeypatch.delenv("HASS_TOKEN", raising=False)
    (tmp_path / ".env").write_text("HASS_TOKEN=added-after-start\n", encoding="utf-8")

    assert get_secret("HASS_TOKEN") in (None, "")

    token = _install_session_secret_scope({"profile_home": None})
    try:
        assert get_secret("HASS_TOKEN") == "added-after-start"
    finally:
        reset_secret_scope(token)

    assert get_secret("HASS_TOKEN") in (None, "")
