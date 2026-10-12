"""A live-session reattach acknowledges the owning profile in ``info.profile_name``.

``session.create`` and the cold/lazy resume paths report ``info.profile_name``; the live-reuse path
(``_live_session_payload`` -> ``_fallback_session_info``) did not, so a client that checks the profile
acknowledgement (the browser controller resuming the chat WS session) read "none".

``_fallback_session_info`` also resolves the session's not-yet-built model route; these tests stub that
out so only the profile acknowledgement is under test.
"""

from __future__ import annotations

from tui_gateway import server


def _stub_session_info_deps(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(server, "_lazy_info_route", lambda _session, _override: {})
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda _cwd: None)
    monkeypatch.setattr(server.git_probe, "branch", lambda _cwd: "")


def test_fallback_session_info_names_the_owning_profile(monkeypatch, tmp_path):
    home = tmp_path / "profiles" / "alpha"
    home.mkdir(parents=True)
    monkeypatch.setattr(server, "profile_name_for_home", lambda h: "alpha" if h == home else None)
    _stub_session_info_deps(monkeypatch, tmp_path)

    assert server._fallback_session_info({"profile_home": home})["profile_name"] == "alpha"


def test_fallback_session_info_launch_profile_without_home(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_current_profile_name", lambda: "launch")
    _stub_session_info_deps(monkeypatch, tmp_path)

    assert server._fallback_session_info({})["profile_name"] == "launch"
