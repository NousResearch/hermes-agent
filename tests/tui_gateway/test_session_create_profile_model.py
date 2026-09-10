"""``session.create`` reports the named profile's model, not the launch-context model.

Why: the desktop's app-global remote mode points every profile at one backend, so
``session.create`` carries a ``profile``. The handler resolves the profile home
correctly (and the AIAgent build re-binds it), but the immediate provisional
``info.model`` it returns fell back to the launch-context ``_resolve_model()``
when no explicit model override was supplied. The composer therefore painted the
launch profile's model for a session that would run under the named profile's.

This is the same bug class as ``terminal.cwd`` (issue #40334): the process-global
launch value belongs to the *launch* profile, so a non-launch profile's value must
be read from ITS config.yaml. Pinned here:

* a named profile with no explicit model override → ``info.model`` is that
  profile's configured model, not the launch-context model;
* the launch profile (no ``profile``) is unchanged → launch-context model;
* an explicit model override still wins over the profile's configured model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tui_gateway import server


@pytest.fixture()
def profile_env(monkeypatch, tmp_path):
    """A disposable launch home + a named profile home with a distinct configured
    model, driving the real ``session.create`` handler."""
    launch = tmp_path / "launch"
    named = tmp_path / "named"
    for home in (launch, named):
        home.mkdir()
    # The named profile's config names a model the launch context does not.
    (named / "config.yaml").write_text("model: named-profile-model\n")

    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(server, "_profile_home", lambda p: named if p == "work" else None)
    # Launch-context model: what the bug used to leak into a named-profile session.
    monkeypatch.setattr(server, "_resolve_model", lambda: "launch-context-model")
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda *a, **k: None)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_completion_cwd", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(server, "_default_session_cwd", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(server, "_load_show_reasoning", lambda: True)
    monkeypatch.setattr(server, "_load_tool_progress_mode", lambda: "all")
    monkeypatch.setattr(server, "_register_session_cwd", lambda *a, **k: None)
    known = set(server._sessions)
    yield {"launch": launch, "named": named}
    with server._sessions_lock:
        for sid in [s for s in server._sessions if s not in known]:
            server._sessions.pop(sid, None)


def _create(**params):
    """Drive the real ``session.create`` handler and return its result envelope."""
    resp = server.handle_request({"id": "1", "method": "session.create", "params": params})
    assert "error" not in resp, resp.get("error")
    return resp["result"]


class TestNamedProfileProvisionalModel:
    def test_named_profile_reports_its_own_model(self, profile_env):
        """A named profile with no explicit model override reports its configured model,
        not the launch-context model."""
        result = _create(profile="work", source="desktop")
        assert result["info"]["model"] == "named-profile-model"

    def test_launch_profile_unchanged(self, profile_env):
        """No ``profile`` → launch-context model (the pre-existing single-profile contract)."""
        result = _create(source="desktop")
        assert result["info"]["model"] == "launch-context-model"

    def test_explicit_model_override_still_wins(self, profile_env):
        """An explicit composer model beats the named profile's configured model."""
        result = _create(profile="work", source="desktop", model="explicit-model")
        assert result["info"]["model"] == "explicit-model"

    def test_named_profile_without_configured_model_falls_back(self, profile_env):
        """A named profile whose config has no model falls back to the launch-context model."""
        (profile_env["named"] / "config.yaml").write_text("terminal:\n  backend: local\n")
        result = _create(profile="work", source="desktop")
        assert result["info"]["model"] == "launch-context-model"
