"""Remote and lazy session status preservation through real gateway entry points."""
from types import SimpleNamespace

import pytest

from tests.tui_gateway.test_tui_gateway_server import _session
from tui_gateway import server


def test_session_status_reports_remote_reasoning_and_fast(monkeypatch):
    server._sessions["sid"] = _session(agent=None, running=False)
    server._sessions["sid"].update(
        {
            "agent": None,
            "_compute_host_active": True,
            "_metadata_mirror": {"model": "gpt-5", "provider": "openai"},
            "create_reasoning_override": {"enabled": True, "effort": "high"},
            "create_service_tier_override": "priority",
        }
    )

    class _DB:
        def get_session(self, _key):
            return {"started_at": 1_700_000_000}

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.status", "params": {"session_id": "sid"}}
        )
    finally:
        server._sessions.pop("sid", None)

    assert isinstance(resp, dict) and "result" in resp, resp
    out = resp["result"]["output"]
    assert "Reasoning: high" in out
    assert "Fast: Yes" in out



def test_fallback_session_info_reports_session_cwd_not_launch_dir(monkeypatch):
    """A lazily-resumed session must report ITS workspace, not the gateway's.

    ``_fallback_session_info`` used ``_default_session_cwd()`` — the directory the
    gateway process happened to start in — so the desktop Files pane painted the
    wrong project for any session resumed without a built agent (#71254).
    """
    monkeypatch.setattr(server, "_default_session_cwd", lambda: "/gateway/launch/dir")
    monkeypatch.setattr(server.git_probe, "branch", lambda cwd: "bb/feature")
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda cwd: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "gpt-5")

    info = server._fallback_session_info(
        {
            "agent": None,
            "cwd": "/projects/session-own-repo",
            "create_reasoning_override": {"enabled": True, "effort": "high"},
            "create_service_tier_override": "priority",
        }
    )

    assert info["cwd"] == "/projects/session-own-repo"
    assert info["branch"] == "bb/feature"
    assert info["reasoning_effort"] == "high"
    assert info["fast"] is True
    assert info["model"] == "gpt-5"



def test_fallback_session_info_uses_own_profile_default_and_preserves_live_model(tmp_path, monkeypatch):
    homes = [tmp_path / "profile-a", tmp_path / "profile-b"]
    models = ["profile-a-model", "profile-b-model"]
    for home, model in zip(homes, models):
        home.mkdir()
        (home / "config.yaml").write_text(f"model:\n  default: {model}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    monkeypatch.setattr(server, "_hermes_home", str(homes[0]))
    monkeypatch.setattr(server.git_probe, "branch", lambda cwd: "")
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda cwd: None)
    for home, expected in [(homes[0], models[0]), (homes[1], models[1]), (homes[0], models[0])]:
        session = {"agent": None, "profile_home": str(home), "cwd": str(tmp_path)}
        assert server._fallback_session_info(session)["model"] == expected
        session["_metadata_mirror"] = {"model": "live-model", "provider": "openai"}
        assert server._fallback_session_info(session)["model"] == "live-model"
        session["pending_model_switch"] = {"display_model": "next-model"}
        assert server._fallback_session_info(session)["model"] == "next-model"


@pytest.mark.parametrize("config,include_reasoning,host_effort,parent_effort", [
    (None, True, "", ""),
    ({}, True, "", ""),
    ({"enabled": False}, True, "none", "none"),
    ({"enabled": True, "effort": "medium"}, True, "medium", "medium"),
    (None, False, "", "high"),
])
def test_remote_reasoning_roundtrip_preserves_default(
    monkeypatch, config, include_reasoning, host_effort, parent_effort,
):
    """A serialized provider default clears an old creation pin; an absent field inherits it."""
    monkeypatch.setattr(server, "_get_db", lambda: None)
    pins = {
        "create_reasoning_override": {"enabled": True, "effort": "high"},
        "create_service_tier_override": "priority",
    }
    live = SimpleNamespace(
        model="gpt-5", provider="openai", reasoning_config=config, service_tier=None,
    )
    host_info = server._session_info(live, pins)
    assert host_info["reasoning_effort"] == host_effort
    assert host_info["service_tier"] == ""
    if not include_reasoning:
        host_info.pop("reasoning_effort")
    parent = _session(agent=None, running=False)
    parent.update(pins, agent=None, _compute_host_active=True, _metadata_mirror=host_info)
    assert parent["agent"] is None
    assert ("reasoning_effort" in parent["_metadata_mirror"]) is include_reasoning
    sid = "reasoning-roundtrip"
    monkeypatch.setitem(server._sessions, sid, parent)

    response = server.handle_request({
        "id": "reasoning", "method": "session.status", "params": {"session_id": sid},
    })
    assert "result" in response, response
    expected_line = f"Reasoning: {parent_effort or 'default'}"
    assert expected_line in response["result"]["output"].splitlines()
    assert "Fast: No" in response["result"]["output"].splitlines()
    info = server._fallback_session_info(parent)
    assert info["reasoning_effort"] == parent_effort
    assert (info["model"], info["provider"]) == ("gpt-5", "openai")
    assert info["service_tier"] == ""
