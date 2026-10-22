"""Model switches disclose an effort carried over without ``--reasoning``."""

from types import SimpleNamespace

import tui_gateway.server as server


def test_model_switch_warns_when_it_keeps_the_current_reasoning_effort(monkeypatch):
    agent = SimpleNamespace(
        model="old/model",
        provider="nous",
        base_url="",
        api_key="",
        api_mode="chat_completions",
        reasoning_config={"enabled": True, "effort": "max"},
        switch_model=lambda **_kw: None,
    )
    result = SimpleNamespace(
        success=True,
        new_model="new/model",
        target_provider="nous",
        base_url="",
        api_key="key",
        api_mode="chat_completions",
        warning_message="",
        model_info=None,
        error_message="",
        runtime_capabilities=None,
    )

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_kw: result)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_persist_live_session_system_prompt", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_append_model_switch_marker", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_emit_session_info", lambda *_a, **_kw: None)
    out = server._apply_model_switch("sid", {"agent": agent}, "new/model --provider nous")

    assert out["warning"] == "reasoning still max"


def test_deferred_model_switch_warns_about_its_carried_reasoning_effort():
    agent = SimpleNamespace(reasoning_config={"enabled": True, "effort": "high"})
    parsed = SimpleNamespace(model_input="new/model", explicit_provider="nous", reasoning_effort="")
    session = {"agent": agent}

    response = server._stash_pending_model_switch("1", "model", "new/model", session, True, parsed)

    assert response["result"]["warning"] == "reasoning still high"
    assert response["result"]["deferred"] is True


def test_explicit_reasoning_override_does_not_report_a_carried_effort():
    agent = SimpleNamespace(reasoning_config={"enabled": True, "effort": "high"})

    assert server._carried_reasoning_notice(agent, "low") == ""
