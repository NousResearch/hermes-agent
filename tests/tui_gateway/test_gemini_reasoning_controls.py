"""AI Studio selections cross the real gateway boundary without paid requests."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tui_gateway.server as server

pytestmark = pytest.mark.usefixtures("gemini_reasoning_catalog")


def make_session(model="gemini-2.5-flash"):
    agent = SimpleNamespace(model=model, provider="gemini", session_id="test-key",
                            service_tier=None, reasoning_config={"enabled": True},
                            _primary_runtime={"provider": "gemini", "model": model})
    return {"session_key": "test-key", "agent": agent}


@pytest.mark.parametrize("effort", ["auto", "budget:-1", "budget:4096", "none"])
def test_reasoning_round_trip_and_primary_runtime(effort):
    from hermes_constants import parse_reasoning_effort
    session = make_session()
    with patch.dict(server._sessions, {"studio-test": session}), \
            patch.object(server, "_write_config_key") as write, \
            patch.object(server, "_persist_live_session_runtime") as persist, \
            patch.object(server, "_emit"):
        response = server._methods["config.set"]("r", {
            "key": "reasoning", "session_id": "studio-test", "value": effort})
        assert response["result"]["value"] == effort
        snapshot = server._session_info(session["agent"])
        assert snapshot["reasoning_effort"] == effort
        assert session["create_reasoning_override"] == parse_reasoning_effort(effort)
        assert session["agent"]._primary_runtime["reasoning_config"] == parse_reasoning_effort(effort)
        persist.assert_called_once()
        write.assert_not_called()


@pytest.mark.parametrize("effort", ["budget:24577", "high", "budget:0"])
def test_invalid_budget_or_effort_does_not_mutate_session(effort):
    session = make_session()
    before = dict(session["agent"].reasoning_config)
    with patch.dict(server._sessions, {"studio-test": session}), \
            patch.object(server, "_write_config_key") as write, \
            patch.object(server, "_persist_live_session_runtime") as persist:
        response = server._methods["config.set"]("r", {
            "key": "reasoning", "session_id": "studio-test", "value": effort})
    assert response["error"]["code"] == 4002
    assert session["agent"].reasoning_config == before
    assert "create_reasoning_override" not in session
    write.assert_not_called()
    persist.assert_not_called()


def test_mandatory_thinking_cannot_be_disabled():
    session = make_session("gemini-3.8-flash")
    with patch.dict(server._sessions, {"studio-test": session}):
        response = server._methods["config.set"]("r", {
            "key": "reasoning", "session_id": "studio-test", "value": "none"})
    assert response["error"]["code"] == 4002
    assert session["agent"].reasoning_config == {"enabled": True}


def test_running_turn_cannot_receive_controls_for_a_pending_different_model():
    session = make_session("gemini-3.8-flash")
    session.update(running=True, pending_model_switch={
        "display_model": "gemini-2.5-flash", "display_provider": "gemini"})
    with patch.dict(server._sessions, {"studio-test": session}):
        response = server._methods["config.set"]("r", {
            "key": "reasoning", "session_id": "studio-test", "value": "budget:4096"})
    assert response["error"]["code"] == 4009
    assert session["agent"].reasoning_config == {"enabled": True}
    assert "create_reasoning_override" not in session
