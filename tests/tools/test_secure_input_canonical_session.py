"""Ephemeral input on the canonical local session authority."""
import json
from types import SimpleNamespace

import pytest

from agent.secure_input_broker import clear_all
from tools.secure_input_tool import (
    check_secure_input_requirements, request_secure_input, set_secure_input_callback,
)


@pytest.fixture(autouse=True)
def isolated_broker():
    clear_all()
    set_secure_input_callback(None)
    yield
    set_secure_input_callback(None)
    clear_all()


@pytest.mark.parametrize("mutation", ["none", "close", "regenerate", "interrupt"])
def test_canonical_owner_scope_is_ephemeral_and_generation_bound(monkeypatch, mutation):
    from gateway.config import Platform
    from gateway.session_secret_prompt import secret_prompt_scope
    from tests.gateway.test_local_viewer_secret_prompt import _turn
    from hermes_cli import config
    turn, _peer = _turn(Platform.LOCAL)
    authority, sid, generation = turn._approval_owner
    row = {"runtime_generation": generation}
    authority.db = SimpleNamespace(get_session=lambda _sid: row)
    agent = SimpleNamespace(_interrupt_requested=False)
    monkeypatch.setattr(config, "save_env_value_secure", lambda *a, **k: pytest.fail("ephemeral input must not persist"))
    requests = []

    def answer(events, session_id, method, params, **kwargs):
        requests.append((session_id, method, params))
        if mutation == "close":
            authority.sessions.pop(sid)
        elif mutation == "regenerate":
            row["runtime_generation"] += 1
        elif mutation == "interrupt":
            agent._interrupt_requested = True
        return "synthetic-canonical-value"

    monkeypatch.setattr("gateway.session_server_requests.ask_viewer", answer)
    with secret_prompt_scope(turn, agent):
        result = json.loads(request_secure_input(purpose="canonical-test"))
        assert result["success"] is (mutation == "none")
        assert "synthetic-canonical-value" not in json.dumps(result)
    assert len(requests) == 1 and requests[0][:2] == (sid, "secret")
    assert not check_secure_input_requirements()


def test_messaging_turn_cannot_borrow_outer_secure_input_callback(monkeypatch):
    from gateway.config import Platform
    from gateway.session_secret_prompt import secret_prompt_scope
    from tests.gateway.test_local_viewer_secret_prompt import _turn
    set_secure_input_callback(lambda *args: pytest.fail("another surface's callback must not run"))
    turn, _peer = _turn(Platform.TELEGRAM)
    with secret_prompt_scope(turn, SimpleNamespace()):
        assert not json.loads(request_secure_input(purpose="test"))["success"]
    assert check_secure_input_requirements()
