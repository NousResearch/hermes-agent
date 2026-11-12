"""``groups.capabilities`` names the first missing lender piece, like the gateway's ``room_link``."""

from __future__ import annotations

import pytest

import hermes_agent.tui_gateway.server as srv
from hermes_agent.tui_gateway import methods_groups


@pytest.fixture
def home(tmp_path, monkeypatch):
    class DurableRunStore:
        durable = True

    path = tmp_path / ".hermes"
    path.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(path))
    monkeypatch.setenv("API_SERVER_KEY", "gateway-api-key-1234567890")
    monkeypatch.setattr(srv, "_run_idempotency_store", DurableRunStore(), raising=False)
    methods_groups.stop_hosted_room_service(timeout=1.0)
    methods_groups.start_hosted_room_service()
    yield path
    methods_groups.stop_hosted_room_service(timeout=1.0)


def test_approvals_off_is_a_policy_refusal_not_a_missing_room_key(home):
    # A remote turn never runs without approval prompts, so the lender's operator must be told to
    # turn approvals on, not to repair a room key that exists (gateway room_link says the same).
    (home / "config.yaml").write_text("approvals:\n  mode: 'off'\n")
    result = srv._methods["groups.capabilities"](1, {})["result"]
    assert result["room_link"] == {"enabled": False, "reason": "execution_policy_unsupported"}
