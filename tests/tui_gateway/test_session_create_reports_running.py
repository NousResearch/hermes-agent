"""``session.create`` reports ``running`` on the reply it hands the client.

Create answers out of the live registry record before any state.db row exists, and the reply used to
omit the liveness key every other attach path reports. Clients gate their turn controls on it and
read an absent key as "this gateway can't report liveness at all": Conduit banners a brand-new chat
("Update this Hermes gateway") and hides send/stop on a session that is merely idle.

The idempotency retry matters as much as the fresh mint — a retry can hand back a session whose turn
is already in flight, and the client must attach to that turn instead of starting a second one.

Both create-shaped replies are minted by the same builder, so `session.branch_stored` needs the field
declared too: the result models are `extra="forbid"`, and an undeclared key raises ContractViolation
on the branch reply alone.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def create(monkeypatch, tmp_path):
    monkeypatch.setattr("hermes_cli.banner.prefetch_update_check", lambda: None)
    from tui_gateway import server

    (tmp_path / "config.yaml").write_text(
        "model:\n  default: claude-opus-5\n  provider: anthropic\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_load_cfg", dict)
    monkeypatch.setattr(server, "_profile_home", lambda *a: None)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *a: None)
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda *a: None)
    return lambda params: server._methods["session.create"]("r1", {"cols": 80, **params})


def test_fresh_create_reports_the_idle_draft_as_not_running(create):
    out = create({})
    assert "error" not in out, out
    assert out["result"]["running"] is False
    assert out["result"]["info"]["lazy"] is True


def test_idempotency_retry_reports_the_turn_already_in_flight(create):
    first = create({"idempotency_key": "k-1"})
    assert "error" not in first, first
    sid = first["result"]["session_id"]
    assert first["result"]["running"] is False

    from tui_gateway import server
    server._sessions[sid]["running"] = True

    retry = create({"idempotency_key": "k-1"})
    assert "error" not in retry, retry
    assert retry["result"]["session_id"] == sid
    assert retry["result"]["running"] is True


def test_create_reports_a_child_run_on_an_agentless_watch_session(create, monkeypatch):
    """A Bot Chat is agent-less until upgraded: the subagent registry is then the only liveness
    signal, so create must consult it rather than answer a flat False."""
    from tui_gateway import server
    monkeypatch.setattr(server, "_child_run_active", lambda child_key, profile_home: True)
    out = create({})
    assert "error" not in out, out
    assert out["result"]["running"] is True


def test_both_create_shaped_results_declare_running():
    """`session.branch_stored` returns the same payload shape as `session.create`; declaring the
    field on only one of them fails closed with ContractViolation on the branch reply."""
    from tui_gateway.contracts.sessions import SessionBranchStoredResult, SessionCreateResult

    for model in (SessionCreateResult, SessionBranchStoredResult):
        assert model.model_fields["running"].annotation == (bool | None)
