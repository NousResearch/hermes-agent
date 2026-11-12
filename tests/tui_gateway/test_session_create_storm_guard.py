"""``session.create`` refuses a sustained create storm before anything is minted.

Regression for the 2026-10-08 incident: a client stuck in a create→drop cycle
minted ~3.4k sessions in one evening, and each create eagerly built a full agent
plus memory provider (~263M provider tokens) with no turn ever run. The guard is
a per-profile budget over a sliding window, so a normal create rate is untouched.
"""

import pytest


@pytest.fixture
def _create(monkeypatch, tmp_path):
    monkeypatch.setattr("hermes_cli.banner.prefetch_update_check", lambda: None)
    from tui_gateway import server

    (tmp_path / "config.yaml").write_text("model:\n  default: claude-opus-5\n  provider: anthropic\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_load_cfg", dict)
    monkeypatch.setattr(server, "_profile_home", lambda *a: None)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *a: None)
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda *a: None)
    # The budget is module state shared with the server namespace: start each test clean.
    server._create_storm_windows.clear()
    server._create_storm_warned.clear()
    return lambda params: (server._methods["session.create"]("r1", {"cols": 80, **params}), server._sessions)


def test_create_storm_is_refused_without_minting_a_session_or_building_an_agent(_create):
    from tui_gateway import server

    built = []
    server._schedule_agent_build = built.append
    budget = server._CREATE_STORM_MAX_PER_WINDOW

    responses = [_create({}) for _ in range(budget)]
    assert all("error" not in response for response, _ in responses), responses
    sessions = responses[-1][1]
    assert len(sessions) == budget
    assert len(built) == budget

    response, sessions = _create({})

    assert response["error"]["code"] == 4035
    # The refused create mints nothing and therefore builds no agent: the whole
    # point is that the storm's cost stops here.
    assert len(sessions) == budget
    assert len(built) == budget


def test_storm_budget_is_per_profile_and_leaves_a_normal_rate_alone():
    from tui_gateway import server

    server._create_storm_windows.clear()
    server._create_storm_warned.clear()
    budget = server._CREATE_STORM_MAX_PER_WINDOW

    for _ in range(budget):
        assert server._create_storm_budget_exceeded("/homes/a") is False

    assert server._create_storm_budget_exceeded("/homes/a") is True
    # A quiet profile is not punished for a neighbour's storm.
    assert server._create_storm_budget_exceeded("/homes/b") is False
