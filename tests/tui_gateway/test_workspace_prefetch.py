"""Deferred builds start workspace prefetch after profile binding, ahead of MCP."""

from types import SimpleNamespace

import pytest

from agent import runtime_cwd, workspace_prefetch
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from tests.tui_gateway.test_session_db_ownership_teardown import _run_build, _session
from tui_gateway import server

pytest_plugins = ["tests.tui_gateway.test_session_db_ownership_teardown"]


@pytest.mark.parametrize("source", ["tui", "desktop"])
@pytest.mark.parametrize("state", ["fresh", "resumed", "raising"])
def test_deferred_build_prefetch(build_env, registered, monkeypatch, tmp_path, source, state):
    repo = tmp_path / "workspace"
    repo.mkdir()
    (repo / "pyproject.toml").write_text('[project]\nname = "prefetch-fixture"\n')
    monkeypatch.setattr(server, "set_hermes_home_override", set_hermes_home_override)
    monkeypatch.setattr(server, "reset_hermes_home_override", reset_hermes_home_override)
    # Restore real cwd binding: the shared DB-ownership fixture intentionally stubs it.
    def bind_cwd(key, cwd=None):
        return runtime_cwd.set_session_cwd(cwd)

    monkeypatch.setattr(server, "_set_session_context", bind_cwd)
    monkeypatch.setattr(server, "_clear_session_context", runtime_cwd.reset_session_cwd)
    monkeypatch.setattr(server, "_session_source", lambda current: current.get("source"))
    monkeypatch.setattr(server, "_wire_session_agent", lambda *a: False)
    monkeypatch.setattr(server, "_announce_built_agent", lambda *a: None)
    order = []
    observed = []
    start = workspace_prefetch.start_workspace_snapshot_prefetch

    def prefetch(**kwargs):
        order.append("prefetch")
        observed.append((str(get_hermes_home()), bool(build_env.opened),
                         runtime_cwd.resolve_context_cwd(), kwargs))
        if state == "raising":
            raise RuntimeError("prefetch startup failed")
        return start(**kwargs)

    monkeypatch.setattr(workspace_prefetch, "start_workspace_snapshot_prefetch", prefetch)
    monkeypatch.setattr("tui_gateway.entry.ensure_mcp_discovery_started", lambda: order.append("mcp"))

    def make_agent(*args, session_db=None, **kwargs):
        order.append("agent")
        return SimpleNamespace(_session_db=session_db, _owns_session_db=False)

    monkeypatch.setattr(server, "_make_agent", make_agent)
    session = _session(build_env.profile_home)
    session.update(cwd=str(repo), source=source)
    if state == "resumed":
        session["resume_session_id"] = "stored-session"
    registered("prefetch-session", session)

    _run_build("prefetch-session", session)

    assert not session.get("agent_error")
    agent = session["agent"]
    assert order == (["mcp", "agent"] if state == "resumed" else ["prefetch", "mcp", "agent"])
    assert observed == ([] if state == "resumed" else [
        (build_env.profile_home, True, repo, {"platform": source})])
    if state == "fresh":
        key, future, _built_at = agent._workspace_snapshot_prefetch
        assert key == str(repo)
        assert "pyproject.toml" in future.result(timeout=15)
    else:
        assert getattr(agent, "_workspace_snapshot_prefetch", None) is None
