"""MCP embedder seams: the stdio child learns its Hermes home, and plugins may transform the
server map (before the suspicious-server filter) and each child's env."""

from pathlib import Path

import pytest

from hermes_cli.plugins import get_plugin_manager
from tools.mcp_tool_config import _build_safe_env, _load_mcp_config


@pytest.fixture
def hooks(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # before the (home-scoped) manager is fetched
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    events: list[tuple[str, dict]] = []

    def register(name, fn):
        def _cb(**kw):
            events.append((name, kw))
            return fn(**kw)
        mgr._hooks.setdefault(name, []).append(_cb)

    try:
        yield register, events
    finally:
        mgr._hooks = saved


def test_stdio_child_env_carries_hermes_home_under_a_redirected_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HOME", str(tmp_path / "elsewhere"))
    env = _build_safe_env(None)
    assert Path(env["HERMES_HOME"]) == tmp_path / "hermes"
    assert _build_safe_env({"HERMES_HOME": "explicit"})["HERMES_HOME"] == "explicit"


def test_child_env_hook_fires_once_per_spawn_and_replaces_env(tmp_path, monkeypatch, hooks):
    register, events = hooks
    register("transform_mcp_child_env", lambda server_name, env, **_: {**env, "SERVER": server_name})
    env = _build_safe_env({"A": "1"}, server_name="docs")
    assert env["SERVER"] == "docs" and env["A"] == "1"
    assert [name for name, _ in events] == ["transform_mcp_child_env"]


def test_servers_hook_may_drop_a_server_before_load(tmp_path, monkeypatch, hooks):
    register, events = hooks
    (tmp_path / "config.yaml").write_text(
        "mcp_servers:\n  keep:\n    command: keep-server\n  drop:\n    command: drop-server\n",
        encoding="utf-8")
    register("transform_mcp_servers", lambda servers, **_: {k: v for k, v in servers.items() if k != "drop"})
    servers = _load_mcp_config()
    assert "keep" in servers and "drop" not in servers
    assert [name for name, _ in events] == ["transform_mcp_servers"]
