"""Saved tool opt-ins must reach the real schema without leaking across platforms."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


def _names(selection, disabled=None):
    from model_tools import get_tool_definitions

    return {
        row["function"]["name"]
        for row in get_tool_definitions(
            selection, disabled_toolsets=disabled, quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        if row["function"]["name"].startswith("kanban_")
    }


@pytest.mark.parametrize("surface", ["cli", "http", "rpc"])
def test_saved_opt_in_roundtrip_reaches_schema_and_board(surface, tmp_path, monkeypatch):
    """Exercise the real config writer, availability gate, skills gate and handler."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    from hermes_cli.config import load_config, save_config
    from hermes_cli.tools_config import _apply_toolset_change, _get_platform_tools
    from tools.registry import registry

    save_config({"platform_toolsets": {"cli": ["file"], "telegram": ["file"]}})

    def selected(platform="cli"):
        return sorted(_get_platform_tools(load_config(), platform, include_default_mcp_servers=False))

    before = _names(selected())  # Warm both cache layers before enabling.
    assert not before
    client = None
    if surface == "http":
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from hermes_cli.web_routers.tools import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

    def toggle(enabled):
        action = "enable" if enabled else "disable"
        if surface == "http":
            response = client.put("/api/tools/toolsets/kanban", json={"enabled": enabled})
            assert response.status_code == 200, response.text
            assert response.json()["enabled"] is enabled
        elif surface == "rpc":
            from tui_gateway.server import _methods

            response = _methods["tools.configure"]("kanban-opt-in", {"action": action, "names": ["kanban"]})
            assert "error" not in response, response
            assert "kanban" in response["result"]["changed"]
        else:
            _apply_toolset_change(load_config(), "cli", ["kanban"], action)

    try:
        toggle(True)
        enabled_names = _names(selected())
        assert {"kanban_list", "kanban_create", "kanban_complete"} <= enabled_names
        assert not _names(selected("telegram")), "CLI opt-in leaked to Telegram"
        assert "file" in selected()
        # A second profile in the same process must not borrow this grant or
        # poison the first profile's cached schema on return.
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        other_home = tmp_path / "profiles" / "observer"
        token = set_hermes_home_override(other_home)
        try:
            save_config({"platform_toolsets": {"cli": ["file"]}})
            assert not _names(selected())
        finally:
            reset_hermes_home_override(token)
        assert _names(selected()) == enabled_names
        from agent.skill_utils import _detect_kanban
        assert _detect_kanban(), "Saved opt-in still hides the Kanban playbook"
        result = json.loads(registry.dispatch("kanban_create", {"title": "opt-in roundtrip", "assignee": "default"}))
        assert result.get("ok"), result
        from hermes_cli.kanban_db_connect import connect_closing
        from hermes_cli.kanban_db import get_task
        with connect_closing() as conn:
            assert get_task(conn, result["task_id"]).title == "opt-in roundtrip"
        toggle(False)
        assert not _names(selected())
        assert not _detect_kanban()
        assert "file" in selected()
        assert enabled_names, "Changing config must not mutate an already-built schema"
    finally:
        if client is not None:
            client.close()


@pytest.mark.parametrize("legacy", [False, True])
def test_selection_is_scoped_and_preserves_worker_and_deny_boundaries(legacy, tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    from hermes_cli.config import load_config, save_config
    from hermes_cli.tools_config import _get_platform_tools
    from agent.delegation_context import delegated_child_context

    save_config({"toolsets": ["kanban"] if legacy else [], "platform_toolsets": {"telegram": ["kanban"]}})
    # The same profile concurrently builds an explicitly opted-in schema and
    # an all/default schema. A platform grant must not become a cached global grant.
    with ThreadPoolExecutor(max_workers=2) as pool:
        named, broad = list(pool.map(_names, [["kanban"], ["hermes-cli"]]))
    assert "kanban_create" in named
    assert bool(broad) is legacy
    assert bool(_names(None)) is legacy
    assert bool(_names(["all"])) is legacy
    assert not _names(["kanban"], ["kanban"])
    assert not _names([])
    assert bool(_names(sorted(_get_platform_tools(load_config(), "cli")))) is legacy
    # An explicitly saved (non-empty) selection is authoritative over the legacy key.
    cfg = load_config()
    cfg["platform_toolsets"]["cli"] = ["file"]
    save_config(cfg)
    assert not _names(sorted(_get_platform_tools(load_config(), "cli")))

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    worker = _names(["file"])
    assert "kanban_complete" in worker
    assert "kanban_list" not in worker
    with delegated_child_context():
        assert not _names(["kanban"])
    assert "kanban_complete" in _names(["file"])
