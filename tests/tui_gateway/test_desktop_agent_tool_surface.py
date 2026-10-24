"""Desktop-only tools must be available before the gateway snapshots an agent schema."""

from tools import desktop_ui
from tools.registry import invalidate_check_fn_cache, registry
from tui_gateway import server


def test_desktop_surface_wires_emitter_and_rechecks_cached_tool(monkeypatch):
    tool_name = "test_desktop_player_surface_probe"
    desktop_ui.set_emitter(None)
    invalidate_check_fn_cache()
    registry.register(
        name=tool_name,
        toolset="desktop_ui",
        schema={"name": tool_name, "description": "probe", "parameters": {"type": "object"}},
        handler=lambda _args, **_kwargs: "{}",
        check_fn=desktop_ui.available,
    )
    monkeypatch.setattr(server, "_wire_desktop_sinks", lambda: desktop_ui.set_emitter(lambda *_: None))
    try:
        assert registry.get_definitions({tool_name}, quiet=True) == []
        server._prepare_desktop_tool_surface("desktop")
        assert desktop_ui.available()
        assert [row["function"]["name"] for row in registry.get_definitions({tool_name}, quiet=True)] == [tool_name]
    finally:
        registry.deregister(tool_name)
        invalidate_check_fn_cache()
        desktop_ui.set_emitter(None)


def test_non_desktop_agents_do_not_wire_desktop_surface(monkeypatch):
    calls = []
    monkeypatch.setattr(server, "_wire_desktop_sinks", lambda: calls.append(True))
    server._prepare_desktop_tool_surface("tui")
    assert calls == []
