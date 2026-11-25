"""``set_theme``: the model hears what the desktop applied, never a success it assumed."""

import json

from tools.registry import discover_builtin_tools, registry


def _dispatch(args: dict, callback) -> dict:
    discover_builtin_tools()
    result = registry.dispatch("set_theme", args, callback=callback)
    assert isinstance(result, str)
    return json.loads(result)


def test_unknown_values_return_the_available_ids_without_reaching_the_renderer():
    calls = []

    accent = _dispatch({"accent": "chartreuse"}, calls.append)
    mode = _dispatch({"mode": "dim"}, calls.append)

    assert accent["success"] is False and "violet" in accent["available"] and "theme" in accent["available"]
    assert mode["success"] is False and mode["available"] == ["light", "dark", "system"]
    assert calls == []


def test_the_renderer_answer_is_the_result_for_values_and_pickers():
    sent = []

    def renderer(payload):
        sent.append(payload)
        return ({"success": True, "picked": {"layout": "terminal-deck"}} if "picker" in payload
                else {"success": False, "error": "Unknown layout", "available": ["sidebar-left"]})

    applied = _dispatch({"layout": " cockpit ", "accent": "#8A2BE2"}, renderer)
    picked = _dispatch({"picker": "layout"}, renderer)

    assert sent == [{"layout": "cockpit", "accent": "#8A2BE2"}, {"picker": "layout"}]
    assert applied == {"success": False, "error": "Unknown layout", "available": ["sidebar-left"]}
    assert picked == {"success": True, "picked": {"layout": "terminal-deck"}}
