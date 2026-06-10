from __future__ import annotations

import model_tools
from tools.browser_tool import BROWSER_TOOL_SCHEMAS
from tools.registry import invalidate_check_fn_cache, registry


def _tool_desc(defs: list[dict], name: str) -> str:
    for tool_def in defs:
        fn = tool_def.get("function", {})
        if fn.get("name") == name:
            return fn.get("description", "")
    raise AssertionError(f"tool {name!r} not found")


def test_browser_navigate_schema_mentions_explicit_browser_and_local_page_cases() -> None:
    desc = next(s["description"] for s in BROWSER_TOOL_SCHEMAS if s["name"] == "browser_navigate")
    assert "use the browser/Chromium" in desc
    assert "inspect console errors" in desc
    assert "localhost, 127.0.0.1, file://" in desc


def test_browser_navigate_schema_drops_web_tool_mentions_when_web_tools_unavailable(monkeypatch) -> None:
    model_tools._tool_defs_cache.clear()
    invalidate_check_fn_cache()

    touched = []
    for name in ("browser_navigate", "browser_snapshot", "browser_click", "browser_type",
                 "browser_scroll", "browser_back", "browser_press", "browser_get_images",
                 "browser_vision", "browser_console", "browser_cdp", "browser_dialog"):
        entry = registry.get_entry(name)
        assert entry is not None, f"missing registry entry for {name}"
        touched.append((entry, entry.check_fn))
        monkeypatch.setattr(entry, "check_fn", lambda: True)

    defs = model_tools.get_tool_definitions(
        enabled_toolsets=["browser"],
        disabled_toolsets=["web"],
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )
    desc = _tool_desc(defs, "browser_navigate")
    assert "web_search" not in desc
    assert "web_extract" not in desc
