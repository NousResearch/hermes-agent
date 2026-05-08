from __future__ import annotations

from unittest.mock import patch

import model_tools


def _browser_description(available_names: set[str]) -> str:
    model_tools._clear_tool_defs_cache()

    schemas = []
    for name in sorted(available_names):
        if name == "browser_navigate":
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "browser_navigate",
                        "description": (
                            "Navigate to a URL in the browser. "
                            "For primarily textual web pages — prefer available lightweight text retrieval tools first. "
                            "Use browser tools only when interaction or rendering is materially needed."
                        ),
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            )
        else:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": f"{name} description",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            )

    with patch.object(model_tools.registry, "get_definitions", return_value=schemas), patch(
        "model_tools.sanitize_tool_schemas", side_effect=lambda tools: tools, create=True
    ):
        tools = model_tools.get_tool_definitions(enabled_toolsets=["browser"], quiet_mode=True)

    return next(
        tool["function"]["description"]
        for tool in tools
        if tool["function"]["name"] == "browser_navigate"
    )


def test_browser_navigate_static_description_avoids_unavailable_tool_names():
    desc = _browser_description({"browser_navigate"})

    assert "prefer available lightweight text retrieval tools first" in desc
    assert "curl via the terminal tool" not in desc
    assert "web_extract" not in desc
    assert "mcporter_call" not in desc


def test_browser_navigate_mentions_terminal_only_when_available():
    desc = _browser_description({"browser_navigate", "terminal"})

    assert "curl via the terminal tool" in desc
    assert "web_extract" not in desc
    assert "mcporter_call" not in desc


def test_browser_navigate_mentions_mcporter_only_when_available():
    desc = _browser_description({"browser_navigate", "mcporter_call"})

    assert "mcporter_call when a configured mcporter retrieval tool" in desc
    assert "curl via the terminal tool" not in desc


def test_browser_navigate_mentions_all_available_retrieval_tools():
    desc = _browser_description(
        {"browser_navigate", "terminal", "web_extract", "mcporter_call"}
    )

    assert "curl via the terminal tool" in desc
    assert "web_extract" in desc
    assert "mcporter_call when a configured mcporter retrieval tool" in desc
