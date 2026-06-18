"""Focused clarify availability/dispatch tests.

This file intentionally keeps the short historical name ``test_clarify.py`` so
focused commands that target it exercise the real ``tools/clarify_tool.py``
implementation instead of failing with "file not found".
"""

import json

from tools.registry import discover_builtin_tools, registry


def test_clarify_tool_is_discoverable_and_registered():
    imported = discover_builtin_tools()

    assert "tools.clarify_tool" in imported
    entry = registry.get_entry("clarify")
    assert entry is not None
    assert entry.toolset == "clarify"
    assert entry.check_fn() is True


def test_clarify_registry_handler_uses_callback_and_returns_response():
    import tools.clarify_tool  # noqa: F401 - ensure registry side effect has run

    entry = registry.get_entry("clarify")
    assert entry is not None

    result = json.loads(
        entry.handler(
            {"question": "Pick?", "choices": ["one", "two"]},
            callback=lambda question, choices: choices[1],
        )
    )

    assert result["question"] == "Pick?"
    assert result["choices_offered"] == ["one", "two"]
    assert result["user_response"] == "two"


def test_clarify_registry_handler_reports_unavailable_without_callback():
    import tools.clarify_tool  # noqa: F401 - ensure registry side effect has run

    entry = registry.get_entry("clarify")
    assert entry is not None

    result = json.loads(entry.handler({"question": "Pick?"}))

    assert "error" in result
    assert "not available" in result["error"]
