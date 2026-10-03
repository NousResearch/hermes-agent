"""Model arguments cannot create an unbounded reaction menu."""

import json

import pytest

from tools.reaction_menu_tool import present_menu_tool


@pytest.mark.parametrize("change", [
    {"prompt": "x" * 501}, {"context_id": "x" * 129}, {"options": []},
    {"options": [{"emoji": "✅", "label": "Route", "payload": "Go"}] * 6},
    {"options": [{"emoji": "✅", "label": "Route", "payload": "Go"}] * 2},
    {"options": [{"emoji": "x" * 33, "label": "Route", "payload": "Go"}]},
    {"options": [{"emoji": "✅", "label": "x" * 121, "payload": "Go"}]},
    {"options": [{"emoji": "✅", "label": "Route", "payload": "x" * 2001}]},
    {"options": [{"emoji": "✅", "label": "Route", "payload": 1}]},
    {"options": [{"emoji": "✅", "label": "Route", "payload": "Go", "terminal": True}]},
])
def test_menu_rejects_unbounded_or_unsupported_arguments(change):
    delivered = []

    def deliver(menu):
        delivered.append(menu)
        return True

    args = {"prompt": "Choose", "options": [{"emoji": "✅", "label": "Route", "payload": "Go"}], **change}
    result = json.loads(present_menu_tool(**args, callback=deliver))
    assert set(result) == {"error"}
    assert delivered == []
