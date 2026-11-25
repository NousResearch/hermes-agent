"""Typed timeline rows take precedence over legacy model-only marker sniffing."""
from copy import deepcopy

import pytest

from tui_gateway import server


@pytest.mark.parametrize("display_kind", ["internal_notification", "model_switch", "personality_switch"])
def test_typed_system_marker_survives_gateway_projection(display_kind):
    row = {
        "role": "user",
        "content": "[System: Background work finished.]",
        "display_kind": display_kind,
        "display_metadata": {"display_text": "Background work finished"},
        "_row_id": 17,
    }
    original = deepcopy(row)

    assert server._history_to_messages([row]) == [{
        "role": "user",
        "text": row["content"],
        "display_kind": display_kind,
        "display_metadata": row["display_metadata"],
        "row_id": 17,
    }]
    assert row == original


@pytest.mark.parametrize("display_kind", [None, "", "hidden"])
def test_model_only_system_markers_remain_hidden(display_kind):
    row = {"role": "user", "content": "[System: Model-only bookkeeping.]"}
    if display_kind is not None:
        row["display_kind"] = display_kind

    assert server._history_to_messages([row]) == []
