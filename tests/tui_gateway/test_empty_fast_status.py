"""``config.set fast status`` reports "normal" for an explicit-normal pin.

A session pinned to normal carries ``""`` (not ``None``) in ``create_service_tier_override``
and, once built, in ``agent.service_tier`` — the status branch echoed that ``""`` back.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tui_gateway.server as server


@pytest.mark.parametrize("tier,expected", [("", "normal"), (None, "normal"), ("priority", "fast")])
@pytest.mark.parametrize("built", [True, False])
def test_fast_status_reports_tier_without_writing_config(tier, expected, built):
    session = ({"agent": SimpleNamespace(service_tier=tier)} if built
               else {"create_service_tier_override": tier})
    with patch.dict(server._sessions, {"s-fast": session}, clear=False), \
            patch.object(server, "_load_service_tier", return_value=tier), \
            patch.object(server, "_write_config_key") as write_key:
        res = server._methods["config.set"]("rid", {"session_id": "s-fast", "key": "fast", "value": "status"})
    assert res["result"]["value"] == expected
    write_key.assert_not_called()
    if built:
        assert session["agent"].service_tier == tier
    else:
        assert session["create_service_tier_override"] == tier
