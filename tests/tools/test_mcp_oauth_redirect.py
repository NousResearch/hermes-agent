from __future__ import annotations

import json
from unittest.mock import patch

import tools.mcp_oauth as mcp_oauth


def test_configure_callback_port_reuses_stored_client_redirect_port(tmp_path):
    storage = mcp_oauth.HermesTokenStorage("worldmonitor", hermes_home=tmp_path)
    client_path = storage._client_info_path()
    client_path.parent.mkdir(parents=True, exist_ok=True)
    client_path.write_text(
        json.dumps(
            {
                "redirect_uris": ["http://127.0.0.1:59061/callback"],
                "client_id": "test-client",
            }
        ),
        encoding="utf-8",
    )

    cfg: dict = {"redirect_port": 0}
    with patch.object(mcp_oauth, "_reserve_callback_port", return_value=99999) as reserve:
        port = mcp_oauth._configure_callback_port(cfg, storage)

    assert port == 59061
    assert cfg["_resolved_port"] == 59061
    reserve.assert_not_called()


def test_configure_callback_port_picks_free_port_without_client_info(tmp_path):
    storage = mcp_oauth.HermesTokenStorage("fresh-server", hermes_home=tmp_path)
    cfg: dict = {"redirect_port": 0}

    with patch.object(mcp_oauth, "_reserve_callback_port", return_value=48123) as reserve:
        port = mcp_oauth._configure_callback_port(cfg, storage)

    assert port == 48123
    assert cfg["_resolved_port"] == 48123
    reserve.assert_called_once_with()
