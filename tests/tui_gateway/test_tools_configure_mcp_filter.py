"""``tools.configure`` must not report an MCP tool toggle the runtime filter will not honour.

When an fnmatch glob in the server's active ``tools.include`` / ``tools.exclude`` list still matches
the tool after the exact-name edit, the RPC refuses, names the pattern, and saves nothing (no
config write, no session reset) instead of listing the target under ``changed``.
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("tools_cfg,action", [
    ({"include": ["create_*"]}, "disable"),
    ({"exclude": ["create_*"]}, "enable"),
], ids=["include-glob", "exclude-glob"])
def test_glob_held_toggle_is_refused_and_not_saved(tools_cfg, action):
    from hermes_cli.config import get_config_path, load_config, save_config
    from tui_gateway import server

    cfg = load_config()
    cfg["mcp_servers"] = {"github": {"command": "x", "tools": dict(tools_cfg)}}
    save_config(cfg)
    before = get_config_path().read_bytes()

    response = server._methods["tools.configure"](1, {"action": action, "names": ["github:create_issue"]})

    assert "'create_*'" in response["error"]["message"]
    assert get_config_path().read_bytes() == before
