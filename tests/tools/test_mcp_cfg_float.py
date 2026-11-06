"""Regression tests for _cfg_float — malformed numeric MCP server config.

A hand-edited config value like ``connect_timeout: "60s"`` or
``keepalive_interval: null`` used to reach a bare ``float(...)`` and raise
``ValueError``/``TypeError``, taking down the connection / keepalive loop.
``_cfg_float`` must coerce valid values and fall back to the default for
malformed ones.
"""
from tools.mcp_tool import _cfg_float


def test_cfg_float_reads_valid_numeric():
    assert _cfg_float({"connect_timeout": 30}, "connect_timeout", 60) == 30.0
    assert _cfg_float({"connect_timeout": 12.5}, "connect_timeout", 60) == 12.5


def test_cfg_float_reads_numeric_string():
    assert _cfg_float({"connect_timeout": "45"}, "connect_timeout", 60) == 45.0


def test_cfg_float_missing_key_uses_default():
    assert _cfg_float({}, "connect_timeout", 60) == 60.0


def test_cfg_float_malformed_string_falls_back():
    # "60s" is not parseable as a float — must not raise.
    assert _cfg_float({"connect_timeout": "60s"}, "connect_timeout", 60) == 60.0


def test_cfg_float_none_value_falls_back():
    # YAML ``keepalive_interval: null`` -> None -> TypeError without the guard.
    assert _cfg_float({"keepalive_interval": None}, "keepalive_interval", 180) == 180.0


def test_stdio_handshake_uses_cfg_float_for_malformed_connect_timeout():
    """stdio initialize handshake path must tolerate connect_timeout: \"60s\" (#51331).

    After 1f6836cd81 the _run_stdio body bounded session.initialize() with
    connect_timeout. A bare float(config.get(...)) raises ValueError for
    hand-edited values like \"60s\"; route through _cfg_float instead.
    """
    import inspect

    from tools import mcp_tool as mcp_mod

    src = inspect.getsource(mcp_mod.MCPServerTask._run_stdio)
    assert "_cfg_float(" in src
    # bare float(config.get("connect_timeout"...)) must not remain
    assert 'float(\n                        config.get("connect_timeout"' not in src
    bad = {"connect_timeout": "60s"}
    resolved = _cfg_float(bad, "connect_timeout", mcp_mod._DEFAULT_CONNECT_TIMEOUT)
    assert resolved == float(mcp_mod._DEFAULT_CONNECT_TIMEOUT)
    good = {"connect_timeout": "12"}
    assert _cfg_float(good, "connect_timeout", mcp_mod._DEFAULT_CONNECT_TIMEOUT) == 12.0
