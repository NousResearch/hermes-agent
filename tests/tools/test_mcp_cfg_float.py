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
