"""Regression tests for #49629 — HTTP MCP timeout must be a timedelta.

The mcp SDK's ``streamablehttp_client`` (notably the deprecated pre-1.24.0
API) expects ``timeout`` as a ``timedelta`` and calls ``.seconds`` on it.
Hermes previously passed a raw ``float``, so every ``url:`` MCP connection
crashed with ``'float' object has no attribute 'seconds'``.
"""

from __future__ import annotations

from datetime import timedelta

from tools.mcp_tool import _as_timeout_timedelta


def test_float_seconds_coerced_to_timedelta():
    assert _as_timeout_timedelta(60.0) == timedelta(seconds=60)


def test_int_seconds_coerced_to_timedelta():
    assert _as_timeout_timedelta(30) == timedelta(seconds=30)


def test_existing_timedelta_passthrough():
    td = timedelta(seconds=120)
    assert _as_timeout_timedelta(td) is td


def test_result_is_timedelta_not_float():
    result = _as_timeout_timedelta(45)
    assert isinstance(result, timedelta)
    # The crash was `.seconds` on a float; confirm the attribute now exists.
    assert result.seconds == 45
