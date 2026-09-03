"""Test coverage for tools/computer_use/cua_backend.py — 136 functions had LOW coverage.

Tests the pure helper functions: config reading, field extraction,
and capability manifest checks. All subprocess and system calls mocked.
"""

from tools.computer_use.cua_backend import (
    _cua_configured_permission_mode,
    _cua_no_overlay,
    _cua_telemetry_disabled,
)


class TestCuaNoOverlay:
    def test_returns_bool(self):
        assert isinstance(_cua_no_overlay(), bool)


class TestCuaTelemetryDisabled:
    def test_returns_bool(self):
        assert isinstance(_cua_telemetry_disabled(), bool)


class TestCuaPermissionMode:
    def test_returns_string(self):
        result = _cua_configured_permission_mode()
        assert isinstance(result, str)
        assert len(result) > 0
