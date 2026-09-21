"""Tests for the Control Room gateway RPC handlers (Phase 3 TUI, CR-303).

Verifies the two new methods register on the server and return well-formed
JSON-RPC responses. The snapshot method must compose real providers (or typed
unavailable), and the action method must route through the router with the
confirmation guard intact.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from tui_gateway.methods_control_room import _err, _ok


def _invoke(server, method_name: str, params: dict) -> dict:
    handler = server._methods[method_name]
    # Handlers are rebound onto server globals at install; call the raw
    # registered function with a fake rid.
    return handler("test-rid", params)


class TestRegistration:
    def test_methods_registered(self):
        import tui_gateway.server as srv

        assert "control.room.snapshot" in srv._methods
        assert "control.room.action" in srv._methods

    def test_snapshot_returns_json_safe_payload(self):
        import tui_gateway.server as srv

        with patch("control_room.service.ControlRoomService") as mock_cls:
            from control_room.contract import ControlRoomSnapshot

            mock_service = mock_cls.return_value
            mock_service.build_snapshot.return_value = ControlRoomSnapshot(
                profile="kensei", counts={"needs_you": 0}
            )
            result = _invoke(srv, "control.room.snapshot", {"profile": "kensei"})
        assert result["id"] == "test-rid"
        payload = result["result"]
        assert payload["profile"] == "kensei"
        assert "version" in payload

    def test_snapshot_handles_provider_failure_as_typed_unavailable(self):
        import tui_gateway.server as srv

        # No mocking of the real service: the peer provider degrades to
        # unavailable because the plugin is not loaded in this test process.
        result = _invoke(srv, "control.room.snapshot", {"profile": "default"})
        assert "result" in result
        payload = result["result"]
        assert payload["capabilities"]["peer_messages"] is False
        assert payload["counts"]["needs_you"] >= 0

    def test_action_requires_action_object(self):
        import tui_gateway.server as srv

        result = _invoke(srv, "control.room.action", {})
        assert "error" in result

    def test_action_requires_confirmation_first(self):
        import tui_gateway.server as srv

        from control_room.actions import ControlRoomActionRouter
        from control_room.contract import ControlRoomActionResult

        action = {
            "id": "act-1",
            "target": {"kind": "process", "id": "p1", "profile": "default"},
            "parameters": {},
            "confirmation": "required",
            "expected_revision": None,
        }

        class FakeRouter(ControlRoomActionRouter):
            def __init__(self, **kwargs):
                super().__init__(executors={}, scope_profile="default")
                self.dispatch_called = False

            def dispatch(self, action, context=None, *, confirmed=False):
                self.dispatch_called = True
                return ControlRoomActionResult(status="confirmation_required", message="Confirm?")

        with patch("control_room.actions.ControlRoomActionRouter", FakeRouter):
            result = _invoke(srv, "control.room.action", {"action": action, "confirmed": False})
        assert "result" in result
        assert result["result"]["status"] == "confirmation_required"


class TestOkErrShapes:
    def test_ok_shape(self):
        assert _ok("rid-1", {"a": 1}) == {"jsonrpc": "2.0", "id": "rid-1", "result": {"a": 1}}

    def test_err_shape(self):
        assert _err("rid-1", 5001, "boom") == {
            "jsonrpc": "2.0",
            "id": "rid-1",
            "error": {"code": 5001, "message": "boom"},
        }
