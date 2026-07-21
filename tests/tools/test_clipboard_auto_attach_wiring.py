"""End-to-end regression for the clipboard auto-attach opt-out (#23984).

Pins the wiring of ``hermes_cli.clipboard.is_clipboard_auto_attach_enabled``
into the two automatic clipboard probe paths:

  * cli.py::_should_auto_attach_clipboard_image_on_paste
    (the CLI's bracketed-paste handler — empty paste → probe clipboard
     for an image and attach it).
  * tui_gateway/server.py::clipboard.paste RPC (auto path)
    (called by the TUI when an empty bracketed-paste arrives, marked
     as auto=True; the gateway must short-circuit the OS clipboard
     probe entirely so Ghostty's privacy prompt does not re-arm).

The unit-level contract for ``is_clipboard_auto_attach_enabled`` itself
lives in ``tests/hermes_cli/test_clipboard_auto_attach.py``; this file
only checks that the wiring at the two probe sites is correct and that
the bug reported in #23984 cannot regress silently.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# CLI: _should_auto_attach_clipboard_image_on_paste
# ---------------------------------------------------------------------------

class TestCliBracketedPasteAutoAttachGate:
    """The bracketed-paste handler must consult the user opt-out."""

    def test_text_paste_never_auto_attaches_regardless_of_gate(self):
        from cli import _should_auto_attach_clipboard_image_on_paste

        with patch(
            "hermes_cli.clipboard.is_clipboard_auto_attach_enabled",
            return_value=True,
        ):
            assert _should_auto_attach_clipboard_image_on_paste("hi") is False

        with patch(
            "hermes_cli.clipboard.is_clipboard_auto_attach_enabled",
            return_value=False,
        ):
            assert _should_auto_attach_clipboard_image_on_paste("hi") is False

    def test_empty_paste_with_gate_enabled_still_auto_attaches(self):
        """Default-enabled users keep the existing behaviour."""
        from cli import _should_auto_attach_clipboard_image_on_paste

        with patch(
            "hermes_cli.clipboard.is_clipboard_auto_attach_enabled",
            return_value=True,
        ):
            assert _should_auto_attach_clipboard_image_on_paste("") is True
            assert _should_auto_attach_clipboard_image_on_paste("   \n\t") is True

    def test_empty_paste_with_gate_disabled_short_circuits(self):
        """The whole point of #23984: opted-out users get zero probes."""
        from cli import _should_auto_attach_clipboard_image_on_paste

        with patch(
            "hermes_cli.clipboard.is_clipboard_auto_attach_enabled",
            return_value=False,
        ):
            assert _should_auto_attach_clipboard_image_on_paste("") is False
            assert _should_auto_attach_clipboard_image_on_paste("   ") is False
            assert _should_auto_attach_clipboard_image_on_paste("\n\t") is False

    def test_gate_failure_falls_back_to_legacy_behaviour(self):
        """Defensive: a broken helper must not break paste handling."""
        from cli import _should_auto_attach_clipboard_image_on_paste

        with patch(
            "hermes_cli.clipboard.is_clipboard_auto_attach_enabled",
            side_effect=RuntimeError("config disk full"),
        ):
            assert _should_auto_attach_clipboard_image_on_paste("") is True
            assert _should_auto_attach_clipboard_image_on_paste("hi") is False


# ---------------------------------------------------------------------------
# TUI gateway: clipboard.paste RPC
# ---------------------------------------------------------------------------

class _FakeSession(dict):
    """Minimal session shape for the gateway RPC."""


def _resolve_clipboard_paste_handler() -> Callable[[Any, Dict[str, Any]], Dict[str, Any]]:
    """Pull the registered ``clipboard.paste`` handler out of tui_gateway.server.

    Importing the server module side-effects: it registers handlers on a
    process-global ``_methods`` dict via the ``@method(...)`` decorator.
    """
    import tui_gateway.server as server_mod

    # The dispatch table is keyed by RPC name. Try several known
    # private attribute names for resilience across refactors.
    for attr in ("_methods", "_METHODS", "_handlers", "_HANDLERS"):
        table = getattr(server_mod, attr, None)
        if isinstance(table, dict) and "clipboard.paste" in table:
            return table["clipboard.paste"]

    # Last resort: walk module attributes for the function literally.
    for name in dir(server_mod):
        obj = getattr(server_mod, name)
        if callable(obj) and getattr(obj, "__name__", "") == "_" \
                and "clipboard.paste" in (getattr(obj, "_method_name", "") or ""):
            return obj

    raise RuntimeError(
        "Could not locate the clipboard.paste handler in tui_gateway.server"
    )


@pytest.fixture
def gateway_paste_handler():
    return _resolve_clipboard_paste_handler()


@pytest.fixture
def stub_sess(monkeypatch):
    """Patch ``_sess`` so we can hand the handler a real-looking session."""
    import tui_gateway.server as server_mod

    sess = _FakeSession()
    sess["sid"] = "test-sid"
    sess["image_counter"] = 0
    sess["attached_images"] = []

    def _fake_sess(_params: dict, _rid: Any):
        return sess, None

    monkeypatch.setattr(server_mod, "_sess", _fake_sess, raising=True)
    return sess


def _ok_payload(response: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the result payload from a JSON-RPC ok-style response."""
    if "result" in response:
        return response["result"]
    return response


class TestGatewayClipboardPasteAutoGate:
    def test_auto_with_gate_disabled_skips_clipboard_probe_entirely(
        self, gateway_paste_handler, stub_sess
    ):
        """The headline #23984 fix: no OS clipboard read at all."""
        save_mock = MagicMock(return_value=True)
        has_mock = MagicMock(return_value=True)
        gate_mock = MagicMock(return_value=False)

        with patch("hermes_cli.clipboard.save_clipboard_image", save_mock), \
             patch("hermes_cli.clipboard.has_clipboard_image", has_mock), \
             patch(
                 "hermes_cli.clipboard.is_clipboard_auto_attach_enabled",
                 gate_mock,
             ):
            response = gateway_paste_handler(
                1,
                {"session_id": "test-sid", "auto": True},
            )

        result = _ok_payload(response)
        assert result["attached"] is False
        assert result.get("skipped") is True
        assert "auto-attach disabled" in result.get("message", "").lower()

        # Critical: neither probe ever ran.
        save_mock.assert_not_called()
        has_mock.assert_not_called()
        # And the gate was consulted.
        gate_mock.assert_called_once()

    def test_auto_with_gate_enabled_still_probes(
        self, gateway_paste_handler, stub_sess
    ):
        """Default-enabled users keep getting auto-attach behaviour."""
        save_mock = MagicMock(return_value=True)

        with patch("hermes_cli.clipboard.save_clipboard_image", save_mock), \
             patch("hermes_cli.clipboard.has_clipboard_image",
                   return_value=False), \
             patch(
                 "hermes_cli.clipboard.is_clipboard_auto_attach_enabled",
                 return_value=True,
             ):
            response = gateway_paste_handler(
                2,
                {"session_id": "test-sid", "auto": True},
            )

        result = _ok_payload(response)
        assert result["attached"] is True
        save_mock.assert_called_once()

    def test_explicit_paste_ignores_gate_and_always_probes(
        self, gateway_paste_handler, stub_sess
    ):
        """Hotkey / right-click pastes are explicit user intent.

        Even if the user opted out of *automatic* probes, an explicit
        Cmd+V on the TUI should still try the clipboard. The TUI marks
        these calls with auto=False (the default).
        """
        save_mock = MagicMock(return_value=True)
        gate_mock = MagicMock(return_value=False)

        with patch("hermes_cli.clipboard.save_clipboard_image", save_mock), \
             patch("hermes_cli.clipboard.has_clipboard_image",
                   return_value=False), \
             patch(
                 "hermes_cli.clipboard.is_clipboard_auto_attach_enabled",
                 gate_mock,
             ):
            response = gateway_paste_handler(
                3,
                {"session_id": "test-sid"},  # no `auto` key → defaults to False
            )

        result = _ok_payload(response)
        assert result["attached"] is True
        save_mock.assert_called_once()
        gate_mock.assert_not_called()

    def test_explicit_auto_false_still_ignores_gate(
        self, gateway_paste_handler, stub_sess
    ):
        save_mock = MagicMock(return_value=False)
        gate_mock = MagicMock(return_value=False)

        with patch("hermes_cli.clipboard.save_clipboard_image", save_mock), \
             patch("hermes_cli.clipboard.has_clipboard_image",
                   return_value=False), \
             patch(
                 "hermes_cli.clipboard.is_clipboard_auto_attach_enabled",
                 gate_mock,
             ):
            response = gateway_paste_handler(
                4,
                {"session_id": "test-sid", "auto": False},
            )

        result = _ok_payload(response)
        assert result["attached"] is False
        assert result.get("skipped") is not True
        save_mock.assert_called_once()
        gate_mock.assert_not_called()

    def test_skipped_response_does_not_increment_image_counter(
        self, gateway_paste_handler, stub_sess
    ):
        """Bug-shape regression: counter must stay at 0 for skipped probes."""
        with patch("hermes_cli.clipboard.save_clipboard_image"), \
             patch("hermes_cli.clipboard.has_clipboard_image"), \
             patch(
                 "hermes_cli.clipboard.is_clipboard_auto_attach_enabled",
                 return_value=False,
             ):
            for _ in range(5):
                gateway_paste_handler(
                    99,
                    {"session_id": "test-sid", "auto": True},
                )

        assert stub_sess["image_counter"] == 0
        assert stub_sess["attached_images"] == []


# ---------------------------------------------------------------------------
# Bug-shape anchor — proves #23984 specifically cannot regress
# ---------------------------------------------------------------------------

class TestBug23984Anchor:
    """Regression anchors that read like the bug report itself.

    If any of these fail, the #23984 spam is back. Don't 'fix' the test
    by deleting the assertion — fix the code path that started probing
    the OS clipboard for an opted-out user again.
    """

    def test_opted_out_user_never_probes_on_empty_bracketed_paste_in_cli(self):
        """The CLI's empty-paste auto-attach must not probe when opted out."""
        from cli import _should_auto_attach_clipboard_image_on_paste

        save_mock = MagicMock()
        with patch(
            "hermes_cli.clipboard.is_clipboard_auto_attach_enabled",
            return_value=False,
        ), patch(
            "hermes_cli.clipboard.save_clipboard_image", save_mock,
        ), patch(
            "hermes_cli.clipboard.has_clipboard_image",
            MagicMock(return_value=True),
        ):
            should_attach = _should_auto_attach_clipboard_image_on_paste("")

        assert should_attach is False, (
            "#23984 regression: opted-out user got an auto-attach probe "
            "from an empty bracketed paste — Ghostty privacy prompt would "
            "fire and 'No image found in clipboard' would spam the TUI."
        )
        save_mock.assert_not_called()

    def test_opted_out_user_gets_no_clipboard_io_on_auto_rpc(
        self, gateway_paste_handler, stub_sess
    ):
        save_mock = MagicMock()
        has_mock = MagicMock()
        with patch(
            "hermes_cli.clipboard.is_clipboard_auto_attach_enabled",
            return_value=False,
        ), patch(
            "hermes_cli.clipboard.save_clipboard_image", save_mock,
        ), patch(
            "hermes_cli.clipboard.has_clipboard_image", has_mock,
        ):
            response = gateway_paste_handler(
                1,
                {"session_id": "test-sid", "auto": True},
            )

        save_mock.assert_not_called()
        has_mock.assert_not_called()
        result = _ok_payload(response)
        assert result.get("skipped") is True

    def test_user_facing_message_does_not_say_no_image_found_when_skipped(
        self, gateway_paste_handler, stub_sess
    ):
        """The 'No image found in clipboard' string is the user-visible
        symptom of the bug. If the user opted out, that string MUST NOT
        appear in any auto-path response.
        """
        with patch(
            "hermes_cli.clipboard.is_clipboard_auto_attach_enabled",
            return_value=False,
        ), patch("hermes_cli.clipboard.save_clipboard_image"), \
             patch("hermes_cli.clipboard.has_clipboard_image"):
            response = gateway_paste_handler(
                1,
                {"session_id": "test-sid", "auto": True},
            )

        msg = (_ok_payload(response).get("message") or "").lower()
        assert "no image found" not in msg, (
            "#23984 regression: the user opted out of auto-attach but the "
            "gateway still reported 'No image found in clipboard' — that "
            "string is the entire user-visible spam."
        )
