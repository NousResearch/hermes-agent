"""Quiet-turn GUI leak regression (2026-11-01): a successful bare ``[SILENT]`` on a machinery
turn must not reach the desktop as message text, a transcript bubble, or a DM body — while
genuine replies, alerts and error diagnostics always keep their text."""
import threading
import unittest
from unittest import mock


class CompletionPayloadBlanking(unittest.TestCase):
    """message.complete text boundary (tui_gateway/prompt_turn.py::_complete_turn_payload)."""

    def _payload(self, final_response: str, display_kind, *, error: str | None = None):
        # Split-module functions are rebound onto server.py's globals (method_ctx.bind_module);
        # the server-bound callables are the canonical ones to exercise.
        from tui_gateway import server

        class _Agent:
            session_id = "s1"
            _session_title_hint = ""

        st = server._TurnRun(agent=_Agent(), one_turn_restore=None,
                             terminal_callback=None, receipt_committed=False)
        st.result = {"final_response": final_response}
        if error is not None:
            st.result["error"] = error
            st.result["success"] = False
        session = {"history_lock": threading.Lock(), "inflight_turn": None}
        with mock.patch.object(server, "_is_bot_mode_session", return_value=False), \
             mock.patch.object(server, "_get_usage", return_value={}), \
             mock.patch.object(server, "_persisted_turn_receipt", return_value=None), \
             mock.patch.object(server, "render_message", return_value=None):
            payload, raw, status = server._complete_turn_payload(
                session, st, None, 80, display_kind=display_kind)
        return payload, status

    def test_silent_machinery_turn_blanks_text(self):
        for kind in ("internal_notification", "process_complete", "async_delegation_complete"):
            payload, status = self._payload("[SILENT]", kind)
            self.assertEqual(status, "complete", kind)
            self.assertEqual(payload["text"], "", kind)

    def test_silent_user_turn_keeps_text(self):
        # A user-addressed turn is NOT silence-eligible: text passes through unchanged so the
        # surface can show the model ignored the user (matching today's platform-adapter policy).
        payload, _ = self._payload("[SILENT]", None)
        self.assertEqual(payload["text"], "[SILENT]")

    def test_real_reply_on_machinery_turn_keeps_text(self):
        payload, _ = self._payload("Disk SMART warning on nas-disk2 — action needed.",
                                   "internal_notification")
        self.assertIn("SMART warning", payload["text"])

    def test_error_turn_fails_open(self):
        payload, status = self._payload("[SILENT]", "internal_notification",
                                        error="provider exploded")
        self.assertEqual(status, "error")
        self.assertNotEqual(payload["text"], "")


class HistoryProjectionHidesBareSilence(unittest.TestCase):
    def _project(self, rows):
        from tui_gateway import server
        return server._history_to_messages(rows)

    def test_bare_silent_assistant_row_hidden(self):
        rows = [
            {"role": "user", "content": "check the thing"},
            {"role": "assistant", "content": "[SILENT]"},
        ]
        out = self._project(rows)
        self.assertEqual([m["role"] for m in out], ["user"])

    def test_genuine_assistant_rows_survive(self):
        rows = [
            {"role": "user", "content": "status?"},
            {"role": "assistant", "content": "All clear. [SILENT] was mentioned mid-prose."},
        ]
        out = self._project(rows)
        self.assertEqual([m["role"] for m in out], ["user", "assistant"])


class BotModeDmBodyStripping(unittest.TestCase):
    def _strip(self, body: str) -> str:
        # Mirror of the delivery-boundary strip in tools/bot_mode_dm.py (same filter function).
        from gateway.response_filters import is_intentional_silence_response
        lines = body.strip().splitlines()
        while lines and is_intentional_silence_response(lines[-1]):
            lines.pop()
        return "\n".join(lines).strip()

    def test_trailing_marker_line_stripped(self):
        self.assertEqual(self._strip("Done, all green.\n[SILENT]"), "Done, all green.")

    def test_mid_text_token_preserved(self):
        text = "Note: replying [SILENT] suppresses delivery."
        self.assertEqual(self._strip(text), text)

    def test_marker_only_message_becomes_empty(self):
        self.assertEqual(self._strip("[SILENT]"), "")


if __name__ == "__main__":
    unittest.main()
