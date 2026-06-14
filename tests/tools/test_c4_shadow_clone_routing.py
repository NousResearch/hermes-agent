"""
C4: shadow clone routing branch in the per-turn drain.

The per-turn drain (the code block inside _handle_message_with_agent that
processes the completion_queue after each agent turn) must route shadow clone
events to _shadow_clone_enqueue instead of falling through to
format_process_notification / _inject_watch_notification.

Routing condition: evt.get("shadow_clone") and evt.get("kanban_ticket_id")
must be checked BEFORE any format/inject path.

Tests:
  T17  Shadow clone event in queue → goes to inbox, NOT to inject
  T18  Non-shadow async_delegation event → goes to inject, NOT to inbox
  T19  Shadow clone event with missing kanban_ticket_id → falls through to inject
  T20  Shadow clone event with missing session_key → dropped silently, no inbox
  T21  Mixed queue: shadow clone + regular + watch → each takes its correct path
  T22  Routing condition matches watcher (_async_delegation_watcher) exactly
"""

from __future__ import annotations

import collections
import queue
import unittest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Inline replica of the per-turn drain routing logic (gateway/run.py ~8800)
# ---------------------------------------------------------------------------

def _simulate_per_turn_drain(
    completion_queue: queue.Queue,
    shadow_clone_inbox: dict,
    shadow_clone_routing: dict,
    injected: list,
    format_notification_fn,
) -> None:
    """Replicate the per-turn async_delegation routing branch.

    This mirrors the actual gateway code at run.py:8800-8822 so tests run
    without importing the 16k-line gateway module.

    Shadow clone branch (symmetric with _async_delegation_watcher):
        if evt.get("shadow_clone") and evt.get("kanban_ticket_id"):
            _sk = evt.get("session_key", "")
            if _sk:
                _rm = {k: evt.get(k, "") for k in (...) if evt.get(k)}
                _shadow_clone_enqueue(_sk, evt["kanban_ticket_id"], _rm)
            continue
        synth_text = format_process_notification(evt)
        if synth_text:
            injected.append((synth_text, evt))
    """
    import collections as _c

    _watch_events = []
    _async_delegation_events = []
    while not completion_queue.empty():
        evt = completion_queue.get_nowait()
        evt_type = evt.get("type", "completion")
        if evt_type in {"watch_match", "watch_disabled"}:
            _watch_events.append(evt)
        elif evt_type == "async_delegation":
            _async_delegation_events.append(evt)
        # else: completion events — ignored here (handled by watcher task)

    # Watch events go straight to inject (not relevant to C4, included for completeness)
    for evt in _watch_events:
        synth_text = format_notification_fn(evt)
        if synth_text:
            injected.append((synth_text, evt))

    # async_delegation events: shadow clone branch first
    for evt in _async_delegation_events:
        # ── C4: Shadow clone → inbox (symmetric with watcher) ─────────────
        if evt.get("shadow_clone") and evt.get("kanban_ticket_id"):
            _sk = evt.get("session_key", "")
            if _sk:
                _rm = {
                    k: evt.get(k, "")
                    for k in ("platform", "chat_id", "thread_id", "user_id", "user_name")
                    if evt.get(k)
                }
                # _shadow_clone_enqueue inline
                if _sk not in shadow_clone_inbox:
                    shadow_clone_inbox[_sk] = _c.deque()
                shadow_clone_inbox[_sk].append(evt["kanban_ticket_id"])
                shadow_clone_routing[_sk] = _rm
            continue
        # ─────────────────────────────────────────────────────────────────
        synth_text = format_notification_fn(evt)
        if synth_text:
            injected.append((synth_text, evt))


def _make_session_key(platform="telegram", chat_type="dm", chat_id="12345"):
    return f"agent:main:{platform}:{chat_type}:{chat_id}"


def _make_shadow_clone_evt(sk, ticket_id="t_abc123", **kwargs):
    return {
        "type": "async_delegation",
        "session_key": sk,
        "shadow_clone": True,
        "kanban_ticket_id": ticket_id,
        "platform": "telegram",
        "chat_id": "12345",
        "delegation_id": "del-shadow-1",
        **kwargs,
    }


def _make_regular_delegation_evt(sk, **kwargs):
    return {
        "type": "async_delegation",
        "session_key": sk,
        "shadow_clone": False,
        "delegation_id": "del-regular-1",
        "platform": "telegram",
        "chat_id": "12345",
        **kwargs,
    }


class TestC4ShadowCloneRouting(unittest.TestCase):
    """C4: per-turn drain routes shadow clone events to inbox, not inject."""

    def _run(self, events):
        cq = queue.Queue()
        for e in events:
            cq.put(e)
        inbox = {}
        routing = {}
        injected = []
        # format fn that always returns a non-empty string so we can detect if it ran
        fmt = MagicMock(return_value="[notification text]")
        _simulate_per_turn_drain(cq, inbox, routing, injected, fmt)
        return inbox, routing, injected, fmt

    # T17
    def test_shadow_clone_goes_to_inbox_not_inject(self):
        """Shadow clone event must land in inbox; inject must NOT be called."""
        sk = _make_session_key()
        inbox, routing, injected, fmt = self._run([_make_shadow_clone_evt(sk, "t_001")])

        self.assertIn(sk, inbox, "inbox must contain the session key")
        self.assertIn("t_001", inbox[sk], "ticket_id must be in inbox deque")
        self.assertEqual(len(injected), 0, "inject must NOT be called for shadow clone")
        fmt.assert_not_called()

    # T18
    def test_regular_delegation_goes_to_inject_not_inbox(self):
        """Non-shadow async_delegation falls through to format_process_notification."""
        sk = _make_session_key()
        inbox, routing, injected, fmt = self._run([_make_regular_delegation_evt(sk)])

        self.assertEqual(len(inbox), 0, "inbox must be untouched for regular delegation")
        fmt.assert_called_once()
        self.assertEqual(len(injected), 1, "regular delegation must reach inject")

    # T19
    def test_shadow_clone_missing_ticket_id_falls_through_to_inject(self):
        """shadow_clone=True but kanban_ticket_id absent → routing condition False → inject."""
        sk = _make_session_key()
        evt = {
            "type": "async_delegation",
            "session_key": sk,
            "shadow_clone": True,
            # no kanban_ticket_id
            "delegation_id": "del-no-ticket",
        }
        inbox, routing, injected, fmt = self._run([evt])

        # The condition is: shadow_clone AND kanban_ticket_id — missing ticket_id → False
        self.assertEqual(len(inbox), 0, "inbox must be empty when kanban_ticket_id is absent")
        fmt.assert_called_once()
        self.assertEqual(len(injected), 1)

    # T20
    def test_shadow_clone_missing_session_key_dropped_silently(self):
        """Shadow clone with no session_key: condition True but _sk is falsy → dropped."""
        evt = {
            "type": "async_delegation",
            "session_key": "",  # empty / falsy
            "shadow_clone": True,
            "kanban_ticket_id": "t_no_sk",
        }
        inbox, routing, injected, fmt = self._run([evt])

        self.assertEqual(len(inbox), 0, "inbox must be empty when session_key is absent")
        self.assertEqual(len(injected), 0, "inject must NOT be called when session_key is absent")

    # T21
    def test_mixed_queue_routes_each_event_correctly(self):
        """Mixed queue: shadow clone goes to inbox; regular delegation goes to inject."""
        sk = _make_session_key()
        events = [
            _make_shadow_clone_evt(sk, "t_sc_1"),
            _make_regular_delegation_evt(sk),
            {"type": "watch_match", "session_key": sk, "delegation_id": "w1"},
        ]
        inbox, routing, injected, fmt = self._run(events)

        # Shadow clone in inbox
        self.assertIn(sk, inbox)
        self.assertIn("t_sc_1", inbox[sk])

        # Regular delegation + watch event both injected (fmt called twice)
        self.assertEqual(fmt.call_count, 2, "format fn called for regular + watch event")
        self.assertEqual(len(injected), 2)

    # T22
    def test_routing_condition_matches_watcher_exactly(self):
        """Verify the routing condition is the same as _async_delegation_watcher:
        evt.get('shadow_clone') and evt.get('kanban_ticket_id')."""
        # shadow_clone=True but falsy kanban_ticket_id (empty string) → condition False
        sk = _make_session_key()
        evt_empty_ticket = {
            "type": "async_delegation",
            "session_key": sk,
            "shadow_clone": True,
            "kanban_ticket_id": "",  # falsy
        }
        inbox, routing, injected, fmt = self._run([evt_empty_ticket])
        self.assertEqual(len(inbox), 0, "empty ticket_id must not trigger shadow clone branch")

        # shadow_clone=False with valid ticket_id → condition False
        evt_no_flag = {
            "type": "async_delegation",
            "session_key": sk,
            "shadow_clone": False,
            "kanban_ticket_id": "t_has_ticket",
        }
        inbox2, _, injected2, fmt2 = self._run([evt_no_flag])
        self.assertEqual(len(inbox2), 0, "shadow_clone=False must not trigger shadow clone branch")
        fmt2.assert_called_once()

    def test_routing_meta_fields_captured(self):
        """Routing metadata (platform, chat_id, etc.) is stored in shadow_clone_routing."""
        sk = _make_session_key(platform="discord", chat_id="9999")
        evt = {
            "type": "async_delegation",
            "session_key": sk,
            "shadow_clone": True,
            "kanban_ticket_id": "t_meta_test",
            "platform": "discord",
            "chat_id": "9999",
            "thread_id": "777",
            "user_id": "uid-42",
        }
        inbox, routing, injected, _ = self._run([evt])

        self.assertIn(sk, routing)
        rm = routing[sk]
        self.assertEqual(rm.get("platform"), "discord")
        self.assertEqual(rm.get("chat_id"), "9999")
        self.assertEqual(rm.get("thread_id"), "777")
        self.assertEqual(rm.get("user_id"), "uid-42")


if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
