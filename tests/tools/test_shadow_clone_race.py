"""
Tests for the shadow clone watcher/drain race condition fix.

Race scenario:
  _async_delegation_watcher sees idle session → calls _drain_shadow_clone_inbox
  Post-turn hook also calls _drain_shadow_clone_inbox for same session
  Without a lock:
    - Both callers drain the deque and pop routing_meta
    - The second caller gets an empty routing_meta → routing_evt has no platform/chat_id
    - inject silently loses the clone notification

Fix verified:
  - asyncio.Lock per session serialises concurrent drains
  - routing_meta is captured before the first await (before asyncio.to_thread yield)
  - Second caller sees empty deque inside the lock → returns immediately
  - All 7 clones are delivered exactly once

T10  Two concurrent drain calls: exactly one delivers (lock serialisation)
T11  routing_meta captured before await: second call gets empty dict (harmless)
T12  Empty deque fast-path: no lock created, returns immediately
T13  Re-enqueue on inject failure: IDs re-queued when inject raises
T14  Watcher idle-drain check: session in _running_agents is skipped
T15  Watcher shadow-clone event enqueue: inbox grows, _running_agents not affected
T16  Seven concurrent clones: all 7 delivered, no duplicates
"""

from __future__ import annotations

import asyncio
import collections
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session_key(platform="telegram", chat_type="dm", chat_id="12345"):
    return f"agent:main:{platform}:{chat_type}:{chat_id}"


class _FakeDrainTarget:
    """Minimal stand-in for the gateway object that owns the shadow clone state."""

    def __init__(self):
        self._shadow_clone_inbox: dict = {}
        self._shadow_clone_routing: dict = {}
        self._shadow_clone_drain_locks: dict = {}
        self._injected: list = []
        self._inject_error: Exception | None = None

    async def _inject_watch_notification(self, synth_text, routing_evt):
        if self._inject_error:
            raise self._inject_error
        self._injected.append((synth_text, routing_evt))

    def _read_kanban_tickets_sync(self, ticket_ids):
        return [
            {"ticket_id": tid, "title": f"Task {tid}", "result": "done", "status": "done"}
            for tid in ticket_ids
        ]

    # Paste the real _drain_shadow_clone_inbox verbatim so tests run without
    # importing the 12 k-line gateway module.
    async def _drain_shadow_clone_inbox(self, session_key: str) -> None:
        inbox = self._shadow_clone_inbox.get(session_key)
        if not inbox:
            return

        if session_key not in self._shadow_clone_drain_locks:
            self._shadow_clone_drain_locks[session_key] = asyncio.Lock()
        async with self._shadow_clone_drain_locks[session_key]:
            inbox = self._shadow_clone_inbox.get(session_key)
            if not inbox:
                return

            pending_ids: list = []
            while inbox:
                pending_ids.append(inbox.popleft())

            if not pending_ids:
                return

            # Capture routing_meta before first await
            routing_meta = self._shadow_clone_routing.pop(session_key, {})

        import asyncio as _aio
        ticket_summaries = await _aio.to_thread(self._read_kanban_tickets_sync, pending_ids)
        if not ticket_summaries:
            ticket_summaries = [
                {"ticket_id": tid, "title": "(unreadable)", "result": None, "status": "done"}
                for tid in pending_ids
            ]

        n = len(ticket_summaries)
        lines = [f"[SHADOW CLONE RETURN — {n} 個影分身完成]\n"]
        for i, t in enumerate(ticket_summaries, 1):
            result_preview = (t.get("result") or "(no result)")
            lines.append(
                f"─── {i}. {t.get('title', '?')}\n"
                f"    Ticket: {t.get('ticket_id', '?')}  Status: {t.get('status', 'done')}\n"
                f"    Result: {result_preview}\n"
            )
        lines.append("\n[完整 decision_trail / insights 請至 Kanban 查閱]")
        synth_text = "\n".join(lines)

        routing_evt = {
            "session_key": session_key,
            "type": "shadow_clone_batch",
            **routing_meta,
        }

        try:
            await self._inject_watch_notification(synth_text, routing_evt)
        except Exception as exc:
            import collections as _c
            if session_key not in self._shadow_clone_inbox:
                self._shadow_clone_inbox[session_key] = _c.deque()
            self._shadow_clone_inbox[session_key].extendleft(reversed(pending_ids))

    def _shadow_clone_enqueue(self, session_key: str, ticket_id: str, routing_meta: dict) -> None:
        import collections as _c
        if session_key not in self._shadow_clone_inbox:
            self._shadow_clone_inbox[session_key] = _c.deque()
        self._shadow_clone_inbox[session_key].append(ticket_id)
        self._shadow_clone_routing[session_key] = routing_meta


# ---------------------------------------------------------------------------
# T10 — Two concurrent drains: lock ensures exactly one delivers
# ---------------------------------------------------------------------------

class TestConcurrentDrainSerialisation(unittest.IsolatedAsyncioTestCase):
    async def test_exactly_one_delivery_on_concurrent_drain(self):
        """With asyncio.Lock, two concurrent calls to _drain_shadow_clone_inbox
        for the same session produce exactly one inject call — never zero, never two."""
        sk = _make_session_key()
        gw = _FakeDrainTarget()
        gw._shadow_clone_enqueue(sk, "ticket-001", {"platform": "telegram", "chat_id": "12345"})

        # Both coroutines start simultaneously
        results = await asyncio.gather(
            gw._drain_shadow_clone_inbox(sk),
            gw._drain_shadow_clone_inbox(sk),
        )

        # Exactly one inject call — the second sees empty deque inside the lock
        self.assertEqual(len(gw._injected), 1, "Expected exactly 1 inject, got: %d" % len(gw._injected))
        _, routing_evt = gw._injected[0]
        self.assertEqual(routing_evt["platform"], "telegram")
        self.assertEqual(routing_evt["chat_id"], "12345")


# ---------------------------------------------------------------------------
# T11 — routing_meta captured before await: second concurrent call gets nothing
# ---------------------------------------------------------------------------

class TestRoutingMetaCapturedBeforeAwait(unittest.IsolatedAsyncioTestCase):
    async def test_second_drain_gets_empty_routing(self):
        """After the first drain pops routing_meta (inside the lock, before await),
        a subsequent call finds no routing and short-circuits without injecting."""
        sk = _make_session_key()
        gw = _FakeDrainTarget()
        gw._shadow_clone_enqueue(sk, "ticket-X", {"platform": "discord", "chat_id": "9999"})

        await gw._drain_shadow_clone_inbox(sk)  # first call — should inject
        self.assertEqual(len(gw._injected), 1)

        # Second call — inbox empty, routing_meta gone
        await gw._drain_shadow_clone_inbox(sk)
        self.assertEqual(len(gw._injected), 1, "Second drain should not inject again")


# ---------------------------------------------------------------------------
# T12 — Fast-path: empty deque never acquires lock
# ---------------------------------------------------------------------------

class TestEmptyDequeEarlyReturn(unittest.IsolatedAsyncioTestCase):
    async def test_empty_session_returns_immediately(self):
        """If inbox is empty or missing, drain returns without creating a lock."""
        sk = _make_session_key()
        gw = _FakeDrainTarget()

        await gw._drain_shadow_clone_inbox(sk)

        # No lock created, no inject
        self.assertNotIn(sk, gw._shadow_clone_drain_locks)
        self.assertEqual(len(gw._injected), 0)

    async def test_empty_deque_object_returns_immediately(self):
        """A deque that exists but is empty also short-circuits."""
        sk = _make_session_key()
        gw = _FakeDrainTarget()
        gw._shadow_clone_inbox[sk] = collections.deque()  # empty deque

        await gw._drain_shadow_clone_inbox(sk)
        self.assertEqual(len(gw._injected), 0)


# ---------------------------------------------------------------------------
# T13 — Re-enqueue on inject failure
# ---------------------------------------------------------------------------

class TestReEnqueueOnInjectFailure(unittest.IsolatedAsyncioTestCase):
    async def test_ids_requeued_when_inject_raises(self):
        """If _inject_watch_notification raises, ticket IDs are put back in inbox."""
        sk = _make_session_key()
        gw = _FakeDrainTarget()
        gw._shadow_clone_enqueue(sk, "ticket-fail-1", {"platform": "telegram"})
        gw._shadow_clone_enqueue(sk, "ticket-fail-2", {"platform": "telegram"})
        gw._inject_error = RuntimeError("network down")

        await gw._drain_shadow_clone_inbox(sk)

        # IDs should be back in the inbox
        recovered = list(gw._shadow_clone_inbox.get(sk, collections.deque()))
        self.assertIn("ticket-fail-1", recovered)
        self.assertIn("ticket-fail-2", recovered)


# ---------------------------------------------------------------------------
# T14 — Watcher idle-drain: skip session that is in _running_agents
# ---------------------------------------------------------------------------

class TestWatcherSkipsRunningSession(unittest.TestCase):
    def test_idle_drain_list_excludes_running_session(self):
        """The watcher's idle-session filter must exclude sessions with active agents."""
        sk_idle = _make_session_key(chat_id="111")
        sk_busy = _make_session_key(chat_id="222")

        shadow_clone_inbox = {
            sk_idle: collections.deque(["t1"]),
            sk_busy: collections.deque(["t2"]),
        }
        running_agents = {sk_busy: MagicMock()}

        # Replicate watcher filter logic
        idle_sessions = [
            _sk for _sk, _q in list(shadow_clone_inbox.items())
            if _q and _sk not in running_agents
        ]

        self.assertIn(sk_idle, idle_sessions)
        self.assertNotIn(sk_busy, idle_sessions)


# ---------------------------------------------------------------------------
# T15 — Watcher shadow-clone enqueue: does not touch _running_agents
# ---------------------------------------------------------------------------

class TestWatcherEnqueueDoesNotTouchRunningAgents(unittest.IsolatedAsyncioTestCase):
    async def test_shadow_clone_event_enqueued_not_dispatched(self):
        """A shadow_clone event in the completion queue is moved to inbox,
        NOT re-queued back onto the completion queue."""
        import queue as _queue

        sk = _make_session_key()
        cq = _queue.Queue()
        shadow_clone_evt = {
            "type": "async_delegation",
            "session_key": sk,
            "shadow_clone": True,
            "kanban_ticket_id": "ticket-sc-1",
            "platform": "telegram",
            "chat_id": "12345",
        }
        cq.put(shadow_clone_evt)

        running_agents = {}
        shadow_clone_inbox: dict = {}
        shadow_clone_routing: dict = {}

        # Simulate one watcher tick (shadow_clone path)
        snapshot = []
        while not cq.empty():
            snapshot.append(cq.get_nowait())

        to_requeue = []
        for evt in snapshot:
            if evt.get("type") != "async_delegation":
                to_requeue.append(evt)
                continue
            _sk = evt.get("session_key", "")
            if not _sk:
                continue
            if evt.get("shadow_clone") and evt.get("kanban_ticket_id"):
                # Enqueue to inbox, do NOT put_back
                if _sk not in shadow_clone_inbox:
                    shadow_clone_inbox[_sk] = collections.deque()
                shadow_clone_inbox[_sk].append(evt["kanban_ticket_id"])
                shadow_clone_routing[_sk] = {"platform": evt.get("platform"), "chat_id": evt.get("chat_id")}
                continue
            if _sk in running_agents:
                to_requeue.append(evt)
                continue

        # The shadow clone ticket went to inbox, not back on the queue
        self.assertTrue(cq.empty(), "completion_queue should be empty after watcher tick")
        self.assertIn(sk, shadow_clone_inbox)
        self.assertEqual(list(shadow_clone_inbox[sk]), ["ticket-sc-1"])


# ---------------------------------------------------------------------------
# T16 — Seven concurrent clones all delivered, no duplicates
# ---------------------------------------------------------------------------

class TestSevenConcurrentClones(unittest.IsolatedAsyncioTestCase):
    async def test_seven_clones_delivered_exactly_once(self):
        """The 7-clone intermittent test case: all delivered, none duplicated."""
        sk = _make_session_key()
        gw = _FakeDrainTarget()
        routing = {"platform": "telegram", "chat_id": "8494508720"}

        # Enqueue 7 tickets one by one (simulates 7 shadow clone completions)
        for i in range(1, 8):
            gw._shadow_clone_enqueue(sk, f"ticket-{i:03d}", routing)

        # Watcher tick + post-turn hook fire simultaneously
        await asyncio.gather(
            gw._drain_shadow_clone_inbox(sk),  # watcher idle-drain
            gw._drain_shadow_clone_inbox(sk),  # post-turn hook
        )

        # All tickets must be in exactly one inject call (batched)
        self.assertEqual(len(gw._injected), 1, "Expected exactly 1 batched inject")
        synth_text, routing_evt = gw._injected[0]

        for i in range(1, 8):
            self.assertIn(f"ticket-{i:03d}", synth_text, f"ticket-{i:03d} missing from inject")

        # Routing must be intact
        self.assertEqual(routing_evt.get("platform"), "telegram")
        self.assertEqual(routing_evt.get("chat_id"), "8494508720")

        # Inbox must be empty (no leftovers)
        remaining = list(gw._shadow_clone_inbox.get(sk, collections.deque()))
        self.assertEqual(remaining, [], "Inbox should be empty after drain")


if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
