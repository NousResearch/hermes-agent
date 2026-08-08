"""Regression tests for /background result delivery through the TUI slash worker.

Background
----------
In the TUI/Desktop route a ``/background`` task finishes AFTER its
``slash.exec`` response was already returned, so the final response used to be
printed to the slash-worker's stdout — a channel nobody reads. The fix routes
the result back as an unsolicited JSON event frame:

- ``slash_worker._emit_background_event`` writes ``{"event": "background.complete",
  ...}`` on the worker's stdout;
- ``SlashWorker._drain_stdout`` splits event frames into ``event_queue`` instead
  of the request/response ``stdout_queue`` (so ``run()`` never has to skip them);
- ``_attach_worker`` starts a drain thread that emits ``background.complete`` to
  the connected client.

These tests cover the frame emission and the queue split without spawning a
real subprocess.
"""

from __future__ import annotations

import io
import json
import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tui_gateway import server
from tui_gateway import slash_worker


class TestEmitBackgroundEvent:
    def test_writes_json_event_frame(self):
        out = io.StringIO()
        slash_worker._emit_background_event("bg_1234_ab", "het resultaat", stream=out)
        frame = json.loads(out.getvalue())
        assert frame == {
            "event": "background.complete",
            "task_id": "bg_1234_ab",
            "text": "het resultaat",
        }

    def test_unicode_survives(self):
        out = io.StringIO()
        slash_worker._emit_background_event("bg_x", "✓ afgerond — klaar", stream=out)
        assert "✓ afgerond — klaar" in json.loads(out.getvalue())["text"]


class TestSlashWorkerEventSplit:
    def _make_worker(self, lines):
        worker = server._SlashWorker.__new__(server._SlashWorker)
        worker.stdout_queue = queue.Queue()
        worker.event_queue = queue.Queue()
        worker.proc = SimpleNamespace(stdout=iter(lines))
        return worker

    def test_event_frames_go_to_event_queue(self):
        worker = self._make_worker(
            [
                json.dumps({"id": 1, "ok": True, "output": "start"}),
                json.dumps({"event": "background.complete", "task_id": "bg_x", "text": "klaar"}),
                json.dumps({"id": 2, "ok": True, "output": "later"}),
            ]
        )
        worker._drain_stdout()

        responses = []
        while True:
            msg = worker.stdout_queue.get_nowait()
            if msg is None:
                break
            responses.append(msg)
        assert [m["id"] for m in responses] == [1, 2]

        events = worker.poll_events()
        assert events == [{"event": "background.complete", "task_id": "bg_x", "text": "klaar"}]
        # poll_events is draining: a second call is empty
        assert worker.poll_events() == []

    def test_garbage_lines_are_skipped(self):
        # The pre-fix behaviour: console output from the background thread hit
        # stdout as non-JSON and was silently dropped by json.loads. It must
        # still not corrupt the protocol queues.
        worker = self._make_worker(["not json at all", json.dumps({"id": 1, "ok": True, "output": "x"})])
        worker._drain_stdout()
        assert worker.stdout_queue.get_nowait()["id"] == 1
        assert worker.poll_events() == []


class TestAttachWorkerEventDrain:
    def test_drain_thread_emits_background_complete(self):
        worker = SimpleNamespace(
            event_queue=queue.Queue(),
            proc=SimpleNamespace(poll=lambda: None),
        )
        session = {"slash_worker": worker}
        emitted = []

        with server._sessions_lock:
            server._sessions["sid-test"] = session
        try:
            with patch.object(server, "_emit", lambda ev, sid, payload: emitted.append((ev, sid, payload))):
                server._attach_worker("sid-test", session, worker)
                worker.event_queue.put(
                    {"event": "background.complete", "task_id": "bg_1", "text": "hallo"}
                )
                deadline = time.monotonic() + 5
                while not emitted and time.monotonic() < deadline:
                    time.sleep(0.05)
                assert emitted == [
                    ("background.complete", "sid-test", {"task_id": "bg_1", "text": "hallo"})
                ]
        finally:
            # Detach so the daemon drain thread exits on its next poll.
            with server._sessions_lock:
                server._sessions.pop("sid-test", None)
            session["slash_worker"] = None
            time.sleep(1.1)

    def test_attach_worker_closes_orphan(self):
        closed = threading.Event()
        worker = SimpleNamespace(close=lambda: closed.set())
        session = {"slash_worker": None}
        with server._sessions_lock:
            server._sessions["sid-orphan"] = {"not": "this session"}
        try:
            server._attach_worker("sid-orphan", session, worker)
            assert closed.is_set()
        finally:
            with server._sessions_lock:
                server._sessions.pop("sid-orphan", None)
