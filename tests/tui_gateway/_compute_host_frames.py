"""Causal, thread-safe compute-host frame capture for protocol tests."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable


class FrameSink:
    """Capture complete JSON-line frames and wake waiters when one is published."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending = ""
        self._frames: list[dict] = []

    def write(self, data: str) -> int:
        with self._condition:
            self._pending += data
            published = False
            while "\n" in self._pending:
                line, self._pending = self._pending.split("\n", 1)
                if line.strip():
                    self._frames.append(json.loads(line))
                    published = True
            if published:
                self._condition.notify_all()
        return len(data)

    def flush(self) -> None:
        pass

    def frames(self) -> list[dict]:
        with self._condition:
            return list(self._frames)

    def wait_for(self, predicate: Callable[[dict], bool], timeout: float = 20.0) -> dict:
        """Wait on frame publication; ``timeout`` is only a deadlock guard."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                for frame in self._frames:
                    if predicate(frame):
                        return frame
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(f"timed out; saw={self._frames}")
                self._condition.wait(remaining)


def start_test_work(target, *, name: str, session: dict | None = None) -> threading.Thread:
    """Start the real turn thread without process-retirement/import side paths."""
    thread = threading.Thread(target=target, name=name)
    if session is not None:
        session["_run_thread"] = thread
    thread.start()
    return thread
