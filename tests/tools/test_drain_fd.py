"""``drain_fd``: the public, platform-picking pipe drain. It must return on ``stop`` while the
process is alive and a writer still holds the pipe — the hand-off case a second reader needs."""

import codecs
import os
import threading
import time
from unittest.mock import MagicMock

from tools.environments.base_output import drain_fd


def test_drain_fd_stops_on_stop_with_a_live_writer():
    r, w = os.pipe()
    try:
        os.write(w, b"first line\n")
        proc = MagicMock()
        proc.poll.return_value = None  # still running: only ``stop`` can end this drain
        sink, stop = [], threading.Event()
        worker = threading.Thread(
            target=drain_fd, args=(proc, r, sink, codecs.getincrementaldecoder("utf-8")("replace"), stop),
            daemon=True)
        worker.start()
        deadline = time.monotonic() + 5
        while "first line" not in "".join(sink) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert worker.is_alive()  # positive control: without stop the drain keeps waiting
        stop.set()
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert "".join(sink) == "first line\n"
    finally:
        os.close(r)
        os.close(w)
