"""The synchronous Thread fake must honour the stdlib's own Thread usage.

Patching ``server.threading.Thread`` replaces it for every caller, so on
Windows ``Popen.communicate()`` starts its pipe readers through the fake
(#137194). Pin the contract that path relies on, host-independently.
"""

import threading

from tests.fakes.immediate_thread import ImmediateThread


def test_reader_thread_call_shape_runs_target_with_its_args(monkeypatch):
    # Mirrors subprocess.Popen._communicate on win32:
    #   t = threading.Thread(target=self._readerthread, args=(fh, buffer))
    #   t.daemon = True; t.start(); ...; t.join(timeout); t.is_alive()
    monkeypatch.setattr(threading, "Thread", ImmediateThread)
    buffer = []

    def readerthread(fh, buf, *, sep=b""):
        buf.append(fh + sep)

    reader = threading.Thread(target=readerthread, args=(b"out", buffer), kwargs={"sep": b"\n"})
    reader.daemon = True
    reader.start()
    reader.join(5)

    assert buffer == [b"out\n"]
    assert not reader.is_alive()


def test_positional_constructor_matches_threading_thread():
    seen = []
    ImmediateThread(None, seen.append, "reader", ("x",)).start()
    assert seen == ["x"]
