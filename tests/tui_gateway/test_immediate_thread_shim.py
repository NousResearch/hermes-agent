"""The shared synchronous Thread fake must satisfy ``threading.Thread``'s contract (#137194).

Patching ``server.threading.Thread`` replaces it for every caller, so on win32
``Popen._communicate()`` starts its pipe readers through the fake:
``Thread(target=self._readerthread, args=(fh, buffer))``, then
``daemon = True``, ``start()``, ``join(timeout)``, ``is_alive()``. The
historical inline copies dropped ``args`` (TypeError at ``start()``) and had
no ``join``/``is_alive`` (AttributeError right after) — Linux reads pipes with
selectors and never noticed. The fake subclasses the real class, so pin the
two shapes the stdlib path and the tests rely on.
"""

import threading

from tests._fixtures.thread_shim import ImmediateThread


def test_reader_thread_call_shape_runs_target_with_its_args(monkeypatch):
    # Mirrors subprocess.Popen._communicate on win32:
    #   t = threading.Thread(target=self._readerthread, args=(fh, buffer))
    #   t.daemon = True; t.start(); ...; t.join(timeout); t.is_alive()
    monkeypatch.setattr(threading, "Thread", ImmediateThread)
    buffer: list[bytes] = []

    def readerthread(fh, buf, *, sep=b""):
        buf.append(fh + sep)

    reader = threading.Thread(target=readerthread, args=(b"out", buffer), kwargs={"sep": b"\n"})
    reader.daemon = True
    reader.start()
    reader.join(5)

    assert buffer == [b"out\n"]
    assert not reader.is_alive()


def test_positional_constructor_and_name_match_threading_thread():
    seen: list[str] = []
    thread = ImmediateThread(None, seen.append, "reader-thread", ("x",))
    thread.start()
    assert seen == ["x"]
    assert thread.name == "reader-thread"
