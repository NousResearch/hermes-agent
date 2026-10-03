"""``_release_finished_handles`` must never close stdout while its reader thread is alive on
Windows: the reader sits in a blocking ``read1()`` holding the BufferedReader lock, so ``close()``
would block forever — under the registry lock — and starve every ``process.list`` RPC."""

import threading

import pytest

import tools.process_registry as module


class _Stream:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _Proc:
    def __init__(self):
        self.stdout, self.stderr, self.stdin = _Stream(), _Stream(), _Stream()


def _session_with_live_reader():
    session = module.ProcessSession.__new__(module.ProcessSession)
    session.process = _Proc()
    session._pty = None
    stop = threading.Event()
    reader = threading.Thread(target=stop.wait, daemon=True)
    reader.start()
    session._reader_thread = reader
    return session, stop


@pytest.mark.parametrize("is_windows, stdout_closed", [(True, False), (False, True)])
def test_release_skips_stdout_only_while_windows_reader_alive(monkeypatch, is_windows, stdout_closed):
    monkeypatch.setattr(module, "_IS_WINDOWS", is_windows)
    session, stop = _session_with_live_reader()
    try:
        started = threading.Event()

        def release():
            started.set()
            module.ProcessRegistry._release_finished_handles(module.ProcessRegistry.__new__(module.ProcessRegistry), session)

        worker = threading.Thread(target=release, daemon=True)
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive(), "release blocked while the reader thread was alive"
        assert session.process.stdout.closed is stdout_closed
        assert session.process.stderr.closed and session.process.stdin.closed
    finally:
        stop.set()


def test_release_closes_stdout_once_reader_finished(monkeypatch):
    monkeypatch.setattr(module, "_IS_WINDOWS", True)
    session, stop = _session_with_live_reader()
    stop.set()
    session._reader_thread.join(timeout=2)
    module.ProcessRegistry._release_finished_handles(module.ProcessRegistry.__new__(module.ProcessRegistry), session)
    assert session.process.stdout.closed
