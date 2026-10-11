"""The TUI gateway must leave a native stack behind when it dies on a fatal signal.

``_log_signal`` only covers signals Python hands to a handler (SIGTERM/SIGHUP).
A *fatal* signal — SIGSEGV from a C extension — kills the interpreter outright,
so without faulthandler the TUI can only report ``child exit signal=SIGSEGV``
with nothing to point at. ``gateway/run.py`` has enabled faulthandler since
#70344; ``tui_gateway/entry.py`` never did, so every crash on the TUI path was
forensically silent.

The subprocess tests are the ones that matter: they drive real signals through a
real interpreter and assert the dump lands in the crash log the TUI already
collects.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap

import pytest


def _run_child(
    hermes_home, body: str, preamble: str = ""
) -> subprocess.CompletedProcess:
    """Import tui_gateway.entry in a fresh interpreter, then run `body`.

    `preamble` runs *before* the import, for tests that need to break
    something the module-level _enable_faulthandler() call depends on.
    """
    script = (
        textwrap.dedent(
            """
            import faulthandler, os, signal, threading, time
            """
        )
        + textwrap.dedent(preamble)
        + textwrap.dedent(
            """
            import tui_gateway.entry as entry
            """
        )
        + textwrap.dedent(body)
    )

    return subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "HERMES_HOME": str(hermes_home)},
        capture_output=True,
        text=True,
        timeout=180,
    )


def _emitted(result: subprocess.CompletedProcess, marker: str) -> bool:
    """Did the child reach `marker`?

    entry.py owns stdout for the JSON-RPC channel and redirects ordinary
    writes to stderr, so a marker can legitimately land on either stream.
    """
    return marker in result.stdout or marker in result.stderr


def _crash_log_text(hermes_home) -> str:
    log = hermes_home / "logs" / "tui_gateway_crash.log"
    assert log.exists(), "no crash log written — faulthandler was not wired up"
    return log.read_text(encoding="utf-8", errors="replace")


def test_fatal_signal_writes_native_stack_to_crash_log(tmp_path):
    """A SIGSEGV must leave an all-thread dump in tui_gateway_crash.log."""
    result = _run_child(
        tmp_path,
        """
        # A second thread proves all_threads=True: the dump must show a frame
        # from a thread other than the one taking the signal, which is what
        # makes these dumps useful for cross-thread teardown races.
        threading.Thread(
            target=lambda: time.sleep(30), daemon=True, name="Bystander"
        ).start()
        time.sleep(0.2)
        faulthandler._sigsegv()
        """,
    )

    # The child must have died on the fault, whatever the platform calls it:
    # -11/139 on POSIX, an NTSTATUS like 0xC0000005 on Windows.
    assert result.returncode != 0, (
        f"child survived a forced fault: {result.stderr[-2000:]}"
    )
    if os.name == "posix":
        assert result.returncode in (-11, 139), (
            f"expected a fatal SIGSEGV, got {result.returncode}: "
            f"{result.stderr[-2000:]}"
        )

    text = _crash_log_text(tmp_path)
    # The header wording is platform-specific — POSIX says "Segmentation
    # fault", Windows reports an access violation. Only the prefix is a
    # contract; what matters is that a dump was written at all.
    assert "Fatal Python error:" in text or "fatal exception:" in text, (
        f"no fatal-signal dump in the crash log: {text[:500]!r}"
    )
    assert "Bystander" in text or text.count("Thread 0x") >= 2, (
        "dump does not cover non-faulting threads — all_threads was not set"
    )


@pytest.mark.skipif(not hasattr(signal, "SIGUSR2"), reason="POSIX-only signal")
def test_sigusr2_dumps_threads_without_killing_the_gateway(tmp_path):
    """``kill -USR2 <pid>`` must dump and keep serving.

    SIGUSR2's default disposition is "terminate", so registering the dump with
    ``chain=True`` would make the diagnostic kill the session it was meant to
    inspect — the same trap #84539 fixes for the messaging gateway.
    """
    result = _run_child(
        tmp_path,
        """
        os.kill(os.getpid(), signal.SIGUSR2)
        time.sleep(0.4)
        print("SURVIVED")
        """,
    )

    assert result.returncode == 0, (
        f"SIGUSR2 killed the process: rc={result.returncode} {result.stderr[-2000:]}"
    )
    assert _emitted(result, "SURVIVED")
    assert "Current thread" in _crash_log_text(tmp_path)


def test_crash_log_handle_is_retained_and_open(tmp_path):
    """The module-global file handle must outlive ``_enable_faulthandler()``.

    faulthandler keeps writing to this fd for the process lifetime; a local
    handle would be garbage-collected and the next fatal signal would dump
    into a closed fd.
    """
    result = _run_child(
        tmp_path,
        """
        assert faulthandler.is_enabled(), "faulthandler not enabled on import"
        handle = entry._FAULTHANDLER_FILE
        assert handle is not None, "no crash-log handle retained"
        assert not handle.closed, "crash-log handle was closed"
        print("OK")
        """,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert _emitted(result, "OK")


def test_unwritable_crash_log_does_not_break_startup(tmp_path):
    """A gateway that cannot open its crash log must still start and serve.

    Forensics are best-effort; losing them must never cost the user a session.
    """
    blocked = tmp_path / "logs"
    blocked.write_text("not a directory", encoding="utf-8")

    result = _run_child(
        tmp_path,
        """
        # Import above already ran _enable_faulthandler() against the blocked
        # path. Reaching this line at all is the contract.
        print("STARTED")
        """,
    )

    assert result.returncode == 0, (
        f"import died on an unwritable crash log: {result.stderr[-2000:]}"
    )
    assert _emitted(result, "STARTED")


def test_failed_enable_does_not_strand_the_crash_log_fd(tmp_path):
    """If ``faulthandler.enable()`` rejects the handle, close it.

    The handle is a module global so faulthandler can keep writing to it for
    the process lifetime — which means nothing will ever collect it if enable()
    bailed. Holding an fd open for a file nothing writes to is a leak.
    """
    result = _run_child(
        tmp_path,
        """
        assert entry._FAULTHANDLER_FILE is None, (
            "handle retained even though enable() failed"
        )
        assert _opened and _opened[0].closed, "crash-log fd was left open"
        print("OK")
        """,
        preamble="""
        # Fail enable() only for the file-backed call entry.py makes, so the
        # stderr fallback is exercised too.
        _opened = []
        _real_open = open

        def _tracking_open(*a, **k):
            fh = _real_open(*a, **k)
            if a and str(a[0]).endswith("tui_gateway_crash.log"):
                _opened.append(fh)
            return fh

        import builtins
        builtins.open = _tracking_open

        _real_enable = faulthandler.enable

        def _boom(*a, **k):
            if "file" in k:
                raise OSError("simulated enable() rejection")
            return _real_enable(*a, **k)

        faulthandler.enable = _boom
        """,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert _emitted(result, "OK")
