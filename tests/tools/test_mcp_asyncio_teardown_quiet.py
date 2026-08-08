"""Regression tests for #81175 — quiet asyncio subprocess-transport teardown.

When an MCP stdio server's subprocess transport is garbage-collected after
the event loop closes, ``BaseSubprocessTransport.__del__`` fires and
``self._check_closed()`` raises ``RuntimeError("Event loop is closed")``.
The transport already finished its work, but CPython's default
``sys.unraisablehook`` prints a multi-line traceback to stderr that
pollutes the chat / TUI.

``tools.mcp_tool`` installs a quiet ``sys.unraisablehook`` on import that
swallows ONLY that benign race and forwards every other unraisable to
the previous hook.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Helpers — drive the detector directly. We do NOT call sys.unraisablehook
# from the tests because CPython type-checks the argument as UnraisableHookArgs
# (which is internal and not exposed for construction in user code on every
# supported Python version). The hook itself delegates to the detector, so
# exercising the detector covers the same logic.
# ---------------------------------------------------------------------------


def _raise_with_asyncio_base_subprocess_frame():
    """Raise inside a function whose frame lives in this test file but
    whose traceback walks through ``asyncio/base_subprocess.py`` by
    importing the module so its code objects are referenced."""
    import asyncio.base_subprocess  # noqa: F401  -- imported for filename match

    # The detector walks up frames. Easiest reliable way to manufacture a
    # matching traceback is to actually call into asyncio.base_subprocess.
    # But base_subprocess doesn't expose a callable that raises the
    # benign RuntimeError directly — that error only fires from
    # ``BaseSubprocessTransport.__del__`` on a closed loop.
    #
    # Instead, we synthesize a fake frame object by constructing an
    # exception from inside this function (so the frame filename is the
    # test file) and then *patching* the frame's f_code.co_filename to
    # point at asyncio/base_subprocess.py. That's hacky; the alternative
    # is to trust the integration test below (real subprocess transport
    # teardown) and only unit-test the "no match" branches here.
    raise RuntimeError("Event loop is closed")


def _capture_traceback_via_helper():
    """Helper that imports asyncio.base_subprocess and re-raises so the
    traceback contains a frame from this helper file plus the import
    line above. We still need to mutate frame.f_code.co_filename for a
    real asyncio/base_subprocess.py match — see the integration test
    for the true end-to-end coverage."""
    return _raise_with_asyncio_base_subprocess_frame()


def test_import_installs_quiet_hook():
    """Importing ``tools.mcp_tool`` must install the quiet hook."""
    from tools import mcp_tool

    assert getattr(sys, "_hermes_mcp_quiet_hook_installed", False) is True
    assert sys.unraisablehook is not None
    assert callable(sys.unraisablehook)


def test_install_is_idempotent():
    """Calling the installer twice does not stack wrappers."""
    from tools import mcp_tool

    hook_before = sys.unraisablehook
    mcp_tool._install_asyncio_del_quiet_hook()
    assert sys.unraisablehook is hook_before


def test_detector_returns_false_for_none():
    """Detector must not crash on a None exc_value."""
    from tools import mcp_tool

    class _Args:
        exc_value = None

    assert mcp_tool._is_asyncio_subprocess_teardown(_Args()) is False


def test_detector_returns_false_for_non_asyncio_runtime_error():
    """A RuntimeError from a non-asyncio frame is not swallowed."""
    from tools import mcp_tool

    try:
        _capture_traceback_via_helper()
    except RuntimeError:
        exc_value = sys.exc_info()[1]
    args = type("_Args", (), {"exc_value": exc_value})()
    assert mcp_tool._is_asyncio_subprocess_teardown(args) is False


def test_detector_returns_false_for_unrelated_value_error():
    """A ValueError from a non-asyncio frame is not swallowed."""
    from tools import mcp_tool

    try:
        raise ValueError("bad value")
    except ValueError:
        exc_value = sys.exc_info()[1]
    args = type("_Args", (), {"exc_value": exc_value})()
    assert mcp_tool._is_asyncio_subprocess_teardown(args) is False


def test_detector_returns_true_for_real_asyncio_base_subprocess_traceback():
    """Walk into ``asyncio.base_subprocess`` and trigger a RuntimeError from
    a function defined there — this produces a traceback whose innermost
    frame's filename genuinely matches ``asyncio/base_subprocess.py``."""
    from tools import mcp_tool
    import asyncio.base_subprocess as base_subprocess_module

    target_filename = base_subprocess_module.__file__

    # The frame that triggers the teardown lives in
    # asyncio/base_subprocess.py::BaseSubprocessTransport.close (called from
    # __del__). We replicate the traceback shape by calling a real
    # coroutine inside an inline ``exec`` whose source is claimed to live
    # in that file. Code objects are immutable, but ``exec`` lets us
    # build a frame whose filename is exactly ``asyncio/base_subprocess.py``
    # by passing ``{"__file__": target_filename}`` implicitly via the
    # ``exec`` first argument. Then we raise inside that exec so the
    # frame's f_code.co_filename matches.
    def _in_asyncio_namespace():
        # exec with a single-statement body that raises. The frame's
        # f_code.co_filename is set by the exec() call's first arg.
        co_filename = target_filename
        code = compile(
            "raise RuntimeError('Event loop is closed')",
            co_filename,
            "exec",
        )
        exec(code, {"__name__": "__main__"})

    try:
        _in_asyncio_namespace()
    except RuntimeError:
        exc_value = sys.exc_info()[1]

    # Sanity check: traceback should contain a frame in asyncio/base_subprocess.py.
    tb = exc_value.__traceback__
    seen_asyncio_frame = False
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == target_filename:
            seen_asyncio_frame = True
            break
        tb = tb.tb_next

    if not seen_asyncio_frame:
        pytest.skip(
            "Could not manufacture a traceback frame inside "
            "asyncio/base_subprocess.py on this Python version; the "
            "real-teardown integration test below covers the detector."
        )

    args = type("_Args", (), {"exc_value": exc_value})()
    assert mcp_tool._is_asyncio_subprocess_teardown(args) is True


def test_detector_accepts_asyncio_events_close_frame():
    """Synthetic ``asyncio/<...>events.py`` filename + ``close`` co_name
    frames are accepted by the detector. We can't easily force a real
    traceback inside proactor_events.py from a unit test, so we use
    ``exec`` with a controlled filename to manufacture a real frame."""
    from tools import mcp_tool

    # Build a tiny ``exec`` whose f_code.co_filename lies inside an
    # asyncio events module and whose top-level co_name is "close".
    target = sys.base_prefix + "/asyncio/unix_events.py"
    code = compile(
        "def close():\n    raise RuntimeError('Event loop is closed')\nclose()",
        target,
        "exec",
    )
    ns = {"__name__": "__main__"}
    try:
        exec(code, ns)
    except RuntimeError:
        exc_value = sys.exc_info()[1]

    args = type("_Args", (), {"exc_value": exc_value})()
    assert mcp_tool._is_asyncio_subprocess_teardown(args) is True


def test_real_asyncio_subprocess_teardown_is_silenced(monkeypatch):
    """End-to-end: spawn a real asyncio subprocess, close the loop, drop
    the transport reference, and verify the quiet hook does not produce
    the benign ``RuntimeError("Event loop is closed")`` traceback."""
    from tools import mcp_tool

    loop = asyncio.new_event_loop()
    try:
        transport_holder = {}

        async def _driver():
            transport, _ = await loop.subprocess_exec(
                lambda: asyncio.SubprocessProtocol(),
                sys.executable, "-c", "pass",
            )
            transport_holder["t"] = transport

        loop.run_until_complete(_driver())

        # Replace sys.unraisablehook with our spy via monkeypatch so the
        # original hook is restored automatically.
        seen = []

        def _spy(unraisable):
            seen.append(unraisable)

        monkeypatch.setattr(sys, "unraisablehook", _spy)

        # The hook installed at import time is what actually swallowed
        # the teardown noise — re-install it on top of our spy so the
        # spy becomes the "previous" hook (which only fires for
        # non-asyncio errors).
        sys._hermes_mcp_quiet_hook_installed = False
        saved_previous = sys.unraisablehook
        mcp_tool._install_asyncio_del_quiet_hook()
        try:
            # Close the loop and GC the transport — this is what triggers
            # BaseSubprocessTransport.__del__ → RuntimeError.
            loop.close()
            transport_holder.clear()
            import gc
            gc.collect()
        finally:
            sys.unraisablehook = saved_previous
            del sys._hermes_mcp_quiet_hook_installed

        # Spy must not have seen the benign teardown.
        benign = [
            u for u in seen
            if isinstance(u.exc_value, RuntimeError)
            and "Event loop is closed" in str(u.exc_value)
        ]
        assert benign == []
    finally:
        if not loop.is_closed():
            loop.close()