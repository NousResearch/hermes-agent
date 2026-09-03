"""Regression test: the stdio TUI entry point prewarms the /model picker cache.

The classic CLI run() loop calls ``prewarm_picker_cache_async()`` during the
idle window after the banner, so the first ``/model`` open hits a warm
provider-models disk cache. The stdio TUI entry point (``entry.main()``) never
did — the first ``/model`` open in a TUI session blocked on serial /v1/models
fetches for every authenticated provider (#72021).

These tests pin the entrypoint wiring itself (the helper's own worker/once
guard is covered in ``tests/hermes_cli/test_picker_prewarm.py``):

- ``main()`` invokes ``hermes_cli.model_switch.prewarm_picker_cache_async``
  exactly once, AFTER the ``gateway.ready`` event is written (banner shown,
  user about to type — the idle window the prewarm is meant to fill).
- The startup path stays non-blocking: with the prewarm spied out, ``main()``
  proceeds into the stdin read loop and returns normally on EOF.
- A prewarm import/start failure is swallowed (fire-and-forget contract) and
  must not prevent ``main()`` from reaching the read loop.

Harness: same style as tests/test_tui_entry_mcp_owner.py — import
``tui_gateway.entry`` and monkeypatch its module attributes, running the real
``main()`` with stubbed I/O collaborators (no subprocess, no real gateway).
"""

from __future__ import annotations

import io

import hermes_cli.model_switch as ms
from tui_gateway import entry


def _run_main(monkeypatch, events, *, prewarm=None):
    """Run entry.main() with stubbed collaborators, recording ordering.

    ``events`` receives ``("write", <event type>)`` for every write_json call
    and ``("prewarm",)`` when the spy fires, in call order.
    """
    monkeypatch.setattr(entry, "_install_sidecar_publisher", lambda: None)
    monkeypatch.setattr(entry, "ensure_mcp_discovery_started", lambda: None)
    monkeypatch.setattr(entry, "resolve_skin", lambda: "default")
    monkeypatch.setattr(entry.server, "_ensure_skin_watcher", lambda: None)
    monkeypatch.setattr(entry, "_log_exit", lambda reason: None)
    # Genuine EOF, no fd-0 forensics in the test process.
    monkeypatch.setattr(entry, "handle_spurious_eof", lambda *a: False)

    def _write_json(payload):
        params = payload.get("params") or {}
        events.append(("write", params.get("type") or payload.get("method")))
        return True

    monkeypatch.setattr(entry, "write_json", _write_json)

    # entry.main() imports the helper lazily from hermes_cli.model_switch,
    # so the spy must live on that module, not on entry.
    if prewarm is None:
        def prewarm():
            events.append(("prewarm",))
            return None  # fire-and-forget handle; never blocks

    monkeypatch.setattr(ms, "prewarm_picker_cache_async", prewarm)

    # Empty stdin -> immediate EOF -> main() returns after entering the loop.
    monkeypatch.setattr(entry.sys, "stdin", io.StringIO(""))

    entry.main()


def test_main_prewarms_picker_cache_after_gateway_ready(monkeypatch):
    """main() must call the prewarm helper once, after gateway.ready is
    written, and still reach the stdin loop (returns on EOF = non-blocking)."""
    events: list[tuple] = []

    _run_main(monkeypatch, events)  # returning at all proves the loop was reached

    prewarm_calls = [e for e in events if e[0] == "prewarm"]
    assert len(prewarm_calls) == 1, (
        f"main() must invoke prewarm_picker_cache_async exactly once, got {events!r}"
    )

    ready_idx = events.index(("write", "gateway.ready"))
    prewarm_idx = events.index(("prewarm",))
    assert ready_idx < prewarm_idx, (
        "prewarm must fire AFTER the gateway.ready write (idle window, "
        f"banner already shown); order was {events!r}"
    )


def test_main_survives_prewarm_failure(monkeypatch):
    """Fire-and-forget contract: a prewarm that raises at start must be
    swallowed and main() must still reach the read loop and exit cleanly."""
    events: list[tuple] = []

    def _boom():
        events.append(("prewarm",))
        raise RuntimeError("provider registry exploded")

    _run_main(monkeypatch, events, prewarm=_boom)  # must not raise

    assert ("prewarm",) in events
    assert ("write", "gateway.ready") in events
