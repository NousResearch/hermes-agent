"""The stdio TUI entry point starts the host toolchain probe while the user types its first message."""

import sys

from tests.tui_gateway.test_entry_picker_prewarm import _run_main
from tools import env_probe


def test_main_warms_the_probe_after_gateway_ready(monkeypatch):
    events: list[tuple] = []
    monkeypatch.setattr(env_probe, "warm_environment_probe_at_boot", lambda: events.append(("probe",)))

    _run_main(monkeypatch, events)

    assert events.count(("probe",)) == 1
    assert events.index(("write", "gateway.ready")) < events.index(("probe",))


def test_main_reaches_the_read_loop_when_the_probe_cannot_import(monkeypatch):
    events: list[tuple] = []
    monkeypatch.setitem(sys.modules, "tools.env_probe", None)  # the lazy import raises ImportError

    _run_main(monkeypatch, events)

    assert ("prewarm",) in events
