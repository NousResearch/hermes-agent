"""The stdio TUI entry point starts the host toolchain probe while the user types its first message."""

from tests.tui_gateway.test_entry_picker_prewarm import _run_main
from tools import env_probe


def test_main_warms_the_probe_after_gateway_ready(monkeypatch):
    events: list[tuple] = []
    monkeypatch.setattr(env_probe, "warm_environment_probe_at_boot", lambda: events.append(("probe",)))

    _run_main(monkeypatch, events)

    assert events.count(("probe",)) == 1
    assert events.index(("write", "gateway.ready")) < events.index(("probe",))
