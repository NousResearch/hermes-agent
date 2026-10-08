"""tui_gateway test fixtures.

Several files here import ``tui_gateway.server`` inside a ``patch.dict("sys.modules", {"hermes_constants":
MagicMock(...)})`` window so the module binds a fixed home. The server's import graph reaches
``agent.process_bootstrap`` → ``hermes_bootstrap``, which is process boot: PM dependency activation reads
the real install root through ``hermes_constants`` and exits the process when that is a MagicMock.
Importing it once here, before any window opens, keeps boot out of the mocked import.
"""

import hermes_bootstrap  # noqa: F401


import pytest


@pytest.fixture(autouse=True)
def _no_turn_alive_ticker(monkeypatch):
    """Every ``handle_ws`` test would start the process-wide ``turn.alive`` ticker, which would then write frames
    into later tests' fake transports. ``tests/tui_gateway/test_turn_alive.py`` drives ``tick`` directly."""
    from tui_gateway import turn_alive

    monkeypatch.setattr(turn_alive, "ensure_started", lambda: None)
