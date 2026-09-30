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
def _clear_parked_notifications():
    """The unowned-notification park is process-global, like the poller registry.

    A park leaked by one test would let a later test's session claim a stale event, or spend the
    cap and refuse a later test's park. Same seam the poller teardown uses.
    """
    from tui_gateway import server

    parked = getattr(server, "_unowned_parked", None)
    if parked is not None:
        parked[:] = []
    yield
    if parked is not None:
        parked[:] = []
