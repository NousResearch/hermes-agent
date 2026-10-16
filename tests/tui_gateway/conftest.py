"""tui_gateway test fixtures.

Several files here import ``tui_gateway.server`` inside a ``patch.dict("sys.modules", {"hermes_constants":
MagicMock(...)})`` window so the module binds a fixed home. The server's import graph reaches
``agent.process_bootstrap`` → ``hermes_bootstrap``, which is process boot: PM dependency activation reads
the real install root through ``hermes_constants`` and exits the process when that is a MagicMock.
Importing it once here, before any window opens, keeps boot out of the mocked import.
"""

import hermes_bootstrap
import pytest


@pytest.fixture
def route_profiles(monkeypatch):
    """``route(server, home_for)``: resolve RPC ``profile`` names through ``home_for(name) -> home | None`` at
    the seam production reads (``server._resolve_profile_home``; ``_profile_home`` wraps it), pairing each
    home with the generation a real resolution captures for it."""
    def route(server, home_for):
        def resolve(profile):
            home = home_for(profile)
            return home, None if home is None else server._capture_profile_incarnation(home)
        monkeypatch.setattr(server, "_resolve_profile_home", resolve)
    return route
