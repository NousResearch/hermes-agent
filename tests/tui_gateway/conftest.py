"""tui_gateway test fixtures.

Several files here import ``tui_gateway.server`` inside a ``patch.dict("sys.modules", {"hermes_constants":
MagicMock(...)})`` window so the module binds a fixed home. The server's import graph reaches
``agent.process_bootstrap`` → ``hermes_bootstrap``, which is process boot: PM dependency activation reads
the real install root through ``hermes_constants`` and exits the process when that is a MagicMock.
Importing it once here, before any window opens, keeps boot out of the mocked import.
"""

import hermes_bootstrap

import pytest


@pytest.fixture(autouse=True)
def _stub_managed_chromium(monkeypatch):
    """Keep the PM store's owning-root borrow out of every tui_gateway test.

    ``pm.environments._unstamped_store`` resolves a source checkout's tools store through
    ``owning_home_root``: a checkout that sits under a real Hermes home (``~/​.hermes/hermes-agent``,
    the documented borrowing case) resolves ``<that home>/tools`` even when the test sandboxed
    ``HERMES_HOME``. Reading the store's payload manifest then touches the real home — which the
    ``home_io_guard`` refuses as a TEST BUG. That is correct product behaviour (a borrowing launch
    must not re-download tools), so tests that reach ``display.*`` (whose status snapshot probes the
    managed Chromium) stub the resolution instead: CI checks out outside any Hermes home and never
    sees this; a source install does.
    """
    from hermes_cli import browser_runtime

    monkeypatch.setattr(browser_runtime, "chromium_executable", lambda **_kwargs: None)
