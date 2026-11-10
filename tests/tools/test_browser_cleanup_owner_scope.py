"""Shutdown teardown runs each session under the profile that owns it.

``cleanup_all_browsers`` (gateway shutdown, atexit) runs outside any profile scope, like the janitor. A cloud
session belongs to the provider and credentials of the profile that created it: closed under the launch
profile's scope instead, a multiplexed profile's Browserbase / Browser Use session is never closed (the launch
profile has no such provider) and keeps billing until the provider's own timeout.
"""

import pytest

from hermes_constants import get_hermes_home
from tools import browser_launcher as launcher
from tools import browser_session as bs
from tools import browser_session_cloud
from tools import browser_session_lifecycle as lifecycle


@pytest.fixture
def isolated_sessions(monkeypatch):
    for name in ("_active_sessions", "_session_last_activity", "_session_owner_homes", "_cleanup_failures"):
        monkeypatch.setattr(bs, name, {})
    monkeypatch.setattr(launcher, "_browsers", {})
    monkeypatch.setattr(lifecycle, "start_cleanup_thread", lambda: None)
    return bs._active_sessions


class _ProfileProvider:
    """The cloud provider the CURRENT profile scope resolves, recording which profile closed what."""

    closed: list = []

    def __init__(self):
        self.home = str(get_hermes_home())

    def close_session(self, session_id):
        _ProfileProvider.closed.append((self.home, session_id))


def test_shutdown_closes_each_cloud_session_under_its_owning_profile(tmp_path, monkeypatch, isolated_sessions):
    launch, routed = tmp_path / "launch", tmp_path / "routed"
    for home in (launch, routed):
        home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(_ProfileProvider, "closed", [])
    monkeypatch.setattr(browser_session_cloud, "get_cloud_provider", _ProfileProvider)
    isolated_sessions["task@routed"] = {"session_name": "bb_1", "bb_session_id": "bb-routed", "features": {},
                                        "cdp_url": "wss://cloud.example/x"}
    bs._session_owner_homes["task@routed"] = str(routed)

    lifecycle.cleanup_all_browsers()

    assert _ProfileProvider.closed == [(str(routed), "bb-routed")]
