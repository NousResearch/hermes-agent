"""A stale active-session lease becomes reclaimable; a live owner keeps its fence.

#112028 (Desktop + mobile dashboard on one stored session) reports the dead end:
the Desktop surface holds the lease, so the dashboard is refused with
``SESSION_NOT_OWNED`` until the Desktop backend restarts. This file covers option
3 of that issue -- reclaim the lease when the owner is *gone* (stopped
heartbeating), never a takeover of a live owner (options 1/2 are shape decisions
for maintainers) -- plus #104691's zombie lease, whose pid is alive but whose
lane stopped vouching for the entry.

The protection direction is asserted here too: a live owner's lease must survive,
because a second writer reasons from a transcript missing the first one's
in-flight turn.
"""

from __future__ import annotations

import time

import pytest

from hermes_cli.active_sessions import (
    LEASE_HEARTBEAT_TIMEOUT_S,
    SESSION_NOT_OWNED,
    active_session_registry_snapshot,
)
from tui_gateway import server


@pytest.fixture(autouse=True)
def _isolated_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_sessions", {})


def _claim(home, surface, live_id, session_id="shared"):
    return server._claim_active_session_slot(
        session_id, live_session_id=live_id, surface=surface, profile_home=home,
    )


def _entry(home) -> dict:
    entries = active_session_registry_snapshot(registry_home=home)
    assert len(entries) == 1, entries
    return entries[0]


def _age(home, seconds: float) -> None:
    """Stop the holder's heartbeat while leaving its pid alive."""
    from hermes_cli.active_sessions import _read_entries, _state_path, _write_entries

    entries = _read_entries(_state_path(home))
    assert entries
    for entry in entries:
        entry["updated_at"] = time.time() - seconds
    _write_entries(_state_path(home), entries)


def test_stale_desktop_lease_goes_to_the_dashboard(tmp_path):
    """The reported dead end: the owner stopped heartbeating, so the claim succeeds."""
    home = tmp_path / "profile"
    desktop_lease, refusal = _claim(home, "desktop", "desktop-runtime")
    assert desktop_lease is not None and refusal is None
    _age(home, LEASE_HEARTBEAT_TIMEOUT_S + 60)

    # No reaper tick has run for the dead lane, so the claimant reclaims it here.
    phone_lease, refusal = _claim(home, "tui", "phone-runtime")

    assert phone_lease is not None, refusal
    assert _entry(home)["surface"] == "tui", "the stale entry must not linger"


def test_fresh_desktop_lease_still_fences_other_surfaces(tmp_path):
    """Desktop's existing lease semantics are unchanged: a live owner is not preempted."""
    home = tmp_path / "profile"
    desktop_lease, _ = _claim(home, "desktop", "desktop-runtime")
    assert desktop_lease is not None

    blocked, refusal = _claim(home, "tui", "phone-runtime")
    assert blocked is None
    assert getattr(refusal, "reason", None) == SESSION_NOT_OWNED
    # ... and the Desktop tab re-claiming its own session is re-entrancy, not a conflict.
    again, refusal = _claim(home, "desktop", "desktop-runtime")
    assert again is not None, refusal
    assert _entry(home)["surface"] == "desktop"


def test_reaper_tick_restamps_the_leases_a_live_backend_holds(tmp_path):
    """The heartbeat comes from the owner's existing reaper tick, not a new subsystem.

    Without it every lease would age out on a long-lived backend and a live owner
    would become stealable -- the hole the reclaim path must not open.
    """
    home = tmp_path / "profile"
    lease, refusal = _claim(home, "desktop", "desktop-runtime")
    assert lease is not None and refusal is None
    server._sessions["sid"] = {"active_session_lease": lease}
    _age(home, LEASE_HEARTBEAT_TIMEOUT_S + 60)

    server._reap_idle_sessions()

    assert time.time() - _entry(home)["updated_at"] < 60, "a held lease must be restamped"
    blocked, refusal = _claim(home, "tui", "phone-runtime")
    assert blocked is None, "a restamped live owner is fenced again"
    assert getattr(refusal, "reason", None) == SESSION_NOT_OWNED
