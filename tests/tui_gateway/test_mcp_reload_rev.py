"""reload.mcp revision-aware coalescing (review on #20379, finding 1).

The TUI's config poll sends the ``mcp_rev`` it observed with each reload
request. The server must guarantee that a success response means THAT
revision (or a newer one) was actually loaded:

- The leader re-hashes the MCP-relevant config after discovery and repeats
  until the hash is stable, so a config edit racing a slow reload can't be
  silently skipped.
- A follower that waited behind a leader coalesces only when the leader's
  loaded revision matches the follower's requested revision; otherwise it
  re-runs the full reload itself.
- A failed reload returns a JSON-RPC error and does not advance the
  generation, so a follower behind a failed leader re-runs too.

Each test file runs in its own subprocess (run_tests.sh isolation), but the
fixtures still restore the module globals they touch.
"""

from __future__ import annotations

import threading

import pytest

import tools.mcp_tool as mcp_tool
import tui_gateway.server as srv


@pytest.fixture()
def reload_env(monkeypatch):
    """Neutralize side effects and expose call-counting fakes."""
    calls = {"discover": 0, "shutdown": 0}
    rev_box = {"rev": "rev-a"}

    monkeypatch.setattr(mcp_tool, "shutdown_mcp_servers", lambda: calls.__setitem__("shutdown", calls["shutdown"] + 1))
    monkeypatch.setattr(mcp_tool, "discover_mcp_tools", lambda: calls.__setitem__("discover", calls["discover"] + 1))
    monkeypatch.setattr(srv, "_compute_mcp_rev", lambda: rev_box["rev"])

    saved = (srv._mcp_reload_gen, srv._mcp_reload_loaded_rev)
    srv._mcp_reload_gen = 0
    srv._mcp_reload_loaded_rev = ""
    yield calls, rev_box
    srv._mcp_reload_gen, srv._mcp_reload_loaded_rev = saved


def _reload(rev: str | None = None, rid: int = 1) -> dict:
    params: dict = {"session_id": "no-such-session", "confirm": True}
    if rev is not None:
        params["rev"] = rev
    return srv._methods["reload.mcp"](rid, params)


def test_success_reports_loaded_rev(reload_env):
    calls, rev_box = reload_env
    rev_box["rev"] = "rev-a"

    envelope = _reload(rev="rev-a")

    assert envelope["result"]["status"] == "reloaded"
    assert envelope["result"]["loaded_rev"] == "rev-a"
    assert calls["discover"] == 1
    assert srv._mcp_reload_gen == 1


def test_failed_reload_is_an_error_and_no_generation_advance(reload_env, monkeypatch):
    """The exact client-facing contract: a failure must NOT look like an ack.
    quietRpc on the TUI side collapses this error to null and keeps the
    revision un-accepted, so the next poll retries."""
    calls, _ = reload_env

    def _boom():
        raise RuntimeError("flapping server")

    monkeypatch.setattr(mcp_tool, "discover_mcp_tools", _boom)

    envelope = _reload(rev="rev-b")

    assert "error" in envelope
    assert srv._mcp_reload_gen == 0
    assert srv._mcp_reload_loaded_rev == ""


def test_leader_rehashes_until_stable_when_config_changes_mid_reload(reload_env, monkeypatch):
    """Revision A starts a reload; the config changes to revision B while
    discovery is connecting servers. The leader must not mark A complete —
    it re-hashes after discovery and reloads again until stable, so the
    reported loaded_rev is what discovery actually read."""
    calls, _ = reload_env

    # Hash sequence: before pass 1 → rev-a; after pass 1 → rev-b (config
    # changed mid-discovery); after pass 2 → rev-b (stable).
    hashes = iter(["rev-a", "rev-b", "rev-b"])
    monkeypatch.setattr(srv, "_compute_mcp_rev", lambda: next(hashes))

    envelope = _reload(rev="rev-a")

    assert envelope["result"]["loaded_rev"] == "rev-b"
    assert calls["discover"] == 2  # pass 1 read stale config, pass 2 converged
    assert srv._mcp_reload_gen == 1


class _WaiterLock:
    """Lock wrapper that signals when a FOLLOWER enters the blocking ``with``
    path. The handler snapshots ``gen_before`` on the line before ``with``,
    so once ``waiting`` fires the snapshot is already taken and it is safe to
    let the leader complete — no sleep-based ordering."""

    def __init__(self):
        self._lock = threading.Lock()
        self.waiting = threading.Event()

    def acquire(self, blocking: bool = True) -> bool:
        return self._lock.acquire(blocking)

    def release(self) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()

    def __enter__(self):
        self.waiting.set()
        self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self._lock.release()
        return False


def _run_leader_follower(reload_env, monkeypatch, follower_rev):
    """Drive the A-then-B overlap deterministically: the leader blocks inside
    discovery until the follower is queued on the lock, then completes."""
    calls, _rev_box = reload_env

    lock = _WaiterLock()
    monkeypatch.setattr(srv, "_mcp_reload_lock", lock)

    leader_in_discovery = threading.Event()
    release_leader = threading.Event()

    def _slow_discover():
        calls["discover"] += 1
        if calls["discover"] == 1:
            leader_in_discovery.set()
            assert release_leader.wait(timeout=10)

    monkeypatch.setattr(mcp_tool, "discover_mcp_tools", _slow_discover)

    results: dict = {}

    lt = threading.Thread(target=lambda: results.__setitem__("leader", _reload(rev="rev-a", rid=1)), daemon=True)
    lt.start()
    assert leader_in_discovery.wait(timeout=10)

    ft = threading.Thread(target=lambda: results.__setitem__("follower", _reload(rev=follower_rev, rid=2)), daemon=True)
    ft.start()
    # The follower has snapshotted gen_before once it blocks on the lock.
    assert lock.waiting.wait(timeout=10)
    release_leader.set()

    lt.join(timeout=10)
    ft.join(timeout=10)
    assert not lt.is_alive() and not ft.is_alive()

    return results, calls


def test_follower_with_matching_rev_coalesces(reload_env, monkeypatch):
    results, calls = _run_leader_follower(reload_env, monkeypatch, follower_rev="rev-a")

    assert results["leader"]["result"]["status"] == "reloaded"
    assert results["follower"]["result"]["status"] == "reloaded"
    assert results["follower"]["result"].get("coalesced") is True
    # Only the leader ran discovery.
    assert calls["discover"] == 1


def test_follower_with_newer_rev_reruns_full_reload(reload_env, monkeypatch):
    """The race from the review: the leader loaded revision A, but this
    follower was triggered by revision B. Coalescing would ack B against A's
    registry — instead the follower must re-run the full reload."""
    results, calls = _run_leader_follower(reload_env, monkeypatch, follower_rev="rev-b")

    assert results["follower"]["result"]["status"] == "reloaded"
    assert results["follower"]["result"].get("coalesced") is None
    # Leader discovery + follower's own re-run.
    assert calls["discover"] == 2


def test_follower_behind_failed_leader_reruns(reload_env, monkeypatch):
    """A failed leader never advances the generation — the follower re-runs
    the full reload instead of acking over an empty/partial registry."""
    calls, _ = reload_env

    lock = _WaiterLock()
    monkeypatch.setattr(srv, "_mcp_reload_lock", lock)

    leader_in_discovery = threading.Event()
    release_leader = threading.Event()

    def _discover():
        calls["discover"] += 1
        if calls["discover"] == 1:
            leader_in_discovery.set()
            assert release_leader.wait(timeout=10)
            raise RuntimeError("flapping server")

    monkeypatch.setattr(mcp_tool, "discover_mcp_tools", _discover)

    results: dict = {}
    lt = threading.Thread(target=lambda: results.__setitem__("leader", _reload(rev="rev-a", rid=1)), daemon=True)
    lt.start()
    assert leader_in_discovery.wait(timeout=10)

    ft = threading.Thread(target=lambda: results.__setitem__("follower", _reload(rev="rev-a", rid=2)), daemon=True)
    ft.start()
    assert lock.waiting.wait(timeout=10)
    release_leader.set()

    lt.join(timeout=10)
    ft.join(timeout=10)

    assert "error" in results["leader"]
    assert results["follower"]["result"]["status"] == "reloaded"
    assert calls["discover"] == 2


def test_legacy_request_without_rev_still_coalesces_on_generation(reload_env, monkeypatch):
    """Manual /reload-mcp and older clients send no rev — generation-only
    coalescing (the pre-existing contract) still applies."""
    results, calls = _run_leader_follower(reload_env, monkeypatch, follower_rev=None)  # type: ignore[arg-type]

    assert results["follower"]["result"].get("coalesced") is True
    assert calls["discover"] == 1
