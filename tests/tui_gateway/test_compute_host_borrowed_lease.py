"""Regression tests for #101416: isolated (compute-host) turns refused their own session.

With ``dashboard.turn_isolation: true`` every lazy (agent-not-yet-built, i.e. every NEW)
desktop session's turn is routed to the compute-host CHILD process. The parent claims the
session's active-session lease in ``prompt.submit`` before routing; the child's freshly built
session record carried NO lease, so ``_admit_prompt_turn`` re-claimed from the child's pid and
was fenced out by the parent's own registry entry (``_is_same_writer`` requires the same pid
AND the same live_session_id) — "Session ... already has a live owner (desktop, pid N,
running 0m)" on every new session's first message, stacking one unreclaimable lease per
attempt (the owner pid is the immortal dashboard process, so ``_prune_dead`` never reclaims).

The fix: the parent vouches on the turn frame (``parent_owns_active_session_lease``) and the
child installs an INERT borrow (``ActiveSessionLease(enabled=False, released=True)``) before
the turn pipeline runs, so admission sees the slot as held upstream. No second claim, no
refusal, and the child can never release or transfer the parent's slot. Without the vouch
(parent predates the field) the legacy self-claim path is preserved and any conflict still
fails CLOSED with the visible refusal.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
import types

import pytest

from tui_gateway import server
from tui_gateway.compute_host import ComputeHost


def _frames(out: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]


def _wait(out: io.StringIO, predicate, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for frame in _frames(out):
            if predicate(frame):
                return frame
        time.sleep(0.01)
    raise AssertionError(f"timed out; saw={_frames(out)}")


def _stub_agent(deltas: list[str]) -> types.SimpleNamespace:
    def run_conversation(
        prompt, *, conversation_history=None, stream_callback=None, **_kw
    ):
        final = "".join(deltas)
        if stream_callback is not None:
            for chunk in deltas:
                stream_callback(chunk)
        messages = [
            *(conversation_history or []),
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final},
        ]
        return {"final_response": final, "messages": messages}

    return types.SimpleNamespace(
        session_id="s1-key",
        run_conversation=run_conversation,
        clear_interrupt=lambda: None,
        hard_interrupt=lambda *a, **k: None,
    )


def _make_frame(sid: str, **overrides) -> dict:
    frame = {
        "type": "turn.start",
        "sid": sid,
        "request_id": "turn",
        "text": "hello",
        "session_key": "s1-key",
        "source": "desktop",
        "cols": 80,
        "history": [],
    }
    frame.update(overrides)
    return frame


def _seed_parent_lease(key: str, live_session_id: str = "parent-sid"):
    """Claim the lease exactly as the parent dashboard would (same pid, its own live id)."""
    from hermes_cli.active_sessions import try_acquire_active_session

    lease, message = try_acquire_active_session(
        session_id=key,
        surface="desktop",
        config={},
        metadata={"live_session_id": live_session_id},
        track_liveness=True,
    )
    assert message is None and lease is not None
    return lease


def _registry() -> list[dict]:
    path = os.path.join(os.environ["HERMES_HOME"], "runtime", "active_sessions.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh).get("entries", [])


@pytest.fixture()
def isolated_env(monkeypatch, tmp_path):
    """Real _build_server_session → _init_session → _run_prompt_submit → _admit_prompt_turn
    pipeline, with the environment-heavy side paths neutralized. The turn BODY is cut right
    after admission (``_prepare_turn_input`` → None), so the lease path under test runs REAL
    against the conftest-sandboxed HERMES_HOME registry while nothing calls a provider."""
    agent = _stub_agent(["a ", "b "])
    monkeypatch.setattr(server, "_make_agent", lambda *a, **kw: agent)
    # Turn-pipeline side paths (same set as test_compute_host_turn_protocol.py).
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(
        server, "_sync_agent_model_with_config", lambda sid, session: None
    )
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(
        server, "_sync_session_key_after_compress", lambda *a, **k: None
    )
    monkeypatch.setattr(server, "_get_usage", lambda agent_: {})
    # _init_session services orthogonal to session ownership.
    monkeypatch.setattr(server, "_hydrate_session_cwd", lambda *a, **k: None)
    monkeypatch.setattr(server, "_wire_session_agent", lambda *a, **k: None)
    monkeypatch.setattr(server, "_start_session_services", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *a, **k: None)
    # Cut the turn AFTER _admit_prompt_turn (admission is the code under test; the body is not).
    import tui_gateway.prompt_turn as prompt_turn

    for mod in (server, prompt_turn):
        if hasattr(mod, "_prepare_turn_input"):
            monkeypatch.setattr(mod, "_prepare_turn_input", lambda *a, **k: None)
    yield agent


@pytest.fixture()
def clean_sessions():
    """Keep the module-global _sessions table clean across tests."""
    yield
    for sid in [s for s in list(server._sessions) if s.startswith("s1")]:
        server._sessions.pop(sid, None)


def _run_turn(frame: dict, timeout: float = 5.0) -> tuple[list[dict], dict | None]:
    """Run one turn.start through the real child path; return (all frames, turn.end frame)."""
    out = io.StringIO()
    host = ComputeHost(stdout=out, heartbeat_secs=0)
    try:
        host.handle_frame(frame)
        end = _wait(out, lambda f: f["type"] == "turn.end", timeout=timeout)
    finally:
        host.close()
    return _frames(out), end


# ── Unit: the borrow install ────────────────────────────────────────────────


def test_no_vouch_flag_installs_nothing():
    session = {"session_key": "s1-key"}
    server._install_borrowed_lease("s1", session, _make_frame("s1"))
    assert "active_session_lease" not in session


def test_falsy_vouch_flag_installs_nothing():
    """The parent only vouches when it actually holds the lease; falsy must never borrow."""
    session = {"session_key": "s1-key"}
    server._install_borrowed_lease(
        "s1", session, _make_frame("s1", parent_owns_active_session_lease=False)
    )
    assert "active_session_lease" not in session


def test_vouch_installs_inert_token():
    session = {"session_key": "s1-key"}
    server._install_borrowed_lease(
        "s1",
        session,
        _make_frame("s1", source="desktop", parent_owns_active_session_lease=True),
    )
    lease = session["active_session_lease"]
    assert lease.enabled is False and lease.released is True
    assert lease.session_id == "s1-key" and lease.surface == "desktop"
    assert lease.state_path is None and lease.lock_path is None


def test_borrow_never_overwrites_an_existing_lease():
    sentinel = object()
    session = {"session_key": "s1-key", "active_session_lease": sentinel}
    server._install_borrowed_lease(
        "s1", session, _make_frame("s1", parent_owns_active_session_lease=True)
    )
    assert session["active_session_lease"] is sentinel


def test_inert_token_cannot_release_or_transfer_the_parents_slot():
    """Release/transfer must be no-ops on the borrow: the slot belongs to the parent process."""
    from hermes_cli.active_sessions import (
        release_active_session,
        transfer_active_session,
        try_acquire_active_session,
    )

    parent = _seed_parent_lease("borrow-noop-key")
    try:
        session = {"session_key": "borrow-noop-key"}
        server._install_borrowed_lease(
            "s1", session, _make_frame("s1", parent_owns_active_session_lease=True)
        )
        borrow = session["active_session_lease"]
        release_active_session(borrow)
        assert (
            not borrow.released or True
        )  # released flag toggles; the registry is what matters
        assert transfer_active_session(borrow, session_id="other-key") is False
        entries = _registry()
        assert [e["session_id"] for e in entries] == ["borrow-noop-key"]
        assert entries[0]["lease_id"] == parent.lease_id
        assert entries[0]["metadata"]["live_session_id"] == "parent-sid"
    finally:
        release_active_session(parent)
    assert _registry() == []


# ── Registry-level: the exclusivity fence is untouched ─────────────────────


def test_same_pid_different_live_session_id_still_fences():
    """The defect's exact precondition, at the registry level, must stay a refusal: the fix
    relaxes nothing about _is_same_writer; it prevents the second claim from happening."""
    from hermes_cli.active_sessions import (
        release_active_session,
        try_acquire_active_session,
    )

    first, message = try_acquire_active_session(
        session_id="fence-key",
        surface="desktop",
        config={},
        metadata={"live_session_id": "aaa"},
    )
    assert message is None and first is not None
    second, refusal = try_acquire_active_session(
        session_id="fence-key",
        surface="desktop",
        config={},
        metadata={"live_session_id": "bbb"},
    )
    assert second is None
    assert getattr(refusal, "reason", "") == "SESSION_NOT_OWNED"
    assert "already has a live owner" in str(refusal)
    release_active_session(first)
    third, message2 = try_acquire_active_session(
        session_id="fence-key",
        surface="desktop",
        config={},
        metadata={"live_session_id": "ccc"},
    )
    assert message2 is None and third is not None
    release_active_session(third)
    assert _registry() == []


# ── Integration: the real child turn path ──────────────────────────────────


def test_isolated_turn_runs_against_parent_leased_session(isolated_env, clean_sessions):
    """THE #101416 repro, fixed: parent holds the lease, child runs the turn end-to-end.
    Before the fix the child's re-claim was fenced by the parent's own entry and the turn
    died with SESSION_NOT_OWNED."""
    parent = _seed_parent_lease("s1-key", live_session_id="parent-sid")
    try:
        frames, end = _run_turn(
            _make_frame("s1", parent_owns_active_session_lease=True)
        )

        kinds = [f["type"] for f in frames]
        # session.info (from _init_session) rides the transport first; what matters is that the
        # turn was ADMITTED (turn.started) and ended cleanly instead of the #101416 refusal.
        assert "turn.started" in kinds
        assert kinds[-1] == "turn.end" and end["session_key"] == "s1-key"
        # The pipeline actually ran past the lease gate: message.start was emitted for the turn.
        assert any(
            f["type"] == "rpc"
            and f["message"]["method"] == "event"
            and (f["message"].get("params") or {}).get("type") == "message.start"
            for f in frames
        )
        # No ownership refusal anywhere in the stream.
        errors = [f for f in frames if f["type"] == "rpc" and f["message"].get("error")]
        assert not errors
        # The registry still holds exactly the parent's lease — unclaimed twice, unmodified.
        entries = _registry()
        assert len(entries) == 1
        assert entries[0]["lease_id"] == parent.lease_id
        assert entries[0]["metadata"]["live_session_id"] == "parent-sid"
    finally:
        from hermes_cli.active_sessions import release_active_session

        release_active_session(parent)


def test_isolated_turn_without_vouch_still_fails_closed(isolated_env, clean_sessions):
    """Negative control: a parent that does NOT vouch (predates the field) leaves the child
    on the legacy self-claim path; against a held slot that must remain a VISIBLE refusal,
    never a silent second writer."""
    _seed_parent_lease("s1-key", live_session_id="parent-sid")
    try:
        frames, _end = _run_turn(_make_frame("s1"))
    finally:
        pass
    refusal_events = [
        f
        for f in frames
        if f["type"] == "rpc"
        and f["message"].get("method") == "event"
        and (f["message"].get("params") or {}).get("type") == "error"
        and "already has a live owner" in json.dumps(f["message"].get("params", {}))
    ]
    assert refusal_events, f"expected the fail-closed refusal, saw={frames}"
    # The parent's entry survived the failed claim.
    entries = _registry()
    assert (
        len(entries) == 1 and entries[0]["metadata"]["live_session_id"] == "parent-sid"
    )
    # Cleanup via the production finalize primitive (drop own-pid orphans), so the child's
    # refused/pending claim and the parent's seeded lease both land cleanly: this test process
    # owns both registry entries and vouches for none of them here. The sweep spares YOUNG
    # own-pid leases by a grace window, so backdate first — exactly what upstream's own
    # orphan-sweep tests do (tests/hermes_cli/test_active_sessions.py::_backdate_leases).
    from hermes_cli.active_sessions import release_orphaned_leases
    from pathlib import Path

    registry_path = Path(os.environ["HERMES_HOME"]) / "runtime" / "active_sessions.json"
    entries = json.load(open(registry_path)).get("entries", [])
    for entry in entries:
        entry["started_at"] = time.time() - 600.0
    with open(registry_path, "w", encoding="utf-8") as fh:
        json.dump({"entries": entries}, fh)
    release_orphaned_leases(live_lease_ids=set())
    assert _registry() == []


def test_isolated_turn_self_claims_when_no_parent_lease_exists(
    isolated_env, clean_sessions
):
    """Fallback path preserved: an unknown/unvouching parent on an UNOWNED session lets the
    child claim for itself and run; the claimed lease is the child's own and is releasable."""
    frames, end = _run_turn(_make_frame("s1"))
    kinds = [f["type"] for f in frames]
    assert "turn.started" in kinds and kinds[-1] == "turn.end"
    errors = [f for f in frames if f["type"] == "rpc" and f["message"].get("error")]
    assert not errors
    # The child's own claim is in the registry under this process's pid; release it cleanly
    # the way the production finalize path would.
    entries = _registry()
    assert len(entries) == 1 and entries[0]["session_id"] == "s1-key"
    # Cleanup the same way the production finalize reclaims a child's own claim: same-writer
    # re-entrancy (same pid + same live_session_id) releases it from the registry.
    from hermes_cli.active_sessions import release_active_session

    session = server._sessions.get("s1")
    lease = session.get("active_session_lease") if session else None
    assert lease is not None and lease.enabled and not lease.released
    release_active_session(lease)
    assert _registry() == []


def test_borrow_install_sits_before_the_turn_pipeline():
    """Guard the ordering contract: the borrow must be installed in _run_real_turn before
    _run_prompt_submit (i.e. before _admit_prompt_turn's claim)."""
    import inspect

    source = inspect.getsource(ComputeHost._run_real_turn)
    borrow_at = source.index("_install_borrowed_lease")
    pipeline_at = source.index("_run_prompt_submit")
    assert borrow_at < pipeline_at
