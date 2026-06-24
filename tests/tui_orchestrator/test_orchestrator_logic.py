"""Prove the inversion LOGIC with injected fakes — no real bun/gateway/ws.

This validates the load-bearing claim ("kill the renderer, lose nothing"):
the gateway (anchor) is spawned ONCE and never torn down when only the
renderer dies; the renderer is respawned within budget and re-attaches.
"""
from __future__ import annotations

import sys
import types

# Stub the port helpers so no real socket work happens in the loop logic test.
import tui_gateway.orchestrator as orch_mod
from tui_gateway.orchestrator import Orchestrator, OrchestratorConfig, _RespawnBudget


class FakeProc:
    """Minimal Popen stand-in. poll() returns None until .die(code) is called."""

    def __init__(self, label: str):
        self.label = label
        self._rc = None
        self.returncode = None
        self.terminated = False

    def die(self, code: int):
        self._rc = code
        self.returncode = code

    def poll(self):
        return self._rc

    def terminate(self):
        self.terminated = True
        if self._rc is None:
            self.die(-15)

    def wait(self, timeout=None):
        return self._rc if self._rc is not None else 0

    def kill(self):
        self.die(-9)


def make_orch(monkeypatch_attrs=None):
    """Build an Orchestrator whose spawn callables record + return FakeProcs,
    and whose port/wait helpers are stubbed to avoid real sockets."""
    spawned = {"gateway": [], "renderer": []}

    def spawn_gateway(host, port, cred):
        p = FakeProc(f"gateway@{host}:{port}")
        spawned["gateway"].append(p)
        return p

    def spawn_renderer(url, resume_sid=None):
        p = FakeProc(f"renderer->{url}")
        p.resume_sid = resume_sid
        spawned["renderer"].append(p)
        return p

    cfg = OrchestratorConfig(
        spawn_gateway=spawn_gateway,
        spawn_renderer=spawn_renderer,
        port=12345,
        poll_interval_s=0.0,  # spin fast in tests
    )
    orch = Orchestrator(cfg)
    return orch, spawned


def run_for(orch, *, steps, mutate):
    """Drive the loop manually: we can't call run() (it blocks), so we replicate
    its body deterministically by stepping the same transitions the loop uses.
    Instead we monkeypatch time.sleep to fire `mutate(i)` each tick and stop
    after `steps`."""
    import tui_gateway.orchestrator as m

    state = {"i": 0}
    orig_sleep = m.time.sleep

    def fake_sleep(_):
        i = state["i"]
        state["i"] += 1
        mutate(i, orch)
        if state["i"] >= steps:
            orch.request_stop()

    m.time.sleep = fake_sleep
    # Stub gateway-ready wait so _start_gateway succeeds without a real port.
    m._wait_for_port = lambda *a, **k: True
    try:
        return orch.run()
    finally:
        m.time.sleep = orig_sleep


def test_kill_renderer_keeps_gateway():
    """THE core claim: a renderer crash respawns the renderer but NEVER the
    gateway. Anchor spawned exactly once; renderer spawned twice."""
    orch, spawned = make_orch()

    def mutate(i, o):
        if i == 1:
            o._renderer.die(1)  # renderer OOM/crash on tick 1

    run_for(orch, steps=4, mutate=mutate)

    assert len(spawned["gateway"]) == 1, f"gateway respawned! {len(spawned['gateway'])}"
    assert len(spawned["renderer"]) == 2, f"renderer not respawned: {len(spawned['renderer'])}"
    print("PASS test_kill_renderer_keeps_gateway: gateway=1 spawn, renderer=2 spawns (re-attach)")


def test_gateway_death_respawns_both():
    """If the anchor itself dies, respawn gateway AND renderer (its ws dropped)."""
    orch, spawned = make_orch()

    def mutate(i, o):
        if i == 1:
            o._gateway.die(1)

    run_for(orch, steps=4, mutate=mutate)

    assert len(spawned["gateway"]) == 2, f"gateway not respawned: {len(spawned['gateway'])}"
    assert len(spawned["renderer"]) == 2, f"renderer not respawned: {len(spawned['renderer'])}"
    print("PASS test_gateway_death_respawns_both: gateway=2, renderer=2")


def test_clean_quit_tears_down():
    """Renderer exit 0 (user /quit) stops the orchestrator; no respawn."""
    orch, spawned = make_orch()

    def mutate(i, o):
        if i == 1:
            o._renderer.die(0)

    run_for(orch, steps=10, mutate=mutate)

    assert len(spawned["renderer"]) == 1, f"respawned after clean quit: {len(spawned['renderer'])}"
    assert orch._renderer_quit is True
    print("PASS test_clean_quit_tears_down: no respawn after exit 0")


def test_respawn_budget_bounds_crashloop():
    """A renderer that crashes every tick is bounded by the budget, not infinite."""
    orch, spawned = make_orch()
    orch.cfg.renderer_respawn = _RespawnBudget(limit=3, window_s=1000.0)

    def mutate(i, o):
        # Kill the renderer every tick it's alive.
        if o._renderer is not None and o._renderer.poll() is None:
            o._renderer.die(1)

    run_for(orch, steps=50, mutate=mutate)

    # initial + at most `limit` respawns, then bail.
    assert len(spawned["renderer"]) <= 1 + 3, f"crashloop not bounded: {len(spawned['renderer'])}"
    print(f"PASS test_respawn_budget_bounds_crashloop: renderer spawns bounded at {len(spawned['renderer'])}")


def test_budget_unit():
    b = _RespawnBudget(limit=2, window_s=10.0)
    assert b.allow(0.0) is True
    assert b.allow(1.0) is True
    assert b.allow(2.0) is False  # 3rd within window denied
    assert b.allow(100.0) is True  # window slid
    print("PASS test_budget_unit: sliding window correct")


def test_respawn_reads_resume_sid_from_active_file():
    """On a crash-respawn the orchestrator reads the live sid from the active-
    session file and passes it as resume_sid so the fresh renderer resumes the
    live session (the core 'recycle lands back on the session' mechanism)."""
    import json
    import tempfile

    orch, spawned = make_orch()
    # Point the orchestrator at a temp active-session file containing a live sid,
    # as the renderer would have written via writeActiveSessionFile.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"session_id": "live-sid-9f"}, fh)
        orch.cfg.active_session_file = fh.name

    def mutate(i, o):
        if i == 1:
            o._renderer.die(1)  # crash → respawn with resume

    run_for(orch, steps=4, mutate=mutate)

    assert len(spawned["renderer"]) == 2
    # First spawn is a cold start (no resume); the respawn carries the sid.
    assert spawned["renderer"][0].resume_sid is None
    assert spawned["renderer"][1].resume_sid == "live-sid-9f", (
        f"respawn didn't resume live sid: {spawned['renderer'][1].resume_sid}"
    )
    print("PASS test_respawn_reads_resume_sid_from_active_file: respawn resumed live-sid-9f")


def test_corrupt_active_file_yields_no_resume():
    """A missing/corrupt active-session file means no resume hint — the fresh
    renderer cold-starts (today's behaviour), never crashes the orchestrator."""
    import tempfile

    orch, spawned = make_orch()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        fh.write("{not json")
        orch.cfg.active_session_file = fh.name

    def mutate(i, o):
        if i == 1:
            o._renderer.die(1)

    run_for(orch, steps=4, mutate=mutate)
    assert spawned["renderer"][1].resume_sid is None
    print("PASS test_corrupt_active_file_yields_no_resume: graceful cold-start fallback")


def test_recycle_exit_code_preserves_session():
    """Recycle-vs-quit disambiguation (the sweeper's #1 ask): a renderer that
    exits with RECYCLE_EXIT_CODE is a DELIBERATE seamless recycle, NOT a user
    /quit. The supervisor must keep the gateway (anchor) alive and respawn a
    fresh renderer that re-attaches — the session survives. This is the exact
    path the old `code == 0` logic destroyed."""
    from tui_gateway.orchestrator import RECYCLE_EXIT_CODE

    orch, spawned = make_orch()

    def mutate(i, o):
        if i == 1:
            o._renderer.die(RECYCLE_EXIT_CODE)  # deliberate recycle

    run_for(orch, steps=4, mutate=mutate)

    # Gateway (durable anchor) NEVER torn down; renderer respawned to re-attach.
    assert len(spawned["gateway"]) == 1, f"gateway torn down on recycle! {len(spawned['gateway'])}"
    assert len(spawned["renderer"]) == 2, f"renderer not respawned on recycle: {len(spawned['renderer'])}"
    # Crucially NOT treated as a user quit.
    assert orch._renderer_quit is False, "recycle wrongly treated as a user /quit"
    print("PASS test_recycle_exit_code_preserves_session: gateway=1, renderer=2, not-a-quit")


def test_recycle_respawn_resumes_live_sid():
    """End-to-end for the recycle path the sweeper flagged: a RECYCLE_EXIT_CODE
    exit respawns the renderer AND passes the live sid (read from the active-
    session file) so the fresh renderer resumes the SAME live session over the
    still-alive gateway. Proves the recycle 'lands back on the session'."""
    import json
    import tempfile

    from tui_gateway.orchestrator import RECYCLE_EXIT_CODE

    orch, spawned = make_orch()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"session_id": "recycle-sid-4c"}, fh)
        orch.cfg.active_session_file = fh.name

    def mutate(i, o):
        if i == 1:
            o._renderer.die(RECYCLE_EXIT_CODE)

    run_for(orch, steps=4, mutate=mutate)

    assert len(spawned["gateway"]) == 1
    assert len(spawned["renderer"]) == 2
    assert spawned["renderer"][0].resume_sid is None  # cold start
    assert spawned["renderer"][1].resume_sid == "recycle-sid-4c", (
        f"recycle respawn didn't resume live sid: {spawned['renderer'][1].resume_sid}"
    )
    print("PASS test_recycle_respawn_resumes_live_sid: recycle re-attached to recycle-sid-4c")


def test_only_exit_zero_tears_down():
    """The disambiguation invariant, stated as a contrast: exit 0 is the ONLY
    code that stops the orchestrator. RECYCLE_EXIT_CODE and an arbitrary crash
    code both keep the session alive. Guards against a regression that would
    make any non-zero code (or, worse, code 0's old collision with recycle)
    tear the session down."""
    from tui_gateway.orchestrator import RECYCLE_EXIT_CODE

    for code, should_quit in [(0, True), (RECYCLE_EXIT_CODE, False), (1, False), (137, False)]:
        orch, spawned = make_orch()

        def mutate(i, o, _code=code):
            if i == 1:
                o._renderer.die(_code)

        run_for(orch, steps=6, mutate=mutate)
        assert orch._renderer_quit is should_quit, (
            f"exit {code}: expected quit={should_quit}, got {orch._renderer_quit}"
        )
        if should_quit:
            assert len(spawned["renderer"]) == 1, f"exit {code} respawned despite being a quit"
        else:
            assert len(spawned["renderer"]) == 2, f"exit {code} did not respawn (session lost)"
            assert len(spawned["gateway"]) == 1, f"exit {code} tore down the gateway"
    print("PASS test_only_exit_zero_tears_down: only code 0 quits; recycle/crash preserve session")


if __name__ == "__main__":
    test_budget_unit()
    test_kill_renderer_keeps_gateway()
    test_gateway_death_respawns_both()
    test_clean_quit_tears_down()
    test_recycle_exit_code_preserves_session()
    test_recycle_respawn_resumes_live_sid()
    test_only_exit_zero_tears_down()
    test_respawn_budget_bounds_crashloop()
    test_respawn_reads_resume_sid_from_active_file()
    test_corrupt_active_file_yields_no_resume()
    print("\nALL ORCHESTRATOR LOGIC TESTS PASSED")
