"""Stage 3 reaper tests — pure decision logic + execute, fully injected.
No real processes touched. Proves every safety invariant.
"""
from __future__ import annotations

import signal

from tui_gateway.reaper import (
    ProcInfo,
    ReapPlan,
    execute_reap,
    plan_reap,
)


MY = 1000  # this orchestrator's pid


def P(pid, owner, role=None, hb=None):
    return ProcInfo(pid=pid, owner_pid=owner, role=role, heartbeat_age_s=hb)


def test_orphan_reaped_when_owner_orchestrator_gone():
    snap = [P(2001, owner=9999, role="renderer")]  # owner 9999 is NOT alive
    plan = plan_reap(snap, my_orchestrator_pid=MY, live_pids=[], alive_orchestrator_pids=[MY])
    assert plan.orphans and plan.orphans[0][0] == 2001
    assert 2001 in plan.all_pids
    print("PASS orphan: owner-gone child reaped")


def test_live_pid_never_reaped_even_if_frozen():
    # A tracked renderer with a very stale heartbeat — but it's a LIVE pid the
    # orchestrator owns. The reaper must NOT race the orchestrator: keep it.
    snap = [P(2002, owner=MY, role="renderer", hb=9999)]
    plan = plan_reap(snap, my_orchestrator_pid=MY, live_pids=[2002], alive_orchestrator_pids=[MY])
    assert plan.is_empty(), f"live pid was reaped: {plan.all_pids}"
    print("PASS safety: live tracked pid never reaped (even frozen)")


def test_own_orchestrator_pid_never_reaped():
    snap = [P(MY, owner=MY, role=None)]
    plan = plan_reap(snap, my_orchestrator_pid=MY, live_pids=[], alive_orchestrator_pids=[MY])
    assert plan.is_empty()
    print("PASS safety: own orchestrator pid never reaped")


def test_unmarked_process_never_touched():
    # owner_pid None ⇒ not an orchestrator child ⇒ ignored entirely.
    snap = [P(3001, owner=None, role="renderer", hb=9999)]
    plan = plan_reap(snap, my_orchestrator_pid=MY, live_pids=[], alive_orchestrator_pids=[MY])
    assert plan.is_empty(), "an unmarked (foreign) process was targeted!"
    print("PASS safety: unmarked/foreign process never targeted")


def test_frozen_renderer_reaped():
    # Owned by us, NOT a live pid (a leftover), heartbeat stale → frozen.
    snap = [P(2003, owner=MY, role="renderer", hb=120)]
    plan = plan_reap(
        snap, my_orchestrator_pid=MY, live_pids=[], alive_orchestrator_pids=[MY], heartbeat_stale_s=90
    )
    assert any(pid == 2003 for pid, _ in plan.frozen + plan.dupes)
    print("PASS frozen: stale-heartbeat leftover renderer reaped")


def test_fresh_heartbeat_leftover_only_deduped_not_frozen():
    # Owned by us, not live, but heartbeat FRESH → not 'frozen'; still a stray
    # leftover renderer so dedup catches it.
    snap = [P(2004, owner=MY, role="renderer", hb=5)]
    plan = plan_reap(
        snap, my_orchestrator_pid=MY, live_pids=[], alive_orchestrator_pids=[MY], heartbeat_stale_s=90
    )
    assert not plan.frozen
    assert any(pid == 2004 for pid, _ in plan.dupes)
    print("PASS dedup: fresh-but-stray renderer deduped, not mislabelled frozen")


def test_other_live_orchestrators_child_not_reaped():
    # owner 2000 is a DIFFERENT alive orchestrator. Not ours → we don't touch it.
    snap = [P(2005, owner=2000, role="renderer", hb=9999)]
    plan = plan_reap(
        snap, my_orchestrator_pid=MY, live_pids=[], alive_orchestrator_pids=[MY, 2000]
    )
    assert plan.is_empty(), "reaped another live orchestrator's child!"
    print("PASS multi-instance: another live orchestrator's child left alone")


def test_no_double_listing_frozen_and_dupe():
    snap = [P(2006, owner=MY, role="renderer", hb=200)]
    plan = plan_reap(
        snap, my_orchestrator_pid=MY, live_pids=[], alive_orchestrator_pids=[MY], heartbeat_stale_s=90
    )
    # all_pids dedupes; ensure 2006 appears once across buckets.
    assert plan.all_pids.count(2006) == 1
    print("PASS hygiene: a pid flagged frozen is not also double-killed as dupe")


def test_execute_reap_sigterms_each_pid_once():
    plan = ReapPlan(orphans=[(11, "o")], frozen=[(12, "f")], dupes=[(13, "d")])
    calls = []
    execute_reap(plan, kill=lambda pid, sig: calls.append((pid, sig)), log=None)
    assert calls == [(11, signal.SIGTERM), (12, signal.SIGTERM), (13, signal.SIGTERM)]
    print("PASS execute: each planned pid SIGTERMed once")


def test_execute_reap_tolerates_already_gone():
    plan = ReapPlan(orphans=[(21, "o"), (22, "o")])
    def kill(pid, sig):
        if pid == 21:
            raise ProcessLookupError
    signalled = execute_reap(plan, kill=kill, log=None)
    assert signalled == [22], f"expected only 22 signalled, got {signalled}"
    print("PASS execute: already-exited pid handled gracefully")


def test_execute_reap_skips_on_permission_error():
    plan = ReapPlan(orphans=[(31, "o")])
    logs = []
    def kill(pid, sig):
        raise PermissionError
    signalled = execute_reap(plan, kill=kill, log=logs.append)
    assert signalled == []
    assert any("not permitted" in m for m in logs)
    print("PASS execute: permission-denied pid skipped + logged, not crashed")


if __name__ == "__main__":
    test_orphan_reaped_when_owner_orchestrator_gone()
    test_live_pid_never_reaped_even_if_frozen()
    test_own_orchestrator_pid_never_reaped()
    test_unmarked_process_never_touched()
    test_frozen_renderer_reaped()
    test_fresh_heartbeat_leftover_only_deduped_not_frozen()
    test_other_live_orchestrators_child_not_reaped()
    test_no_double_listing_frozen_and_dupe()
    test_execute_reap_sigterms_each_pid_once()
    test_execute_reap_tolerates_already_gone()
    test_execute_reap_skips_on_permission_error()
    print("\nALL REAPER TESTS PASSED")
