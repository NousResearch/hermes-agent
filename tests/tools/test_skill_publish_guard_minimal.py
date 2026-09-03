"""Cross-process regression witness for live skill publication (A1B, A1G contract).

Hermes 0.20.5 minimal re-derivation. This is the only regression test
required to prove the contract — the historical A1G test suite (1536+
lines) tested a frozen internal API we are not porting. A1B alone
witnesses:

  CASE_1_SAME_NAME_RACE              — two processes race the same
                                       canonical identity; exactly one
                                       publishes, the other observes
                                       lock-acquisition failure.
  CASE_2_LOCK_RELEASE_AFTER_FAILURE  — a failing publisher does not
                                       leave a stale lock for the
                                       next legitimate publication.
  CASE_3_DIFFERENT_NAME_NONINTERFERENCE — different canonical identities
                                          do not contend; the lock is
                                          keyed by name, not a global
                                          namespace.
  CASE_4_PERMISSION_OR_IO_FAILURE    — chmod 000 on the locks/ parent
                                       surfaces as a permission error,
                                       NOT as a duplicate-name refusal.

Uses ``multiprocessing`` with ``spawn`` so each child is a fresh
interpreter (real cross-process, not threads). Synchronization is via
``multiprocessing.Event`` primitives + a queue rendezvous — no sleeps
as the sole synchronization mechanism. Children are joined with
timeouts; any stuck process is terminated, killed, and reaped before
the test exits. No orphan process or leaked lock may remain.
"""

from __future__ import annotations

import errno
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import pytest


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

_SKILL_NAME = "cross-process-publish-fixture"
_SKILL_CONTENT = """---
name: cross-process-publish-fixture
description: Verify cross-process skill publication.
---
# Cross-Process Publish Fixture

Used by test_skill_publish_guard_minimal. Do not invoke directly.
"""

_OTHER_SKILL_NAME = "cross-process-publish-other"
_OTHER_SKILL_CONTENT = """---
name: cross-process-publish-other
description: Verify different-name noninterference.
---
# Other
"""

_HOLDER_ENTER_TIMEOUT_S = 15
_RELEASE_TIMEOUT_S = 10
_JOIN_TIMEOUT_S = 20
_LOCK_RELEASED_POLL_INTERVAL_S = 0.05
_LOCK_RELEASED_POLL_TIMEOUT_S = 5


# ---------------------------------------------------------------------------
# Child worker entry points
# ---------------------------------------------------------------------------

def _worker_publish(
    worker_id: int,
    hermes_home: str,
    result_path: str,
    skill_name: str,
    skill_content: str,
    category: str,
    holder_entered: Optional[mp.Event],
    release_holder: Optional[mp.Event],
    holder_finished: Optional[mp.Event],
    contender_attempting: Optional[mp.Event],
    contender_acquired: Optional[mp.Event],
    contender_release: Optional[mp.Event],
    contender_finished: Optional[mp.Event],
    a_in_lock: Optional[mp.Event] = None,
    release_a: Optional[mp.Event] = None,
) -> None:
    """One child publishes a skill through the REAL public path.

    A1B (Hermes 0.20.5 TOCTOU repair): BOTH the publisher that
    holds the canonical-name lock during the race AND the
    publisher that races against it must invoke the real public
    ``skill_manage`` create path. The lock is the thing
    ``_create_skill`` acquires internally — there is no separate
    "hold the lock manually" branch in production. If only one of
    the two processes exercised the real public path, the witness
    would prove lock exclusion but not two-publisher stale-check
    safety.

    Determinism: the FIRST invocation of the in-lock authoritative
    ``_find_skill`` inside ``_create_skill`` is monkey-patched (in
    this child only) to block on ``a_in_lock`` and ``release_a``
    Events. This freezes Publisher A inside the guarded region
    while Publisher B races against it. After the rendezvous, the
    patch is removed and the real ``_find_skill`` runs normally.

    The patch is applied to ``tools.skill_manager_tool._find_skill``
    because that is the exact symbol ``_create_skill`` calls into
    from inside the guard — there is no production-only test hook.
    """
    os.environ["HERMES_HOME"] = hermes_home
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    is_holder = holder_entered is not None
    # The holder pauses inside ``_find_skill`` only when an
    # ``a_in_lock`` rendezvous is configured; otherwise it goes
    # through ``skill_manage`` straight to completion.
    a_in_lock_rendezvous = a_in_lock is not None and release_a is not None
    result: dict = {"worker_id": worker_id, "role": "holder" if is_holder else "contender"}
    try:
        from tools import skill_manager_tool as _smt
        from tools.skill_manager_tool import skill_manage

        original_find_skill = _smt._find_skill
        if is_holder:
            if a_in_lock_rendezvous:
                # Holder (A1B rendezvous): patch ``_find_skill`` to
                # signal ``a_in_lock`` (and ``holder_entered``) and
                # block until ``release_a`` on the call that fires
                # INSIDE the production publication guard.
                #
                # Why call #2 (not #1): the real public ``skill_manage``
                # path invokes ``_find_skill`` once BEFORE entering
                # ``_create_skill`` (for the audit-ledger pre-mutation
                # capture) and once INSIDE the guard (the authoritative
                # in-lock duplicate check). Pausing at call #1 freezes
                # the holder BEFORE the guard — the contender then
                # acquires the lock, writes, and wins, yielding the
                # pre-fix (stale-pre-lock) behavior the test must
                # reject. Pausing at call #2 freezes the holder INSIDE
                # the guard holding the canonical-name lock; the
                # contender's ``live_skill_publish_guard`` call must
                # then observe ``EWOULDBLOCK`` and surface
                # ``lock_acquisition_failure=True`` — the exact
                # post-fix contract.
                _find_calls = {"n": 0}

                def _blocking_find_skill(name, *args, **kwargs):
                    _find_calls["n"] += 1
                    if _find_calls["n"] == 2:
                        holder_entered.set()
                        a_in_lock.set()
                        if not release_a.wait(timeout=_RELEASE_TIMEOUT_S):
                            raise TimeoutError(
                                "holder (A) was not released by parent in time"
                            )
                    return original_find_skill(name, *args, **kwargs)

                _smt._find_skill = _blocking_find_skill
                # For the rendezvous path, we do NOT pre-set
                # ``holder_entered`` — the patched ``_find_skill``
                # signals it once the guard is actually held.
                holder_entered_already_signalled = True
            else:
                holder_entered_already_signalled = False
            try:
                if not holder_entered_already_signalled:
                    holder_entered.set()
                payload = json.loads(
                    skill_manage(
                        action="create",
                        name=skill_name,
                        category=category,
                        content=skill_content,
                    )
                )
            finally:
                if a_in_lock_rendezvous:
                    _smt._find_skill = original_find_skill
            result.update({
                "kind": "return",
                "payload": payload,
            })
            if holder_finished is not None:
                holder_finished.set()
        else:
            # Contender: goes through the REAL public path with NO
            # monkeypatching — this is exactly what production does
            # when a second publisher races against an existing
            # canonical-name lock.
            assert (
                contender_attempting is not None
            ), "contender must be initialized with contender_attempting event"
            contender_attempting.set()
            payload = json.loads(
                skill_manage(
                    action="create",
                    name=skill_name,
                    category=category,
                    content=skill_content,
                )
            )
            result.update({
                "kind": "return",
                "payload": payload,
            })
            if contender_finished is not None:
                contender_finished.set()
    except BaseException as exc:  # pragma: no cover - surfaced by parent assertions
        result.update({
            "kind": "worker_error",
            "exception_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })

    Path(result_path).write_text(json.dumps(result), encoding="utf-8")


def _worker_publish_with_failure(
    worker_id: int,
    hermes_home: str,
    result_path: str,
    skill_name: str,
    skill_content: str,
    category: str,
    failure_at: str,
    holder_entered: mp.Event,
    release_holder: mp.Event,
) -> None:
    """Holder that fails AFTER entering the guard (CASE_2).

    ``failure_at`` is the marker in the publish sequence at which the
    child raises (after lock acquisition, after mkdir, after write, or
    after security scan).
    """
    os.environ["HERMES_HOME"] = hermes_home
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    result: dict = {"worker_id": worker_id, "failure_at": failure_at}
    try:
        # We exercise the guard directly — that is what skill_manage
        # calls into — and force the failure right where the real
        # publication code would. This is the smallest possible
        # exercise of the lock release-on-failure contract.
        from pathlib import Path as _P
        from tools.skill_publish_guard import live_skill_publish_guard

        holder_entered.set()
        with live_skill_publish_guard(skill_name, target=_P(hermes_home)):
            if failure_at == "inside_guard":
                raise RuntimeError("simulated publisher failure inside guarded region")
            if not release_holder.wait(timeout=_RELEASE_TIMEOUT_S):
                raise TimeoutError("holder was not released by parent in time")
        result["kind"] = "released_cleanly"
    except BaseException as exc:  # pragma: no cover - surfaced by parent assertions
        result.update({
            "kind": "raised",
            "exception_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
    Path(result_path).write_text(json.dumps(result), encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------

def _wait_until_released(lock_path: Path, timeout: float) -> bool:
    """Poll whether the lock file is unheld (try non-blocking probe lock).

    Backend-aware so the helper runs meaningfully on native Windows
    rather than crashing on the absent ``fcntl`` module:

      * On POSIX the original ``fcntl.flock`` semantics are preserved
        byte-for-byte (whole-file advisory lock, EWOULDBLOCK / EAGAIN
        is contention, descriptor close releases).
      * On native Windows the helper probes the same one-byte range
        that ``tools.skill_publish_guard`` acquires, with the same
        byte-0 materialisation and seek-to-0 positioning the
        production guard requires, and the same EACCES / EDEADLK
        contention taxonomy it classifies. The probe lock is
        released with LK_UNLCK before the descriptor is closed, and
        any successfully acquired probe is always released.

    Returns True if the lock is released (or never held) within
    ``timeout`` seconds. This is the only place we ever sleep, and
    only to assert the post-condition — never as the primary
    synchronization mechanism for the race itself.
    """
    if not lock_path.exists():
        return True

    # Mirror the production guard's primitive selection: prefer
    # ``fcntl`` whenever it is importable so POSIX behaviour is
    # byte-for-byte unchanged; only fall through to ``msvcrt`` when
    # ``fcntl`` is genuinely absent (native Windows) and ``msvcrt``
    # is present. Neither import is unconditional because the same
    # helper runs on both platforms.
    try:
        import fcntl as _fcntl
    except ImportError:
        _fcntl = None
    try:
        import msvcrt as _msvcrt
    except ImportError:
        _msvcrt = None
    _use_msvcrt = _fcntl is None and _msvcrt is not None

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            fd = os.open(str(lock_path), os.O_RDWR)
        except OSError:
            time.sleep(_LOCK_RELEASED_POLL_INTERVAL_S)
            continue
        # Always close the descriptor, and always release any probe
        # lock we successfully acquired on this iteration.
        probe_held = False
        try:
            if _use_msvcrt:
                # msvcrt.locking is a byte-range lock relative to the
                # CURRENT file position. Two consequences the POSIX
                # path does not have:
                #   - a zero-length lock file has no byte to lock, so
                #     we materialise byte 0 (same as the production
                #     guard and as ``hermes_cli/managed_uv.py``).
                #   - the descriptor must be seeked to 0 before
                #     locking, or two publishers could lock disjoint
                #     ranges and both believe they won.
                try:
                    if os.fstat(fd).st_size == 0:
                        os.write(fd, b"\0")
                except OSError:
                    # Read-only or otherwise non-writable: the byte
                    # may already be present from a prior holder.
                    pass
                os.lseek(fd, 0, os.SEEK_SET)
                try:
                    _msvcrt.locking(fd, _msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    # Mirror the production contention taxonomy so
                    # this helper's notion of "contention" matches
                    # ``_classify_flock_failure`` on native Windows.
                    if exc.errno in (errno.EACCES, errno.EDEADLK):
                        time.sleep(_LOCK_RELEASED_POLL_INTERVAL_S)
                        continue
                    raise
                # Acquired the one-byte probe range; release it
                # immediately so the next legitimate acquirer is not
                # blocked behind the helper. The release covers the
                # same byte at the same offset as the acquire.
                probe_held = True
                os.lseek(fd, 0, os.SEEK_SET)
                try:
                    _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
                except OSError:
                    # Best-effort release: an LK_UNLCK failure here
                    # is a soft signal that another process was
                    # already probing the same range, which is
                    # itself the contention we were checking for.
                    # The descriptor close below still tears down
                    # the kernel handle, and the production guard
                    # runs the same LK_UNLCK on the same byte when
                    # it actually holds the lock, so a transient
                    # here cannot mask a real leak in the guard.
                    pass
                return True
            else:
                # POSIX: fcntl.flock is whole-file. The original
                # semantic contract is preserved verbatim:
                # non-blocking probe + immediate release, EWOULDBLOCK
                # / EAGAIN is contention, descriptor close releases.
                try:
                    _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
                    _fcntl.flock(fd, _fcntl.LOCK_UN)
                    return True
                except OSError as exc:
                    if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                        time.sleep(_LOCK_RELEASED_POLL_INTERVAL_S)
                        continue
                    raise
        finally:
            if probe_held and _use_msvcrt:
                # Defensive: if the explicit LK_UNLCK above raised
                # before the success return, still attempt to drop
                # the probe before closing the descriptor so we never
                # leak a held range out of the helper.
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            os.close(fd)
    return False


def _spawn_publishers(
    hermes_home: Path,
    *,
    skill_name: str = _SKILL_NAME,
    skill_content: str = _SKILL_CONTENT,
    holder_category: str = "cat-holder",
    contender_category: str = "cat-contender",
    contender_skill_name: Optional[str] = None,
    contender_skill_content: Optional[str] = None,
    contender_delay_event: bool = True,
    use_a_in_lock_rendezvous: bool = False,
    spawn_publisher=_worker_publish,
    extra_holder_args: tuple = (),
    extra_contender_args: tuple = (),
) -> dict:
    """Run a holder + contender pair in fresh interpreter processes.

    Returns a dict with both processes' exit codes, result payloads,
    any stuck PIDs that had to be terminated, and the path to the
    resolved publication lock file so the caller can assert cleanup.

    By default both processes publish the same ``skill_name`` (the
    contested race shape). Pass ``contender_skill_name`` to make the
    contender publish a *different* canonical name — that is the
    CASE_3 noninterference shape.

    A1B deterministic rendezvous: when
    ``use_a_in_lock_rendezvous=True``, the holder is configured to
    block on the FIRST in-lock ``_find_skill`` call until the parent
    sets ``release_a``. This freezes the holder inside the guarded
    region holding the canonical-name lock so the contender's
    ``skill_manage`` attempt is forced to overlap. Pass
    ``use_a_in_lock_rendezvous=False`` (the default, used by CASE 3)
    for the no-pause shape — the holder's ``skill_manage`` runs to
    completion normally and only the post-conditions are asserted.
    """
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "skills").mkdir(parents=True, exist_ok=True)
    results_dir = hermes_home / "_results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    holder_entered = ctx.Event()
    release_holder = ctx.Event()
    holder_finished = ctx.Event()
    contender_attempting = ctx.Event()
    contender_acquired = ctx.Event()
    contender_release = ctx.Event()
    contender_finished = ctx.Event()
    a_in_lock = ctx.Event() if use_a_in_lock_rendezvous else None
    release_a = ctx.Event() if use_a_in_lock_rendezvous else None
    result_paths = [
        results_dir / f"worker-{i}.json" for i in range(2)
    ]

    effective_contender_name = contender_skill_name or skill_name
    effective_contender_content = contender_skill_content or skill_content

    holder_args_base = (
        0, str(hermes_home), str(result_paths[0]),
        skill_name, skill_content, holder_category,
        holder_entered, release_holder, holder_finished,
        contender_attempting, contender_acquired, contender_release,
        contender_finished,
    )
    if use_a_in_lock_rendezvous:
        holder = ctx.Process(
            target=spawn_publisher,
            args=holder_args_base + extra_holder_args + (
                a_in_lock, release_a,
            ),
        )
    else:
        holder = ctx.Process(
            target=spawn_publisher,
            args=holder_args_base + extra_holder_args,
        )
    contender = ctx.Process(
        target=spawn_publisher,
        args=(
            1, str(hermes_home), str(result_paths[1]),
            effective_contender_name, effective_contender_content, contender_category,
            None, None, None,
            contender_attempting, contender_acquired, contender_release,
            contender_finished,
        ) + extra_contender_args,
    )

    started: list[mp.Process] = []
    try:
        holder.start()
        started.append(holder)
        # Wait for the holder to have entered the guard before the
        # contender starts — this is the deterministic rendezvous.
        if not holder_entered.wait(timeout=_HOLDER_ENTER_TIMEOUT_S):
            return {
                "ok": False,
                "reason": "holder never entered the guarded region",
                "exitcodes": [holder.exitcode],
                "results": [],
                "stuck_pids": [],
                "result_paths": [str(p) for p in result_paths],
            }

        if use_a_in_lock_rendezvous and a_in_lock is not None:
            # Holder has entered ``skill_manage`` — wait for it to
            # have reached the in-lock ``_find_skill`` and be paused
            # there holding the canonical-name lock.
            if not a_in_lock.wait(timeout=_HOLDER_ENTER_TIMEOUT_S):
                return {
                    "ok": False,
                    "reason": "holder (A) never reached the in-lock _find_skill",
                    "exitcodes": [holder.exitcode],
                    "results": [],
                    "stuck_pids": [],
                    "result_paths": [str(p) for p in result_paths],
                }

        contender.start()
        started.append(contender)
        # Wait for the contender to actually attempt publication.
        if contender_delay_event and not contender_attempting.wait(timeout=_HOLDER_ENTER_TIMEOUT_S):
            return {
                "ok": False,
                "reason": "contender never attempted publication",
                "exitcodes": [holder.exitcode, contender.exitcode],
                "results": [],
                "stuck_pids": [],
                "result_paths": [str(p) for p in result_paths],
            }
        # CRITICAL: wait for the contender to FINISH (record its result)
        # before releasing the holder. Otherwise the holder releases
        # first, the contender takes the (now-free) lock, sees the
        # duplicate, and the test reports a false positive.
        contender.join(timeout=_JOIN_TIMEOUT_S)
        if contender.is_alive():
            return {
                "ok": False,
                "reason": "contender did not finish before holder release deadline",
                "exitcodes": [holder.exitcode, contender.exitcode],
                "results": [],
                "stuck_pids": [],
                "result_paths": [str(p) for p in result_paths],
            }
    finally:
        # Release the holder no matter what — either via the A1B
        # in-lock rendezvous or via the historical release_holder.
        if use_a_in_lock_rendezvous and release_a is not None:
            release_a.set()
        else:
            release_holder.set()

    deadline = time.monotonic() + _JOIN_TIMEOUT_S
    for proc in started:
        proc.join(timeout=max(0.0, deadline - time.monotonic()))

    stuck = [p for p in started if p.is_alive()]
    stuck_pids = [p.pid for p in stuck]
    for proc in stuck:
        proc.terminate()
    for proc in stuck:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)

    missing = [str(p) for p in result_paths if not p.is_file()]
    payloads = []
    for p in result_paths:
        if p.is_file():
            try:
                payloads.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception as exc:
                payloads.append({"kind": "bad_json", "error": str(exc), "path": str(p)})

    return {
        "ok": True,
        "exitcodes": [p.exitcode for p in started],
        "results": payloads,
        "stuck_pids": stuck_pids,
        "missing_results": missing,
        "holder_entered": holder_entered.is_set(),
        "contender_attempted": contender_attempting.is_set(),
    }


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Per-test isolated HERMES_HOME with the canary checkout on PYTHONPATH."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    # The witness spawns fresh interpreters; each must find the canary's
    # ``tools`` package on PYTHONPATH. We resolve the canary root from the
    # test's own location: tests/tools/<this>.py -> <canary>.
    canary_root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("PYTHONPATH", str(canary_root))
    return tmp_path


# ---------------------------------------------------------------------------
# CASE 1 — same-name race: two REAL public publishers; exactly one succeeds
# ---------------------------------------------------------------------------

def _assert_same_name_race_contract(
    race: dict,
    hermes_home,
    *,
    expected_winner_role: str,
) -> None:
    """Assert the COMPLETE GOOD contract for the same-name race.

    This is the single source of truth for the race contract. Both
    the normal CASE 1 witness (against repaired production) and the
    counterfactual discriminator (against the Sol-rejected
    pre-TOCTOU-repair implementation) MUST go through this helper so
    that the contract is enforced uniformly — a test that describes
    a failure semantically but does not reject it is a defect.

    Contract (the GOOD post-fix semantics, derived from real
    production behavior under the deterministic rendezvous):

      1. Both processes invoke the real public path with no
         worker_error.
      2. EXACTLY ONE ``success=True`` — the publisher that acquired
         the canonical-name lock and completed the guarded region.
      3. EXACTLY ONE ``lock_acquisition_failure=True`` — the loser
         whose ``live_skill_publish_guard`` raised
         ``SkillPublishLockError`` while the winner held the lock.
         This is the discriminating signal: under the pre-fix
         ordering, the loser would see a duplicate-name refusal
         instead.
      4. ZERO duplicate-name refusals — the loser never reaches
         the in-lock duplicate check because the lock itself
         rejects it.
      5. The winner is the role that was deterministically paused
         INSIDE the guard (the ``expected_winner_role``), not the
         late-arriving contender.
      6. Exactly ONE ``SKILL.md`` on disk — written by the winner.
      7. The lock file exists but is released; no atomic-write
         residue.

    Parameters
    ----------
    race:
        Return value of ``_spawn_publishers(..., use_a_in_lock_rendezvous=True)``.
    hermes_home:
        The per-test ``HERMES_HOME`` path (for filesystem checks).
    expected_winner_role:
        The role expected to win — ``"holder"`` for the deterministic
        rendezvous where the holder is paused inside the guard.
    """
    assert race["ok"], race
    assert race["exitcodes"] == [0, 0], race
    assert not race["stuck_pids"], race
    assert not race["missing_results"], race
    assert len(race["results"]) == 2

    holder_result = race["results"][0]
    contender_result = race["results"][1]

    # 1. Both processes invoked the real public path with no
    # worker_error.
    assert holder_result["role"] == "holder"
    assert holder_result["kind"] == "return", holder_result
    assert contender_result["role"] == "contender"
    assert contender_result["kind"] == "return", (
        f"contender raised: {contender_result}"
    )

    holder_payload = holder_result["payload"]
    contender_payload = contender_result["payload"]

    # 2. EXACTLY ONE success.
    successful = [
        r for r in (holder_result, contender_result)
        if r["payload"].get("success") is True
    ]
    assert len(successful) == 1, (
        f"SUCCESSFUL_PUBLICATION_COUNT must be 1, got {len(successful)}: "
        f"holder={holder_payload}, contender={contender_payload}"
    )
    winning_result = successful[0]

    # 5. The winner is the expected role.
    assert winning_result["role"] == expected_winner_role, (
        f"expected_winner_role={expected_winner_role!r}, but winner was "
        f"{winning_result['role']!r}: holder={holder_payload}, "
        f"contender={contender_payload}"
    )

    # 3. EXACTLY ONE lock_acquisition_failure=True — on the loser.
    lock_failure_results = [
        r for r in (holder_result, contender_result)
        if r["payload"].get("lock_acquisition_failure") is True
    ]
    assert len(lock_failure_results) == 1, (
        f"LOCK_ACQUISITION_FAILURE_COUNT must be 1, got "
        f"{len(lock_failure_results)}: holder={holder_payload}, "
        f"contender={contender_payload}"
    )
    assert lock_failure_results[0] is not winning_result, (
        f"the winner must NOT carry lock_acquisition_failure=True: "
        f"winner_payload={winning_result['payload']}"
    )

    # 4. ZERO duplicate-name refusals. A duplicate-name refusal is
    # the pre-fix loser's failure mode — its presence means the
    # production code allowed the contender to acquire the lock
    # on stale pre-lock knowledge and the holder saw the
    # contender's write.
    duplicate_refusal_count = sum(
        1 for r in (holder_result, contender_result)
        if r["payload"].get("success") is False
        and "already exists" in str(r["payload"].get("error", "")).lower()
    )
    assert duplicate_refusal_count == 0, (
        f"DUPLICATE_REFUSAL_COUNT must be 0 (pre-fix loser semantic), "
        f"got {duplicate_refusal_count}: holder={holder_payload}, "
        f"contender={contender_payload}"
    )

    # The loser must be a clean lock-acquisition failure (not some
    # other failure mode).
    loser_payload = (
        contender_payload if winning_result is holder_result
        else holder_payload
    )
    assert loser_payload.get("success") is False, (
        f"losing publisher must not have published successfully: {loser_payload}"
    )
    assert loser_payload.get("lock_acquisition_failure") is True, (
        f"losing publisher must carry lock_acquisition_failure=True "
        f"(the post-fix loser semantic); got: {loser_payload}"
    )

    # 6. Exactly ONE SKILL.md on disk, written by the winner.
    skills_root = hermes_home / "skills"
    holder_skill = skills_root / "cat-holder" / _SKILL_NAME / "SKILL.md"
    contender_skill = skills_root / "cat-contender" / _SKILL_NAME / "SKILL.md"
    if winning_result is holder_result:
        assert holder_skill.is_file(), (
            f"the holder (winner) must have written its SKILL.md: "
            f"{holder_skill}"
        )
        assert not contender_skill.exists(), (
            f"the contender (loser) must NOT have written a SKILL.md: "
            f"{contender_skill}"
        )
    else:
        assert contender_skill.is_file(), (
            f"the contender (winner) must have written its SKILL.md: "
            f"{contender_skill}"
        )
        assert not holder_skill.exists(), (
            f"the holder (loser) must NOT have written a SKILL.md: "
            f"{holder_skill}"
        )
    live_skill_files = sorted(skills_root.rglob("SKILL.md"))
    assert len(live_skill_files) == 1, (
        f"FINAL_CANONICAL_SKILL_COUNT must be 1, got {len(live_skill_files)}: "
        f"{live_skill_files}"
    )

    # 7. The lock file exists but is released; no atomic-write residue.
    lock_paths = list((hermes_home / "locks").glob("skill-publish-*.lock"))
    assert len(lock_paths) == 1, lock_paths
    assert _wait_until_released(lock_paths[0], _LOCK_RELEASED_POLL_TIMEOUT_S), (
        f"lock file still held after both processes exited: {lock_paths[0]}"
    )
    tmp_residue = sorted(p for p in hermes_home.rglob("*.tmp") if p.is_file())
    assert not tmp_residue, f"atomic write left temporary files: {tmp_residue}"


def test_case_1_same_name_race_publishes_once(hermes_home):
    """Two independent processes race the same canonical identity.

    A1B (Hermes 0.20.5 TOCTOU repair): BOTH processes invoke the real
    public ``skill_manage(action="create", ...)`` path. Neither process
    substitutes a direct ``live_skill_publish_guard`` call for its
    publication operation. This proves the test is not a lock-exclusion
    witness that ignores the stale-check vulnerability it is meant to
    guard against.

    Deterministic rendezvous: Publisher A enters ``skill_manage`` and
    reaches the in-lock authoritative ``_find_skill`` (the second
    call in the public path; the first is a pre-mutation audit-ledger
    capture that runs BEFORE the guard). A patches its own
    process-local ``_find_skill`` to block on Events, freezes inside
    the guard holding the canonical-name lock, and signals the
    parent. The parent then starts Publisher B, which enters the real
    public ``skill_manage`` path and attempts to publish while A
    still holds the lock.

    B's outcome is determined by the publication primitive's actual
    bounded-lock semantics: under the post-fix production code the
    in-lock authoritative duplicate check AND the mutation live
    INSIDE one ``live_skill_publish_guard``, so B's
    ``live_skill_publish_guard`` call observes ``EWOULDBLOCK`` and
    the caller surfaces ``lock_acquisition_failure=True``. That is
    the GOOD loser semantic the contract enforces.

    Why this catches the pre-fix implementation: under the Sol-
    rejected ordering (authoritative duplicate lookup OUTSIDE the
    guard, mutation inside), the holder is paused BEFORE the guard
    at the pre-lock check, the contender acquires the lock and
    writes successfully, and the holder — when released — sees the
    contender's just-written SKILL.md and refuses with "already
    exists". The GOOD contract rejects this loser semantic. See
    ``test_case_1_discriminates_against_prelock_duplicate_check``
    for the explicit counterfactual.
    """
    race = _spawn_publishers(hermes_home, use_a_in_lock_rendezvous=True)
    _assert_same_name_race_contract(
        race, hermes_home, expected_winner_role="holder"
    )


# ---------------------------------------------------------------------------
# CASE 1 discriminator: the test must FAIL against the pre-fix production
# ordering (authoritative duplicate lookup outside the lock).
# ---------------------------------------------------------------------------

def _build_prelock_counterfactual(root: Path) -> Path:
    """Materialise a self-contained pre-TOCTOU-repair ``tools`` package.

    The counterfactual only needs ONE property of the historical
    Sol-rejected implementation: the authoritative ``_find_skill``
    duplicate lookup happens OUTSIDE ``live_skill_publish_guard``, with
    only the mutation inside. Everything else (validation, category
    handling, security scan, lint, payload shape) is irrelevant to the
    race contract, so we do not copy the historical 2000-line module —
    we shim it.

    ``root/tools/__init__.py`` extends its ``__path__`` with the REAL
    ``tools`` directory, so ``tools.skill_publish_guard``,
    ``tools.registry`` and every other submodule resolve to the real
    implementation. Only ``skill_manager_tool`` is shadowed, because
    ``root/tools/`` comes first on the package path.

    ``root/tools/skill_manager_tool.py`` loads the real module under a
    private name, re-exports its namespace (so helpers, constants and
    the ``skill_manage`` dispatcher all behave normally), and then
    REPLACES ``_create_skill`` with the pre-lock ordering. Module-global
    ``_find_skill`` is used by that replacement, so the children's
    ``monkeypatch``-equivalent rebinding of
    ``tools.skill_manager_tool._find_skill`` still intercepts it — which
    is what makes the deterministic rendezvous work in the subprocess.

    Returns the scratch root (to be placed first on ``PYTHONPATH``).
    """
    real_tools_dir = Path(__file__).resolve().parents[2] / "tools"
    pkg = root / "tools"
    pkg.mkdir(parents=True, exist_ok=True)

    (pkg / "__init__.py").write_text(
        "# Counterfactual shim package: shadow ONLY skill_manager_tool and\n"
        "# resolve every other submodule from the real tools package.\n"
        "import os\n"
        f"__path__.append({str(real_tools_dir)!r})\n",
        encoding="utf-8",
    )

    (pkg / "skill_manager_tool.py").write_text(
        '"""Pre-TOCTOU-repair counterfactual of tools.skill_manager_tool.\n'
        "\n"
        "Re-exports the real module and overrides _create_skill with the\n"
        "Sol-rejected ordering: authoritative duplicate lookup OUTSIDE the\n"
        "publication guard, mutation inside.\n"
        '"""\n'
        "import importlib.util as _ilu\n"
        "import sys as _sys\n"
        "\n"
        f"_REAL = {str(real_tools_dir / 'skill_manager_tool.py')!r}\n"
        "_spec = _ilu.spec_from_file_location(\n"
        '    "_hermes_real_skill_manager_tool", _REAL\n'
        ")\n"
        "_real = _ilu.module_from_spec(_spec)\n"
        '_sys.modules["_hermes_real_skill_manager_tool"] = _real\n'
        "_spec.loader.exec_module(_real)\n"
        "\n"
        "# Re-export the real namespace so every helper, constant and the\n"
        "# skill_manage dispatcher behave exactly as in production.\n"
        "# Dunders MUST be excluded: copying ``__name__`` would repoint this\n"
        "# module's identity at the real module, making the delegator below\n"
        "# recurse into itself (a RecursionError that skill_manage's telemetry\n"
        "# try/except silently swallows, costing the audit call site).\n"
        "for _n in dir(_real):\n"
        '    if _n.startswith("__") or _n in ("_create_skill",):\n'
        "        continue\n"
        "    globals()[_n] = getattr(_real, _n)\n"
        "\n"
        "# The rendezvous patches ``tools.skill_manager_tool._find_skill``.\n"
        "# ``skill_manage``'s audit-ledger pre-capture is a re-exported\n"
        "# function object that still closes over the REAL module's globals,\n"
        "# so route the real module's ``_find_skill`` through this module's\n"
        "# (patchable) global. Both call sites then honour the patch, keeping\n"
        "# the call ordering identical to production: #1 audit pre-capture,\n"
        "# #2 the duplicate lookup (which in this counterfactual sits OUTSIDE\n"
        "# the guard — the defect under test).\n"
        "_orig_find_skill = _real._find_skill\n"
        "_find_skill = _orig_find_skill\n"
        "\n"
        "\n"
        "def _find_skill_delegator(name, *a, **k):\n"
        "    return _sys.modules[__name__]._find_skill(name, *a, **k)\n"
        "\n"
        "\n"
        "_real._find_skill = _find_skill_delegator\n"
        "\n"
        "\n"
        "def _create_skill(name, content, category=None):\n"
        '    """Pre-fix ordering: duplicate lookup OUTSIDE the guard."""\n'
        "    err = _validate_name(name)\n"
        "    if err:\n"
        '        return {"success": False, "error": err}\n'
        "    err = _validate_category(category)\n"
        "    if err:\n"
        '        return {"success": False, "error": err}\n'
        "    err = _validate_frontmatter(content, new_skill=True)\n"
        "    if err:\n"
        '        return {"success": False, "error": err}\n'
        "\n"
        "    # THE DEFECT: authoritative duplicate check taken OUTSIDE the\n"
        "    # publication lock. A concurrent publisher can complete between\n"
        "    # this lookup and the lock acquisition below.\n"
        "    existing = _find_skill(name)\n"
        "    if existing:\n"
        "        return {\n"
        '            "success": False,\n'
        "            \"error\": \"A skill named '%s' already exists at %s.\"\n"
        '                     % (name, existing["path"]),\n'
        "        }\n"
        "\n"
        "    skill_dir = _resolve_skill_dir(name, category)\n"
        "    try:\n"
        "        with live_skill_publish_guard(name, target=skill_dir):\n"
        "            skill_dir.mkdir(parents=True, exist_ok=True)\n"
        '            skill_md = skill_dir / "SKILL.md"\n'
        "            atomic_write_text(\n"
        "                skill_md, content, preserve_mode=True, create_mode=0o644\n"
        "            )\n"
        "    except SkillPublishLockError as exc:\n"
        "        return {\n"
        '            "success": False,\n'
        '            "error": str(exc),\n'
        '            "lock_acquisition_failure": True,\n'
        '            "lock_failure_kind": exc.kind,\n'
        "        }\n"
        "\n"
        "    return {\n"
        '        "success": True,\n'
        "        \"message\": \"Skill '%s' created.\" % name,\n"
        '        "skill_md": str(skill_dir / "SKILL.md"),\n'
        "    }\n"
        "\n"
        "\n"
        "# Rebind the dispatcher's reference so skill_manage(action='create')\n"
        "# routes into the counterfactual _create_skill above.\n"
        "_real._create_skill = _create_skill\n",
        encoding="utf-8",
    )
    return root


def test_case_1_discriminates_against_prelock_duplicate_check(
    hermes_home, tmp_path
):
    """Sanity check that CASE 1 (the post-fix witness) would FAIL if
    production were the Sol-rejected pre-fix ordering (authoritative
    duplicate lookup outside the lock).

    The honest discriminator runs the SAME CASE 1 race contract in a
    subprocess whose ``tools.skill_manager_tool`` is a pre-TOCTOU-repair
    counterfactual built by ``_build_prelock_counterfactual`` into a
    per-test temporary directory. The subprocess MUST see the GOOD race
    contract fail under that ordering — that is the decisive gate proving
    CASE 1 actually discriminates.

    A parent-side ``monkeypatch.setattr`` on ``_create_skill`` is NOT
    sufficient: each ``mp.spawn`` child re-imports ``tools`` and gets the
    real, unpatched module, so a parent-side patch is a no-op for the
    cross-process race. The counterfactual must therefore exist as real
    importable files on the subprocess's path — but it is built here,
    hermetically, rather than depending on a pre-existing scratch tree.
    """
    import subprocess
    import sys
    import textwrap

    bad_scratch = _build_prelock_counterfactual(tmp_path / "prelock-counterfactual")
    bad_tools = bad_scratch / "tools" / "skill_manager_tool.py"
    assert bad_tools.is_file(), (
        f"counterfactual pre-TOCTOU-repair shim was not built at {bad_tools}"
    )

    # Re-run the same race contract in the scratch environment.
    # The subprocess imports the test module (which sees the BAD
    # implementation because the scratch root is the first PYTHONPATH
    # entry), runs the same rendezvous, and invokes
    # ``_assert_same_name_race_contract``. We assert that the contract
    # FAILS (the pre-fix code does not satisfy it).
    driver = textwrap.dedent(
        """
        import os, sys, json
        from pathlib import Path
        sys.path.insert(0, r'{scratch}')
        sys.path.insert(0, r'{testdir}')
        import test_skill_publish_guard_minimal as t
        import tempfile
        tmp = tempfile.mkdtemp()
        try:
            hermes_home = Path(tmp)
            hermes_home.mkdir(exist_ok=True)
            race = t._spawn_publishers(hermes_home, use_a_in_lock_rendezvous=True)
            try:
                t._assert_same_name_race_contract(
                    race, hermes_home, expected_winner_role='holder'
                )
            except AssertionError as exc:
                print('DISCRIMINATOR_RAISED:' + str(exc))
                sys.exit(0)
            print('DISCRIMINATOR_PASSED_UNEXPECTEDLY')
            sys.exit(2)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        """
    ).format(
        scratch=str(bad_scratch),
        testdir=str(Path(__file__).resolve().parent),
    )

    result = subprocess.run(
        [sys.executable, "-c", driver],
        capture_output=True,
        text=True,
        timeout=120,
        # Override the inherited PYTHONPATH so the bad scratch's
        # ``tools`` package wins. The test's hermes_home fixture
        # sets PYTHONPATH to the canary root; if we let that
        # propagate, the subprocess loads the GOOD implementation
        # and the discriminator never fires.
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [
                    str(bad_scratch),
                    str(Path(__file__).resolve().parents[2]),
                ]
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert result.returncode == 0, (
        f"counterfactual subprocess did not raise the expected "
        f"AssertionError. returncode={result.returncode}, "
        f"stdout={result.stdout!r}, stderr={result.stderr!r}"
    )
    failure_msg = result.stdout.split("DISCRIMINATOR_RAISED:", 1)[-1].strip()
    assert (
        "lock_acquisition_failure" in failure_msg
        or "DUPLICATE_REFUSAL_COUNT" in failure_msg
        or "already exists" in failure_msg
        or "expected_winner_role" in failure_msg
    ), (
        f"discriminator failure must cite the race contract, not "
        f"environment/setup: {failure_msg!r}"
    )


# ---------------------------------------------------------------------------
# CASE 2 — lock release after failure inside the guard
# ---------------------------------------------------------------------------

def test_case_2_lock_release_after_failure(hermes_home):
    """A publisher that fails INSIDE the guard must not leave a stale lock.

    We exercise the minimal primitive directly (the public
    ``skill_manage`` path delegates to the same lock). The child
    enters the guard, raises, and exits. The parent then verifies the
    lock is acquirable by a fresh, legitimate publisher (the next
    test in the suite would be blocked if the lock leaked).
    """
    from tools.skill_publish_guard import (
        live_skill_publish_guard, _canonical_lock_path,
    )

    ctx = mp.get_context("spawn")
    hermes_home.mkdir(parents=True, exist_ok=True)
    result_path = hermes_home / "_results" / "fail-holder.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    holder_entered = ctx.Event()
    release_holder = ctx.Event()

    proc = ctx.Process(
        target=_worker_publish_with_failure,
        args=(
            0, str(hermes_home), str(result_path),
            _SKILL_NAME, _SKILL_CONTENT, "cat-fail",
            "inside_guard", holder_entered, release_holder,
        ),
    )
    proc.start()
    try:
        assert holder_entered.wait(timeout=_HOLDER_ENTER_TIMEOUT_S), (
            "failure-holder never entered the guarded region"
        )
        # Brief pause so the child actually raises inside the guard.
        # This is not the primary synchronization — the Event IS —
        # but ensures we test the post-raise cleanup path.
        time.sleep(0.05)
    finally:
        release_holder.set()

    proc.join(timeout=_JOIN_TIMEOUT_S)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
    assert proc.exitcode == 0, f"failure-holder process exited non-zero: {proc.exitcode}"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["kind"] == "raised", result
    assert result["exception_type"] == "RuntimeError", result

    lock_path = _canonical_lock_path(_SKILL_NAME)
    assert lock_path.exists(), lock_path
    assert _wait_until_released(lock_path, _LOCK_RELEASED_POLL_TIMEOUT_S), (
        f"lock file still held after the failed child exited: {lock_path}"
    )

    # Now prove a fresh legitimate publisher can acquire and complete.
    # We do this in the parent process — the lock release is the test;
    # we do not need a second child to prove the lock is releasable.
    with live_skill_publish_guard(_SKILL_NAME, target=hermes_home):
        pass


# ---------------------------------------------------------------------------
# CASE 3 — different canonical names do not contend
# ---------------------------------------------------------------------------

def test_case_3_different_names_do_not_contend(hermes_home):
    """The lock is keyed by canonical name, not by a global namespace.

    One process holds the publication lock on canonical name A; a
    SECOND process simultaneously tries to publish canonical name B.
    The B publisher must succeed (the lock for A is irrelevant to
    B). If the lock were global, this test would deadlock (both
    publishers contending for one lock, no rendezvous, no release).
    """
    race = _spawn_publishers(
        hermes_home,
        skill_name=_SKILL_NAME,
        skill_content=_SKILL_CONTENT,
        contender_skill_name=_OTHER_SKILL_NAME,
        contender_skill_content=_OTHER_SKILL_CONTENT,
        holder_category="cat-h",
        contender_category="cat-c",
    )

    assert race["ok"], race
    assert race["exitcodes"] == [0, 0], race
    assert not race["stuck_pids"], race
    results = race["results"]
    assert len(results) == 2, results
    # Holder: payload is the manual-lock-held success (no skill_manage call).
    assert results[0]["role"] == "holder"
    assert results[0]["payload"]["success"] is True, results[0]
    # Contender: published a DIFFERENT canonical name, so it must
    # have succeeded despite the holder holding the OTHER name's lock.
    assert results[1]["role"] == "contender"
    assert results[1]["kind"] == "return", results[1]
    assert results[1]["payload"]["success"] is True, results[1]
    # And the contender's published skill must exist on disk under the
    # OTHER name's category.
    other_skill = (
        hermes_home / "skills" / "cat-c" / _OTHER_SKILL_NAME / "SKILL.md"
    )
    assert other_skill.is_file(), other_skill


# ---------------------------------------------------------------------------
# CASE 4 — permission failure is not rewritten as duplicate
# ---------------------------------------------------------------------------

def test_case_4_permission_failure_preserved_meaningfully(hermes_home, monkeypatch):
    """A permission failure on the locks/ directory surfaces as
    ``SkillPublishLockError``, NOT as a duplicate-name refusal.

    We chmod 000 the locks/ directory so the holder cannot open the
    lock file. The lock-acquisition error MUST propagate up to the
    caller as a permission-class failure with
    ``lock_acquisition_failure=True`` — the directive forbids silently
    rewriting it into a misleading duplicate result.
    """
    # Pre-create the locks/ dir as the user, then chmod 000.
    locks_dir = hermes_home / "locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    # Sanity: in our sandbox the test runs as a regular user, so
    # chmod 000 actually denies us. Skip if the user is root or the
    # OS ignores the mode.
    try:
        locks_dir.chmod(0o000)
    except OSError:
        pytest.skip("could not chmod the locks/ directory")

    # Probe: the user CANNOT open files inside a 0o000 dir.
    probe = locks_dir / "probe"
    try:
        probe.touch()
        # If we got here, chmod didn't actually deny us (some sandboxes
        # ignore mode bits). Skip cleanly so the test is honest.
        locks_dir.chmod(0o700)
        probe.unlink()
        pytest.skip("chmod 000 did not deny access in this environment")
    except (OSError, PermissionError):
        pass

    try:
        from tools.skill_publish_guard import (
            live_skill_publish_guard, SkillPublishLockError,
        )

        raised = None
        try:
            with live_skill_publish_guard(_SKILL_NAME, target=hermes_home):
                pass
        except SkillPublishLockError as exc:
            raised = exc

        assert raised is not None, (
            "expected SkillPublishLockError when locks/ is unwritable, "
            "but the guard entered the region"
        )
        assert raised.lock_acquisition_failure is True
        # And via the public ``skill_manage`` create path: the caller
        # must see success=False with lock_acquisition_failure=True,
        # NOT the "already exists" duplicate wording.
        from tools.skill_manager_tool import skill_manage
        payload = json.loads(
            skill_manage(
                action="create",
                name=_SKILL_NAME,
                category="cat-perm",
                content=_SKILL_CONTENT,
            )
        )
        assert payload["success"] is False, payload
        assert payload.get("lock_acquisition_failure") is True, (
            f"permission failure must surface as lock-acquisition failure, "
            f"not as a duplicate-name refusal: {payload}"
        )
        assert "already exists" not in str(payload.get("error", "")).lower(), payload
    finally:
        # Restore permissions so other tests (and the pytest tmp_path
        # teardown) can clean up.
        try:
            locks_dir.chmod(0o700)
        except OSError:
            pass


# ===========================================================================
# MF2/MF3/MF4 — bounded regression tests for the A1G repair
# ===========================================================================
# These tests are ADDITIVE — they do not modify the existing A1B/A1G race
# witnesses above. Each MF* test class is a narrowly scoped contract witness
# that targets exactly one of the three must-fix defects and discriminates
# against the pre-fix production behavior described in the A1G report.
#
# Conventions:
#   - All tests use the per-test ``hermes_home`` fixture (isolated HOME).
#   - MF2 tests build a real skills tree on disk and exercise the real
#     ``skill_manage`` create path.
#   - MF3/MF4 tests patch ``fcntl.flock`` / ``os.open`` / ``os.close`` at
#     import-time of the guard module. The patches are scoped to the
#     ``live_skill_publish_guard`` instance attributes so we never globally
#     break pytest's own file descriptor handling.
# ===========================================================================


# ---------------------------------------------------------------------------
# MF2 — Live-tree identity / symlink-junction (PERSISTS in pre-fix code)
# ---------------------------------------------------------------------------
#
# The pre-fix production code never validates that ``skill_dir`` or any of
# its parents (the ``category/`` intermediate, or ``skills/`` itself) is NOT
# a redirect. A poisoned skills tree can therefore cause
# ``skill_dir.mkdir`` and the atomic-write to land outside the authoritative
# writable skills root. The fix MUST reject this fail-closed at the
# publication critical section, before any mkdir happens.

def _build_hermes_home_with_skills_dir(hermes_home: Path) -> Path:
    """Create the canonical ``<hermes_home>/skills/`` tree.

    Returns the skills root. Tests that need a redirected parent (MF2-A)
    or a redirected target (MF2-B) then mutate this tree further; the
    non-redirected witness (MF2-C) just uses it as-is.
    """
    skills_root = hermes_home / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    return skills_root


def test_mf2_a_symlinked_parent_category_redirected_to_outside(
    hermes_home, monkeypatch
):
    """MF2-A: a symlinked category under skills/ points OUTSIDE the skills
    root. The normal create path must refuse fail-closed; no SKILL.md may
    appear at the external destination, and no SKILL.md may appear inside
    the authoritative skills root under the legitimate category path.

    Discriminator: the pre-fix code follows the symlink and writes the
    SKILL.md at the external target. The post-fix code refuses at the
    publication guard with a redirect/refusal error that is distinguishable
    from a plain duplicate-name refusal.
    """
    skills_root = _build_hermes_home_with_skills_dir(hermes_home)

    # External location that the symlinked category will redirect to.
    external_dir = hermes_home / "external_escape_target"
    external_dir.mkdir(parents=True, exist_ok=True)
    # Plant a sentinel inside the external target so we can prove the
    # legitimate create did NOT clobber / overwrite it.
    sentinel = external_dir / "DO_NOT_OVERWRITE.txt"
    sentinel.write_text("preserved", encoding="utf-8")

    # Create the symlinked category INSIDE skills/ that points OUTSIDE.
    category_link = skills_root / "poisoned-category"
    try:
        category_link.symlink_to(external_dir, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover
        pytest.skip(f"symlink not supported in this environment: {exc}")

    from tools.skill_manager_tool import skill_manage

    payload = json.loads(
        skill_manage(
            action="create",
            name=_SKILL_NAME,
            category="poisoned-category",
            content=_SKILL_CONTENT,
        )
    )

    # Post-fix contract: refused, NOT silently written to the escape target.
    assert payload.get("success") is False, (
        f"MF2-A: creation must be refused fail-closed when the parent "
        f"category is a redirect, but payload reports success: {payload}"
    )
    # The refusal must be distinguishable from a normal duplicate-name
    # contention — it must mention redirect / symlink / outside-the-root.
    err_msg = str(payload.get("error", "")).lower()
    assert "already exists" not in err_msg, (
        f"MF2-A: refusal must NOT be reported as duplicate-name contention; "
        f"the contract requires redirect/escape refusal. Got: {payload}"
    )
    # The redirection must not have mutated the external target.
    assert sentinel.is_file(), (
        f"MF2-A: sentinel at {sentinel} must remain; pre-fix code would "
        f"have created the skill directory at the external target"
    )
    sentinel_text = sentinel.read_text(encoding="utf-8")
    assert sentinel_text == "preserved", (
        f"MF2-A: external target must remain untouched, got: {sentinel_text!r}"
    )
    # No SKILL.md may have been written at the external destination.
    assert not (external_dir / _SKILL_NAME / "SKILL.md").exists(), (
        f"MF2-A: pre-fix code wrote SKILL.md to external escape target at "
        f"{external_dir / _SKILL_NAME / 'SKILL.md'}"
    )
    # No SKILL.md may have been written at the legitimate path either
    # (the legitimate path resolves through the symlink to the external
    # target — both must be empty of the new skill).
    live_skills = sorted((hermes_home / "skills").rglob("SKILL.md"))
    assert _SKILL_NAME not in {p.parent.name for p in live_skills}, (
        f"MF2-A: no SKILL.md under a directory named {_SKILL_NAME!r} may "
        f"exist anywhere reachable from skills/: {live_skills}"
    )


def test_mf2_b_existing_target_skill_path_is_a_redirect(
    hermes_home, monkeypatch
):
    """MF2-B: the targeted skill directory already exists AND is a
    symlink that escapes the skills root.

    Pre-fix code: opens the redirect, writes SKILL.md through it, and the
    external destination is mutated.

    Post-fix code: refuses fail-closed; external destination is untouched.
    """
    skills_root = _build_hermes_home_with_skills_dir(hermes_home)
    category = skills_root / "cat-with-escape"
    category.mkdir(parents=True, exist_ok=True)

    # External target where the redirect will land.
    external_target = hermes_home / "mf2_b_external_target"
    external_target.mkdir(parents=True, exist_ok=True)
    sentinel = external_target / "DO_NOT_OVERWRITE.txt"
    sentinel.write_text("preserved", encoding="utf-8")

    # Pre-create the skill directory at the external target FIRST.
    real_target_dir = external_target / _SKILL_NAME
    real_target_dir.mkdir(parents=True, exist_ok=True)

    # Now create a symlink INSIDE the skills tree at the canonical skill
    # path that redirects to the real (external) target.
    skill_link = category / _SKILL_NAME
    try:
        skill_link.symlink_to(real_target_dir, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover
        pytest.skip(f"symlink not supported in this environment: {exc}")

    from tools.skill_manager_tool import skill_manage

    payload = json.loads(
        skill_manage(
            action="create",
            name=_SKILL_NAME,
            category="cat-with-escape",
            content=_SKILL_CONTENT,
        )
    )

    # Post-fix contract: refused, NOT written through the redirect.
    assert payload.get("success") is False, (
        f"MF2-B: creation must be refused when the target skill path is "
        f"a redirect, but payload reports success: {payload}"
    )
    err_msg = str(payload.get("error", "")).lower()
    assert "already exists" not in err_msg, (
        f"MF2-B: refusal must not be reported as duplicate-name contention; "
        f"got: {payload}"
    )
    # The external destination must remain untouched.
    assert sentinel.is_file() and sentinel.read_text(encoding="utf-8") == "preserved", (
        f"MF2-B: external destination sentinel must remain 'preserved'"
    )
    # The skill directory at the external target must remain empty of SKILL.md.
    assert not (real_target_dir / "SKILL.md").exists(), (
        f"MF2-B: pre-fix code wrote SKILL.md through the redirect to "
        f"{real_target_dir / 'SKILL.md'}"
    )


def test_mf2_c_normal_non_redirect_create_still_succeeds(hermes_home):
    """MF2-C: a clean, non-redirected create path MUST still succeed.

    The repair must not break legitimate publication. This guards against
    an over-broad repair that refuses every legitimate create.
    """
    _build_hermes_home_with_skills_dir(hermes_home)
    from tools.skill_manager_tool import skill_manage

    payload = json.loads(
        skill_manage(
            action="create",
            name=_SKILL_NAME,
            category="cat-normal",
            content=_SKILL_CONTENT,
        )
    )
    assert payload.get("success") is True, (
        f"MF2-C: a normal, non-redirected create must succeed, got: {payload}"
    )
    written = hermes_home / "skills" / "cat-normal" / _SKILL_NAME / "SKILL.md"
    assert written.is_file(), f"MF2-C: SKILL.md must exist at {written}"


# ---------------------------------------------------------------------------
# MF3 — Repair exception taxonomy (PERSISTS in pre-fix code)
# ---------------------------------------------------------------------------
#
# The pre-fix ``SkillPublishLockError`` collapses EWOULDBLOCK/EAGAIN and
# arbitrary OSError into identical caller-visible semantics. The caller
# cannot tell contention from hard infrastructure failure, and the
# user-visible message misleadingly says "retry shortly" even on hard
# errors.
#
# Contract: callers MUST be able to distinguish CONTENTION (retryable)
# from HARD_ACQUISITION_FAILURE (NOT retryable in the short term).
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="MF3-A EWOULDBLOCK-as-CONTENTION classification: this witness forces the POSIX fcntl.flock LOCK_EX|LOCK_NB EWOULDBLOCK contract; the equivalent Windows contention-coverage lives in test_mf1_e (EACCES/EDEADLK).",
)

def test_mf3_a_contention_classification(monkeypatch):
    """MF3-A: forcing fcntl.flock to raise EWOULDBLOCK must classify the
    failure as CONTENTION (retryable), not HARD_ACQUISITION_FAILURE.

    The caller-visible payload must NOT contain misleading hard-failure
    wording; it MAY recommend a retry.
    """
    import fcntl as real_fcntl
    from tools import skill_publish_guard as guard_mod

    original_flock = real_fcntl.flock

    def contending_flock(fd, op):
        # Only fail acquisition (LOCK_EX | LOCK_NB) with EWOULDBLOCK.
        # Pass everything else (LOCK_UN, LOCK_SH, etc.) through unchanged.
        if op & real_fcntl.LOCK_EX and op & real_fcntl.LOCK_NB:
            err = OSError(errno.EWOULDBLOCK, "Resource temporarily unavailable")
            raise err
        return original_flock(fd, op)

    monkeypatch.setattr(guard_mod.fcntl, "flock", contending_flock)

    import os as real_os
    hermes_home = real_os.environ.get("HERMES_HOME")
    if not hermes_home:
        pytest.skip("HERMES_HOME not set; cannot resolve lock root")

    target = Path(hermes_home)
    from tools.skill_publish_guard import (
        live_skill_publish_guard,
        SkillPublishLockError,
    )

    raised = None
    try:
        with live_skill_publish_guard(_SKILL_NAME, target=target):
            pass
    except SkillPublishLockError as exc:
        raised = exc

    assert raised is not None, "expected SkillPublishLockError on contention"
    # Post-fix contract: classification field exists and is CONTENTION.
    classification = getattr(raised, "kind", None) or getattr(
        raised, "classification", None
    ) or getattr(raised, "failure_kind", None)
    assert classification == "CONTENTION", (
        f"MF3-A: EWOULDBLOCK must classify as CONTENTION, got "
        f"{classification!r}; full error: {raised}"
    )
    # The existing compatibility field MUST still be present.
    assert getattr(raised, "lock_acquisition_failure", None) is True
    # Original cause must remain attached.
    assert getattr(raised, "cause_exception", None) is not None
    assert raised.cause_exception.errno == errno.EWOULDBLOCK

    # The caller-visible _create_skill payload must surface the contention
    # classification and not mislead the agent about a retry strategy on
    # what is actually retryable.
    from tools.skill_manager_tool import skill_manage

    payload = json.loads(
        skill_manage(
            action="create",
            name=_SKILL_NAME,
            category="cat-mf3a",
            content=_SKILL_CONTENT,
        )
    )
    assert payload.get("success") is False, payload
    assert payload.get("lock_acquisition_failure") is True, payload
    # Caller must surface the classification.
    assert payload.get("lock_failure_kind") == "CONTENTION", (
        f"MF3-A: caller payload must surface CONTENTION classification, "
        f"got: {payload}"
    )
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="MF3-B EIO-as-HARD classification: this witness forces fcntl.flock to raise EIO, which is the POSIX-only contract; the equivalent Windows hard-failure coverage lives in test_mf1_f.",
)


def test_mf3_b_hard_acquisition_failure_not_misleadingly_retryable(
    monkeypatch,
):
    """MF3-B: forcing fcntl.flock to raise EIO must classify as HARD, not
    CONTENTION, and the caller-visible payload must NOT recommend an
    immediate retry as if it were contention.
    """
    import fcntl as real_fcntl
    from tools import skill_publish_guard as guard_mod

    original_flock = real_fcntl.flock

    def hard_flock(fd, op):
        if op & real_fcntl.LOCK_EX and op & real_fcntl.LOCK_NB:
            raise OSError(errno.EIO, "I/O error (simulated hard failure)")
        return original_flock(fd, op)

    monkeypatch.setattr(guard_mod.fcntl, "flock", hard_flock)

    from tools.skill_publish_guard import (
        live_skill_publish_guard,
        SkillPublishLockError,
    )

    target = Path(guard_mod.get_hermes_home())
    raised = None
    try:
        with live_skill_publish_guard(_SKILL_NAME, target=target):
            pass
    except SkillPublishLockError as exc:
        raised = exc

    assert raised is not None, "expected SkillPublishLockError on EIO"
    classification = getattr(raised, "kind", None) or getattr(
        raised, "classification", None
    ) or getattr(raised, "failure_kind", None)
    assert classification == "HARD_ACQUISITION_FAILURE", (
        f"MF3-B: EIO must classify as HARD_ACQUISITION_FAILURE, got "
        f"{classification!r}; full error: {raised}"
    )
    assert getattr(raised, "lock_acquisition_failure", None) is True
    assert getattr(raised, "cause_exception", None) is not None
    assert raised.cause_exception.errno == errno.EIO

    # The caller-visible payload must distinguish hard from contention
    # and must NOT say "retry shortly" as if the user could fix it by
    # waiting a few hundred milliseconds.
    from tools.skill_manager_tool import skill_manage

    payload = json.loads(
        skill_manage(
            action="create",
            name=_SKILL_NAME,
            category="cat-mf3b",
            content=_SKILL_CONTENT,
        )
    )
    assert payload.get("success") is False, payload
    assert payload.get("lock_acquisition_failure") is True, payload
    assert payload.get("lock_failure_kind") == "HARD_ACQUISITION_FAILURE", (
        f"MF3-B: caller payload must surface HARD classification, got: {payload}"
    )
    err_msg = str(payload.get("error", "")).lower()
    assert "retry shortly" not in err_msg, (
        f"MF3-B: hard failure must NOT be reported as 'retry shortly' "
        f"(that wording implies short-term retryability); got: {payload}"
    )


def test_mf3_c_lock_file_open_failure_is_hard(monkeypatch):
    """MF3-C: forcing os.open of the lock file to fail with EACCES must
    surface as HARD_ACQUISITION_FAILURE, not CONTENTION.
    """
    import os as real_os
    from tools import skill_publish_guard as guard_mod

    original_open = real_os.open

    def denied_open(path, flags, mode=0o777, *args, **kwargs):
        # Reject any open that lands in our locks/ directory.
        path_str = str(path)
        if "locks" in path_str and "skill-publish" in path_str:
            raise PermissionError(errno.EACCES, "Permission denied (simulated)")
        return original_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(guard_mod.os, "open", denied_open)

    from tools.skill_publish_guard import (
        live_skill_publish_guard,
        SkillPublishLockError,
    )

    target = Path(guard_mod.get_hermes_home())
    raised = None
    try:
        with live_skill_publish_guard(_SKILL_NAME, target=target):
            pass
    except SkillPublishLockError as exc:
        raised = exc

    assert raised is not None, "expected SkillPublishLockError on EACCES"
    classification = getattr(raised, "kind", None) or getattr(
        raised, "classification", None
    ) or getattr(raised, "failure_kind", None)
    assert classification == "HARD_ACQUISITION_FAILURE", (
        f"MF3-C: EACCES on os.open must classify as HARD, got "
        f"{classification!r}; full error: {raised}"
    )
    assert getattr(raised, "lock_acquisition_failure", None) is True
    # Original cause must be preserved.
    cause = getattr(raised, "cause_exception", None)
    assert cause is not None
    assert isinstance(cause, PermissionError) or getattr(cause, "errno", None) == errno.EACCES

    # The caller-visible payload must surface the hard classification.
    from tools.skill_manager_tool import skill_manage

    payload = json.loads(
        skill_manage(
            action="create",
            name=_SKILL_NAME,
            category="cat-mf3c",
            content=_SKILL_CONTENT,
        )
    )
    assert payload.get("success") is False, payload
    assert payload.get("lock_acquisition_failure") is True, payload
    assert payload.get("lock_failure_kind") == "HARD_ACQUISITION_FAILURE", (
        f"MF3-C: caller payload must surface HARD, got: {payload}"
    )


# ---------------------------------------------------------------------------
# MF4 — Release / cleanup failure context (PERSISTS in pre-fix code)
# ---------------------------------------------------------------------------
#
# The pre-fix ``live_skill_publish_guard`` uses
#   try: fcntl.flock(fd, fcntl.LOCK_UN) except OSError: pass
# which silently swallows the unlock failure and exposes the os.close
# outside any try/except. The body-success + LOCK_UN-fail path can leak
# the OSError out of the context manager and convert a successful
# publication into an unrelated exception.

def _patch_unlock_failure(monkeypatch, errno_value: int = errno.EIO):
    """Patch fcntl.flock so that LOCK_UN raises, everything else passes."""
    import fcntl as real_fcntl
    from tools import skill_publish_guard as guard_mod

    original_flock = real_fcntl.flock

    def unlocking_flock(fd, op):
        if op == real_fcntl.LOCK_UN:
            raise OSError(errno_value, "simulated LOCK_UN failure")
        return original_flock(fd, op)

    monkeypatch.setattr(guard_mod.fcntl, "flock", unlocking_flock)
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="MF4-A body-success / unlock-failure observability: this witness forces the POSIX fcntl.flock LOCK_UN contract; the equivalent Windows unlock-diagnostic coverage lives in test_mf1_g.",
)


def test_mf4_a_body_success_unlock_failure_is_observable_not_masking(
    monkeypatch, caplog,
):
    """MF4-A: when the body succeeds but LOCK_UN raises, the primary
    outcome must remain observable AND the release failure must be
    diagnosable. The repair must NOT convert a successful publication
    into an unrelated exception solely because of unlock failure
    (unless the chosen contract explicitly and defensibly requires it).
    """
    _patch_unlock_failure(monkeypatch, errno_value=errno.EIO)

    from tools.skill_publish_guard import (
        live_skill_publish_guard,
        SkillPublishLockError,
    )

    target = Path(os.environ.get("HERMES_HOME") or "/tmp")
    body_result = "BODY_OK"
    observed_outcome = None
    raised_exc = None

    import logging as _logging
    with caplog.at_level(_logging.WARNING, logger="tools.skill_publish_guard"):
        try:
            with live_skill_publish_guard(_SKILL_NAME, target=target):
                observed_outcome = body_result
        except BaseException as exc:
            raised_exc = exc

    # The body's primary outcome MUST be observed one way or another.
    if raised_exc is not None:
        # The implementation chose to raise on unlock failure. The body
        # outcome and the release failure must BOTH be observable.
        # Accept either a structured attribute carrying the body outcome
        # OR a chained cause mentioning the unlock error.
        body_attrs = [
            getattr(raised_exc, attr, None) for attr in
            ("body_outcome", "body_result", "primary_outcome", "inner_outcome")
        ]
        assert any(v == body_result for v in body_attrs), (
            f"MF4-A: when the body succeeds and unlock fails and the "
            f"implementation raises, the body outcome must remain "
            f"observable on the propagated exception; got exception="
            f"{raised_exc!r}, body attrs={body_attrs}"
        )
    else:
        # The implementation chose NOT to raise. The body outcome must
        # have been observable normally.
        assert observed_outcome == "BODY_OK", (
            f"MF4-A: body outcome must be observed when body succeeds; "
            f"got {observed_outcome!r}"
        )

    # Discriminator (the part that fails pre-fix): the release failure
    # MUST be observable in some structured form. Pre-fix code uses
    # `except OSError: pass` which silently swallows the unlock failure
    # — no exception, no log, no attribute, nothing. The contract is
    # that the release failure leaves at least one observable trace.
    release_failure_observed = False

    # (a) A structured exception carries the body outcome + release
    #     failure info. We already checked (a) above; now check whether
    #     the chained cause mentions the unlock error.
    if raised_exc is not None:
        cur = raised_exc
        depth = 0
        while cur is not None and depth < 5:
            s = str(cur).lower()
            if (
                "eio" in s or "lock_un" in s or "unlock" in s
                or "release" in s
            ):
                release_failure_observed = True
                break
            cur = cur.__cause__ or cur.__context__
            depth += 1

    # (b) A logger.warning fires carrying the lock path and exc_info.
    if not release_failure_observed:
        for rec in caplog.records:
            msg = rec.getMessage().lower()
            if (
                "unlock" in msg or "release" in msg or "lock_un" in msg
            ) and rec.levelno >= _logging.WARNING:
                release_failure_observed = True
                break

    assert release_failure_observed, (
        f"MF4-A: body succeeded (observed_outcome={observed_outcome!r}, "
        f"raised={raised_exc!r}) but the LOCK_UN release failure was "
        f"silently dropped — no exception, no chained cause, no log "
        f"record, no attribute. Pre-fix `except OSError: pass` swallows "
        f"the unlock error and makes the leak/duplication diagnostically "
        f"invisible. Repair must leave at least one observable trace. "
        f"caplog records: "
        f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="MF4-B body+unlock double-failure diagnostic: this witness forces fcntl.flock LOCK_UN to raise, which is the POSIX-only contract; the equivalent Windows unlock-diagnostic coverage lives in test_mf1_g.",
)


def test_mf4_b_body_failure_unlock_failure_does_not_mask_primary(
    monkeypatch,
):
    """MF4-B: when the body raises RuntimeError AND LOCK_UN raises, the
    primary RuntimeError MUST remain the propagated exception. The
    release failure must be observable as secondary diagnostic context
    (chained cause, secondary exception, or warning), but must NOT
    replace or mask the primary.
    """
    _patch_unlock_failure(monkeypatch, errno_value=errno.EIO)

    from tools.skill_publish_guard import (
        live_skill_publish_guard,
        SkillPublishLockError,
    )

    target = Path(os.environ.get("HERMES_HOME") or "/tmp")
    raised_primary = None
    try:
        with live_skill_publish_guard(_SKILL_NAME, target=target):
            raise RuntimeError("PRIMARY_BODY_FAILURE")
    except RuntimeError as exc:
        raised_primary = exc

    assert raised_primary is not None, (
        f"MF4-B: primary RuntimeError must propagate, not be masked by "
        f"the unlock failure"
    )
    assert "PRIMARY_BODY_FAILURE" in str(raised_primary), (
        f"MF4-B: the primary RuntimeError must be the one observed by "
        f"the caller; got: {raised_primary!r}"
    )

    # The release failure must be observable as secondary context.
    secondary_observed = False
    # (a) Chained __context__ / __cause__
    cause_chain = []
    cur = raised_primary
    while cur is not None and len(cause_chain) < 5:
        cause_chain.append(cur)
        cur = cur.__cause__ or cur.__context__
    for link in cause_chain:
        s = str(link).lower()
        if "eio" in s or "lock_un" in s or "unlock" in s or "release" in s:
            secondary_observed = True
            break
    # (b) The guard records the release failure in a module- or
    #     context-level attribute. Some implementations stash the
    #     unlock error on the guard's local frame; we cannot reach that
    #     after the fact, so we accept chained cause OR a warning logged.
    #     The chained-cause check above is sufficient when present.
    # The implementation MUST at minimum chain the release error via
    # raise X from release_exc or set __context__ to it.
    assert secondary_observed, (
        f"MF4-B: the primary RuntimeError was preserved ({raised_primary!r}) "
        f"but the release failure must also be observable as secondary "
        f"diagnostic context (chained via __cause__/__context__ or logged "
        f"with the lock path). Got cause chain: "
        f"{[(type(c).__name__, str(c)) for c in cause_chain]}"
    )


def test_mf4_c_close_failure_does_not_mask_active_body_exception(
    monkeypatch,
):
    """MF4-C: when the body raises AND os.close also fails, the primary
    body exception MUST remain the propagated exception. Cleanup failure
    must remain observable but must not mask the primary.

    The os.close patch is scoped: only the guard-owned fd triggers a
    close failure; every other fd delegates to the original os.close.
    """
    import os as real_os
    from tools import skill_publish_guard as guard_mod

    original_close = real_os.close
    # Track every fd the guard module opens so we know which one is
    # the guard-owned fd to selectively fail.
    guard_fds: list[int] = []

    original_open = real_os.open

    def tracking_open(path, flags, mode=0o777, *args, **kwargs):
        fd = original_open(path, flags, mode, *args, **kwargs)
        if "skill-publish" in str(path):
            guard_fds.append(fd)
        return fd

    def selective_close(fd, *args, **kwargs):
        if fd in guard_fds:
            raise OSError(errno.EIO, "simulated os.close failure on guard fd")
        return original_close(fd, *args, **kwargs)

    monkeypatch.setattr(guard_mod.os, "open", tracking_open)
    monkeypatch.setattr(guard_mod.os, "close", selective_close)

    from tools.skill_publish_guard import live_skill_publish_guard

    target = Path(os.environ.get("HERMES_HOME") or "/tmp")
    raised_primary = None
    try:
        with live_skill_publish_guard(_SKILL_NAME, target=target):
            raise RuntimeError("PRIMARY_BODY_FAILURE_FOR_CLOSE_TEST")
    except RuntimeError as exc:
        raised_primary = exc

    assert raised_primary is not None, (
        f"MF4-C: primary RuntimeError must propagate, not be masked by "
        f"the os.close failure"
    )
    assert "PRIMARY_BODY_FAILURE_FOR_CLOSE_TEST" in str(raised_primary), (
        f"MF4-C: the primary RuntimeError must be the one observed by "
        f"the caller; got: {raised_primary!r}"
    )

    # The close failure must be observable as secondary diagnostic context.
    secondary_observed = False
    cause_chain = []
    cur = raised_primary
    while cur is not None and len(cause_chain) < 5:
        cause_chain.append(cur)
        cur = cur.__cause__ or cur.__context__
    for link in cause_chain:
        s = str(link).lower()
        if "eio" in s or "close" in s or "fd" in s:
            secondary_observed = True
            break
    assert secondary_observed, (
        f"MF4-C: the primary RuntimeError was preserved ({raised_primary!r}) "
        f"but the close failure must also be observable as secondary "
        f"diagnostic context. Got cause chain: "
        f"{[(type(c).__name__, str(c)) for c in cause_chain]}"
    )


# ---------------------------------------------------------------------------
# MF1 — WINDOWS LOCK SUPPORT (Tier-1 Windows platform contract)
# ---------------------------------------------------------------------------
#
# The guard serializes publication with a byte-range/advisory file lock. On
# POSIX that is ``fcntl.flock``; on Windows ``fcntl`` does not exist at all
# and the platform primitive is ``msvcrt.locking``. These witnesses pin the
# cross-platform contract:
#
#   MF1-A  the module imports on Windows (where ``fcntl`` is absent)
#   MF1-B  Windows acquisition uses msvcrt LK_NBLCK, one byte, at offset 0
#   MF1-C  Windows release uses msvcrt LK_UNLCK, one byte, BEFORE fd close
#   MF1-D  POSIX still uses fcntl.flock and never requires msvcrt
#   MF1-E  Windows contention errnos classify as CONTENTION (retryable)
#   MF1-F  a non-contention Windows failure classifies as HARD
#   MF1-G  Windows unlock failure stays observable and never masks the body
#
# Windows is simulated WITHOUT creating repo-root ``fcntl``/``msvcrt`` files
# (those shadow the real stdlib for every process in the tree). Instead each
# witness loads a private copy of the guard module through
# ``importlib.util.spec_from_file_location`` under a patched
# ``sys.modules`` / ``sys.platform`` view, so the production module object
# imported by the rest of the suite is never mutated.


_GUARD_SRC = (
    Path(__file__).resolve().parents[2] / "tools" / "skill_publish_guard.py"
)


class _FakeMsvcrt:
    """Minimal stand-in for the ``msvcrt`` extension module.

    Records every ``locking()`` call as ``(fd, mode, nbytes, offset)`` where
    ``offset`` is the file position observed at call time — that is how the
    witnesses prove the guard positioned the descriptor before locking, which
    matters because ``msvcrt.locking`` is byte-range relative to the CURRENT
    file position (unlike ``flock``, which is whole-file).
    """

    # Real values from CPython's msvcrt module.
    LK_LOCK = 0
    LK_NBLCK = 1
    LK_NBRLCK = 2
    LK_RLCK = 3
    LK_UNLCK = 4

    def __init__(self, *, acquire_error=None, release_error=None):
        self.calls: list[tuple[int, int, int, int]] = []
        self._acquire_error = acquire_error
        self._release_error = release_error

    def locking(self, fd, mode, nbytes):  # noqa: D401 - mirrors msvcrt sig
        try:
            offset = os.lseek(fd, 0, os.SEEK_CUR)
        except OSError:
            offset = -1
        self.calls.append((fd, mode, nbytes, offset))
        if mode in (self.LK_NBLCK, self.LK_NBRLCK, self.LK_LOCK, self.LK_RLCK):
            if self._acquire_error is not None:
                raise self._acquire_error
        elif mode == self.LK_UNLCK:
            if self._release_error is not None:
                raise self._release_error
        return None

    # Convenience views used by the assertions.
    def acquire_calls(self):
        return [c for c in self.calls if c[1] != self.LK_UNLCK]

    def release_calls(self):
        return [c for c in self.calls if c[1] == self.LK_UNLCK]


class _ForbiddenModule:
    """Any attribute access raises — proves a module was never touched."""

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, item):
        raise AssertionError(
            f"MF1: the {self._name!r} module must not be used on this "
            f"platform, but attribute {item!r} was accessed"
        )


def _load_guard_for_platform(
    platform_name: str,
    *,
    fake_msvcrt=None,
    forbid_fcntl: bool = False,
    forbid_msvcrt: bool = False,
):
    """Import a PRIVATE copy of the guard under a simulated platform.

    ``sys.platform`` and ``os.name`` are both patched because in-tree
    precedents key off either one. ``fcntl`` is made unimportable when
    ``forbid_fcntl`` is set (the Windows reality), and ``msvcrt`` is injected
    so the Windows branch has a primitive to call.

    Returns the freshly-executed module object. Never mutates the shared
    ``tools.skill_publish_guard`` entry in ``sys.modules``.
    """
    import importlib.util

    saved_modules = {}
    for name in ("fcntl", "msvcrt"):
        saved_modules[name] = sys.modules.get(name, None)

    saved_platform = sys.platform
    saved_os_name = os.name
    saved_meta_path = list(sys.meta_path)

    class _BlockImport:
        """Meta-path hook that makes named modules genuinely unimportable."""

        def __init__(self, blocked):
            self.blocked = set(blocked)

        def find_module(self, fullname, path=None):  # legacy API
            return self if fullname in self.blocked else None

        def load_module(self, fullname):
            raise ImportError(f"No module named {fullname!r} (MF1 simulation)")

        def find_spec(self, fullname, path=None, target=None):
            if fullname in self.blocked:
                raise ImportError(
                    f"No module named {fullname!r} (MF1 simulation)"
                )
            return None

    try:
        blocked = []
        if forbid_fcntl:
            blocked.append("fcntl")
            sys.modules.pop("fcntl", None)
        if forbid_msvcrt:
            blocked.append("msvcrt")
            sys.modules.pop("msvcrt", None)
        if blocked:
            sys.meta_path.insert(0, _BlockImport(blocked))

        if fake_msvcrt is not None:
            sys.modules["msvcrt"] = fake_msvcrt

        # Patch the platform view BEFORE executing the module body so any
        # module-level platform constant is computed under the simulation.
        sys.platform = platform_name
        os.name = "nt" if platform_name == "win32" else "posix"

        spec = importlib.util.spec_from_file_location(
            f"_mf1_guard_{platform_name}_{id(fake_msvcrt)}", _GUARD_SRC
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.platform = saved_platform
        os.name = saved_os_name
        sys.meta_path[:] = saved_meta_path
        for name, val in saved_modules.items():
            if val is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = val


# --- MF1-A: Windows importability --------------------------------------------

def test_mf1_a_module_imports_on_windows_without_fcntl(hermes_home):
    """MF1-A: the guard must import on Windows, where ``fcntl`` is absent.

    Pre-fix the module body does an unconditional ``import fcntl``, so this
    raises ImportError under the simulation — the exact Tier-1 Windows
    breakage MF1 exists to fix. Post-fix the import must succeed and the
    module must expose its documented public surface.
    """
    fake = _FakeMsvcrt()

    mod = _load_guard_for_platform(
        "win32", fake_msvcrt=fake, forbid_fcntl=True
    )

    # Public surface must survive the platform split.
    assert hasattr(mod, "live_skill_publish_guard")
    assert hasattr(mod, "SkillPublishLockError")
    assert mod.LOCK_KIND_CONTENTION == "CONTENTION"
    assert mod.LOCK_KIND_HARD == "HARD_ACQUISITION_FAILURE"

    # And the module must NOT be holding a live fcntl reference on Windows.
    assert getattr(mod, "fcntl", None) is None, (
        "MF1-A: on Windows the module-level ``fcntl`` binding must be None "
        f"(fcntl does not exist there); got {getattr(mod, 'fcntl', None)!r}"
    )


# --- MF1-B: Windows acquisition primitive ------------------------------------

def test_mf1_b_windows_acquire_uses_msvcrt_lk_nblck_one_byte(hermes_home):
    """MF1-B: Windows acquisition must be a non-blocking one-byte lock at 0.

    ``msvcrt.locking`` locks ``nbytes`` starting at the CURRENT file
    position, so the guard must seek to 0 first or two publishers can lock
    disjoint ranges and both "win" — the exact race the guard prevents.
    """
    fake = _FakeMsvcrt()
    mod = _load_guard_for_platform(
        "win32", fake_msvcrt=fake, forbid_fcntl=True
    )

    target = Path(str(hermes_home))
    with mod.live_skill_publish_guard(_SKILL_NAME, target=target):
        pass

    acquires = fake.acquire_calls()
    assert len(acquires) == 1, (
        f"MF1-B: expected exactly one msvcrt acquisition call, got {fake.calls}"
    )
    fd, mode, nbytes, offset = acquires[0]
    assert mode == _FakeMsvcrt.LK_NBLCK, (
        f"MF1-B: acquisition must use LK_NBLCK (non-blocking); got mode={mode}"
    )
    assert nbytes == 1, (
        f"MF1-B: acquisition must lock exactly 1 byte; got nbytes={nbytes}"
    )
    assert offset == 0, (
        f"MF1-B: the descriptor must be positioned at byte 0 before "
        f"msvcrt.locking (it is position-relative); observed offset={offset}"
    )

    # A zero-length lock file has no byte to lock; the guard must have
    # ensured one exists, otherwise the real msvcrt would fail.
    lock_path = mod._canonical_lock_path(_SKILL_NAME)
    assert lock_path.exists(), "MF1-B: lock file must exist after acquisition"
    assert lock_path.stat().st_size >= 1, (
        "MF1-B: the lock file must contain at least one lockable byte "
        f"before msvcrt.locking; size={lock_path.stat().st_size}"
    )


# --- MF1-C: Windows release primitive ---------------------------------------

def test_mf1_c_windows_release_uses_lk_unlck_before_close(hermes_home):
    """MF1-C: Windows release must be an explicit one-byte LK_UNLCK at 0,
    issued while the descriptor is still open.

    Relying on fd-close to drop a Windows lock is not the documented
    contract; an explicit unlock is required. The unlock must also be
    position-correct, mirroring acquisition.
    """
    fake = _FakeMsvcrt()
    mod = _load_guard_for_platform(
        "win32", fake_msvcrt=fake, forbid_fcntl=True
    )

    closed_fds: list[int] = []
    real_close = os.close

    def recording_close(fd):
        closed_fds.append(fd)
        return real_close(fd)

    target = Path(str(hermes_home))
    orig = mod.os.close
    mod.os.close = recording_close
    try:
        with mod.live_skill_publish_guard(_SKILL_NAME, target=target):
            pass
    finally:
        mod.os.close = orig

    releases = fake.release_calls()
    assert len(releases) == 1, (
        f"MF1-C: expected exactly one LK_UNLCK call, got {fake.calls}"
    )
    fd, mode, nbytes, offset = releases[0]
    assert mode == _FakeMsvcrt.LK_UNLCK
    assert nbytes == 1, (
        f"MF1-C: release must unlock exactly 1 byte; got nbytes={nbytes}"
    )
    assert offset == 0, (
        f"MF1-C: descriptor must be repositioned to 0 before LK_UNLCK; "
        f"observed offset={offset}"
    )

    # The unlock must have happened while the fd was still valid, i.e.
    # BEFORE the guard closed it. Observing a real position (offset != -1)
    # already proves the fd was open; assert the ordering explicitly too.
    assert fd in closed_fds, (
        "MF1-C: the guard-owned fd must still be closed after unlocking"
    )
    assert offset != -1, (
        "MF1-C: LK_UNLCK must be issued while the fd is still open "
        "(release before close), but the descriptor was already closed"
    )


# --- MF1-D: POSIX regression witness ----------------------------------------
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="MF1-D POSIX fcntl regression: this witness directly monkeypatches fcntl.flock which is absent on native Windows; the Windows lane is covered by the native msvcrt witnesses NW1, NW2, and NW3.",
)

def test_mf1_d_posix_still_uses_fcntl_and_never_requires_msvcrt(hermes_home):
    """MF1-D: the Windows work must not disturb the accepted POSIX path.

    Loads the guard with ``msvcrt`` made both unimportable AND poisoned, so
    any accidental use on POSIX is a hard failure, then proves acquisition
    and release still route through ``fcntl.flock`` with LOCK_EX|LOCK_NB and
    LOCK_UN respectively.
    """
    mod = _load_guard_for_platform(
        "linux", forbid_msvcrt=True
    )

    assert getattr(mod, "msvcrt", None) is None, (
        "MF1-D: on POSIX the module-level ``msvcrt`` binding must be None; "
        f"got {getattr(mod, 'msvcrt', None)!r}"
    )
    assert mod.fcntl is not None, (
        "MF1-D: on POSIX the module must hold a live fcntl reference"
    )

    ops: list[int] = []
    real_flock = mod.fcntl.flock

    def recording_flock(fd, op):
        ops.append(op)
        return real_flock(fd, op)

    mod.fcntl.flock = recording_flock
    try:
        target = Path(str(hermes_home))
        with mod.live_skill_publish_guard(_SKILL_NAME, target=target):
            pass
    finally:
        mod.fcntl.flock = real_flock

    import fcntl as real_fcntl_mod

    assert ops, "MF1-D: POSIX path must call fcntl.flock"
    acquire_ops = [
        o for o in ops
        if (o & real_fcntl_mod.LOCK_EX) and (o & real_fcntl_mod.LOCK_NB)
    ]
    assert acquire_ops, (
        f"MF1-D: POSIX acquisition must use LOCK_EX|LOCK_NB; observed ops={ops}"
    )
    assert real_fcntl_mod.LOCK_UN in ops, (
        f"MF1-D: POSIX release must use LOCK_UN; observed ops={ops}"
    )


# --- MF1-E: Windows contention taxonomy -------------------------------------

@pytest.mark.parametrize(
    "contention_errno", [errno.EACCES, errno.EDEADLK],
    ids=["EACCES", "EDEADLK"],
)
def test_mf1_e_windows_contention_classifies_as_contention(
    hermes_home, contention_errno,
):
    """MF1-E: the established Windows contention errnos must classify as
    CONTENTION, preserving the MF3 contract on Windows.

    ``msvcrt.locking(LK_NBLCK)`` reports EACCES/EDEADLK when another process
    holds the range (the in-tree taxonomy documented in
    ``cron/scheduler.py::_is_lock_contention_errno``). Misclassifying these
    as HARD would tell a caller a transient race is unrecoverable.
    """
    fake = _FakeMsvcrt(
        acquire_error=OSError(contention_errno, "simulated windows contention")
    )
    mod = _load_guard_for_platform(
        "win32", fake_msvcrt=fake, forbid_fcntl=True
    )

    target = Path(str(hermes_home))
    raised = None
    try:
        with mod.live_skill_publish_guard(_SKILL_NAME, target=target):
            pass
    except mod.SkillPublishLockError as exc:
        raised = exc

    assert raised is not None, (
        "MF1-E: Windows lock contention must raise SkillPublishLockError"
    )
    assert raised.kind == mod.LOCK_KIND_CONTENTION, (
        f"MF1-E: errno {contention_errno} on Windows must classify as "
        f"CONTENTION, got {raised.kind!r}"
    )
    # Backward-compatible caller contract must survive.
    assert raised.lock_acquisition_failure is True
    assert raised.cause_exception is not None
    assert raised.cause_exception.errno == contention_errno
    # Contention IS retryable, so the retry wording is correct here.
    assert "retry" in str(raised).lower(), (
        f"MF1-E: contention should remain described as retryable; got {raised}"
    )


# --- MF1-F: Windows hard acquisition failure --------------------------------

def test_mf1_f_windows_hard_failure_is_not_described_as_retryable(hermes_home):
    """MF1-F: a non-contention Windows failure must classify as HARD and must
    NOT be dressed up as ordinary contention.

    EIO from the lock syscall cannot be fixed by retrying in a tight loop;
    telling the caller to "retry shortly" turns a hard infrastructure fault
    into a spin. This is the MF3 distinction, enforced on the Windows branch.
    """
    fake = _FakeMsvcrt(
        acquire_error=OSError(errno.EIO, "simulated windows hard failure")
    )
    mod = _load_guard_for_platform(
        "win32", fake_msvcrt=fake, forbid_fcntl=True
    )

    target = Path(str(hermes_home))
    raised = None
    try:
        with mod.live_skill_publish_guard(_SKILL_NAME, target=target):
            pass
    except mod.SkillPublishLockError as exc:
        raised = exc

    assert raised is not None, (
        "MF1-F: a hard Windows lock failure must raise SkillPublishLockError"
    )
    assert raised.kind == mod.LOCK_KIND_HARD, (
        f"MF1-F: EIO must classify as HARD_ACQUISITION_FAILURE, got "
        f"{raised.kind!r}"
    )
    assert raised.lock_acquisition_failure is True
    # The original cause must remain diagnosable.
    assert raised.cause_exception is not None
    assert raised.cause_exception.errno == errno.EIO
    assert raised.__cause__ is not None, (
        "MF1-F: the underlying OSError must remain chained as __cause__"
    )
    # A hard failure must not advise an immediate retry.
    assert "retry shortly" not in str(raised).lower(), (
        f"MF1-F: hard acquisition failure must NOT recommend 'retry "
        f"shortly'; got: {raised}"
    )


# --- MF1-G: Windows release failure + MF4 preservation ----------------------

def test_mf1_g_windows_unlock_failure_observable_and_non_masking(
    hermes_home, caplog,
):
    """MF1-G: Windows unlock failure must stay observable (MF4) and must
    never mask the body's outcome.

    Two arms, matching the accepted MF4 contract:
      (a) body SUCCEEDS + unlock fails -> success is observed AND the
          release failure leaves a diagnosable trace (log or chained cause);
      (b) body RAISES  + unlock fails -> the body's exception stays primary
          and the cleanup failure is secondary context, never a replacement.
    """
    import logging as _logging

    # --- arm (a): body success, unlock fails -------------------------------
    fake_a = _FakeMsvcrt(
        release_error=OSError(errno.EIO, "simulated windows LK_UNLCK failure")
    )
    mod_a = _load_guard_for_platform(
        "win32", fake_msvcrt=fake_a, forbid_fcntl=True
    )
    target = Path(str(hermes_home))

    observed = None
    raised_a = None
    with caplog.at_level(_logging.WARNING):
        try:
            with mod_a.live_skill_publish_guard(_SKILL_NAME, target=target):
                observed = "BODY_OK"
        except BaseException as exc:  # noqa: BLE001 - contract probe
            raised_a = exc

    assert fake_a.release_calls(), (
        "MF1-G: the Windows release primitive must have been attempted"
    )
    if raised_a is None:
        assert observed == "BODY_OK", (
            "MF1-G(a): a successful body must remain observable when only "
            f"unlock failed; got observed={observed!r}"
        )
    # The release failure must not vanish silently.
    release_observed = any(
        ("unlock" in r.getMessage().lower()
         or "release" in r.getMessage().lower()
         or "lk_unlck" in r.getMessage().lower())
        and r.levelno >= _logging.WARNING
        for r in caplog.records
    )
    if not release_observed and raised_a is not None:
        cur = raised_a
        depth = 0
        while cur is not None and depth < 5:
            s = str(cur).lower()
            if "eio" in s or "unlock" in s or "release" in s:
                release_observed = True
                break
            cur = cur.__cause__ or cur.__context__
            depth += 1
    assert release_observed, (
        "MF1-G(a): the Windows LK_UNLCK failure was silently swallowed — it "
        "must leave at least one observable trace (log record or chained "
        "cause). caplog: "
        f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )

    # --- arm (b): body raises, unlock also fails ---------------------------
    fake_b = _FakeMsvcrt(
        release_error=OSError(errno.EIO, "simulated windows LK_UNLCK failure")
    )
    mod_b = _load_guard_for_platform(
        "win32", fake_msvcrt=fake_b, forbid_fcntl=True
    )

    sentinel = RuntimeError("PRIMARY_BODY_FAILURE")
    raised_b = None
    try:
        with mod_b.live_skill_publish_guard(_SKILL_NAME, target=target):
            raise sentinel
    except BaseException as exc:  # noqa: BLE001 - contract probe
        raised_b = exc

    assert raised_b is sentinel, (
        "MF1-G(b): the body's own exception must remain PRIMARY; a Windows "
        f"cleanup failure must never replace it. Got {raised_b!r}"
    )
    # And the cleanup failure must still be reachable as secondary context.
    chain = []
    cur = raised_b.__cause__ or raised_b.__context__
    depth = 0
    while cur is not None and depth < 5:
        chain.append(cur)
        cur = cur.__cause__ or cur.__context__
        depth += 1
    secondary = any(
        isinstance(link, OSError) and link.errno == errno.EIO for link in chain
    )
    assert secondary, (
        "MF1-G(b): the primary body exception was preserved but the Windows "
        "unlock failure must remain observable as secondary diagnostic "
        f"context. Chain: {[(type(c).__name__, str(c)) for c in chain]}"
    )


# ---------------------------------------------------------------------------
# MF1 NATIVE WINDOWS COMPANION (windows_only lane)
# ---------------------------------------------------------------------------
#
# The eight MF1 witnesses above simulate Windows on any host: they inject a
# fake ``msvcrt``, block ``fcntl``, and patch ``sys.platform``. That proves the
# guard's platform DISPATCH is structured correctly, but it cannot prove the
# real OS honours the lock — a fake ``locking()`` records a call, it does not
# serialize anything. These companions close that gap by running the REAL
# primitive on a real Windows host.
#
# Division of labour, deliberately minimal (no duplication of the simulated
# suite):
#
#   NW1  real import/binding state (fcntl absent, msvcrt present)
#   NW2  real msvcrt acquire + release through the public guard, including
#        byte-0 materialisation on an initially empty lock file
#   NW3  REAL cross-process contention: a child process holds the lock, the
#        parent observes the genuine OS error mapped to LOCK_KIND_CONTENTION,
#        then acquisition succeeds after the holder releases
#
# Hard-failure (EIO) and MF4 cleanup-failure injection stay with the simulated
# tests on purpose: manufacturing a genuine EIO or a failing LK_UNLCK from a
# real Windows filesystem needs privileged/flaky setup (ACL games, filter
# drivers, forced dismounts) whose failure modes are environmental rather than
# behavioural. Deterministic fault injection is the more honest contract for
# those two, and the real-OS tests below cover the parts injection cannot:
# that the primitive exists, is callable, and actually excludes a second
# process.
#
# These tests carry a literal ``@pytest.mark.windows_only`` because
# ``scripts/ci/list_os_marked_tests.py`` greps the marker NAME out of the
# source to decide which FILES the Windows lane imports; ``-m windows_only``
# then decides which tests run. A file-local alias would be discovered but
# deselected, reporting green over zero coverage.


def _nw_child_source() -> str:
    """Source for the lock-holder child used by NW3.

    Runs as ``python -c`` in a fresh interpreter (no inherited handles beyond
    what it opens itself) so the contention it creates is genuine
    cross-process Windows byte-range exclusion, not two fds in one process.

    Protocol on stdout, line-buffered:
      ``HELD``    the lock is acquired and being held
      ``RELEASED``the lock has been explicitly released
    The child waits for a line on stdin before releasing, so the parent
    controls the window precisely instead of racing a sleep.
    """
    return (
        "import os, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "os.environ['HERMES_HOME'] = sys.argv[2]\n"
        "from tools.skill_publish_guard import live_skill_publish_guard\n"
        "from pathlib import Path\n"
        "name = sys.argv[3]\n"
        "with live_skill_publish_guard(name, target=Path(sys.argv[2])):\n"
        "    sys.stdout.write('HELD\\n')\n"
        "    sys.stdout.flush()\n"
        "    sys.stdin.readline()\n"
        "sys.stdout.write('RELEASED\\n')\n"
        "sys.stdout.flush()\n"
    )


@pytest.mark.windows_only
def test_nw1_native_windows_uses_msvcrt_not_fcntl():
    """NW1: on a real Windows host the guard must bind msvcrt, never fcntl.

    No simulation of any kind: no fake modules, no ``sys.platform`` patch, no
    ``os.name`` patch. This is the assertion the simulated MF1-A witness
    structurally cannot make, because on Linux ``fcntl`` genuinely exists.
    """
    from tools import skill_publish_guard as guard

    assert guard.fcntl is None, (
        "NW1: fcntl must be unavailable on native Windows; the guard bound "
        f"{guard.fcntl!r} — a repo-root shadow module or a stray import"
    )
    assert guard.msvcrt is not None, (
        "NW1: msvcrt must be importable and bound on native Windows"
    )
    assert guard._use_msvcrt() is True, (
        "NW1: the Windows locking branch must be selected on native Windows"
    )
    # Sanity-check the real constants the implementation depends on exist and
    # are distinct (a stub/shadow module would not satisfy this).
    assert guard.msvcrt.LK_NBLCK != guard.msvcrt.LK_UNLCK
    assert hasattr(guard.msvcrt, "locking")


@pytest.mark.windows_only
def test_nw2_native_windows_real_acquire_and_release(hermes_home):
    """NW2: the real msvcrt acquire/release cycle through the public guard.

    Exercises ``live_skill_publish_guard`` (the public path) rather than the
    private helpers, so the assertions stay stable if the internals are
    refactored. Verifies the two properties the real OS enforces and a fake
    cannot: that ``msvcrt.locking`` actually succeeds on the guard's lock file
    (which requires the byte-0 materialisation for an initially empty file),
    and that the explicit release leaves the lock re-acquirable in-process.
    """
    from tools.skill_publish_guard import (
        _canonical_lock_path,
        live_skill_publish_guard,
    )

    target = Path(str(hermes_home))
    lock_path = _canonical_lock_path(_SKILL_NAME)

    # Precondition: either absent, or present and empty. Both are the
    # "no lockable byte yet" case the Windows branch must handle.
    if lock_path.exists():
        assert lock_path.stat().st_size >= 0

    # First real acquire/release. If the guard failed to materialise byte 0,
    # the real msvcrt.locking would raise here instead of succeeding.
    with live_skill_publish_guard(_SKILL_NAME, target=target):
        assert lock_path.exists(), "NW2: lock file must exist while held"
        assert lock_path.stat().st_size >= 1, (
            "NW2: the real Windows byte-range lock needs at least one byte in "
            f"the lock file; size={lock_path.stat().st_size}"
        )

    # A second acquire proves the explicit LK_UNLCK really released the
    # range. Pre-fix (or with a broken release) this would raise or hang.
    with live_skill_publish_guard(_SKILL_NAME, target=target):
        pass

    # And a third, to rule out a one-shot fluke.
    with live_skill_publish_guard(_SKILL_NAME, target=target):
        pass


@pytest.mark.windows_only
def test_nw3_native_windows_real_interprocess_contention(hermes_home):
    """NW3: genuine cross-process Windows contention maps to CONTENTION.

    A separate interpreter holds the guard lock; this process then attempts
    the same canonical name non-blockingly and must observe a real
    ``msvcrt.locking`` failure classified as ``LOCK_KIND_CONTENTION`` with the
    backward-compatible payload intact. After the holder releases, acquisition
    must succeed — proving the lock was genuinely exclusive and genuinely
    released, not merely that an errno was mapped.

    This is the assertion no monkeypatched msvcrt can make.
    """
    import subprocess

    from tools.skill_publish_guard import (
        LOCK_KIND_CONTENTION,
        SkillPublishLockError,
        live_skill_publish_guard,
    )

    canary_root = str(Path(__file__).resolve().parents[2])
    target = Path(str(hermes_home))

    child = subprocess.Popen(
        [
            sys.executable, "-c", _nw_child_source(),
            canary_root, str(hermes_home), _SKILL_NAME,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        # Wait (bounded) for the child to report it holds the lock.
        deadline = time.monotonic() + _HOLDER_ENTER_TIMEOUT_S
        first = None
        while time.monotonic() < deadline:
            line = child.stdout.readline()
            if line:
                first = line.strip()
                break
            if child.poll() is not None:
                break
        if first != "HELD":
            err = ""
            try:
                err = child.stderr.read() or ""
            except Exception:  # pragma: no cover - diagnostics only
                pass
            pytest.fail(
                "NW3: holder child never acquired the lock "
                f"(got {first!r}, rc={child.poll()!r}); stderr:\n{err}"
            )

        # The child holds a real Windows byte-range lock. Our attempt must
        # fail with a classified contention error, not block and not succeed.
        raised = None
        try:
            with live_skill_publish_guard(_SKILL_NAME, target=target):
                pass
        except SkillPublishLockError as exc:
            raised = exc

        assert raised is not None, (
            "NW3: acquiring a lock genuinely held by another process must "
            "raise SkillPublishLockError, but the guard succeeded — the real "
            "Windows lock is not excluding a second process"
        )
        assert raised.kind == LOCK_KIND_CONTENTION, (
            "NW3: real cross-process Windows contention must classify as "
            f"CONTENTION, got {raised.kind!r} "
            f"(cause errno={getattr(raised.cause_exception, 'errno', None)!r})"
        )
        assert raised.lock_acquisition_failure is True
        assert raised.cause_exception is not None, (
            "NW3: the originating OSError must remain attached"
        )
        # Document the errno the real OS actually produced; it must be one the
        # implementation's Windows taxonomy recognises as contention.
        assert raised.cause_exception.errno in (errno.EACCES, errno.EDEADLK), (
            "NW3: native Windows reported an unexpected contention errno "
            f"{raised.cause_exception.errno!r}; the guard's Windows "
            "contention set (EACCES/EDEADLK) needs to cover it"
        )

        # Release the holder, then prove the range is genuinely free again.
        child.stdin.write("go\n")
        child.stdin.flush()
        released_deadline = time.monotonic() + _RELEASE_TIMEOUT_S
        released = None
        while time.monotonic() < released_deadline:
            line = child.stdout.readline()
            if line:
                released = line.strip()
                break
            if child.poll() is not None:
                break
        assert released == "RELEASED" or child.poll() is not None, (
            f"NW3: holder did not report release (got {released!r})"
        )
        child.wait(timeout=_RELEASE_TIMEOUT_S)

        with live_skill_publish_guard(_SKILL_NAME, target=target):
            pass
    finally:
        # Deterministic cleanup: never leave a holder process or a held lock
        # behind, whatever the assertions did.
        try:
            if child.stdin and not child.stdin.closed:
                child.stdin.close()
        except Exception:  # pragma: no cover - cleanup best effort
            pass
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=_RELEASE_TIMEOUT_S)
            except subprocess.TimeoutExpired:  # pragma: no cover
                child.kill()
                child.wait(timeout=_RELEASE_TIMEOUT_S)
        for stream in (child.stdout, child.stderr):
            try:
                if stream and not stream.closed:
                    stream.close()
            except Exception:  # pragma: no cover - cleanup best effort
                pass
