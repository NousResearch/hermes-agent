"""Verification evidence for finished background processes.

Split out of ``tools/process_registry.py`` (which is past the code-health
file-size cap) so the registry only carries the call site.
"""

import logging

logger = logging.getLogger("tools.process_registry")


def record_verification_evidence(session) -> None:
    """Record a finished background command as verification evidence.

    The foreground terminal path records evidence inline once the command
    returns. A background spawn returns immediately with a synthetic
    ``exit_code: 0`` and never reaches that code, so before this hook a
    test suite too slow for the foreground timeout — i.e. exactly the
    suites that must run in the background — could never record evidence
    at all. ``verification_status()`` then stayed ``stale`` forever and the
    verify-on-stop nudge replayed an older foreground run indefinitely.

    This runs on the single completion convergence point, under the
    ``was_running`` guard, so every exit route (reader thread, PTY reader,
    kill, reconcile) records exactly once and a re-entrant
    ``_move_to_finished`` cannot double-insert.

    Only genuinely finished runs are eligible. A killed or lost process
    proves nothing about the tree, so it is skipped rather than recorded
    with its (possibly zero) exit code. ``_kill_requested`` is checked in
    addition to ``completion_reason`` because a kill races the reader
    thread: the process dies from the signal and the reader can publish it
    as "exited" before kill_process() stamps "killed".
    """
    if session._kill_requested:
        return
    if session.completion_reason not in ("exited", "already_exited"):
        return
    if session.exit_code is None:
        return
    try:
        from agent.verification_evidence import record_terminal_result
        from tools.ansi_strip import strip_ansi

        record_terminal_result(
            command=session.command,
            cwd=session.cwd,
            # Mirror the foreground path's identity fallback so evidence
            # lands under the key verification_status() later reads.
            session_id=(
                session.parent_session_id
                or session.session_key
                or session.task_id
                or "default"
            ),
            exit_code=int(session.exit_code),
            output=strip_ansi(session.output_buffer or ""),
        )
    except Exception:
        # Evidence is advisory; never let it disturb process bookkeeping.
        logger.debug(
            "verification evidence recording failed for %s",
            session.id,
            exc_info=True,
        )
