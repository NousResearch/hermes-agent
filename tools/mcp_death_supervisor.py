#!/usr/bin/env python3
"""One parent-death supervisor per Hermes process, shared by all stdio MCP servers.

Why this exists
---------------
When Hermes dies without running its cleanup path (SIGKILL, OOM killer, a hard
crash), stdio MCP servers it spawned are reparented to init and keep running
forever.  macOS has no ``PR_SET_PDEATHSIG``, so something has to outlive Hermes
and reap them.

The predecessor (``mcp_stdio_watchdog.py``) solved this with one CPython
*per MCP server*, wrapping each server command and polling ``getppid()`` every
two seconds.  That costs ~10 MB of resident memory per server and detects death
up to one poll interval late.  This module replaces the whole fleet of pollers
with a single supervisor per Hermes process:

* **Death detection is a blocking read on a pipe.**  Hermes holds the only write
  end.  When Hermes dies -- by any means, including SIGKILL -- the write end
  closes and the read returns EOF.  Exact, instant, and free.
* **Servers are spawned unwrapped.**  The MCP SDK already spawns stdio children
  with ``start_new_session=True``, so each one is its own process-group leader
  and ``killpg`` still reaches its descendants.  Removing the wrapper also
  removes the signal-forwarding layer the wrapper needed to avoid inverting the
  bug it fixed.

Protocol (line-based, on stdin)
-------------------------------
    register <pgid>\n     start reaping this process group on parent death
    unregister <pgid>\n   stop reaping it (its server shut down cleanly)

On EOF the supervisor SIGTERMs every still-registered process group, waits a
short grace period, SIGKILLs the survivors, and exits.  A registered group that
Hermes never unregistered *is* the orphan set, so a clean Hermes shutdown --
which unregisters as it tears each server down -- ends with nothing to kill.

Unparseable lines are ignored rather than fatal: a corrupted byte on the control
pipe must not cost us the reaping guarantee for every other server.

Residual risk: process-group reuse
----------------------------------
We reap by pgid, so a registration is only as meaningful as the group's
identity.  A group we deliberately keep registered -- an orphan that teardown
failed to kill, such as the ``node`` ``mcp-remote`` leaves behind -- can
eventually exit on its own, after which the kernel is free to hand that pgid to
an unrelated process owned by the same user.  If Hermes then dies ungracefully
while the registration is still stale, we would signal a stranger.
``_is_safe_target`` cannot catch this: the value is stale, not invalid.

Two things narrow the window.  Hermes prunes registrations whose group has no
members left (``_prune_dead_supervised_pgids``) on every registration change,
and the orphan sweep unregisters whatever it reaps.  Neither closes it -- a
group can die and its pgid be recycled between two probes -- so the exposure is
real but bounded to that gap, and requires an ungraceful death inside it.

Closing it completely means proving group identity at reap time, e.g. stamping
MCP children with a boot-unique env marker and checking that some member still
carries it before signalling.  That was judged not worth putting a ``ps`` parse
into the one process whose job is to stay simple enough to always work; it is
the obvious next step if this class of bug ever actually bites.  Note the same
exposure already exists in Hermes's own killpg-based orphan cleanup, which this
module did not introduce (see upstream issue #88350).
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

# Matches the grace period the per-server watchdog used before it escalated.
_TERM_GRACE_S = 3.0
# How often we re-check for survivors during that grace period.
_REAP_POLL_S = 0.1
# A command is "unregister <pgid>" -- around 20 characters. The cap only has to
# be generous enough for a legitimate line; see _serve for why it exists.
_MAX_LINE_CHARS = 256


def _is_safe_target(pgid: int, *, own_pgid: int, parent_pgid: int) -> bool:
    """Return True if ``pgid`` is a process group we may signal.

    Defensive only -- Hermes already filters non-MCP children before it
    registers anything (see ``_filter_mcp_children`` in ``tools/mcp_tool.py``).
    But this process signals whole process *groups*, so a bad value here is
    unusually expensive: ``killpg(0, ...)`` signals our own group, and pgid 1
    is init.  A caller bug should cost us one unreaped server, never the
    Hermes process tree or the session.
    """
    if pgid <= 1:
        return False
    if pgid == own_pgid or pgid == parent_pgid:
        return False
    return True


def _reap(pgids: set[int]) -> None:
    """SIGTERM every group, then SIGKILL whatever is still alive.

    Every process-group call below is POSIX-only by construction: this whole
    module only ever runs as a child of ``_update_death_supervisor``, which
    returns early unless ``os.name == "posix"``, so the supervisor is never
    spawned on Windows in the first place.
    """
    if not pgids:
        return

    alive = set()
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGTERM)  # windows-footgun: ok — POSIX-only process
            alive.add(pgid)
        except (ProcessLookupError, PermissionError, OSError):
            # Already gone, or not ours to signal. Either way, nothing to reap.
            pass

    deadline = time.monotonic() + _TERM_GRACE_S
    while alive and time.monotonic() < deadline:
        time.sleep(_REAP_POLL_S)
        for pgid in list(alive):
            try:
                # Signal 0 probes liveness: succeeds iff some member survives.
                os.killpg(pgid, 0)  # windows-footgun: ok — POSIX-only process
            except (ProcessLookupError, PermissionError, OSError):
                alive.discard(pgid)

    for pgid in alive:
        try:
            os.killpg(pgid, signal.SIGKILL)  # windows-footgun: ok — POSIX-only
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _serve(stream, *, own_pgid: int, parent_pgid: int) -> set[int]:
    """Read control lines until EOF; return the groups still registered.

    Reads are length-capped rather than newline-terminated. Iterating the
    stream instead lets a writer that never sends a newline grow this process
    without bound -- feeding it ``/dev/zero`` reached 15 GB before it was
    stopped. Nothing in Hermes can produce that today, but this process is the
    last line of defense against leaked servers, so it must not be the thing
    that dies under memory pressure. A line truncated by the cap fails to parse
    and is skipped; the remainder resyncs at the next newline.
    """
    registered: set[int] = set()
    while True:
        line = stream.readline(_MAX_LINE_CHARS)
        if not line:
            break  # EOF: the parent is gone.
        if not line.endswith("\n"):
            # Truncated by the cap, or an unterminated tail at EOF. Either way
            # it is not a command we are willing to act on.
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        verb, raw = parts
        try:
            pgid = int(raw)
        except ValueError:
            continue
        if verb == "register":
            if _is_safe_target(pgid, own_pgid=own_pgid, parent_pgid=parent_pgid):
                registered.add(pgid)
        elif verb == "unregister":
            registered.discard(pgid)
    return registered


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Reap registered process groups when the parent dies."
    )
    parser.add_argument(
        "--parent-pgid",
        type=int,
        required=True,
        help="Process group of the spawning Hermes process; never signalled.",
    )
    args = parser.parse_args(argv)

    # The parent may be torn down with killpg on its own group. We are spawned
    # with start_new_session=True precisely so that sweep cannot take us with
    # it before we have reaped -- assert that here rather than trust the caller.
    own_pgid = os.getpgid(0)
    if own_pgid == args.parent_pgid:
        print(
            "mcp_death_supervisor: refusing to run inside the parent's process "
            "group (a killpg of the parent would kill us before we can reap)",
            file=sys.stderr,
        )
        return 2

    # A dying parent's SIGINT/SIGHUP must not preempt the reap; the pipe's EOF
    # is our only shutdown signal. SIGHUP is POSIX-only, which is fine here --
    # this process is never spawned on Windows (see _reap's docstring).
    for sig in (signal.SIGINT, signal.SIGHUP):  # windows-footgun: ok — POSIX-only process
        try:
            signal.signal(sig, signal.SIG_IGN)
        except (ValueError, OSError):
            pass

    registered = _serve(sys.stdin, own_pgid=own_pgid, parent_pgid=args.parent_pgid)
    _reap(registered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
