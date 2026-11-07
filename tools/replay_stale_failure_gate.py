#!/usr/bin/env python3
"""AC5 live replay: prove the gate against a COPY of the real production board.

Production is never mutated:

* the board is copied with SQLite's online ``.backup()`` API (a consistent
  snapshot even while the live board is being written), and
* the copy is then opened strictly read-only
  (``sqlite3.connect("file:...?mode=ro", uri=True)``).

The replay re-runs the real audit population from the card body: every
failure event that landed AFTER ``tasks.completed_at``, on every board, and
asks the gate what it would do with each one.

Usage:
    python3 replay_stale_failure_gate.py <board.db> [<board.db> ...]
"""

import sqlite3
import sys
from pathlib import Path

FAILURE_KINDS = ("gave_up", "crashed", "timed_out")
TERMINAL_NOTIFY_STATUSES = frozenset({"done", "archived", "cancelled"})


def backup_to(src: Path, dst: Path) -> Path:
    """Consistent online copy of ``src`` via SQLite's .backup() API."""
    src_conn = sqlite3.connect(f"{src.resolve().as_uri()}?mode=ro", uri=True)
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst_conn = sqlite3.connect(str(dst))
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()
    return dst


def stale_failure_events(conn: sqlite3.Connection):
    """Failure events that landed AFTER the card was completed."""
    rows = conn.execute(
        """
        SELECT e.id, e.task_id, e.kind, e.created_at,
               t.status, t.completed_at, t.title
        FROM task_events e
        JOIN tasks t ON t.id = e.task_id
        WHERE e.kind IN ('gave_up', 'crashed', 'timed_out')
          AND t.completed_at IS NOT NULL
          AND e.created_at > t.completed_at
        ORDER BY e.id
        """
    ).fetchall()
    return rows


def gate_decision(status, has_completed_at):
    """Mirror of the notifier's gate: would this wake be dropped?"""
    if status is None:
        return False  # fail OPEN
    return status in TERMINAL_NOTIFY_STATUSES or bool(has_completed_at)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    scratch = Path(
        "/home/serveradmin/.hermes/profiles/ultron/cache/scratch/t_f4423952_replay"
    )
    grand_total = grand_suppressed = grand_delivered = 0
    for board_arg in argv[1:]:
        board = Path(board_arg)
        if not board.exists():
            print(f"skip (missing): {board}")
            continue
        copy = scratch / f"{board.parent.name}-{board.name}.copy"
        backup_to(board, copy)
        # Open the COPY strictly read-only.
        conn = sqlite3.connect(f"{copy.resolve().as_uri()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = stale_failure_events(conn)
            by_status = {}
            for r in rows:
                decision = gate_decision(r["status"], r["completed_at"])
                by_status.setdefault(r["status"], [0, 0])
                by_status[r["status"]][0] += 1
                if decision:
                    by_status[r["status"]][1] += 1
            total = len(rows)
            suppressed = sum(1 for r in rows if gate_decision(r["status"], r["completed_at"]))
            delivered = total - suppressed
            grand_total += total
            grand_suppressed += suppressed
            grand_delivered += delivered
            print(f"\n=== {board.parent.name} ({board.name}) ===")
            print(f"  stale failure events after completed_at: {total}")
            print(f"  would be SUPPRESSED (gate):              {suppressed}")
            print(f"  would be DELIVERED (gate):                {delivered}")
            for status in sorted(by_status, key=lambda s: (s is None, s)):
                n, sup = by_status[status]
                print(f"    live status {status!r:14} -> {sup}/{n} suppressed")
        finally:
            conn.close()
    print("\n=== TOTAL ===")
    print(f"  stale failure events: {grand_total}")
    print(f"  suppressed by gate:   {grand_suppressed}")
    print(f"  delivered by gate:    {grand_delivered}")
    print("\n  production was NOT mutated (replay ran on .backup() copies, opened mode=ro)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
