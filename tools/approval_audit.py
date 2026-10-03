"""Append-only approval / guarded-command audit sink.

Records one row per approval decision and per guarded command evaluation, keyed on
``trace_id``, so "did anything attempt to reach a protected path?" is answerable from
data instead of inferred from a log sweep.

Storage: daily-partitioned SQLite files under ``<hermes root>/audit/``
(``approval_events-YYYY-MM-DD.db``), one HMAC key file shared by every partition.

Append-only is enforced at three layers:

1. The Python write surface exposes nothing but :func:`record_event` — there is no
   update or delete function.
2. Two SQLite triggers ``RAISE(ABORT)`` on ``UPDATE`` and ``DELETE`` for every reader
   of the file, including ``sqlite3`` from a shell.
3. Every row's ``row_hash`` is ``HMAC(key, partition || prev_hash || row fields)``, so
   a rewrite performed by dropping the triggers (or restoring an older file) is
   detectable with :func:`verify_partitions`.

Retention operates on whole day-partition files and never on rows: a partition older
than ``security.audit.retention_days`` is unlinked. No surviving row is ever mutated,
which is what keeps guarantee (2) unconditional.

``trace_id`` resolution order: an explicitly bound context id
(:func:`set_current_trace_id`, used by an embedding host), then the session-scoped
``HERMES_SESSION_TRACE_ID`` (so a dispatcher that spawns a worker can propagate the
Company OS delegation chain into the agent process), then a stable
``sess:<session_key>[/turn:<turn_id>]`` fallback so a row is never untraceable.

Every function here fails soft: an audit failure must never change an approval
decision or interrupt a turn.
"""

from __future__ import annotations

import datetime as _dt
import functools
import hashlib
import hmac
import os
import re
import sqlite3
import threading
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from hermes_constants import get_default_hermes_root, get_hermes_home, profile_name_for_home

AUDIT_DIR_NAME = "audit"
EVENT_FILE_PREFIX = "approval_events-"
KEY_FILE_NAME = "approval_hmac.key"
DEFAULT_RETENTION_DAYS = 180

#: Partitions are named ``approval_events-YYYY-MM-DD.db``; the regex doubles as the
#: guard that keeps pruning from touching anything else in the directory.
_PARTITION_RE = re.compile(r"^approval_events-(\d{4}-\d{2}-\d{2})\.db$")

DECISIONS = ("allow", "deny", "prompt")

_APPEND_ONLY_ABORT = "approval_events is append-only: this statement is not permitted"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS approval_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ns        INTEGER NOT NULL,
    ts           TEXT    NOT NULL,
    trace_id     TEXT    NOT NULL,
    profile      TEXT    NOT NULL,
    session_key  TEXT    NOT NULL,
    surface      TEXT    NOT NULL,
    decision     TEXT    NOT NULL,
    outcome      TEXT    NOT NULL,
    class_key    TEXT    NOT NULL,
    pattern_id   TEXT    NOT NULL,
    target_digest TEXT   NOT NULL,
    target_len   INTEGER NOT NULL,
    prev_hash    TEXT    NOT NULL,
    row_hash     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_approval_events_ts_ns ON approval_events(ts_ns);
CREATE INDEX IF NOT EXISTS idx_approval_events_trace_id ON approval_events(trace_id);
CREATE INDEX IF NOT EXISTS idx_approval_events_class_key ON approval_events(class_key);
CREATE TRIGGER IF NOT EXISTS approval_events_no_update
BEFORE UPDATE ON approval_events
BEGIN
    SELECT RAISE(ABORT, 'approval_events is append-only: UPDATE is not permitted');
END;
CREATE TRIGGER IF NOT EXISTS approval_events_no_delete
BEFORE DELETE ON approval_events
BEGIN
    SELECT RAISE(ABORT, 'approval_events is append-only: DELETE is not permitted');
END;
"""

# --- Gate bookkeeping -------------------------------------------------------------------------------------------
# The three public guards nest (check_dangerous_command -> _run_approval_gate), and the card's contract is
# exactly ONE row per evaluation. The classifier leaves a note describing what fired; the decorator records it
# only when the outermost gate unwinds.

_audit_depth: ContextVar[int] = ContextVar("approval_audit_depth", default=0)
_audit_note: ContextVar[Optional[dict]] = ContextVar("approval_audit_note", default=None)
_trace_id_var: ContextVar[str] = ContextVar("approval_audit_trace_id", default="")

_key_lock = threading.Lock()
_key_cache: Optional[bytes] = None
_conn_local = threading.local()

# ``security.audit`` is read on every guarded call, and the config loader costs ~3.5 ms
# per call (it re-merges DEFAULT_CONFIG). Cache it briefly: 5 s of staleness on a knob
# that defaults to "on" is invisible, and an approval call must not pay a YAML parse.
_AUDIT_CFG_TTL = 5.0
_audit_cfg_cache: Optional[tuple[float, dict]] = None


def clear_audit_config_cache() -> None:
    """Drop the TTL cache (tests, and anything that just rewrote config.yaml)."""
    global _audit_cfg_cache
    _audit_cfg_cache = None


def audit_note(*, surface: str, pattern_id: str = "", class_key: str = "",
               outcome: str = "") -> None:
    """Declare that the current gate evaluation must be recorded.

    Called by the classifier, which is the only place that knows which rule fired.
    Nested gates overwrite the note, so the innermost (most specific) classification wins.
    """
    _audit_note.set({"surface": surface, "pattern_id": pattern_id or "",
                     "class_key": class_key or "", "outcome": outcome or ""})


def audit_outcome(outcome: str) -> None:
    """Refine the outcome on the pending note (``bypass:yolo``, ``session_approved``, ...)."""
    note = _audit_note.get()
    if note is not None:
        note["outcome"] = outcome


def take_audit_note() -> Optional[dict]:
    """Consume and clear the pending note (outermost gate only)."""
    note = _audit_note.get()
    _audit_note.set(None)
    return note


def audit_gate(target_of):
    """Decorator: record exactly ONE audit row for the outermost guarded evaluation.

    ``target_of(args, kwargs)`` returns the target to digest (the command, the script, or
    the gate's display label). Gates nest — ``check_dangerous_command`` delegates to
    ``_run_approval_gate`` — and the classifier inside leaves a single note, so depth
    tracking is what makes "one row per approval decision" literally true.
    """
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            depth = _audit_depth.get()
            if depth == 0:
                _audit_note.set(None)
            _audit_depth.set(depth + 1)
            try:
                result = fn(*args, **kwargs)
            except BaseException:
                _audit_depth.set(depth)
                if depth == 0:
                    _audit_note.set(None)
                raise
            _audit_depth.set(depth)
            if depth == 0:
                note = take_audit_note()
                if note is not None:
                    decision, derived = decision_from_result(result)
                    record_event(
                        surface=note["surface"], target=target_of(args, kwargs),
                        decision=decision, outcome=note["outcome"] or derived,
                        pattern_id=note["pattern_id"], class_key=note["class_key"])
            return result
        return wrapper
    return decorate


def set_current_trace_id(trace_id: str) -> Any:
    """Bind an explicit ``trace_id`` for the current context; returns the prior token."""
    return _trace_id_var.set(trace_id or "")


def reset_current_trace_id(token: Any) -> None:
    _trace_id_var.reset(token)


def current_trace_id() -> str:
    """The correlation id this row is keyed on (see the module docstring for precedence)."""
    trace_id = (_trace_id_var.get() or "").strip()
    if trace_id:
        return trace_id
    try:
        from gateway.session_context import get_session_env
        trace_id = (get_session_env("HERMES_SESSION_TRACE_ID", "") or "").strip()
    except Exception:
        trace_id = ""
    if trace_id:
        return trace_id
    session_key, turn_id = _session_and_turn()
    trace_id = f"sess:{session_key or '-'}"
    if turn_id:
        trace_id = f"{trace_id}/turn:{turn_id}"
    return trace_id


def current_profile() -> str:
    """Active profile name; session-scoped binding wins over the HOME-derived one."""
    try:
        from gateway.session_context import get_session_env
        session_profile = get_session_env("HERMES_SESSION_PROFILE", "") or ""
    except Exception:
        session_profile = ""
    if session_profile:
        return session_profile
    try:
        return profile_name_for_home(get_hermes_home()) or "default"
    except Exception:
        return "default"


def _session_and_turn() -> tuple[str, str]:
    try:
        from tools import approval_context
        session_key = approval_context.get_current_session_key("")
        turn_id = approval_context._approval_turn_id.get() or ""
    except Exception:
        session_key, turn_id = "", ""
    return session_key, turn_id


# --- Config -------------------------------------------------------------------------------------------------------

def _audit_config() -> dict:
    global _audit_cfg_cache
    cached = _audit_cfg_cache
    now = time.monotonic()
    if cached is not None and now - cached[0] < _AUDIT_CFG_TTL:
        return cached[1]
    try:
        from hermes_cli.config import load_config_readonly
        cfg = ((load_config_readonly() or {}).get("security") or {}).get("audit") or {}
    except Exception:
        cfg = {}
    _audit_cfg_cache = (now, cfg)
    return cfg


def audit_enabled() -> bool:
    """``security.audit.enabled`` (default true — an audit log nobody turns on answers nothing)."""
    try:
        return bool(_audit_config().get("enabled", True))
    except Exception:
        return True


def retention_days() -> int:
    """``security.audit.retention_days`` (default 180; 0 keeps every partition)."""
    try:
        return max(int(_audit_config().get("retention_days", DEFAULT_RETENTION_DAYS)), 0)
    except Exception:
        return DEFAULT_RETENTION_DAYS


# --- Paths and key ------------------------------------------------------------------------------------------------

def audit_dir() -> Path:
    """Root-anchored, so every profile appends to (and queries) the SAME store.

    A per-profile store would answer the security review's cross-profile question only
    by merging N databases, and would let activity in profile A hide from a query run
    in profile B. Root is also where the review looked and found nothing.
    """
    return get_default_hermes_root() / AUDIT_DIR_NAME


def partition_path(day: Optional[str] = None) -> Path:
    """``approval_events-YYYY-MM-DD.db`` for ``day`` (UTC, default: today)."""
    day = day or _utc_day()
    return audit_dir() / f"{EVENT_FILE_PREFIX}{day}.db"


def list_partitions() -> list[Path]:
    """Every partition file, oldest first. Unreadable directories yield nothing."""
    try:
        entries = [p for p in audit_dir().iterdir() if _PARTITION_RE.match(p.name)]
    except OSError:
        return []
    return sorted(entries, key=lambda p: p.name)


def _utc_day(ts_ns: Optional[int] = None) -> str:
    if ts_ns is None:
        return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    return _dt.datetime.fromtimestamp(ts_ns / 1e9, _dt.timezone.utc).strftime("%Y-%m-%d")


def _load_key() -> bytes:
    """32 random bytes persisted next to the partitions so digests stay correlatable across runs."""
    global _key_cache
    if _key_cache is not None:
        return _key_cache
    with _key_lock:
        if _key_cache is not None:
            return _key_cache
        path = audit_dir() / KEY_FILE_NAME
        try:
            existing = path.read_bytes()
            if len(existing) >= 32:
                _key_cache = existing[:32]
                return _key_cache
        except OSError:
            pass
        key = os.urandom(32)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_private(path, key)
            _key_cache = key
        except OSError:
            # An unwritable key file must not disable the audit: fall back to a per-process
            # key. Digests stop correlating across processes, but rows still land.
            _key_cache = key
        return _key_cache


def _write_private(path: Path, payload: bytes) -> None:
    """Create ``path`` with owner-only permissions where the platform supports them."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def target_digest(target: str) -> tuple[str, int]:
    """``(hmac_sha256[:16] hex, raw length)`` — never the target itself.

    Salted (keyed) so two runs of the same command correlate while the store cannot be
    brute-forced offline for low-entropy command shapes. Raw secret material never
    reaches the file: only the digest and the length.
    """
    raw = target if isinstance(target, str) else str(target)
    encoded = raw.encode("utf-8", "replace")
    digest = hmac.new(_load_key(), encoded, hashlib.sha256).hexdigest()[:16]
    return digest, len(encoded)


# --- Writing ------------------------------------------------------------------------------------------------------

def decision_from_result(result: Any) -> tuple[str, str]:
    """``(decision, outcome)`` for an approval-gate result dict.

    ``decision`` is one of :data:`DECISIONS`. ``prompt`` means the gate escalated and no
    human has answered yet (``pending_approval`` / ``approval_required``).
    """
    if not isinstance(result, dict):
        return "deny", "unknown"
    if result.get("approved"):
        if result.get("smart_approved"):
            return "allow", "smart_approved"
        if result.get("user_approved"):
            return "allow", "user_approved"
        return "allow", "approved"
    status = str(result.get("status") or "")
    if status in ("pending_approval", "approval_required"):
        return "prompt", status
    outcome = str(result.get("outcome") or "").strip()
    if not outcome:
        if result.get("user_deny"):
            outcome = "user_deny"
        elif result.get("hardline"):
            outcome = "hardline"
        elif result.get("smart_denied"):
            outcome = "smart_denied"
        elif "message" in result:
            outcome = "blocked"
        else:
            outcome = "denied"
    return "deny", outcome


def record_event(*, surface: str, target: str, decision: str, outcome: str = "",
                 pattern_id: str = "", class_key: str = "", trace_id: str = "",
                 profile: str = "") -> bool:
    """Append one event. Returns True when the row landed; never raises.

    The single write API: there is deliberately no update/delete counterpart.
    """
    if decision not in DECISIONS:
        decision = "deny"
    try:
        if not audit_enabled():
            return False
        session_key, _turn = _session_and_turn()
        digest, length = target_digest(target)
        ts_ns = time.time_ns()
        row = {
            "ts_ns": ts_ns,
            "ts": _iso(ts_ns),
            "trace_id": trace_id or current_trace_id(),
            "profile": profile or current_profile(),
            "session_key": session_key or "",
            "surface": surface,
            "decision": decision,
            "outcome": outcome or "",
            "class_key": class_key or "",
            "pattern_id": pattern_id or "",
            "target_digest": digest,
            "target_len": length,
        }
        _append(row)
        _maybe_prune()
        return True
    except Exception:
        import logging
        logging.getLogger("tools.approval").debug("approval audit append failed", exc_info=True)
        return False


def _iso(ts_ns: int) -> str:
    stamp = _dt.datetime.fromtimestamp(ts_ns / 1e9, _dt.timezone.utc)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.") + f"{stamp.microsecond // 1000:03d}Z"


def _open(path: Path, *, create: bool) -> sqlite3.Connection:
    """Open (creating when asked) a partition.

    The schema script is DDL and DDL is not free: running ``executescript`` on every
    append cost ~16 ms/call in the first measurement of this store. One probe of
    ``sqlite_master`` decides whether the append-only triggers are already there, which
    keeps an append in the tens-of-microseconds range while still failing safe — a
    partition someone created without its triggers gets them re-applied, not trusted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0, isolation_level=None)
    try:
        # journal_mode is persistent in the file (this line is a cheap read afterwards);
        # synchronous is per-connection, so it must be set every time.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=10000")
        if create:
            guarded = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='trigger'"
                " AND name='approval_events_no_update'").fetchone()
            if guarded is None:
                conn.executescript(_SCHEMA)
    except Exception:
        conn.close()
        raise
    return conn


def _genesis(partition: str, key: bytes) -> str:
    return hmac.new(key, b"genesis|" + partition.encode("ascii"), hashlib.sha256).hexdigest()


def _row_hash(key: bytes, partition: str, prev_hash: str, row: dict) -> str:
    material = "\x1f".join((
        partition, prev_hash, str(row["ts_ns"]), row["trace_id"], row["profile"],
        row["session_key"], row["surface"], row["decision"], row["outcome"],
        row["class_key"], row["pattern_id"], row["target_digest"], str(row["target_len"]),
    ))
    return hmac.new(key, material.encode("utf-8", "replace"), hashlib.sha256).hexdigest()


def _pooled_connection(path: Path) -> sqlite3.Connection:
    """Reuse one connection per partition per thread.

    A cold ``sqlite3.connect`` on Windows costs ~2.8 ms and closing the last handle
    checkpoints-and-removes the WAL, so a connect/append/close loop paid that on every
    single approval decision. Connections are thread-local: SQLite connections are not
    thread-safe, and ``BEGIN IMMEDIATE`` still serialises writers across threads and
    processes through the busy timeout.
    """
    pool = getattr(_conn_local, "pool", None)
    if pool is None:
        pool = _conn_local.pool = {}
    conn = pool.get(str(path))
    if conn is not None:
        try:
            conn.execute("SELECT 1")
            return conn
        except sqlite3.Error:
            with _suppress_oserror():
                conn.close()
            pool.pop(str(path), None)
    conn = _open(path, create=True)
    pool[str(path)] = conn
    return conn


def _close_pooled(path: Path) -> None:
    """Release this thread's handle so a partition can actually be unlinked on Windows."""
    pool = getattr(_conn_local, "pool", None)
    if not pool:
        return
    conn = pool.pop(str(path), None)
    if conn is not None:
        with _suppress_oserror():
            conn.close()


def _append(row: dict) -> None:
    key = _load_key()
    day = _utc_day(row["ts_ns"])
    partition = f"{EVENT_FILE_PREFIX}{day}.db"
    conn = _pooled_connection(audit_dir() / partition)
    # BEGIN IMMEDIATE so two writers cannot both read the same tail hash and fork the chain.
    conn.execute("BEGIN IMMEDIATE")
    try:
        tail = conn.execute(
            "SELECT row_hash FROM approval_events ORDER BY id DESC LIMIT 1").fetchone()
        prev_hash = tail[0] if tail else _genesis(partition, key)
        row["prev_hash"] = prev_hash
        row["row_hash"] = _row_hash(key, partition, prev_hash, row)
        conn.execute(
            "INSERT INTO approval_events (ts_ns, ts, trace_id, profile, session_key,"
            " surface, decision, outcome, class_key, pattern_id, target_digest,"
            " target_len, prev_hash, row_hash)"
            " VALUES (:ts_ns, :ts, :trace_id, :profile, :session_key, :surface,"
            " :decision, :outcome, :class_key, :pattern_id, :target_digest,"
            " :target_len, :prev_hash, :row_hash)", row)
        conn.execute("COMMIT")
    except BaseException:
        with _suppress_oserror():
            conn.execute("ROLLBACK")
        raise


# --- Retention ----------------------------------------------------------------------------------------------------

_prune_lock = threading.Lock()
_last_prune = 0.0
_PRUNE_INTERVAL_S = 3600.0


def _maybe_prune(force: bool = False) -> None:
    """Unlink day-partitions past ``retention_days``. Whole files only, at most once an hour per process."""
    global _last_prune
    with _prune_lock:
        now = time.monotonic()
        if not force and now - _last_prune < _PRUNE_INTERVAL_S:
            return
        _last_prune = now
    if retention_days() <= 0:
        return
    prune_partitions(retention_days())


def prune_partitions(days: int) -> list[str]:
    """Remove partitions dated more than ``days`` days ago (UTC). Returns removed file names."""
    if days <= 0:
        return []
    cutoff = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=days)).strftime("%Y-%m-%d")
    removed = []
    for path in list_partitions():
        match = _PARTITION_RE.match(path.name)
        if match and match.group(1) < cutoff:
            _close_pooled(path)  # Windows cannot unlink a file this thread still holds open
            try:
                path.unlink()
                removed.append(path.name)
            except OSError:
                # A held-open WAL or a read-only mount keeps the file: retention retries next hour.
                pass
            for sidecar in (f"{path.name}-wal", f"{path.name}-shm"):
                with _suppress_oserror():
                    (audit_dir() / sidecar).unlink()
    return removed


class _suppress_oserror:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return exc_type is not None and issubclass(exc_type, OSError)


# --- Reading ------------------------------------------------------------------------------------------------------

COLUMNS = ("id", "ts_ns", "ts", "trace_id", "profile", "session_key", "surface",
           "decision", "outcome", "class_key", "pattern_id", "target_digest",
           "target_len", "prev_hash", "row_hash")


def iter_events(*, partitions: Optional[Iterable[Path]] = None, since_days: Optional[int] = None,
                trace_id: str = "", decision: str = "", surface: str = "",
                class_like: Any = "", limit: int = 0) -> Iterator[dict]:
    """Yield events newest-first across the selected partitions.

    ``class_like`` is a SQL ``LIKE`` pattern over ``class_key`` — the filter that answers
    "which attempts targeted a protected path?" (e.g. ``%Hermes secrets%``). A sequence of
    patterns is OR'd together, which is what ``--protected`` needs: one pattern per rule
    family, all in one pass.
    """
    if partitions is None:
        partitions = list_partitions()
    if since_days:
        cutoff = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=since_days))
        cutoff = cutoff.replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d")
        partitions = [p for p in partitions if (m := _PARTITION_RE.match(p.name)) and m.group(1) >= cutoff]
    clauses, params = [], []
    if trace_id:
        clauses.append("trace_id = ?")
        params.append(trace_id)
    if decision:
        clauses.append("decision = ?")
        params.append(decision)
    if surface:
        clauses.append("surface = ?")
        params.append(surface)
    if class_like:
        patterns = [class_like] if isinstance(class_like, str) else list(class_like)
        if patterns:
            clauses.append(f"({' OR '.join('class_key LIKE ?' for _ in patterns)})")
            params.extend(patterns)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM approval_events{where} ORDER BY ts_ns DESC, id DESC"
    # Newest partition first, rows already DESC inside it: the stream is globally newest-first
    # and stops as soon as ``limit`` rows are out, so a big store costs one partition, not all.
    emitted = 0
    for path in reversed(list(partitions)):
        rows = _read_rows(path, sql, params)
        for row in rows:
            yield row
            emitted += 1
            if limit and emitted >= limit:
                return


def _read_rows(path: Path, sql: str, params: list) -> list[dict]:
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error:
        return []
    try:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error:
            return []
        return [dict(row) for row in rows]
    finally:
        conn.close()


def read_events(**kwargs) -> list[dict]:
    """Materialised :func:`iter_events`, oldest-first (convenient for verification)."""
    return list(reversed(list(iter_events(**kwargs))))


def verify_partitions(partitions: Optional[Iterable[Path]] = None) -> dict:
    """Recompute each row's HMAC chain. Returns ``{partition: "ok"|"broken: <row id>"}``.

    Detects edits that the UPDATE/DELETE triggers cannot stop — a dropped trigger, an
    out-of-band ``sqlite3`` write, or a swapped-in older file.
    """
    key = _load_key()
    report: dict[str, str] = {}
    for path in (list(partitions) if partitions is not None else list_partitions()):
        match = _PARTITION_RE.match(path.name)
        label = match.group(1) if match else path.name
        if not path.exists():
            report[label] = "missing"
            continue
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(
                "SELECT * FROM approval_events ORDER BY id ASC").fetchall()]
            conn.close()
        except sqlite3.Error:
            report[label] = "unreadable"
            continue
        expected_prev = _genesis(path.name, key)
        status = "ok"
        for row in rows:
            if row["prev_hash"] != expected_prev:
                status = f"broken: row {row['id']} prev_hash does not chain"
                break
            material_row = {k: row[k] for k in COLUMNS}
            if _row_hash(key, path.name, row["prev_hash"], material_row) != row["row_hash"]:
                status = f"broken: row {row['id']} row_hash mismatch"
                break
            expected_prev = row["row_hash"]
        report[label] = status
    return report


def audit_summary(since_days: Optional[int] = None) -> dict:
    """Aggregate counts the query surface prints: by decision, by class_key, by profile."""
    totals: dict[str, int] = {}
    by_class: dict[str, int] = {}
    by_profile: dict[str, int] = {}
    by_surface: dict[str, int] = {}
    rows = 0
    for row in iter_events(since_days=since_days):
        rows += 1
        totals[row["decision"]] = totals.get(row["decision"], 0) + 1
        key = row["class_key"] or "(unclassified)"
        by_class[key] = by_class.get(key, 0) + 1
        by_profile[row["profile"] or "-"] = by_profile.get(row["profile"] or "-", 0) + 1
        by_surface[row["surface"]] = by_surface.get(row["surface"], 0) + 1
    return {"rows": rows, "by_decision": totals, "by_class_key": by_class,
            "by_profile": by_profile, "by_surface": by_surface,
            "retention_days": retention_days(), "enabled": audit_enabled(),
            "store": str(audit_dir())}
