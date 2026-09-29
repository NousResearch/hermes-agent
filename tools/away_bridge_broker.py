"""Away-Mode Telegram Bridge — broker library (design: Rev 7).

Owns ``$HERMES_HOME/away-bridge/bridge.db``: armed-session rows, per-request
state, duplicate acks, and the sweep/GC/report operations shared by the CLI
(``hermes away-bridge …``) and the Telegram pre-dispatch interceptor
(``gateway/platforms/away_bridge_interceptor.py``).

Normative sources (design doc planning/capabilities/AWAY-MODE-TELEGRAM-BRIDGE.md):
  - §4.1  state path, profile scope, identity model (arm-time key inheritance,
    lineage-relative two-sided lookup), schema
  - §4.2  first-answer-wins conditional-update claim + idempotent duplicate ack
  - §4.3  prepare/start/publish ordering; waiter poll set (JOIN armed_sessions)
  - §4.6  disarm: chat flips every lineage-matching row; Telegram disarms all
  - §4.7  reconcile is a reporter; GC; identity-validated waiter liveness

Every connection sets ``PRAGMA foreign_keys=ON`` and WAL; busy timeouts are
per-caller (short for the gateway event loop path, longer for CLI).
"""

from __future__ import annotations

import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

FULL_ID_PREFIX = "ab"
TOKEN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"  # RFC 4648 base32, no padding
TOKEN_LEN = 16  # 80 bits of display entropy, per §4.8

DEFAULT_POLL_SECONDS = 2.0
DEFAULT_SWEEP_WINDOW_HOURS = 1.0
DEFAULT_RETENTION_DAYS = 30
# Short dedicated busy timeout for interceptor-side broker I/O (§4.4): the
# gateway event loop must never wait long on a desktop-side write lock.
DEFAULT_INTERCEPTOR_BUSY_MS = 1500
# CLI-side busy timeout: chat claims may contend with the interceptor.
DEFAULT_CLI_BUSY_MS = 5000

RETURN_INTENTS = ("i'm back", "im back", "חזרתי")


class BrokerError(Exception):
    """Base class for broker errors surfaced to callers."""


class NotArmedError(BrokerError):
    """prepare() found no armed row lineage-matching the caller."""


class AlreadyAwaitingError(BrokerError):
    """Single-awaiting invariant (§4.5, R5-2): a request is already open."""


class UnknownRequestError(BrokerError):
    """The request key does not resolve to a row."""


class InvalidStateError(BrokerError):
    """The row is in a state that forbids the requested transition."""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def bridge_dir(hermes_home: Optional[Path] = None) -> Path:
    """Resolve ``$HERMES_HOME/away-bridge`` (profile-scoped, never hardcoded)."""
    if hermes_home is None:
        hermes_home = _hermes_home_from_env()
    return Path(hermes_home) / "away-bridge"


def bridge_db_path(hermes_home: Optional[Path] = None) -> Path:
    return bridge_dir(hermes_home) / "bridge.db"


def state_db_path(hermes_home: Optional[Path] = None) -> Path:
    if hermes_home is None:
        hermes_home = _hermes_home_from_env()
    return Path(hermes_home) / "state.db"


def _hermes_home_from_env() -> Path:
    val = os.environ.get("HERMES_HOME", "").strip()
    if val:
        return Path(val)
    return Path.home() / ".hermes"


def load_away_bridge_config(hermes_home: Optional[Path] = None) -> Dict[str, Any]:
    """Read the ``away_bridge`` block from ``config.yaml`` (tolerant).

    Uses ``hermes_cli.config.load_config_readonly()`` (the repo's config-read
    guard requires behavioral reads through the owner modules). Tolerant of
    missing config/errors: returns defaults. The waiter and the gateway
    interceptor both call this on hot paths.
    """
    defaults: Dict[str, Any] = {
        "enabled": False,
        "waiter_poll_seconds": DEFAULT_POLL_SECONDS,
        "sweep_window_hours": DEFAULT_SWEEP_WINDOW_HOURS,
        "retention_days": DEFAULT_RETENTION_DAYS,
        "broker_busy_ms": DEFAULT_INTERCEPTOR_BUSY_MS,
        "telegram": {"owner_chat_id": "", "owner_user_id": ""},
    }
    try:
        from hermes_cli.config import load_config_readonly

        data = load_config_readonly() or {}
    except Exception:
        return defaults
    cfg = data.get("away_bridge") or {}
    if not isinstance(cfg, dict):
        return defaults
    merged = dict(defaults)
    for key in ("enabled", "waiter_poll_seconds", "sweep_window_hours", "retention_days", "broker_busy_ms"):
        if key in cfg:
            merged[key] = cfg[key]
    tg = cfg.get("telegram") or {}
    if isinstance(tg, dict):
        merged["telegram"] = {
            "owner_chat_id": str(tg.get("owner_chat_id", "") or ""),
            "owner_user_id": str(tg.get("owner_user_id", "") or ""),
        }
    return merged


# ---------------------------------------------------------------------------
# connection handling
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS armed_sessions (
    session_key      TEXT PRIMARY KEY,
    armed_at         REAL NOT NULL,
    surface          TEXT NOT NULL,
    status           TEXT NOT NULL CHECK(status IN ('armed','disarmed')),
    disarmed_at      REAL
);

CREATE TABLE IF NOT EXISTS requests (
    req_key          TEXT PRIMARY KEY,
    session_key      TEXT NOT NULL,
    display_token    TEXT UNIQUE,
    question         TEXT NOT NULL,
    state            TEXT NOT NULL CHECK(state IN
                        ('prepared','awaiting','claimed','cancelled')),
    winning_channel  TEXT CHECK(winning_channel IN ('chat','telegram')),
    answer           TEXT,
    created_at       REAL NOT NULL,
    published_at     REAL,
    claimed_at       REAL,
    cancelled_at     REAL,
    delivered        INTEGER NOT NULL DEFAULT 0,
    waiter_pid       INTEGER,
    waiter_start_time INTEGER,
    last_swept_at    REAL,
    FOREIGN KEY(session_key) REFERENCES armed_sessions(session_key)
);

CREATE TABLE IF NOT EXISTS duplicate_acks (
    req_key          TEXT NOT NULL,
    late_channel     TEXT NOT NULL CHECK(late_channel IN ('chat','telegram')),
    sent_at          REAL NOT NULL,
    PRIMARY KEY(req_key, late_channel)
);

CREATE INDEX IF NOT EXISTS idx_requests_state ON requests(state);
CREATE INDEX IF NOT EXISTS idx_requests_session ON requests(session_key, state);
"""


def _connect(db_path: Path, busy_ms: int) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=busy_ms / 1000.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=%d" % busy_ms)
    conn.executescript(SCHEMA)
    return conn


def _token() -> str:
    """Display token: ``A#`` + 16 base32 chars (≥80 bits, §4.3/§4.8).

    Stored WITH the prefix so lookups key on the exact owner-visible form.
    """
    return "A#" + "".join(secrets.choice(TOKEN_ALPHABET) for _ in range(TOKEN_LEN))


# ---------------------------------------------------------------------------
# identity resolution (§4.1)
# ---------------------------------------------------------------------------

def resolve_lineage_tip(key: str, hermes_home: Optional[Path] = None) -> str:
    """Forward-resolve a session key/id to its compression-lineage tip.

    Uses ``SessionDB.resolve_resume_session_id`` (hermes_state.py:8566+),
    exactly as the gateway poller does (server.py:8941, 9015). Fail-open:
    any error returns the input unchanged, matching the poller's except path.
    """
    if not key:
        return key
    try:
        from hermes_state import SessionDB

        db_path = state_db_path(hermes_home)
        if not db_path.exists():
            return key
        db = SessionDB(db_path, read_only=True)
        try:
            return db.resolve_resume_session_id(key) or key
        finally:
            try:
                db.close()
            except Exception:
                pass
    except Exception:
        return key


def current_session_key(explicit: Optional[str] = None) -> str:
    """The caller's session key: explicit arg, else bridged env (§4.1)."""
    if explicit:
        return explicit
    return os.environ.get("HERMES_SESSION_KEY", "").strip()


# ---------------------------------------------------------------------------
# broker operations
# ---------------------------------------------------------------------------

class AwayBridgeBroker:
    """All broker state transitions; one instance per process is fine."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        *,
        busy_ms: int = DEFAULT_CLI_BUSY_MS,
        hermes_home: Optional[Path] = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path else bridge_db_path(hermes_home)
        self.hermes_home = Path(hermes_home) if hermes_home else None
        self.busy_ms = busy_ms

    # -- arm (§4.1, dedupe rule normative) -----------------------------------

    def arm(self, session_key: str, surface: str = "chat") -> Dict[str, Any]:
        now = time.time()
        tip = resolve_lineage_tip(session_key, self.hermes_home)
        with _connect(self.db_path, self.busy_ms) as conn:
            conn.execute("BEGIN IMMEDIATE")
            refreshed = None
            for row in conn.execute(
                "SELECT session_key, status FROM armed_sessions WHERE status='armed'"
            ).fetchall():
                if resolve_lineage_tip(row["session_key"], self.hermes_home) == tip:
                    refreshed = row["session_key"]
                    break
            if refreshed is not None:
                conn.execute(
                    "UPDATE armed_sessions SET armed_at=?, status='armed', disarmed_at=NULL "
                    "WHERE session_key=?",
                    (now, refreshed),
                )
                conn.commit()
                return {"session_key": refreshed, "action": "refreshed"}
            # A disarmed row for this exact key still occupies the UNIQUE
            # slot (§4.6 keeps rows for audit): re-arm flips it back instead
            # of inserting a duplicate.
            conn.execute(
                "INSERT INTO armed_sessions(session_key, armed_at, surface, status, disarmed_at) "
                "VALUES (?,?,?,'armed',NULL) "
                "ON CONFLICT(session_key) DO UPDATE SET armed_at=excluded.armed_at, "
                "status='armed', disarmed_at=NULL, surface=excluded.surface",
                (session_key, now, surface),
            )
            conn.commit()
            return {"session_key": session_key, "action": "inserted"}

    # -- prepare (§4.1 two-sided lookup; §4.5 single-awaiting invariant) -----

    def prepare(self, session_key: Optional[str], question: str) -> Dict[str, Any]:
        caller_key = current_session_key(session_key)
        if not caller_key:
            raise NotArmedError("no session key (pass --session or set HERMES_SESSION_KEY)")
        caller_tip = resolve_lineage_tip(caller_key, self.hermes_home)
        now = time.time()
        with _connect(self.db_path, self.busy_ms) as conn:
            conn.execute("BEGIN IMMEDIATE")
            armed_row = None
            for row in conn.execute(
                "SELECT session_key, armed_at FROM armed_sessions WHERE status='armed' "
                "ORDER BY armed_at DESC"
            ).fetchall():
                if resolve_lineage_tip(row["session_key"], self.hermes_home) == caller_tip:
                    if armed_row is None or row["armed_at"] > armed_row["armed_at"]:
                        armed_row = row
            if armed_row is None:
                conn.commit()
                raise NotArmedError(
                    "session is not armed (no armed row resolves to this lineage)"
                )
            inherit_key = armed_row["session_key"]
            open_row = conn.execute(
                "SELECT req_key, state FROM requests WHERE session_key=? AND "
                "state IN ('prepared','awaiting') LIMIT 1",
                (inherit_key,),
            ).fetchone()
            if open_row is not None:
                conn.commit()
                raise AlreadyAwaitingError(
                    "session already has an open request %s (batch or defer, §4.5)"
                    % open_row["req_key"]
                )
            req_key = FULL_ID_PREFIX + secrets.token_hex(16)
            while True:
                token = _token()
                try:
                    conn.execute(
                        "INSERT INTO requests(req_key, session_key, display_token, question, "
                        "state, created_at) VALUES (?,?,?,?,'prepared',?)",
                        (req_key, inherit_key, token, question, now),
                    )
                    break
                except sqlite3.IntegrityError:
                    token = _token()
            conn.commit()
            return {"req_key": req_key, "token": token, "session_key": inherit_key}

    # -- publish (§4.3 step 4) -----------------------------------------------

    def publish(
        self,
        req_key: str,
        waiter_pid: Optional[int] = None,
        waiter_start_time: Optional[int] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        with _connect(self.db_path, self.busy_ms) as conn:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                "UPDATE requests SET state='awaiting', published_at=?, waiter_pid=?, "
                "waiter_start_time=? WHERE req_key=? AND state='prepared'",
                (now, waiter_pid, waiter_start_time, req_key),
            )
            if cur.rowcount != 1:
                row = conn.execute(
                    "SELECT state FROM requests WHERE req_key=?", (req_key,)
                ).fetchone()
                conn.commit()
                if row is None:
                    raise UnknownRequestError(req_key)
                raise InvalidStateError(
                    "request %s is '%s', expected 'prepared'" % (req_key, row["state"])
                )
            row = conn.execute(
                "SELECT display_token, question FROM requests WHERE req_key=?",
                (req_key,),
            ).fetchone()
            conn.commit()
            return {
                "req_key": req_key,
                "token": row["display_token"],
                "question": row["question"],
            }

    def cancel(self, req_key: str) -> None:
        now = time.time()
        with _connect(self.db_path, self.busy_ms) as conn:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                "UPDATE requests SET state='cancelled', cancelled_at=? "
                "WHERE req_key=? AND state IN ('prepared','awaiting')",
                (now, req_key),
            )
            conn.commit()
            if cur.rowcount != 1:
                row = conn.execute(
                    "SELECT state FROM requests WHERE req_key=?", (req_key,)
                ).fetchone()
                if row is None:
                    raise UnknownRequestError(req_key)
                if row["state"] not in ("cancelled",):
                    raise InvalidStateError(
                        "request %s is '%s'; cannot cancel" % (req_key, row["state"])
                    )

    # -- claim + duplicate ack (§4.2) -----------------------------------------

    def claim(self, req_key: str, channel: str, answer: str) -> Dict[str, Any]:
        if channel not in ("chat", "telegram"):
            raise ValueError("channel must be 'chat' or 'telegram'")
        now = time.time()
        with _connect(self.db_path, self.busy_ms) as conn:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                "UPDATE requests SET state='claimed', winning_channel=?, answer=?, "
                "claimed_at=? WHERE req_key=? AND state='awaiting'",
                (channel, answer, now, req_key),
            )
            if cur.rowcount == 1:
                conn.commit()
                return {"outcome": "won", "channel": channel}
            row = conn.execute(
                "SELECT state, winning_channel FROM requests WHERE req_key=?",
                (req_key,),
            ).fetchone()
            conn.commit()
            if row is None:
                return {"outcome": "reject", "reason": "unknown"}
            if row["state"] == "claimed":
                return {"outcome": "lost", "winner": row["winning_channel"]}
            return {"outcome": "reject", "reason": row["state"]}

    def ack_duplicate(self, req_key: str, late_channel: str) -> bool:
        """Idempotent ack claim: True iff THIS caller should send the ack."""
        if late_channel not in ("chat", "telegram"):
            raise ValueError("late_channel must be 'chat' or 'telegram'")
        now = time.time()
        with _connect(self.db_path, self.busy_ms) as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO duplicate_acks(req_key, late_channel, sent_at) "
                "VALUES (?,?,?)",
                (req_key, late_channel, now),
            )
            conn.commit()
            return cur.rowcount == 1

    # -- fetch (§4.3, idempotent under delivered) -----------------------------

    def fetch(self, req_key: str) -> Dict[str, Any]:
        with _connect(self.db_path, self.busy_ms) as conn:
            row = conn.execute(
                "SELECT req_key, session_key, display_token, question, state, "
                "winning_channel, answer, delivered FROM requests WHERE req_key=?",
                (req_key,),
            ).fetchone()
            if row is None:
                raise UnknownRequestError(req_key)
            first_delivery = False
            if row["state"] == "claimed" and not row["delivered"]:
                conn.execute(
                    "UPDATE requests SET delivered=1 WHERE req_key=?", (req_key,)
                )
                conn.commit()
                first_delivery = True
            return {
                "req_key": row["req_key"],
                "session_key": row["session_key"],
                "token": row["display_token"],
                "state": row["state"],
                "winning_channel": row["winning_channel"],
                "answer": row["answer"],
                "delivered": bool(row["delivered"]) or first_delivery,
            }

    # -- waiter poll set (§4.3, complete) --------------------------------------

    def poll_request(self, req_key: str) -> Dict[str, Any]:
        """One indexed point query for the waiter. Returns waiter-poll state.

        Raises BrokerError on DB errors (busy beyond timeout, missing/corrupt)
        — the waiter maps any exception to its ERROR exit, never silent spins.
        """
        with _connect(self.db_path, self.busy_ms) as conn:
            row = conn.execute(
                "SELECT r.state AS state, r.winning_channel AS winning_channel, "
                "a.status AS armed_status "
                "FROM requests r JOIN armed_sessions a USING(session_key) "
                "WHERE r.req_key=?",
                (req_key,),
            ).fetchone()
            if row is None:
                return {"outcome": "ERROR", "reason": "no row"}
            state = row["state"]
            armed = row["armed_status"]
            if state == "prepared":
                if armed == "armed":
                    return {"outcome": "KEEP_WAITING"}
                return {"outcome": "DISARMED"}
            if state == "awaiting":
                if armed == "armed":
                    return {"outcome": "KEEP_WAITING"}
                return {"outcome": "DISARMED"}
            if state == "claimed":
                return {
                    "outcome": "TELEGRAM_WON"
                    if row["winning_channel"] == "telegram"
                    else "CHAT_WON"
                }
            return {"outcome": "CANCELLED"}

    # -- disarm (§4.6) ---------------------------------------------------------

    def disarm_chat(self, session_key: Optional[str] = None) -> Dict[str, Any]:
        """Flip EVERY armed row whose tip equals the caller's tip (R4-1)."""
        caller_key = current_session_key(session_key)
        if not caller_key:
            raise NotArmedError("no session key (pass --session or set HERMES_SESSION_KEY)")
        caller_tip = resolve_lineage_tip(caller_key, self.hermes_home)
        now = time.time()
        flipped: List[str] = []
        with _connect(self.db_path, self.busy_ms) as conn:
            conn.execute("BEGIN IMMEDIATE")
            for row in conn.execute(
                "SELECT session_key FROM armed_sessions WHERE status='armed'"
            ).fetchall():
                if resolve_lineage_tip(row["session_key"], self.hermes_home) == caller_tip:
                    conn.execute(
                        "UPDATE armed_sessions SET status='disarmed', disarmed_at=? "
                        "WHERE session_key=?",
                        (now, row["session_key"]),
                    )
                    flipped.append(row["session_key"])
            conn.commit()
        return {"flipped": flipped}

    def disarm_all(self) -> Dict[str, Any]:
        """Telegram global disarm: every armed row in THIS profile's broker."""
        now = time.time()
        with _connect(self.db_path, self.busy_ms) as conn:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                "UPDATE armed_sessions SET status='disarmed', disarmed_at=? "
                "WHERE status='armed'",
                (now,),
            )
            conn.commit()
            return {"disarmed_count": cur.rowcount}

    def has_armed(self) -> bool:
        with _connect(self.db_path, self.busy_ms) as conn:
            row = conn.execute(
                "SELECT 1 FROM armed_sessions WHERE status='armed' LIMIT 1"
            ).fetchone()
            return row is not None

    # -- sweep (§4.4): rate-limited re-ask candidates + orphan-row GC ----------

    def sweep_candidates(self, window_hours: float = DEFAULT_SWEEP_WINDOW_HOURS) -> List[Dict[str, Any]]:
        """Awaiting requests due for a re-ask; updates last_swept_at.

        Rate limit is claimed transactionally per row so a crash-looping
        gateway cannot spam the owner. Callers re-check state='awaiting'
        immediately before each send (§4.4).
        """
        now = time.time()
        due: List[Dict[str, Any]] = []
        with _connect(self.db_path, self.busy_ms) as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT req_key, display_token, question, last_swept_at FROM requests "
                "WHERE state='awaiting' ORDER BY created_at"
            ).fetchall()
            for row in rows:
                last = row["last_swept_at"] or 0.0
                if now - last < window_hours * 3600.0:
                    continue
                conn.execute(
                    "UPDATE requests SET last_swept_at=? WHERE req_key=? AND state='awaiting'",
                    (now, row["req_key"]),
                )
                due.append(
                    {
                        "req_key": row["req_key"],
                        "token": row["display_token"],
                        "question": row["question"],
                    }
                )
            conn.commit()
        return due

    def gc(self, retention_days: float = DEFAULT_RETENTION_DAYS) -> Dict[str, Any]:
        """Orphan-armed-row GC + retention pruning (§4.4, §4.7).

        An armed row is orphaned when its lineage tip is neither live nor
        resumable: the tip id has no row in the profile's state.db (the
        pragmatic V1 liveness test — a live or resumable session always has
        its row in ``sessions``; reaped/absent ones do not).
        """
        now = time.time()
        orphaned: List[str] = []
        pruned = 0
        with _connect(self.db_path, self.busy_ms) as conn:
            conn.execute("BEGIN IMMEDIATE")
            for row in conn.execute(
                "SELECT session_key FROM armed_sessions WHERE status='armed'"
            ).fetchall():
                tip = resolve_lineage_tip(row["session_key"], self.hermes_home)
                if self._tip_exists(tip):
                    continue
                open_req = conn.execute(
                    "SELECT 1 FROM requests WHERE session_key=? AND state IN "
                    "('prepared','awaiting') LIMIT 1",
                    (row["session_key"],),
                ).fetchone()
                if open_req is not None:
                    continue
                conn.execute(
                    "UPDATE armed_sessions SET status='disarmed', disarmed_at=? "
                    "WHERE session_key=? AND status='armed'",
                    (now, row["session_key"]),
                )
                orphaned.append(row["session_key"])
            cutoff = now - retention_days * 86400.0
            cur = conn.execute(
                "DELETE FROM requests WHERE state='claimed' AND delivered=1 AND "
                "claimed_at < ?",
                (cutoff,),
            )
            pruned = cur.rowcount
            conn.execute(
                "DELETE FROM duplicate_acks WHERE req_key NOT IN "
                "(SELECT req_key FROM requests)"
            )
            conn.commit()
        return {"orphaned": orphaned, "pruned_requests": pruned}

    def _tip_exists(self, tip: str) -> bool:
        try:
            from hermes_state import SessionDB

            db_path = state_db_path(self.hermes_home)
            if not db_path.exists():
                return False
            db = SessionDB(db_path, read_only=True)
            try:
                row = db._conn.execute(
                    "SELECT 1 FROM sessions WHERE id=? LIMIT 1", (tip,)
                ).fetchone()
                return row is not None
            finally:
                try:
                    db._conn.close()
                except Exception:
                    pass
        except Exception:
            # Fail open: an unresolvable state.db must not disarm live rows.
            return True

    # -- reconcile (§4.7): REPORTER ONLY ----------------------------------------

    def reconcile(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "undelivered_winners": [],
            "awaiting_waiter_dead": [],
            "prepared_stuck": [],
            "disarmed_with_awaiting": [],
        }
        with _connect(self.db_path, self.busy_ms) as conn:
            undelivered = conn.execute(
                "SELECT req_key, display_token, winning_channel, answer, claimed_at "
                "FROM requests WHERE state='claimed' AND delivered=0"
            ).fetchall()
            for row in undelivered:
                report["undelivered_winners"].append(
                    {
                        "req_key": row["req_key"],
                        "token": row["display_token"],
                        "winning_channel": row["winning_channel"],
                        "answer": row["answer"],
                    }
                )
            awaiting = conn.execute(
                "SELECT req_key, display_token, question, waiter_pid, waiter_start_time "
                "FROM requests WHERE state='awaiting'"
            ).fetchall()
            for row in awaiting:
                if not self._waiter_live(row["waiter_pid"], row["waiter_start_time"]):
                    report["awaiting_waiter_dead"].append(
                        {
                            "req_key": row["req_key"],
                            "token": row["display_token"],
                            "question": row["question"],
                            "waiter_pid": row["waiter_pid"],
                        }
                    )
            prepared = conn.execute(
                "SELECT req_key, display_token, question, created_at FROM requests "
                "WHERE state='prepared'"
            ).fetchall()
            for row in prepared:
                report["prepared_stuck"].append(
                    {
                        "req_key": row["req_key"],
                        "token": row["display_token"],
                        "question": row["question"],
                        "created_at": row["created_at"],
                    }
                )
            disarmed = conn.execute(
                "SELECT r.req_key, r.display_token FROM requests r "
                "JOIN armed_sessions a USING(session_key) "
                "WHERE r.state='awaiting' AND a.status='disarmed'"
            ).fetchall()
            for row in disarmed:
                report["disarmed_with_awaiting"].append(
                    {"req_key": row["req_key"], "token": row["display_token"]}
                )
        return report

    @staticmethod
    def _waiter_live(pid: Optional[int], start_time: Optional[int]) -> bool:
        """Identity-validated liveness (mirrors _host_pid_is_ours, §4.7)."""
        if not pid:
            return False
        try:
            psutil = __import__("psutil")
            proc = psutil.Process(int(pid))
            if not proc.is_running():
                return False
        except Exception:
            return False
        if start_time is None:
            return True  # legacy row without a captured start time
        try:
            from gateway.status import get_process_start_time

            return get_process_start_time(int(pid)) == int(start_time)
        except Exception:
            return False

    # -- lookup helpers ----------------------------------------------------------

    # -- read-only snapshot ------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Full status view for ``away-bridge status``.

        Ensures the DB/schema exist first (fresh installs have neither), so a
        status probe on an untouched broker returns empty lists instead of
        crashing — status must never be the command that creates state beyond
        the (idempotent) schema.
        """
        with _connect(self.db_path, self.busy_ms) as conn:
            armed = [dict(r) for r in conn.execute(
                "SELECT * FROM armed_sessions ORDER BY armed_at DESC"
            ).fetchall()]
            requests = [dict(r) for r in conn.execute(
                "SELECT * FROM requests ORDER BY created_at DESC"
            ).fetchall()]
        return {"armed_sessions": armed, "requests": requests}

    def get_request_by_token(self, token: str) -> Optional[Dict[str, Any]]:
        with _connect(self.db_path, self.busy_ms) as conn:
            row = conn.execute(
                "SELECT req_key, session_key, display_token, question, state, "
                "winning_channel FROM requests WHERE display_token=?",
                (token,),
            ).fetchone()
            if row is None:
                return None
            return {
                "req_key": row["req_key"],
                "session_key": row["session_key"],
                "token": row["display_token"],
                "question": row["question"],
                "state": row["state"],
                "winning_channel": row["winning_channel"],
            }

    def get_request(self, req_key: str) -> Optional[Dict[str, Any]]:
        with _connect(self.db_path, self.busy_ms) as conn:
            row = conn.execute(
                "SELECT req_key, session_key, display_token, question, state, "
                "winning_channel, answer FROM requests WHERE req_key=?",
                (req_key,),
            ).fetchone()
            if row is None:
                return None
            return {
                "req_key": row["req_key"],
                "session_key": row["session_key"],
                "token": row["display_token"],
                "question": row["question"],
                "state": row["state"],
                "winning_channel": row["winning_channel"],
                "answer": row["answer"],
            }

    def waiter_start_time_for(self, pid: int) -> Optional[int]:
        """Kernel start time of a waiter PID (recorded at publish, §4.3)."""
        try:
            from gateway.status import get_process_start_time

            return get_process_start_time(int(pid))
        except Exception:
            return None
