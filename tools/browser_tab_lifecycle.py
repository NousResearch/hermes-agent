"""Per-turn ownership records for a resolved local browser endpoint.

The browser architecture owns backend selection, CDP discovery, and safety
classification in :mod:`tools.browser_tool`.  This module receives an already
validated loopback HTTP control endpoint and only manages tab ownership,
leases, and verified cleanup.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:  # POSIX cross-process serialization; Windows keeps the in-process lock.
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None  # type: ignore[assignment]


_BLANK_URLS = {
    "",
    "about:blank",
    "chrome://newtab/",
    "chrome://new-tab-page/",
}
_PROCESS_LOCK = threading.Lock()


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso(value: Optional[dt.datetime] = None) -> str:
    return (value or _utc_now()).isoformat()


def _redact_url(url: str) -> str:
    """Keep origin/path for debugging; remove credentials, query, fragment."""
    try:
        parsed = urllib.parse.urlsplit(str(url or ""))
    except ValueError:
        return "[invalid-url]"
    if not parsed.scheme:
        return str(url or "").split("?", 1)[0].split("#", 1)[0]
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urllib.parse.urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def _live_browser_key(endpoint: str, *, timeout_s: float = 5.0) -> str:
    """Resolve the identity of the browser currently bound to ``endpoint``."""
    with urllib.request.urlopen(
        str(endpoint).rstrip("/") + "/json/version", timeout=timeout_s
    ) as response:
        raw = response.read()
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("CDP /json/version returned invalid JSON") from exc
    websocket_url = payload.get("webSocketDebuggerUrl") if isinstance(payload, dict) else None
    if not isinstance(websocket_url, str) or not websocket_url.strip():
        raise RuntimeError("CDP /json/version did not return webSocketDebuggerUrl")
    return hashlib.sha256(websocket_url.strip().encode("utf-8")).hexdigest()


def _default_state_dir() -> Path:
    try:
        from hermes_constants import get_hermes_home

        base = Path(get_hermes_home())
    except Exception:
        base = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
    return base / "state" / "resource-hygiene"


def _private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        path.chmod(0o700)
    return path


class _ExclusiveFileLock:
    """Thread + file lock with a bounded wait."""

    def __init__(self, path: Path, timeout_s: float = 60.0):
        self.path = path
        self.timeout_s = max(0.1, float(timeout_s))
        self._handle = None
        self._thread_acquired = False

    def acquire(self) -> None:
        deadline = time.monotonic() + self.timeout_s
        while not _PROCESS_LOCK.acquire(blocking=False):
            if time.monotonic() >= deadline:
                raise TimeoutError("another managed browser_exec call is still running")
            time.sleep(0.05)
        self._thread_acquired = True
        try:
            _private_dir(self.path.parent)
            self._handle = open(self.path, "a+", encoding="utf-8")
            with contextlib.suppress(OSError):
                os.chmod(self.path, 0o600)
            if fcntl is None:
                return
            while True:
                try:
                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("another managed browser_exec process is still running")
                    time.sleep(0.05)
        except Exception:
            self.release()
            raise

    def release(self) -> None:
        if self._handle is not None:
            if fcntl is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                self._handle.close()
            self._handle = None
        if self._thread_acquired:
            self._thread_acquired = False
            _PROCESS_LOCK.release()


class BrowserTabLifecycleGuard:
    """Own targets created by one local browser_exec call.

    ``start`` acquires a process-wide/cross-process lock, reuses an unexpired
    lease for this browser+turn when present, otherwise creates one dedicated
    blank target, and snapshots the browser. ``finish`` leases or closes only
    this scope's target plus targets created while the scope was active.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        endpoint: str,
        browser_key: str,
        owner_key: str,
        lease_minutes: int = 0,
        lease_reason: str = "",
        state_dir: Optional[Path] = None,
        lock_timeout_s: float = 60.0,
        request_timeout_s: float = 5.0,
    ):
        self.enabled = bool(enabled)
        self.endpoint = str(endpoint or '').rstrip('/') or None
        self.browser_key = str(browser_key or '')
        self.owner_key = str(owner_key or "browser-exec-default")[:200]
        self.lease_minutes = max(0, int(lease_minutes or 0))
        self.lease_reason = str(lease_reason or "").strip()[:500]
        self.state_dir = _private_dir(state_dir or _default_state_dir())
        self.request_timeout_s = max(0.2, float(request_timeout_s))
        self.run_id = uuid.uuid4().hex
        self._lock = _ExclusiveFileLock(self.state_dir / "browser.lock", lock_timeout_s)
        self._before: Dict[str, Dict[str, Any]] = {}
        self.target_id: Optional[str] = None
        self._started = False

    @property
    def managed(self) -> bool:
        return self.enabled and self.endpoint is not None

    def _request_json(self, path: str, *, method: str = "GET") -> Any:
        if self.endpoint is None:
            raise RuntimeError("managed browser endpoint is unavailable")
        request = urllib.request.Request(self.endpoint + path, method=method)
        with urllib.request.urlopen(request, timeout=self.request_timeout_s) as response:
            raw = response.read()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw.decode("utf-8", errors="replace")

    def _snapshot(self) -> Dict[str, Dict[str, Any]]:
        raw = self._request_json("/json/list")
        if not isinstance(raw, list):
            raise RuntimeError("CDP /json/list returned a non-list response")
        result: Dict[str, Dict[str, Any]] = {}
        for target in raw:
            if not isinstance(target, dict) or target.get("type") != "page":
                continue
            target_id = str(target.get("id") or target.get("targetId") or "")
            if target_id:
                result[target_id] = target
        return result

    def _require_live_browser_identity(self) -> None:
        live_key = _live_browser_key(
            self.endpoint or "", timeout_s=self.request_timeout_s
        )
        if live_key != self.browser_key:
            raise RuntimeError(
                "browser identity changed at the managed CDP endpoint; cleanup skipped"
            )

    def _connect(self) -> sqlite3.Connection:
        db_path = self.state_dir / "resources.sqlite3"
        connection = sqlite3.connect(str(db_path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            """
        )
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version < 2:
            # v1 keyed only by target_id. Those rows cannot safely survive an
            # endpoint/browser change, so discard the ephemeral ownership
            # ledger rather than risk closing a target in another browser.
            connection.executescript(
                """
                DROP TABLE IF EXISTS browser_resources;
                DROP TABLE IF EXISTS browser_runs;
                PRAGMA user_version=2;
                """
            )
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS browser_resources (
                browser_key TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                target_id TEXT NOT NULL,
                owner_key TEXT NOT NULL,
                run_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                heartbeat_at TEXT NOT NULL,
                state TEXT NOT NULL,
                lease_expires_at TEXT,
                lease_reason TEXT,
                url_redacted TEXT,
                title TEXT,
                closed_at TEXT,
                close_verified INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                PRIMARY KEY (browser_key, target_id)
            );
            CREATE INDEX IF NOT EXISTS idx_browser_resources_owner_state
                ON browser_resources(browser_key, owner_key, state);
            CREATE TABLE IF NOT EXISTS browser_runs (
                run_id TEXT PRIMARY KEY,
                browser_key TEXT NOT NULL,
                owner_key TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                requested_lease_minutes INTEGER NOT NULL,
                created_count INTEGER NOT NULL DEFAULT 0,
                repurposed_count INTEGER NOT NULL DEFAULT 0,
                closed_count INTEGER NOT NULL DEFAULT 0,
                leased_count INTEGER NOT NULL DEFAULT 0,
                remaining_pages INTEGER,
                ok INTEGER,
                error TEXT
            );
            """
        )
        with contextlib.suppress(OSError):
            os.chmod(db_path, 0o600)
        return connection

    def _insert_run_start(self) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO browser_runs "
                "(run_id, browser_key, owner_key, started_at, requested_lease_minutes) "
                "VALUES (?, ?, ?, ?, ?)",
                (self.run_id, self.browser_key, self.owner_key, _iso(), self.lease_minutes),
            )

    def start(self) -> Optional[str]:
        if not self.enabled:
            return None
        if self.endpoint is None or not self.browser_key:
            return "tab lifecycle requires a resolved local browser identity"
        try:
            self._lock.acquire()
            reap_expired_browser_tab_leases(
                state_dir=self.state_dir, acquire_lock=False
            )
            self._require_live_browser_identity()
            self._before = self._snapshot()
            with self._connect() as db:
                reusable = self._owned_active_ids(db) & set(self._before)
            if reusable:
                self.target_id = sorted(reusable)[0]
            else:
                created = self._request_json("/json/new?about%3Ablank", method="PUT")
                if not isinstance(created, dict):
                    raise RuntimeError("CDP did not return the dedicated target")
                self.target_id = str(created.get("id") or created.get("targetId") or "")
                if not self.target_id:
                    raise RuntimeError("CDP dedicated target has no id")
            self._insert_run_start()
            self._started = True
            return None
        except Exception as exc:
            self._lock.release()
            return f"tab lifecycle could not prepare the local browser: {exc}"

    def _owned_active_ids(self, db: sqlite3.Connection) -> set[str]:
        rows = db.execute(
            """SELECT target_id FROM browser_resources
               WHERE browser_key = ? AND owner_key = ? AND state = 'leased'
                 AND lease_expires_at > ?""",
            (self.browser_key, self.owner_key, _iso()),
        ).fetchall()
        return {str(row[0]) for row in rows}

    @staticmethod
    def _is_blank(target: Dict[str, Any]) -> bool:
        return str(target.get("url") or "").strip() in _BLANK_URLS

    def _upsert_resources(
        self,
        db: sqlite3.Connection,
        targets: Dict[str, Dict[str, Any]],
        target_ids: Iterable[str],
        *,
        state: str,
        lease_expires_at: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        now = _iso()
        for target_id in target_ids:
            target = targets.get(target_id, {})
            db.execute(
                """
                INSERT INTO browser_resources
                    (browser_key, endpoint, target_id, owner_key, run_id,
                     created_at, heartbeat_at, state, lease_expires_at,
                     lease_reason, url_redacted, title, last_error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(browser_key, target_id) DO UPDATE SET
                    endpoint=excluded.endpoint,
                    owner_key=excluded.owner_key,
                    run_id=excluded.run_id,
                    heartbeat_at=excluded.heartbeat_at,
                    state=excluded.state,
                    lease_expires_at=excluded.lease_expires_at,
                    lease_reason=excluded.lease_reason,
                    url_redacted=excluded.url_redacted,
                    title=excluded.title,
                    last_error=excluded.last_error
                """,
                (
                    self.browser_key,
                    self.endpoint,
                    target_id,
                    self.owner_key,
                    self.run_id,
                    now,
                    now,
                    state,
                    lease_expires_at,
                    self.lease_reason or None,
                    _redact_url(str(target.get("url") or "")),
                    str(target.get("title") or "")[:500],
                    error,
                ),
            )

    def _close_target(self, target_id: str) -> Optional[str]:
        try:
            self._request_json(f"/json/close/{urllib.parse.quote(target_id, safe='')}")
            return None
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            return str(exc)

    def _ensure_blank_baseline(self, snapshot: Dict[str, Dict[str, Any]]) -> Optional[str]:
        if any(self._is_blank(target) for target in snapshot.values()):
            return None
        try:
            self._require_live_browser_identity()
            self._request_json("/json/new?about%3Ablank", method="PUT")
            return None
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            return f"could not create blank baseline page: {exc}"

    def _finish_run(
        self,
        db: sqlite3.Connection,
        *,
        created: int,
        repurposed: int,
        closed: int,
        leased: int,
        remaining: Optional[int],
        errors: list[str],
    ) -> None:
        db.execute(
            """
            UPDATE browser_runs SET finished_at=?, created_count=?, repurposed_count=?,
                closed_count=?, leased_count=?, remaining_pages=?, ok=?, error=?
            WHERE run_id=?
            """,
            (
                _iso(),
                created,
                repurposed,
                closed,
                leased,
                remaining,
                0 if errors else 1,
                "; ".join(errors)[:2000] if errors else None,
                self.run_id,
            ),
        )

    def finish(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"managed": False, "reason": "disabled"}
        if not self._started:
            return {"managed": False, "reason": "not-started"}

        errors: list[str] = []
        created_ids: set[str] = set()
        repurposed_ids: set[str] = set()
        leased_count = 0
        closed_count = 0
        remaining: Optional[int] = None
        try:
            self._require_live_browser_identity()
            after = self._snapshot()
            created_ids = set(after) - set(self._before)
            scope_ids = {self.target_id} if self.target_id and self.target_id in after else set()
            repurposed_ids = set()
            with self._connect() as db:
                previously_owned = self._owned_active_ids(db) & set(after)
                candidates = created_ids | previously_owned | scope_ids
                if self.lease_minutes:
                    expiry = _iso(_utc_now() + dt.timedelta(minutes=self.lease_minutes))
                    self._upsert_resources(
                        db, after, candidates, state="leased", lease_expires_at=expiry
                    )
                    leased_count = len(candidates)
                    remaining = len(after)
                else:
                    self._upsert_resources(db, after, candidates, state="closing")
                    close_errors: Dict[str, str] = {}
                    for target_id in sorted(candidates):
                        err = self._close_target(target_id)
                        if err:
                            close_errors[target_id] = err
                    deadline = time.monotonic() + 5.0
                    current = after
                    while time.monotonic() < deadline:
                        current = self._snapshot()
                        if not (candidates & set(current)):
                            break
                        time.sleep(0.1)
                    survivors = candidates & set(current)
                    for target_id in candidates - survivors:
                        db.execute(
                            """UPDATE browser_resources SET state='closed', closed_at=?,
                               close_verified=1, last_error=NULL
                               WHERE browser_key=? AND target_id=?""",
                            (_iso(), self.browser_key, target_id),
                        )
                    for target_id in survivors:
                        error = close_errors.get(target_id) or "target still present after close"
                        db.execute(
                            """UPDATE browser_resources SET state='close_failed',
                               close_verified=0, last_error=?
                               WHERE browser_key=? AND target_id=?""",
                            (error[:1000], self.browser_key, target_id),
                        )
                        errors.append(f"target {target_id[:12]} was not closed")
                    closed_count = len(candidates - survivors)
                    baseline_error = self._ensure_blank_baseline(current)
                    if baseline_error:
                        errors.append(baseline_error)
                    final = self._snapshot()
                    remaining = len(final)
                self._finish_run(
                    db,
                    created=len(created_ids),
                    repurposed=len(repurposed_ids),
                    closed=closed_count,
                    leased=leased_count,
                    remaining=remaining,
                    errors=errors,
                )
        except Exception as exc:
            errors.append(str(exc))
            with contextlib.suppress(Exception):
                with self._connect() as db:
                    self._finish_run(
                        db,
                        created=len(created_ids),
                        repurposed=len(repurposed_ids),
                        closed=closed_count,
                        leased=leased_count,
                        remaining=remaining,
                        errors=errors,
                    )
        finally:
            self._lock.release()
            self._started = False

        return {
            "managed": True,
            "ok": not errors,
            "created": len(created_ids),
            "repurposed": len(repurposed_ids),
            "closed": closed_count,
            "leased": leased_count,
            "remaining_pages": remaining,
            "errors": errors,
        }


def _close_ledger_rows(rows: Iterable[sqlite3.Row], db: sqlite3.Connection) -> Dict[str, Any]:
    closed = 0
    failed = 0
    errors: list[str] = []
    grouped: Dict[tuple[str, str], list[sqlite3.Row]] = {}
    for row in rows:
        key = (str(row["browser_key"]), str(row["endpoint"]).rstrip("/"))
        grouped.setdefault(key, []).append(row)

    for (browser_key, endpoint), browser_rows in grouped.items():
        try:
            live_key = _live_browser_key(endpoint)
            if live_key != browser_key:
                raise RuntimeError("browser identity changed; cleanup skipped")
        except Exception as exc:
            for row in browser_rows:
                target_id = str(row["target_id"])
                db.execute(
                    """UPDATE browser_resources SET state='close_failed',
                       close_verified=0, last_error=?
                       WHERE browser_key=? AND target_id=?""",
                    (str(exc)[:1000], browser_key, target_id),
                )
                failed += 1
                errors.append(f"target {target_id[:12]}: {exc}")
            continue

        browser_closed = 0
        for row in browser_rows:
            target_id = str(row["target_id"])
            try:
                request = urllib.request.Request(
                    endpoint + f"/json/close/{urllib.parse.quote(target_id, safe='')}",
                    method="GET",
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    response.read()
                with urllib.request.urlopen(endpoint + "/json/list", timeout=5) as response:
                    raw_targets = response.read()
                if not raw_targets:
                    raise RuntimeError("CDP /json/list returned an empty response")
                targets = json.loads(raw_targets)
                if not isinstance(targets, list):
                    raise RuntimeError("CDP /json/list returned a non-list response")
                if any(
                    str(target.get("id") or target.get("targetId") or "") == target_id
                    for target in targets
                    if isinstance(target, dict)
                ):
                    raise RuntimeError("target still present after close")
                db.execute(
                    """UPDATE browser_resources SET state='closed', closed_at=?,
                       close_verified=1, last_error=NULL
                       WHERE browser_key=? AND target_id=?""",
                    (_iso(), browser_key, target_id),
                )
                closed += 1
                browser_closed += 1
            except Exception as exc:
                db.execute(
                    """UPDATE browser_resources SET state='close_failed',
                       close_verified=0, last_error=?
                       WHERE browser_key=? AND target_id=?""",
                    (str(exc)[:1000], browser_key, target_id),
                )
                failed += 1
                errors.append(f"target {target_id[:12]}: {exc}")

        # A baseline is a write to the live browser. Only perform it when this
        # cleanup closed something and the endpoint still identifies the same
        # browser that owns the ledger rows.
        if browser_closed == 0:
            continue
        try:
            if _live_browser_key(endpoint) != browser_key:
                raise RuntimeError("browser identity changed before baseline write")
            with urllib.request.urlopen(endpoint + "/json/list", timeout=5) as response:
                raw_targets = response.read()
            if not raw_targets:
                raise RuntimeError("CDP /json/list returned an empty response")
            targets = json.loads(raw_targets)
            if not isinstance(targets, list):
                raise RuntimeError("CDP /json/list returned a non-list response")
            has_blank = any(
                isinstance(target, dict)
                and str(target.get("url") or "").strip() in _BLANK_URLS
                for target in targets
            )
            if not has_blank:
                request = urllib.request.Request(
                    endpoint + "/json/new?about%3Ablank", method="PUT"
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    response.read()
        except Exception as exc:
            failed += 1
            errors.append(f"blank baseline: {exc}")
    return {"closed": closed, "failed": failed, "errors": errors}


def _ledger_connection(state_dir: Optional[Path] = None) -> Optional[sqlite3.Connection]:
    db_path = (state_dir or _default_state_dir()) / "resources.sqlite3"
    if not db_path.exists():
        return None
    db = sqlite3.connect(str(db_path), timeout=10)
    db.row_factory = sqlite3.Row
    columns = {str(row[1]) for row in db.execute("PRAGMA table_info(browser_resources)")}
    if not {"browser_key", "endpoint", "target_id"}.issubset(columns):
        db.close()
        return None
    return db


def reap_expired_browser_tab_leases(
    *, state_dir: Optional[Path] = None, acquire_lock: bool = True
) -> Dict[str, Any]:
    """Close leases whose bounded expiry has passed."""
    lock = _ExclusiveFileLock((state_dir or _default_state_dir()) / "browser.lock", 5)
    if acquire_lock:
        lock.acquire()
    try:
        db = _ledger_connection(state_dir)
        if db is None:
            return {"closed": 0, "failed": 0, "errors": []}
        try:
            rows = db.execute(
                """SELECT browser_key, endpoint, target_id FROM browser_resources
                   WHERE state='leased' AND lease_expires_at <= ?""",
                (_iso(),),
            ).fetchall()
            report = _close_ledger_rows(rows, db)
            db.commit()
            return report
        finally:
            db.close()
    finally:
        if acquire_lock:
            lock.release()


def finalize_browser_tab_owner(
    owner_key: str, *, state_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Close every still-leased target owned by one completed turn."""
    key = str(owner_key or "").strip()
    if not key:
        return {"closed": 0, "failed": 0, "errors": []}
    lock = _ExclusiveFileLock((state_dir or _default_state_dir()) / "browser.lock", 10)
    lock.acquire()
    try:
        db = _ledger_connection(state_dir)
        if db is None:
            return {"closed": 0, "failed": 0, "errors": []}
        try:
            rows = db.execute(
                """SELECT browser_key, endpoint, target_id FROM browser_resources
                   WHERE owner_key=? AND state='leased'""",
                (key,),
            ).fetchall()
            report = _close_ledger_rows(rows, db)
            db.commit()
            return report
        finally:
            db.close()
    finally:
        lock.release()
