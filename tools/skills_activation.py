"""
Layer-2 skills activation API (spec 309, plan 02-01, upstream-PR-quality).

Implements the NET-NEW managed-skills-activation contract on top of the
existing Hermes Agent aiohttp API server:

- ``POST /v1/skills/activations``   — activate/refresh a managed skill
- ``DELETE /v1/skills/activations/{skillId}?mode=clean|rollback`` — cleanup
- ``GET /v1/skills?scope=managed``  — managed-only observed inventory

Design notes (see 02-01-PLAN.md <locked_decisions> for the adjudicated
rationale):

- Scope is bound to the DEPLOYMENT (LD-1) — never derived from request
  body fields. A body-carried scope assertion is validated against the
  configured deployment scope; mismatch is a 403, never a silent override.
- The managed-ownership ledger is a NEW, separate WAL-mode SQLite file
  (``skills_activation.db``) under the injected home, decoupled from
  ``hermes_state.py``'s session-schema evolution (LD-3). The WAL-with-
  NFS-fallback helper below is a copied/adapted shape from
  ``hermes_state.py:143-203`` (COPIED, not imported, by design).
- Idempotency has two tiers: a DURABLE tier (the ledger's
  ``idempotency_key``/``fingerprint`` columns, checked with a plain read
  before touching the in-memory cache) and an IN-FLIGHT tier (the
  injected ``_IdempotencyCache`` instance's ``get_or_set``, which
  collapses genuinely concurrent identical requests into one execution
  via its shielded-task pattern). Because aiohttp/asyncio here is single-
  threaded and the write-critical-section below contains no ``await``,
  the section is *already* atomic with respect to the event loop; the
  ledger's own SQLite transaction (``BEGIN IMMEDIATE``) is the durable,
  audit-honest backstop, not a workaround for a race that couldn't
  otherwise occur in this process.
- Replay lifetime is scoped to "the current managed row": DELETE (clean
  or any rollback that removes the row) bumps a per-skill in-memory
  epoch counter, which is folded into the in-flight cache's key
  namespace. This makes a post-delete cache hit unreachable without
  reaching into ``_IdempotencyCache``'s private store (no modification
  to that class needed).
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - aiohttp is an exact-pinned fork dep
    web = None  # type: ignore[assignment]

SCHEMA_VERSION = 1
MANAGED_BY = "paperclip-skill-contract/v1"
MAX_CONTENT_BYTES = 1024 * 1024  # 1 MiB
MAX_MANAGED_ENTRIES = 500
MAX_BODY_DEPTH = 8
CONTRACT_VERSION = "1"

_SKILL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9._-]{1,32}$")
_CONTENT_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDEMPOTENCY_KEY_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

_PATH_KEY_DENYLIST = {
    "path", "filePath", "file_path", "sourcePath", "source_path",
    "localPath", "local_path", "dir", "directory",
}

# ---------------------------------------------------------------------------
# WAL-mode-with-NFS-fallback (copied/adapted shape from hermes_state.py:143-203,
# LD-3 — decoupled deliberately, not imported).
# ---------------------------------------------------------------------------

_WAL_INCOMPAT_MARKERS = ("locking protocol", "not authorized")


def _apply_wal_with_fallback(conn: sqlite3.Connection, *, db_label: str = "skills_activation.db") -> str:
    """Set journal_mode=WAL, falling back to DELETE on WAL-incompatible filesystems."""
    try:
        current = conn.execute("PRAGMA journal_mode").fetchone()
        if current and current[0] == "wal":
            return "wal"
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        return "wal"
    except sqlite3.OperationalError as exc:
        msg = str(exc).lower()
        if not any(marker in msg for marker in _WAL_INCOMPAT_MARKERS):
            raise
        conn.execute("PRAGMA journal_mode=DELETE")
        return "delete"


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    schema_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS managed_skills (
    skill_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    version TEXT,
    scope_json TEXT NOT NULL,
    managed_by TEXT NOT NULL,
    activated_at TEXT NOT NULL,
    previous_state_json TEXT,
    idempotency_key TEXT NOT NULL,
    fingerprint TEXT NOT NULL
);
"""


def _safe_rollback(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ROLLBACK")
    except sqlite3.OperationalError:
        pass  # no transaction was active — already committed/rolled back


def _open_ledger(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None)  # autocommit; we BEGIN explicitly
    conn.row_factory = sqlite3.Row
    _apply_wal_with_fallback(conn, db_label=db_path.name)
    conn.executescript(_SCHEMA_SQL)
    conn.execute(
        "INSERT OR IGNORE INTO schema_meta (id, schema_version) VALUES (1, ?)",
        (SCHEMA_VERSION,),
    )
    return conn


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class _ActivationError(Exception):
    """Carries an HTTP status + house-shaped error envelope."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _error_response(status: int, code: str, message: str) -> "web.Response":
    return web.json_response(
        {"error": {"message": message, "type": "invalid_request_error", "code": code}},
        status=status,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _contains_path_like(value: str) -> bool:
    return "/" in value or "\\" in value or ".." in value


def _scan_body_for_path_payloads(node: Any, *, depth: int, skip_key: Optional[str] = None) -> None:
    """Raise _ActivationError(422) if any denylisted key or path-like string value is found.

    ``skip_key`` names the ONE key (skill.content) whose string value is
    exempt from the path-like-value scan (it legitimately contains
    arbitrary skill markdown, which may include '/' or '..' in prose).
    """
    if depth > MAX_BODY_DEPTH:
        raise _ActivationError(422, "invalid_request", "request body nesting too deep")
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _PATH_KEY_DENYLIST:
                raise _ActivationError(422, "invalid_request", f"disallowed path-shaped key: {key}")
            child_skip = "content" if key == skip_key else None
            _scan_body_for_path_payloads(value, depth=depth + 1, skip_key=child_skip)
    elif isinstance(node, list):
        for item in node:
            _scan_body_for_path_payloads(item, depth=depth + 1, skip_key=skip_key)
    elif isinstance(node, str):
        if skip_key == "content":
            return
        if _contains_path_like(node):
            raise _ActivationError(422, "invalid_request", "path-shaped value rejected")


def _canonical_fingerprint(body: Dict[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_activation_body(body: Any) -> Dict[str, Any]:
    if not isinstance(body, dict):
        raise _ActivationError(422, "invalid_request", "request body must be a JSON object")

    # Scan for path-shaped denylisted keys/values BEFORE anything else,
    # so a filesystem-path payload is rejected regardless of what other
    # fields are also missing/malformed.
    _scan_body_for_path_payloads(body, depth=0, skip_key="skill")

    contract_version = body.get("contractVersion")
    if contract_version != CONTRACT_VERSION:
        raise _ActivationError(422, "unsupported_contract_version", "unknown or missing contractVersion")

    idempotency_key = body.get("idempotencyKey")
    if not isinstance(idempotency_key, str) or not _IDEMPOTENCY_KEY_RE.match(idempotency_key):
        raise _ActivationError(422, "invalid_request", "idempotencyKey missing or malformed")

    skill = body.get("skill")
    if not isinstance(skill, dict):
        raise _ActivationError(422, "invalid_request", "skill object missing")

    skill_id = skill.get("skillId")
    if not isinstance(skill_id, str) or not _SKILL_ID_RE.match(skill_id):
        raise _ActivationError(422, "invalid_request", "skillId missing or malformed")

    version = skill.get("version")
    if version is not None and (not isinstance(version, str) or not _VERSION_RE.match(version)):
        raise _ActivationError(422, "invalid_request", "version malformed")

    content_hash = skill.get("contentHash")
    if not isinstance(content_hash, str) or not _CONTENT_HASH_RE.match(content_hash):
        raise _ActivationError(422, "invalid_request", "contentHash missing or malformed")

    content = skill.get("content")
    if not isinstance(content, str) or len(content) == 0:
        raise _ActivationError(422, "invalid_request", "content missing or empty")
    if len(content.encode("utf-8")) > MAX_CONTENT_BYTES:
        raise _ActivationError(422, "invalid_request", "content exceeds maximum size")

    # Optional body-asserted scope fields (LD-1): validated, never derived.
    scope_assert: Dict[str, str] = {}
    for field in ("agentId", "tenantId", "companyId"):
        val = body.get(field)
        if val is not None:
            if not isinstance(val, str):
                raise _ActivationError(422, "invalid_request", f"{field} must be a string")
            scope_assert[field] = val

    return {
        "idempotency_key": idempotency_key,
        "skill_id": skill_id,
        "version": version,
        "content_hash": content_hash,
        "content": content,
        "scope_assert": scope_assert,
    }


_SCOPE_ASSERT_TO_LEDGER_FIELD = {
    "agentId": "agentId",
    "tenantId": "tenantId",
    "companyId": "companyId",
}


class SkillsActivationService:
    """Constructor-injected Layer-2 activation service.

    Parameters mirror the plan's <artifacts> symbol contract exactly:
    ``home`` (Path — deployment home, defaulting production callers to
    ``get_hermes_home()``), ``check_auth`` (the exact ``_check_auth``
    contract: returns ``None`` on OK, a 401 ``web.Response`` otherwise),
    ``scope`` (the deployment's configured scope dict — LD-1), and
    ``idem_cache`` (an ``_IdempotencyCache`` INSTANCE, injected to avoid
    a circular import with ``api_server.py``).
    """

    def __init__(
        self,
        home: Path,
        check_auth: Callable[[Any], Optional[Any]],
        scope: Dict[str, str],
        idem_cache: Any,
    ) -> None:
        self._home = Path(home)
        self._check_auth = check_auth
        self._scope = dict(scope)
        self._idem_cache = idem_cache
        self._skills_dir = self._home / "skills"
        self._db_path = self._home / "skills_activation.db"
        self._epoch: Dict[str, int] = {}

    # -- shared plumbing ---------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        return _open_ledger(self._db_path)

    def _cache_key(self, skill_id: str, idempotency_key: str) -> str:
        epoch = self._epoch.get(skill_id, 0)
        return f"{skill_id}#{epoch}#{idempotency_key}"

    def _skill_md_path(self, skill_id: str) -> Path:
        target = (self._skills_dir / skill_id / "SKILL.md").resolve()
        containment_root = self._skills_dir.resolve()
        if containment_root not in target.parents and target != containment_root:
            raise _ActivationError(422, "invalid_request", "skillId resolves outside the managed skills root")
        return target

    def _is_native_on_disk(self, skill_id: str, conn: sqlite3.Connection) -> bool:
        """True when a SKILL.md exists on disk for this id with NO ledger row."""
        row = conn.execute(
            "SELECT 1 FROM managed_skills WHERE skill_id = ?", (skill_id,)
        ).fetchone()
        if row is not None:
            return False
        skill_md = self._skills_dir / skill_id / "SKILL.md"
        return skill_md.exists()

    def _fetch_row(self, skill_id: str, conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM managed_skills WHERE skill_id = ?", (skill_id,)
        ).fetchone()

    def _upsert_managed_row(
        self,
        conn: sqlite3.Connection,
        *,
        skill_id: str,
        content_hash: str,
        version: Optional[str],
        previous_state: Optional[Dict[str, Any]],
        idempotency_key: str,
        fingerprint: str,
        activated_at: str,
    ) -> None:
        """The single ledger-write step — a dedicated method so fault-injection
        tests can monkeypatch exactly this call to simulate a ledger failure
        after the file write (before-image restoration test)."""
        conn.execute(
            "INSERT INTO managed_skills "
            "(skill_id, content_hash, version, scope_json, managed_by, activated_at, "
            " previous_state_json, idempotency_key, fingerprint) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(skill_id) DO UPDATE SET "
            " content_hash=excluded.content_hash, version=excluded.version, "
            " activated_at=excluded.activated_at, previous_state_json=excluded.previous_state_json, "
            " idempotency_key=excluded.idempotency_key, fingerprint=excluded.fingerprint",
            (
                skill_id,
                content_hash,
                version,
                json.dumps(self._scope),
                MANAGED_BY,
                activated_at,
                json.dumps(previous_state) if previous_state is not None else None,
                idempotency_key,
                fingerprint,
            ),
        )

    def _write_skill_file_atomic(self, skill_id: str, content: str) -> None:
        skill_md = self._skill_md_path(skill_id)
        skill_md.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = skill_md.with_suffix(skill_md.suffix + f".tmp-{os.getpid()}-{time.time_ns()}")
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, skill_md)

    def _restore_before_image(self, skill_id: str, before_content: Optional[str]) -> None:
        skill_md = self._skill_md_path(skill_id)
        if before_content is None:
            skill_md.unlink(missing_ok=True)
        else:
            self._write_skill_file_atomic(skill_id, before_content)

    def _ledger_entry_from_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "skillId": row["skill_id"],
            "contentHash": row["content_hash"],
            "version": row["version"],
            "scope": json.loads(row["scope_json"]),
            "managedBy": row["managed_by"],
            "activatedAt": row["activated_at"],
            "previousState": json.loads(row["previous_state_json"]) if row["previous_state_json"] else None,
            "idempotencyKey": row["idempotency_key"],
        }

    def _ack_from_row(self, row: sqlite3.Row, *, reloaded: bool = True) -> Dict[str, Any]:
        return {
            "skillId": row["skill_id"],
            "contentHash": row["content_hash"],
            "activatedAt": row["activated_at"],
            "reloaded": reloaded,
        }

    def _check_scope_assert(self, scope_assert: Dict[str, str]) -> Optional[str]:
        for field, ledger_field in _SCOPE_ASSERT_TO_LEDGER_FIELD.items():
            if field in scope_assert:
                configured = self._scope.get(ledger_field)
                if scope_assert[field] != configured:
                    return field
        return None

    # -- POST /v1/skills/activations ---------------------------------------

    async def handle_activate(self, request: "web.Request") -> "web.Response":
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            raw_body = await request.text()
        except Exception:
            return _error_response(422, "invalid_request", "unreadable request body")

        try:
            parsed = json.loads(raw_body)
        except (json.JSONDecodeError, ValueError):
            return _error_response(422, "invalid_request", "request body is not valid JSON")

        try:
            validated = _validate_activation_body(parsed)
        except _ActivationError as exc:
            return _error_response(exc.status, exc.code, exc.message)

        mismatched_field = self._check_scope_assert(validated["scope_assert"])
        if mismatched_field:
            return _error_response(403, "wrong_scope", f"scope assertion mismatch: {mismatched_field}")

        skill_id = validated["skill_id"]
        idempotency_key = validated["idempotency_key"]
        fingerprint = _canonical_fingerprint(parsed)

        conn = self._conn()
        try:
            # Native-ownership guard: an on-disk skill with NO ledger row
            # must never be hijacked by activation.
            if self._is_native_on_disk(skill_id, conn):
                return _error_response(409, "native_conflict", "skillId belongs to a remote-native skill")

            # Durable idempotency replay — plain read, no cache involved.
            existing = self._fetch_row(skill_id, conn)
            if existing is not None and existing["idempotency_key"] == idempotency_key:
                if existing["fingerprint"] == fingerprint:
                    return web.json_response(
                        {
                            "ack": self._ack_from_row(existing),
                            "ledgerEntry": self._ledger_entry_from_row(existing),
                        },
                        status=200,
                    )
                return _error_response(409, "idempotency_conflict", "idempotencyKey reused with a different body")
        finally:
            conn.close()

        async def compute() -> Tuple[int, Dict[str, Any]]:
            return self._activate_compute(skill_id, validated, fingerprint)

        cache_key = self._cache_key(skill_id, idempotency_key)
        try:
            status, payload = await self._idem_cache.get_or_set(cache_key, fingerprint, compute)
        except _ActivationErrorAsResponse as exc:
            return _error_response(exc.status, exc.code, exc.message)
        return web.json_response(payload, status=status)

    def _activate_compute(
        self, skill_id: str, validated: Dict[str, Any], fingerprint: str
    ) -> Tuple[int, Dict[str, Any]]:
        content = validated["content"]
        supplied_hash = validated["content_hash"]

        recomputed = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(recomputed, supplied_hash):
            raise _ActivationErrorAsResponse(409, "integrity_mismatch", "contentHash does not match delivered content")

        conn = self._conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                # Re-check the durable-idempotency-replay decision INSIDE
                # the transaction — closes the concurrent
                # different-fingerprint race window (adjudication #5).
                existing = self._fetch_row(skill_id, conn)
                if existing is not None and existing["idempotency_key"] == validated["idempotency_key"]:
                    _safe_rollback(conn)
                    if existing["fingerprint"] == fingerprint:
                        return 200, {
                            "ack": self._ack_from_row(existing),
                            "ledgerEntry": self._ledger_entry_from_row(existing),
                        }
                    raise _ActivationErrorAsResponse(409, "idempotency_conflict", "idempotencyKey reused with a different body")

                is_new_row = existing is None
                if is_new_row:
                    count = conn.execute("SELECT COUNT(*) AS n FROM managed_skills").fetchone()["n"]
                    if count >= MAX_MANAGED_ENTRIES:
                        _safe_rollback(conn)
                        raise _ActivationErrorAsResponse(409, "ledger_full", "managed-skill ledger is at capacity")

                activated_at = _iso_now()

                if existing is not None and existing["content_hash"] == recomputed:
                    # Same-content no-op refresh: bookkeeping only, no
                    # file rewrite, previousState untouched.
                    conn.execute(
                        "UPDATE managed_skills SET idempotency_key = ?, fingerprint = ?, activated_at = ? "
                        "WHERE skill_id = ?",
                        (validated["idempotency_key"], fingerprint, activated_at, skill_id),
                    )
                    conn.execute("COMMIT")
                    row = self._fetch_row(skill_id, conn)
                    return 201, {
                        "ack": self._ack_from_row(row, reloaded=True),
                        "ledgerEntry": self._ledger_entry_from_row(row),
                    }

                before_content: Optional[str] = None
                skill_md = self._skill_md_path(skill_id)
                if skill_md.exists():
                    before_content = skill_md.read_text(encoding="utf-8")

                previous_state = None
                if existing is not None:
                    previous_state = {
                        "contentHash": existing["content_hash"],
                        "version": existing["version"],
                        "content": before_content,
                    }

                self._write_skill_file_atomic(skill_id, content)

                try:
                    self._upsert_managed_row(
                        conn,
                        skill_id=skill_id,
                        content_hash=recomputed,
                        version=validated["version"],
                        previous_state=previous_state,
                        idempotency_key=validated["idempotency_key"],
                        fingerprint=fingerprint,
                        activated_at=activated_at,
                    )
                    conn.execute("COMMIT")
                except Exception:
                    _safe_rollback(conn)
                    # Ledger write failed after the file write — restore
                    # the before-image so a refresh failure never
                    # destroys the previously-managed file.
                    self._restore_before_image(skill_id, before_content)
                    raise

                row = self._fetch_row(skill_id, conn)
                return 201, {
                    "ack": self._ack_from_row(row, reloaded=True),
                    "ledgerEntry": self._ledger_entry_from_row(row),
                }
            except _ActivationErrorAsResponse:
                raise
            except Exception:
                _safe_rollback(conn)
                raise
        finally:
            conn.close()

    # -- DELETE /v1/skills/activations/{skillId} ---------------------------

    async def handle_deactivate(self, request: "web.Request") -> "web.Response":
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        skill_id = request.match_info.get("skill_id", "")
        if not _SKILL_ID_RE.match(skill_id):
            return _error_response(422, "invalid_request", "skillId malformed")

        mode_values = request.query.getall("mode", [])
        if len(mode_values) == 0:
            mode = "clean"
        elif len(mode_values) == 1 and mode_values[0] in ("clean", "rollback"):
            mode = mode_values[0]
        else:
            return _error_response(422, "invalid_request", "mode must be exactly one of clean|rollback")

        conn = self._conn()
        try:
            row = self._fetch_row(skill_id, conn)
            if row is None:
                if self._is_native_on_disk(skill_id, conn):
                    return _error_response(403, "not_managed", "skillId belongs to a remote-native skill")
                return _error_response(404, "not_found", "no managed skill with that id")

            at = _iso_now()
            if mode == "clean":
                self._remove_skill_file_and_dir(skill_id)
                conn.execute("DELETE FROM managed_skills WHERE skill_id = ?", (skill_id,))
                self._bump_epoch(skill_id)
                return web.json_response({"ack": {"skillId": skill_id, "removed": True, "at": at}}, status=200)

            # mode == "rollback"
            previous_state = json.loads(row["previous_state_json"]) if row["previous_state_json"] else None
            if previous_state is None:
                self._remove_skill_file_and_dir(skill_id)
                conn.execute("DELETE FROM managed_skills WHERE skill_id = ?", (skill_id,))
                self._bump_epoch(skill_id)
                return web.json_response({"ack": {"skillId": skill_id, "removed": True, "at": at}}, status=200)

            prior_content = previous_state.get("content")
            if prior_content is not None:
                self._write_skill_file_atomic(skill_id, prior_content)
            conn.execute(
                "UPDATE managed_skills SET content_hash = ?, version = ?, previous_state_json = NULL, "
                "activated_at = ? WHERE skill_id = ?",
                (previous_state.get("contentHash"), previous_state.get("version"), at, skill_id),
            )
            return web.json_response({"ack": {"skillId": skill_id, "restored": True, "at": at}}, status=200)
        finally:
            conn.close()

    def _remove_skill_file_and_dir(self, skill_id: str) -> None:
        """Remove SKILL.md and, if now empty, its containing skill directory.

        A managed skill removed by clean/rollback-to-absence should leave no
        trace — an empty ``home/skills/<skillId>/`` directory would silently
        litter the skills tree and could confuse a later live scan.
        """
        skill_md = self._skill_md_path(skill_id)
        skill_md.unlink(missing_ok=True)
        skill_dir = skill_md.parent
        try:
            skill_dir.rmdir()
        except OSError:
            pass  # non-empty (unexpected extra files) or already gone — leave it

    def _bump_epoch(self, skill_id: str) -> None:
        self._epoch[skill_id] = self._epoch.get(skill_id, 0) + 1

    # -- GET /v1/skills?scope=managed ---------------------------------------

    async def handle_managed_inventory(self, request: "web.Request") -> "web.Response":
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        conn = self._conn()
        try:
            rows = conn.execute("SELECT * FROM managed_skills ORDER BY skill_id").fetchall()
        finally:
            conn.close()

        data: List[Dict[str, Any]] = []
        for row in rows:
            skill_md = self._skills_dir / row["skill_id"] / "SKILL.md"
            if not skill_md.exists():
                continue
            data.append({
                "name": row["skill_id"],
                "description": "",
                "category": None,
                "contentHash": row["content_hash"],
                "managedBy": row["managed_by"],
                "scope": json.loads(row["scope_json"]),
            })

        return web.json_response({"object": "list", "data": data})


class _ActivationErrorAsResponse(Exception):
    """Raised from inside a transaction to unwind to an HTTP error response."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
