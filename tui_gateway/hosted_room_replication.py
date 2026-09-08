"""Bounded, passive history publisher. Coverage is an acknowledgement, not failover safety.

Two workers rotate durable targets, one page per turn. Shared checkpoints
retain pending page coordinates (not duplicate event bodies), so a lost reply or
restart retries the same page. No policy/driver lock or transaction spans HTTP.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gateway import hosted_room_links as links
from gateway import hosted_rooms as rooms
from gateway import hosted_room_replica_retirement as retirement
from gateway import hosted_room_passive_lineage as lineage
from gateway.hosted_room_passive_protocol import supports_lineage
from gateway.hosted_room_peer import PROTOCOL_VERSION, _split_token, gateway_room_grant_secret
from gateway.hosted_rooms_common import open_sqlite, table_exists
from gateway.status import _release_file_lock, _try_acquire_file_lock
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError

POLL_SECONDS = 5.0
PAGE_TIMEOUT_SECONDS = 3.0
PAGE_LIMIT = 32
WORKERS = 2
_TABLE = "hosted_room_replication_publishers"
_TARGET_TABLE = "hosted_room_replication_targets"
_BLOCKED = {"needs_reauthorization", "replica_rejected", "invalid_ack", "source_gap", "unsupported_lineage"}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _generation(link: links.StoredRoomLink, room: dict) -> str:
    record = link.as_record()
    # Health timestamps change during ordinary execution and are not route identity.
    for key in ("status", "updated_at"):
        record.pop(key)
    return _digest([record, room["authority_gateway_id"], room["authority_epoch"], room["members"]]
                   + ([room["lineage_sha256"]] if room["authority_epoch"] != 1 else []))


def _attach_lineage(conn, room):
    if room["authority_epoch"] != 1:
        history, digest = lineage.source_locked(conn, room["room_id"], {
            "gateway_id": room["authority_gateway_id"], "epoch": room["authority_epoch"]})
        room.update(authority_history=history, lineage_sha256=digest)
    return room


def _replication_hint(token: str) -> dict:
    if not isinstance(token, str) or len(token) > links.MAX_GRANT_CHARS:
        return {}
    try:
        hint = json.loads(_split_token(token)[0].decode("ascii"))
    except (ValueError, UnicodeError, RecursionError):
        return {}
    if isinstance(hint, dict) and isinstance(hint.get("permissions"), list) and "replicate" in hint["permissions"]:
        return hint
    return {}


def _eligible(link: links.StoredRoomLink, room: dict, local_id: str) -> bool:
    if (
        room["authority_gateway_id"] != local_id
        or link.status == "needs_reauthorization" or room.get("safety_status")
        or link.catalog.installation_id == local_id
        or PROTOCOL_VERSION not in link.catalog.protocol_versions
        or link.catalog.execution_policy.target_profile != link.target_profile
    ):
        return False
    hint = _replication_hint(link.grant)
    expected = {
        "version": PROTOCOL_VERSION, "room_id": link.room_id, "member_id": link.member_id,
        "home_install_id": local_id, "authority_gateway_id": local_id, "authority_epoch": room["authority_epoch"],
        "target_install_id": link.catalog.installation_id, "target_profile": link.target_profile,
        "execution_policy_digest": link.catalog.execution_policy.policy_digest,
    }
    # Hints only avoid ineligible sends. The target authenticates signature, live
    # reservation, revocation and its fixed status horizon; no unsigned timing here.
    if any(hint.get(k) != v for k, v in expected.items()):
        return False
    matching = [m for m in room["members"] if m.get("member_id") == link.member_id]
    if len(matching) != 1:
        return False
    target = matching[0].get("target", {})
    if not isinstance(target, dict) or target.get("profile") != link.target_profile:
        return False
    return target.get("kind") == "peer" and target.get("installation_id") == link.catalog.installation_id


@dataclass(frozen=True)
class _Route:
    key: tuple[str, str]
    link: links.StoredRoomLink = field(repr=False)
    room: dict = field(repr=False)
    generation: str = field(repr=False)


@dataclass(frozen=True)
class _RetirementWork:
    enrollment_id: str


_Work = tuple[str, str] | _RetirementWork


class HostedRoomReplicationPublisher:
    """Gateway-owned asynchronous copy, deliberately independent of execution."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.local_id = rooms.local_authority_gateway_id()
        links.load_room_links_tolerant(self.db_path)  # initialize the existing root schema
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._routes: deque[_Work] = deque()
        self._inflight: set[_Work] = set()
        self._due: dict[_Work, float] = {}
        self._retirement_delays: dict[_RetirementWork, float] = {}
        self._scan_at = 0.0
        self._error: str | None = None
        with self._transaction() as conn:
            conn.execute(f"""CREATE TABLE IF NOT EXISTS {_TABLE} (
                room_id TEXT NOT NULL, member_id TEXT NOT NULL, generation TEXT NOT NULL,
                target_install_id TEXT NOT NULL, target_profile TEXT NOT NULL,
                authority_gateway_id TEXT NOT NULL, authority_epoch INTEGER NOT NULL,
                acked_seq INTEGER NOT NULL DEFAULT 0, source_latest_seq INTEGER NOT NULL DEFAULT 0,
                pending_end INTEGER, pending_latest INTEGER, pending_name TEXT,
                status TEXT NOT NULL DEFAULT 'pending', updated_at REAL NOT NULL,
                PRIMARY KEY(room_id, member_id))""")
            conn.execute(f"""CREATE TABLE IF NOT EXISTS {_TARGET_TABLE} (
                room_id TEXT NOT NULL, target_install_id TEXT NOT NULL, lineage TEXT NOT NULL,
                selected_member_id TEXT NOT NULL, acked_seq INTEGER NOT NULL DEFAULT 0,
                source_latest_seq INTEGER NOT NULL DEFAULT 0, pending_end INTEGER,
                pending_latest INTEGER, pending_name TEXT, status TEXT NOT NULL DEFAULT 'pending',
                updated_at REAL NOT NULL, PRIMARY KEY(room_id, target_install_id))""")
            for table in (_TABLE, _TARGET_TABLE):
                columns = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
                if "pending_lineage_sha256" not in columns:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN pending_lineage_sha256 TEXT")
            if "passive_version" not in {r["name"] for r in conn.execute(f"PRAGMA table_info({_TABLE})")}:
                conn.execute(f"ALTER TABLE {_TABLE} ADD COLUMN passive_version INTEGER")

    @contextmanager
    def _transaction(self):
        conn = open_sqlite(self.db_path, timeout=0.25)
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def start(self) -> None:
        with self._condition:
            if any(t.is_alive() for t in self._threads):
                return  # A timed-out stop must not spawn replacement workers.
            self._stop.clear()
            self._scan_at = 0.0
            self._error = None
            self._threads = [threading.Thread(
                target=self._worker, name=f"hosted-room-replication-{i}", daemon=True,
            ) for i in range(WORKERS)]
            try:
                for thread in self._threads:
                    thread.start()
            except (RuntimeError, OSError):
                self._error = "publisher_start_failed"
                self._stop.set()
                self._threads = [thread for thread in self._threads if thread.ident is not None]
                self._condition.notify_all()

    def stop(self, *, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
            threads = tuple(self._threads)
        for thread in threads:
            if thread.ident is not None:
                thread.join(max(0.0, deadline - time.monotonic()))
        return not any(t.is_alive() for t in threads)

    def _scan(self, now: float) -> None:
        with self._transaction() as conn:
            has_retirement = table_exists(conn, retirement.HOME_TABLE)
            keys = [(row["room_id"], row["member_id"]) for row in conn.execute(
                "SELECT room_id, member_id, grant FROM hosted_room_links ORDER BY room_id, member_id LIMIT ?",
                (links.MAX_LINKS,),
            ) if _replication_hint(row["grant"])]
            conn.execute(f"""UPDATE {_TABLE} SET status='stopped_route_removed'
                WHERE status!='stopped_route_removed' AND NOT EXISTS (
                    SELECT 1 FROM hosted_room_links AS l
                    WHERE l.room_id={_TABLE}.room_id AND l.member_id={_TABLE}.member_id)""")
            conn.execute(f"""UPDATE {_TABLE} SET source_latest_seq=MAX(source_latest_seq, COALESCE(
                (SELECT next_seq-1 FROM hosted_rooms WHERE room_id={_TABLE}.room_id), source_latest_seq))
                WHERE status='stopped_route_removed'""")
            # Keep a bounded diagnostic tail after revocation, not an ever-growing outbox.
            conn.execute(f"""DELETE FROM {_TABLE} WHERE rowid IN (
                SELECT rowid FROM {_TABLE} WHERE status='stopped_route_removed'
                ORDER BY updated_at DESC LIMIT -1 OFFSET ?)""", (links.MAX_LINKS,))
            conn.execute(f"""DELETE FROM {_TARGET_TABLE} WHERE NOT EXISTS (
                SELECT 1 FROM {_TABLE} AS r WHERE r.room_id={_TARGET_TABLE}.room_id
                    AND r.target_install_id={_TARGET_TABLE}.target_install_id)""")
        if has_retirement:
            keys.extend(_RetirementWork(value) for value in retirement.pending_notice_ids(
                self.db_path, local_gateway_id=self.local_id))
        current = set(keys)
        self._routes = deque([k for k in self._routes if k in current])
        self._routes.extend(k for k in keys if k not in self._routes)
        self._due = {k: due for k, due in self._due.items() if k in current}
        self._retirement_delays = {k: delay for k, delay in self._retirement_delays.items() if k in current}
        self._scan_at = now + POLL_SECONDS

    def _take(self) -> _Work | None:
        with self._condition:
            while not self._stop.is_set():
                now = time.monotonic()
                if now >= self._scan_at:
                    self._scan(now)
                for _ in range(len(self._routes)):
                    key = self._routes.popleft()
                    self._routes.append(key)
                    if key not in self._inflight and self._due.get(key, 0) <= now:
                        self._inflight.add(key)
                        return key
                wake_at = min([self._scan_at, *(
                    self._due.get(k, now) for k in self._routes if k not in self._inflight
                )])
                self._condition.wait(timeout=min(POLL_SECONDS, max(0.01, wake_at - now)))
        return None

    def _worker(self) -> None:
        while not self._stop.is_set():
            key, more = None, False
            try:
                key = self._take()
                if key is None:
                    return
                more = self._publish_retirement(key) if isinstance(key, _RetirementWork) else self._publish_one(key)
                self._error = None
            except Exception:
                # Never include transport exceptions, grant material, URLs or raw rows.
                self._error = "publisher_local_error"
                self._stop.wait(POLL_SECONDS)
            finally:
                if key is not None:
                    with self._condition:
                        self._inflight.discard(key)
                        delay = self._retirement_delays.get(key, POLL_SECONDS) if isinstance(key, _RetirementWork) else POLL_SECONDS
                        self._due[key] = time.monotonic() + (0 if more else delay)
                        self._condition.notify_all()

    def _publish_retirement(self, work: _RetirementWork) -> bool:
        route = retirement.notice_route(self.db_path, enrollment_id=work.enrollment_id, local_gateway_id=self.local_id)
        if route is None or self._stop.is_set():
            return False
        path = self.db_path.parent / "room_replication_locks" / (
            _digest([route["room_id"], route["target_install_id"]]) + ".lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8") as handle:
            if not _try_acquire_file_lock(handle):
                return False
            try:
                if retirement.notice_route(self.db_path, enrollment_id=work.enrollment_id, local_gateway_id=self.local_id) is None:
                    return False
                notice = retirement.materialize_notice(
                    self.db_path, enrollment_id=work.enrollment_id, local_gateway_id=self.local_id,
                    secret_loader=gateway_room_grant_secret,
                )
                if self._stop.is_set():
                    return False
                client = PeerRunsHTTPClient(base_url=notice.endpoint, api_key="", timeout_seconds=PAGE_TIMEOUT_SECONDS)
                response = client.retire_replica(notice)
                retirement.acknowledge_notice(self.db_path, notice=notice, response=response)
                self._retirement_delays.pop(work, None)
            except (retirement.RetirementError, PeerRunsHTTPError, OSError) as exc:
                code = "retirement_key_unavailable" if isinstance(exc, retirement.RetirementKeyUnavailable) else "retirement_delivery_unconfirmed"
                retirement.record_delivery_error(self.db_path, enrollment_id=work.enrollment_id, code=code)
                self._retirement_delays[work] = min(300.0, self._retirement_delays.get(work, POLL_SECONDS / 2) * 2)
            finally:
                _release_file_lock(handle)
        return False

    def _load_route(self, key: tuple[str, str]) -> _Route | None:
        with closing(open_sqlite(self.db_path, timeout=0.25)) as conn:
            raw = conn.execute(
                "SELECT * FROM hosted_room_links WHERE room_id=? AND member_id=?", key,
            ).fetchone()
        if raw is None or not _replication_hint(raw["grant"]):
            return None
        try:
            link = links.StoredRoomLink.from_record(raw)
            room = rooms.room_state(self.db_path, room_id=key[0], include_disbanded=True)
            with rooms._transaction(self.db_path) as conn:
                _attach_lineage(conn, room)
            if _eligible(link, room, self.local_id):
                return _Route(key, link, room, _generation(link, room))
        except (rooms.HostedRoomError, ValueError):
            pass
        with self._transaction() as conn:
            conn.execute(f"""UPDATE {_TABLE} SET status='stopped_ineligible'
                WHERE room_id=? AND member_id=? AND status NOT IN ({','.join('?' for _ in _BLOCKED)})""",
                         (*key, *sorted(_BLOCKED)))
        return None

    def _current(self, conn, route: _Route) -> bool:
        raw = conn.execute("SELECT * FROM hosted_room_links WHERE room_id=? AND member_id=?", route.key).fetchone()
        room = conn.execute("SELECT * FROM hosted_rooms WHERE room_id=?", (route.key[0],)).fetchone()
        quarantine = conn.execute(
            "SELECT 1 FROM hosted_room_quarantine WHERE room_id=?", (route.key[0],),
        ).fetchone()
        if raw is None or room is None or quarantine is not None:
            return False
        try:
            link = links.StoredRoomLink.from_record(raw)
            state = _attach_lineage(conn, {**dict(room), "members": json.loads(room["members_json"])})
            return _eligible(link, state, self.local_id) and _generation(link, state) == route.generation
        except (ValueError, rooms.HostedRoomError):
            return False

    def _checkpoint(self, route: _Route) -> dict | None:
        with self._transaction() as conn:
            if not self._current(conn, route):
                return None
            conn.execute(f"""INSERT INTO {_TABLE} (
                room_id, member_id, generation, target_install_id, target_profile,
                authority_gateway_id, authority_epoch, updated_at) VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(room_id,member_id) DO UPDATE SET generation=excluded.generation,
                target_install_id=excluded.target_install_id, target_profile=excluded.target_profile,
                authority_gateway_id=excluded.authority_gateway_id, authority_epoch=excluded.authority_epoch,
                acked_seq=0, source_latest_seq=0, pending_end=NULL, pending_latest=NULL,
                pending_name=NULL, pending_lineage_sha256=NULL, passive_version=NULL,
                status='pending', updated_at=excluded.updated_at
                WHERE generation!=excluded.generation""", (
                    *route.key, route.generation, route.link.catalog.installation_id, route.link.target_profile,
                    route.room["authority_gateway_id"], route.room["authority_epoch"], time.time(),
                ))
            row = conn.execute(f"SELECT * FROM {_TABLE} WHERE room_id=? AND member_id=?", route.key).fetchone()
            return dict(row)

    def _save(self, route: _Route, checkpoint: dict, **values) -> bool:
        with self._transaction() as conn:
            if not self._current(conn, route):
                return False
            values["updated_at"] = time.time()
            saved = conn.execute(
                f"UPDATE {_TARGET_TABLE} SET " + ",".join(f"{k}=?" for k in values)
                + " WHERE room_id=? AND target_install_id=? AND lineage=? AND acked_seq=?"
                + " AND pending_end IS ? AND pending_latest IS ?",
                (*values.values(), route.key[0], route.link.catalog.installation_id,
                 checkpoint["lineage"], checkpoint["acked_seq"],
                 checkpoint["pending_end"], checkpoint["pending_latest"]),
            ).rowcount == 1
            if saved:
                conn.execute(
                    f"UPDATE {_TABLE} SET " + ",".join(f"{k}=?" for k in values)
                    + " WHERE room_id=? AND member_id=? AND generation=?",
                    (*values.values(), *route.key, route.generation),
                )
            return saved

    def _lock_path(self, route: _Route) -> Path:
        return self.db_path.parent / "room_replication_locks" / (
            _digest([route.key[0], route.link.catalog.installation_id]) + ".lock")

    def _select_route(self, initial: _Route) -> _Route | None:
        """Select a live authorized profile while holding the target's OS lock."""
        with closing(open_sqlite(self.db_path, timeout=0.25)) as conn:
            candidates = conn.execute(
                "SELECT * FROM hosted_room_links WHERE room_id=? ORDER BY member_id LIMIT ?",
                (initial.key[0], links.MAX_LINKS),
            ).fetchall()
        selected = []
        room = rooms.room_state(self.db_path, room_id=initial.key[0], include_disbanded=True)
        with rooms._transaction(self.db_path) as conn:
            _attach_lineage(conn, room)
        for raw in candidates:
            if not _replication_hint(raw["grant"]):
                continue
            try:
                link = links.StoredRoomLink.from_record(raw)
            except ValueError:
                continue
            if link.catalog.installation_id != initial.link.catalog.installation_id or not _eligible(link, room, self.local_id):
                continue
            route = _Route((link.room_id, link.member_id), link, room, _generation(link, room))
            checkpoint = self._checkpoint(route)
            if checkpoint is not None and checkpoint["status"] not in _BLOCKED:
                selected.append((checkpoint["status"] == "unavailable", route.key, route))
        return min(selected, key=lambda item: item[:2])[2] if selected else None

    def _target_checkpoint(self, route: _Route) -> dict | None:
        lineage = _digest([route.room["authority_gateway_id"], route.room["authority_epoch"], route.room["members"]]
            + ([route.room["lineage_sha256"]] if route.room["authority_epoch"] != 1 else []))
        key = (route.key[0], route.link.catalog.installation_id)
        with self._transaction() as conn:
            if not self._current(conn, route):
                return None
            conn.execute(f"""INSERT INTO {_TARGET_TABLE}
                (room_id,target_install_id,lineage,selected_member_id,updated_at) VALUES (?,?,?,?,?)
                ON CONFLICT(room_id,target_install_id) DO UPDATE SET lineage=excluded.lineage,
                acked_seq=0, source_latest_seq=0, pending_end=NULL, pending_latest=NULL,
                pending_name=NULL, pending_lineage_sha256=NULL, status='pending', updated_at=excluded.updated_at
                WHERE lineage!=excluded.lineage""", (*key, lineage, route.key[1], time.time()))
            conn.execute(f"UPDATE {_TARGET_TABLE} SET selected_member_id=? WHERE room_id=? AND target_install_id=?",
                         (route.key[1], *key))
            return dict(conn.execute(
                f"SELECT * FROM {_TARGET_TABLE} WHERE room_id=? AND target_install_id=?", key,
            ).fetchone())

    def _publish_one(self, key: tuple[str, str]) -> bool:
        route = self._load_route(key)
        if route is None or self._stop.is_set():
            return False
        path = self._lock_path(route)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Never unlink lock files: replacing a locked inode permits two owners.
        # Reuse gateway's supported flock/msvcrt helpers, not a TTL lease.
        with path.open("a+", encoding="utf-8") as handle:
            if not _try_acquire_file_lock(handle):
                return False
            try:
                selected = self._select_route(route)
                return self._publish_locked(selected) if selected is not None else False
            finally:
                _release_file_lock(handle)

    def _negotiate_lineage(self, route, checkpoint, client):
        cached = self._checkpoint(route)
        if cached is None or cached["status"] in _BLOCKED:
            return False, None
        if cached["passive_version"] == 2:
            return True, None
        try:
            proof = client.probe(grant=route.link.grant)
        except PeerRunsHTTPError as exc:
            self._http_failure(route, checkpoint, exc)
            return False, None
        if not supports_lineage(proof):
            self._save(route, checkpoint, status="unsupported_lineage")
            return False, None
        with self._transaction() as conn:
            if not self._current(conn, route):
                return False, None
            conn.execute(f"UPDATE {_TABLE} SET passive_version=2 WHERE room_id=? AND member_id=? AND generation=?",
                         (*route.key, route.generation))
        return True, proof

    def _publish_locked(self, route: _Route) -> bool:
        key = route.key
        checkpoint = self._target_checkpoint(route)
        if checkpoint is None:
            return False
        client = PeerRunsHTTPClient(
            base_url=route.link.target_url, api_key="",
            target_profile=route.link.target_profile if route.link.target_profile != "default" else None,
            timeout_seconds=PAGE_TIMEOUT_SECONDS,
        )
        v2 = route.room["authority_epoch"] != 1
        probe = None
        if v2:
            supported, probe = self._negotiate_lineage(route, checkpoint, client)
            if not supported:
                return False
        enrollment = retirement.current_home_enrollment(
            self.db_path, room_id=key[0], target_install_id=route.link.catalog.installation_id,
        )
        if v2 and (enrollment is None or enrollment.get("version") != 2
                or enrollment.get("lineage_sha256") != route.room["lineage_sha256"]
                or enrollment["authority_gateway_id"] != self.local_id
                or enrollment["authority_epoch"] != route.room["authority_epoch"]):
            self._save(route, checkpoint, status="retirement_enrollment_required")
            return False
        if enrollment is not None:
            if enrollment["state"] == "prepared":
                try:
                    proof = (probe if probe is not None else client.probe(grant=route.link.grant)).get("retirement_enrollment")
                except PeerRunsHTTPError as exc:
                    return self._http_failure(route, checkpoint, exc)
                if not retirement.confirm_home_enrollment(
                    self.db_path, enrollment_id=enrollment["enrollment_id"], proof=proof,
                ):
                    return False
            elif enrollment["state"] != "enrolled":
                return False
        cursor, pending = checkpoint["acked_seq"], checkpoint["pending_end"]
        if pending is None and cursor >= route.room["latest_seq"] and checkpoint["status"] == "acked":
            return False
        limit = PAGE_LIMIT if pending is None else max(1, pending - cursor)
        page = rooms.read_events(
            self.db_path, room_id=key[0], since_seq=cursor, limit=limit, include_disbanded=True,
            **({"replica_version": 2} if v2 else {}),
        )
        expected_authority = {"gateway_id": route.room["authority_gateway_id"], "epoch": route.room["authority_epoch"]}
        if (page["authority"] != expected_authority
                or (v2 and page["lineage_sha256"] != route.room["lineage_sha256"])):
            return False
        name = route.room["name"]
        if pending is not None:
            # New source events cannot change the in-flight retry's coverage claim.
            if pending == cursor:
                page.update(events=[], cursor=cursor)
            if page["cursor"] != pending or (v2 and checkpoint["pending_lineage_sha256"] != page["lineage_sha256"]):
                self._save(route, checkpoint, status="source_gap")
                return False
            page.update(latest_seq=checkpoint["pending_latest"], has_more=pending < checkpoint["pending_latest"])
            name = checkpoint["pending_name"]
        else:
            values = dict(pending_end=page["cursor"], pending_latest=page["latest_seq"], pending_name=name,
                          source_latest_seq=page["latest_seq"], status="pending",
                          pending_lineage_sha256=page.get("lineage_sha256"))
            if not self._save(route, checkpoint, **values):
                return False
            checkpoint.update(values)
        if self._stop.is_set():
            return False
        try:
            reply = client.replicate_page(
                grant=route.link.grant, target_profile=route.link.target_profile,
                room_id=key[0], room_name=name, members=route.room["members"], page=page,
            )
        except PeerRunsHTTPError as exc:
            return self._http_failure(route, checkpoint, exc)
        if (
            not isinstance(reply, dict) or reply.get("room_id") != key[0]
            or reply.get("authority") != expected_authority
            or type(reply.get("stored_seq")) is not int or reply["stored_seq"] < page["cursor"]
            or reply["stored_seq"] > page["latest_seq"]
            or (v2 and (type(reply.get("replica_version")) is not int or reply["replica_version"] != 2
                or reply.get("lineage_sha256") != page["lineage_sha256"]
                or reply.get("lineage_status") != ("verified" if reply["stored_seq"] >=
                    route.room["authority_history"][-1]["from_seq"] else "pending")))
        ):
            self._save(route, checkpoint, status="invalid_ack")
            return False
        saved = self._save(
            route, checkpoint, acked_seq=page["cursor"], source_latest_seq=page["latest_seq"],
            pending_end=None, pending_latest=None, pending_name=None, pending_lineage_sha256=None,
            status="pending" if page["has_more"] else "acked",
        )
        return saved and page["has_more"]

    def _http_failure(self, route: _Route, checkpoint: dict, exc: PeerRunsHTTPError) -> bool:
        if exc.status_code == 409 and exc.error_code == "room_replica_gap":
            self._save(route, checkpoint, acked_seq=0, pending_end=None, pending_latest=None,
                       pending_name=None, status="replica_gap")
            return False
        status = "unavailable"
        if exc.status_code in {401, 403}:
            status = "needs_reauthorization"
        elif exc.status_code is not None and 400 <= exc.status_code < 500 and exc.status_code not in {408, 429}:
            status = "replica_rejected"
        self._save(route, checkpoint, status=status)
        return False

    def status(self, room_id: str | None = None) -> dict:
        import sqlite3
        error = self._error
        retirements = []
        try:
            with closing(open_sqlite(self.db_path, timeout=0.25)) as conn:
                has_retirement = table_exists(conn, retirement.HOME_TABLE)
                rows = conn.execute(f"""SELECT r.room_id, member_id, r.target_install_id, target_profile,
                    authority_gateway_id, authority_epoch, r.acked_seq, r.source_latest_seq, r.status, r.updated_at,
                    t.selected_member_id, t.acked_seq AS target_acked_seq, t.status AS target_status,
                    t.source_latest_seq AS target_source_latest_seq
                    FROM {_TABLE} AS r LEFT JOIN {_TARGET_TABLE} AS t
                    ON r.room_id=t.room_id AND r.target_install_id=t.target_install_id
                    WHERE (? IS NULL OR r.room_id=?) ORDER BY r.room_id, member_id""", (room_id, room_id))
                routes = []
                for row in rows:
                    role = "selected" if row["selected_member_id"] == row["member_id"] else "alternate"
                    if row["status"] in _BLOCKED or row["status"].startswith("stopped"):
                        role = "blocked"
                    routes.append({**dict(row), "delivery_unconfirmed": row["status"] != "acked", "role": role})
            if has_retirement:
                retirements = retirement.home_status(self.db_path, room_id=room_id)
        except (OSError, sqlite3.Error):
            routes, retirements, error = None, None, "publisher_status_unavailable"
        return {
            "running": any(t.is_alive() for t in self._threads), "stopping": self._stop.is_set(),
            "workers": sum(t.is_alive() for t in self._threads), "routes": routes,
            "retirements": retirements,
            "error": error, "mode": "passive_async_copy", "source_loss_safe": False,
        }
