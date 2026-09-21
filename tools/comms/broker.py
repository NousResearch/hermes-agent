"""Small persistent loopback broker for the inter-agent tool.

The broker is deliberately a separate process so several Hermes conversations can
share one queue. It binds only to loopback and stores messages in the installation
control home so profile-local workers can interoperate.
"""

from __future__ import annotations

import getpass
import json
import hmac
import os
from pathlib import Path
import secrets
import sqlite3
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import psutil

from hermes_constants import get_default_hermes_root
from tools.comms import BROKER_PROTOCOL_VERSION

HOST = "127.0.0.1"
PORT = 0
MAX_QUERY_LIMIT = 1000
RECEIVE_WAIT_SECONDS = 25.0
RECEIVE_POLL_INTERVAL_SECONDS = 0.1
_WINDOWS_ACL_SECURED: set[tuple[Path, bool]] = set()


def _state_path(name: str) -> Path:
    return get_default_hermes_root() / name


def _secure_state_permissions(
    path: Path, *, directory: bool = False, force: bool = False
) -> None:
    """Keep broker state owner-only, including inherited Windows ACLs."""
    path.chmod(0o700 if directory else 0o600)
    cache_key = (path, directory)
    if os.name != "nt" or (not force and cache_key in _WINDOWS_ACL_SECURED):
        return
    account = getpass.getuser()
    if not account:
        raise PermissionError(f"unable to identify broker state owner: {path}")
    permissions = "(OI)(CI)F" if directory else "F"
    result = subprocess.run(
        ["icacls", str(path), "/inheritance:r", "/grant:r", f"{account}:{permissions}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise PermissionError(f"unable to secure broker state ACL: {path}")
    _WINDOWS_ACL_SECURED.add(cache_key)


def _broker_token() -> str:
    home = get_default_hermes_root()
    home.mkdir(parents=True, exist_ok=True)
    _secure_state_permissions(home, directory=True)
    path = _state_path("inter-agent-broker.token")
    try:
        token = path.read_text(encoding="utf-8").strip()
        if token:
            _secure_state_permissions(path)
            return token
    except FileNotFoundError:
        pass
    token = secrets.token_urlsafe(32)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(token)
        os.replace(temporary, path)
        _secure_state_permissions(path)
    finally:
        temporary.unlink(missing_ok=True)
    return token


class _BrokerServer(ThreadingHTTPServer):
    broker_id: str


def _write_endpoint(server: _BrokerServer) -> Path:
    path = _state_path("inter-agent-broker.json")
    path.write_text(
        json.dumps({
            "port": server.server_port,
            "broker_id": server.broker_id,
            "protocol_version": BROKER_PROTOCOL_VERSION,
            "pid": os.getpid(),
            "pid_start_time_us": int(
                round(psutil.Process(os.getpid()).create_time() * 1_000_000)
            ),
        }),
        encoding="utf-8",
    )
    _secure_state_permissions(path)
    return path


def _remove_endpoint_if_owned(path: Path, broker_id: str) -> None:
    """Remove the endpoint only when it still names this broker instance."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return
    if payload.get("broker_id") != broker_id:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _restrict_database_permissions(path: Path, *, force: bool = False) -> None:
    for candidate in (
        path,
        Path(f"{path}-journal"),
        Path(f"{path}-wal"),
        Path(f"{path}-shm"),
    ):
        try:
            _secure_state_permissions(candidate, force=force)
        except FileNotFoundError:
            continue


def _limit(query: dict[str, list[str]]) -> int:
    try:
        requested = int(query.get("limit", [100])[0])
    except (TypeError, ValueError):
        requested = 100
    return max(1, min(requested, MAX_QUERY_LIMIT))


def _commit_database(connection: sqlite3.Connection, path: Path) -> None:
    connection.commit()
    _restrict_database_permissions(path, force=True)


def _database() -> sqlite3.Connection:
    home = get_default_hermes_root()
    home.mkdir(parents=True, exist_ok=True)
    _secure_state_permissions(home, directory=True)
    path = home / "inter-agent-messages.db"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        os.close(fd)
    _restrict_database_permissions(path)
    connection = sqlite3.connect(path, timeout=10)
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS messages ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "sender TEXT NOT NULL, recipient TEXT NOT NULL, body TEXT NOT NULL,"
            "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS messages_recipient_id_idx "
            "ON messages (recipient, id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS messages_sender_id_idx "
            "ON messages (sender, id)"
        )
        _commit_database(connection, path)
    finally:
        _restrict_database_permissions(path)
    return connection


class _Handler(BaseHTTPRequestHandler):
    def _authorized(self) -> bool:
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {_broker_token()}"
        return hmac.compare_digest(supplied, expected)

    def _write(self, status: int, payload: object) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not self._authorized():
            self._write(401, {"error": "unauthorized"})
            return
        if parsed.path == "/health":
            self._write(
                200,
                {
                    "ok": True,
                    "broker_id": self.server.broker_id,
                    "protocol_version": BROKER_PROTOCOL_VERSION,
                },
            )
            return
        if parsed.path == "/receive":
            query = parse_qs(parsed.query)
            recipient = query.get("to", [""])[0]
            try:
                since = max(0, int(query.get("since", [0])[0]))
            except (TypeError, ValueError):
                self._write(400, {"error": "since must be an integer"})
                return
            limit = _limit(query)
            deadline = time.monotonic() + RECEIVE_WAIT_SECONDS
            rows = []
            while True:
                with _database() as db:
                    rows = db.execute(
                        "SELECT id, sender, recipient, body, created_at FROM messages "
                        "WHERE recipient = ? AND id > ? ORDER BY id LIMIT ?",
                        (recipient, since, limit),
                    ).fetchall()
                if rows or time.monotonic() >= deadline:
                    break
                time.sleep(
                    min(
                        RECEIVE_POLL_INTERVAL_SECONDS,
                        max(0.0, deadline - time.monotonic()),
                    )
                )
            self._write(200, [
                {"id": row[0], "from": row[1], "to": row[2], "body": row[3], "created_at": row[4]}
                for row in rows
            ])
            return
        if parsed.path == "/history":
            query = parse_qs(parsed.query)
            peer = query.get("with_", [""])[0]
            limit = _limit(query)
            with _database() as db:
                if peer:
                    rows = db.execute(
                        "SELECT id, sender, recipient, body, created_at FROM messages "
                        "WHERE sender = ? OR recipient = ? ORDER BY id DESC LIMIT ?",
                        (peer, peer, limit),
                    ).fetchall()
                else:
                    rows = db.execute(
                        "SELECT id, sender, recipient, body, created_at FROM messages "
                        "ORDER BY id DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
            rows = list(reversed(rows))
            self._write(200, [
                {"id": row[0], "from": row[1], "to": row[2], "body": row[3], "created_at": row[4]}
                for row in rows
            ])
            return
        self._write(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/send":
            self._write(404, {"error": "not found"})
            return
        if not self._authorized():
            self._write(401, {"error": "unauthorized"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            sender = str(payload["from"])
            recipient = str(payload["to"])
            body = str(payload["body"])
            if not sender or not recipient or not body:
                raise ValueError("from, to, and body are required")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._write(400, {"error": str(exc)})
            return
        with _database() as db:
            cursor = db.execute(
                "INSERT INTO messages(sender, recipient, body) VALUES (?, ?, ?)",
                (sender, recipient, body),
            )
            message_id = cursor.lastrowid
            _commit_database(db, _state_path("inter-agent-messages.db"))
        self._write(200, {"id": message_id})

    def log_message(self, _format: str, *_args: object) -> None:
        return


def main() -> None:
    server = _BrokerServer((HOST, PORT), _Handler)
    server.broker_id = secrets.token_urlsafe(24)
    endpoint = _write_endpoint(server)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        _remove_endpoint_if_owned(endpoint, server.broker_id)


if __name__ == "__main__":
    main()
