"""Process ownership for superseded, non-shared Docker sandboxes (#84969).

The private namespace participates in the runtime fingerprint: clients which do
not register ownership cannot attach to these containers. SQLite transactions
serialize reaping with claims; kernel-held file locks survive idle periods and
are released on process death. Compatible persistent containers are never
retired just for being idle. Shared keys and pre-protocol containers remain
operator-managed.
"""

import hashlib
import json
import logging
import sqlite3
import time
import uuid
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from hermes_cli.active_sessions import _FileLock
from tools.environments.docker_ownership import strip_sandbox_privileges
from tools.environments.remote_common import run_capture

logger = logging.getLogger(__name__)
OWNER_LABEL = "hermes-reuse-owners"
_DATABASE = "docker-reuse-owners.sqlite3"
# ``hermes-task-id`` truncates at 63 chars, so ``<long-base>:ws:<digest>`` tasks can
# share its prefix; ``hermes-workspace`` carries the collision-proof digest instead.
_SCOPE_LABELS = ("hermes-task-id", "hermes-profile", "hermes-egress", "hermes-workspace")


def _retirement_scope(labels: dict[str, str]) -> str:
    return json.dumps([labels.get(key) for key in _SCOPE_LABELS])


@dataclass
class ReuseLease:
    token: str
    lock: _FileLock


class ReuseOwners:
    def __init__(self, home: Path):
        self.path = home / "runtime" / "docker-reuse-owners" / _DATABASE
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS containers (name TEXT PRIMARY KEY, container_id TEXT, "
                       "scope TEXT NOT NULL, profile TEXT NOT NULL, fingerprint TEXT NOT NULL, "
                       "superseded REAL, retired INTEGER NOT NULL DEFAULT 0)")
            db.execute("CREATE TABLE IF NOT EXISTS owners (token TEXT PRIMARY KEY, name TEXT NOT NULL)")
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('namespace', ?)", (uuid.uuid4().hex,))
            nonce = db.execute("SELECT value FROM metadata WHERE key='namespace'").fetchone()[0]
            identity = self.path.parent.stat()
            # A copied runtime registry is not proof of ownership on another store: a copy
            # lands on a new directory inode. st_dev is deliberately excluded — it names
            # the mount, which a remount of the same store changes along with nothing else.
            self.namespace = hashlib.sha256(
                f"{nonce}:{identity.st_ino}".encode()).hexdigest()[:32]

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        db = sqlite3.connect(self.path, timeout=35)
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _lease(self) -> ReuseLease:
        token = uuid.uuid4().hex
        lock = _FileLock(self.path.parent / (token + ".lock"), timeout=0)
        lock.__enter__()
        return ReuseLease(token, lock)

    def reserve(self, name: str, labels: dict[str, str], fingerprint: str) -> ReuseLease:
        scope = _retirement_scope(labels)
        lease = self._lease()
        try:
            with self.transaction() as db:
                db.execute("INSERT INTO containers (name, scope, profile, fingerprint) VALUES (?, ?, ?, ?)",
                           (name, scope, labels["hermes-profile"], fingerprint))
                db.execute("INSERT INTO owners VALUES (?, ?)", (lease.token, name))
        except BaseException:
            lease.lock.__exit__(None, None, None)
            raise
        return lease

    def finish(self, name: str, container_id: str) -> None:
        with self.transaction() as db:
            db.execute("UPDATE containers SET container_id=? WHERE name=?", (container_id, name))

    def claim(self, container_id: str, start: Callable[[], bool]) -> ReuseLease | None:
        lease = self._lease()
        claimed = False
        try:
            with self.transaction() as db:
                row = db.execute("SELECT name FROM containers WHERE container_id=? AND retired=0",
                                 (container_id,)).fetchone()
                if row is None or not start():
                    return None
                db.execute("INSERT INTO owners VALUES (?, ?)", (lease.token, row[0]))
                db.execute("UPDATE containers SET superseded=NULL WHERE name=?", (row[0],))
            claimed = True
            return lease
        finally:
            if not claimed:
                lease.lock.__exit__(None, None, None)

    def release(self, lease: ReuseLease) -> None:
        try:
            with self.transaction() as db:
                db.execute("DELETE FROM owners WHERE token=?", (lease.token,))
            # The lease lock file is the abandonment evidence: it may only vanish once
            # the row is committed. On a failed transaction both survive, so a retry can
            # still release (or the reaper keeps protecting) the container.
            lease.lock.path.unlink(missing_ok=True)
        finally:
            lease.lock.__exit__(None, None, None)

    def supersede(self, labels: dict[str, str], fingerprint: str) -> None:
        scope = _retirement_scope(labels)
        with self.transaction() as db:
            db.execute("UPDATE containers SET superseded=COALESCE(superseded, ?) "
                       "WHERE scope=? AND fingerprint<>? AND retired=0", (time.time(), scope, fingerprint))

    def reap(self, docker: str, max_age_seconds: int, profile: str | None) -> int:
        removed = 0
        with self.transaction() as db:
            candidates = db.execute("SELECT name FROM containers WHERE retired=0 "
                              "AND superseded<=? AND (? IS NULL OR profile=?)",
                              (time.time() - max_age_seconds, profile, profile)).fetchall()
        for (name,) in candidates:
            with ExitStack() as locks:
                # A claimant may have revived this container since the inventory.
                with self.transaction() as db:
                    row = db.execute("SELECT container_id, fingerprint FROM containers WHERE name=? "
                                     "AND retired=0 AND superseded<=?", (name, time.time() - max_age_seconds)).fetchone()
                    if row is None:
                        continue
                    if not self._owners_abandoned(db, name, locks):
                        continue
                    container_id, fingerprint = row
                # Docker waits run without a write transaction: a held BEGIN IMMEDIATE
                # would starve every other registry process for the whole stop/remove.
                if not self._remove(docker, container_id or name, name, fingerprint):
                    continue
                with self.transaction() as db:
                    # Re-check nothing changed underneath (a claim clears superseded):
                    # a container revived in between is not retired by the dead row.
                    if db.execute("SELECT 1 FROM containers WHERE name=? AND retired=0 "
                                  "AND superseded IS NOT NULL AND fingerprint=?",
                                  (name, fingerprint)).fetchone() is None:
                        continue
                    db.execute("UPDATE containers SET retired=1 WHERE name=?", (name,))
                    db.execute("DELETE FROM owners WHERE name=?", (name,))
                    removed += 1
        return removed

    def _owners_abandoned(self, db: sqlite3.Connection, name: str, locks: ExitStack) -> bool:
        for (token,) in db.execute("SELECT token FROM owners WHERE name=?", (name,)):
            path = self.path.parent / (token + ".lock")
            if not path.is_file() or path.is_symlink():
                return False
            try:
                locks.enter_context(_FileLock(path, timeout=0))
            except (OSError, RuntimeError):
                return False  # busy or unprovable is never evidence of abandonment
        return True

    def _remove(self, docker: str, target: str, name: str, fingerprint: str) -> bool:
        import subprocess

        try:
            result = run_capture([docker, "inspect", "--format", "{{json .}}", target], timeout=10, check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            # A confirmed-absent container is not a failure to preserve: its row is
            # retired so dead entries do not linger in every sweep. A daemon-unavailable
            # (or every other inspect failure) keeps the row for a later pass.
            output = exc.stderr if isinstance(exc, subprocess.SubprocessError) else str(exc)
            if "no such container" in output.lower():
                logger.info("Reaped superseded container %s (already absent)", target)
                return True
            logger.debug("Could not verify/reap superseded container %s: %s", target, exc)
            return False
        try:
            record = json.loads(result.stdout)
            labels = record["Config"]["Labels"]
            if (not isinstance(labels, dict) or not isinstance(record["Id"], str)
                    or not isinstance(record["State"]["Running"], bool)
                    or record["Name"] not in (name, "/" + name) or labels.get(OWNER_LABEL) != self.namespace
                    or labels.get("hermes-runtime-fingerprint") != fingerprint):
                return False
            container_id = record["Id"]
            strip_sandbox_privileges(docker, container_id)
            if record["State"]["Running"]:
                run_capture([docker, "stop", "--time", "10", container_id], timeout=20, check=True)
            # No -f: daemon-side concurrent restarts still protect the container.
            run_capture([docker, "rm", container_id], timeout=10, check=True)
            logger.info("Reaped superseded container %s after its last owner exited", container_id[:12])
            return True
        except (OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError) as exc:
            logger.debug("Could not verify/reap superseded container %s: %s", target, exc)
            return False


def reap_owned_containers(home: Path, docker: str, max_age_seconds: int, profile: str | None) -> int:
    if not (home / "runtime" / "docker-reuse-owners" / _DATABASE).is_file():
        return 0
    try:
        return ReuseOwners(home).reap(docker, max_age_seconds, profile)
    except (OSError, sqlite3.Error) as exc:
        logger.warning("Docker ownership records unavailable; preserving containers: %s", exc)
        return 0
