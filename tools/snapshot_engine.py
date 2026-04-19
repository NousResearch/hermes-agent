"""
Runtime Snapshot Engine — Self-versioning state snapshots for Hermes.

Content-addressed storage (like git-lite) for the internal state of ~/.hermes.
Snapshots are deduplicated, debounced, and fully independent of git.

Key concepts:
  - objects/: SHA256-addressed blob store (deduplication)
  - snapshots/: Per-snapshot manifest.json + meta.json
  - history.db: SQLite index for fast queries
  - wal/: Write-ahead log for zero-loss between snapshots
  - Debounced auto-snapshots (30s default, 15s for memory writes)
  - Branching for safe upgrade experiments

Usage:
  from tools.snapshot_engine import SnapshotEngine, auto_snapshot, safe_run

  engine = SnapshotEngine()
  snap_id = engine.snapshot(label="pre-upgrade")

  # Restore
  engine.restore(snap_id)

  # Auto (debounced)
  auto_snapshot(trigger="memory_write")

  # Transaction
  result = safe_run(my_function, label="dangerous-op")

  # Branching
  engine.create_branch("pre-update-v2")
  engine.switch_branch("pre-update-v2")
  engine.list_branches()
"""

import hashlib
import json
import logging
import shutil
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Directories/files to include in snapshots (relative to hermes_home)
INCLUDE_FILES: Set[str] = {
    "state.db",
    "config.yaml",
    "auth.json",
    "cron/jobs.json",
    "gateway_state.json",
    "channel_directory.json",
    "processes.json",
}

# Patterns to exclude from ALL walks
EXCLUDE_PATTERNS: Set[str] = {
    ".snapshots",
    "hermes-agent",
    "venv",
    ".venv",
    "checkpoints",
    "sessions",
    "logs",
    "cache",
    "skills",
    "optional-skills",
    "node_modules",
    "__pycache__",
    ".git",
    "broken",
    "pastes",
    "temp_vision_images",
    "tests",
    "website",
    "bin",
    "acp_adapter",
    "acp_registry",
    "agent",
    "assets",
    "datagen-config-examples",
    "docker",
    "docs",
    "environments",
    "gateway",
    "hermelinChat",
    "hermes_cli",
    "honcho_integration",
    "landingpage",
    "nix",
    "packaging",
    "plans",
    ".plans",
    "plugins",
    "scripts",
    "tinker-atropos",
    "tools",
}

# SQLite WAL/SHM files — never snapshot these directly
SQLITE_WAL_PATTERNS = {".db-wal", ".db-shm"}

# Debounce windows (seconds)
_DEBOUNCE_DEFAULT = 30
_DEBOUNCE_MEMORY = 15

# ---------------------------------------------------------------------------
# Module-level state for debouncing
# ---------------------------------------------------------------------------
_last_snapshot_time: float = 0
_last_state_hash: str = ""
_debounce_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Core Engine
# ---------------------------------------------------------------------------

def _update_state_hash(h: str) -> None:
    """Update the module-level state hash (must be called from within SnapshotEngine)."""
    global _last_state_hash
    _last_state_hash = h


class SnapshotEngine:
    """Content-addressed snapshot engine for Hermes runtime state."""

    def __init__(self, hermes_home: Optional[Path] = None):
        self.hermes_home = hermes_home or get_hermes_home()
        self.store = self.hermes_home / ".snapshots"
        self.objects = self.store / "objects"
        self.snapshots_dir = self.store / "snapshots"
        self.head_file = self.store / "HEAD"
        self.history_db = self.store / "history.db"
        self.wal_dir = self.store / "wal"
        self.branches_dir = self.store / "branches"
        self.branch_head = self.store / "BRANCH_HEAD"

        # Ensure directories exist
        self.objects.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.wal_dir.mkdir(parents=True, exist_ok=True)
        self.branches_dir.mkdir(parents=True, exist_ok=True)

        # Initialize history DB
        self._init_history_db()

        # Initialize default branch if none exists
        if not self.branch_head.exists():
            self.branch_head.write_text("main")

    def _init_history_db(self) -> None:
        """Create the snapshot index database."""
        conn = sqlite3.connect(str(self.history_db))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                label TEXT,
                trigger TEXT,
                file_count INTEGER,
                total_size INTEGER,
                unique_objects INTEGER,
                created_at REAL NOT NULL,
                branch TEXT DEFAULT 'main'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS snapshot_files (
                snapshot_id TEXT NOT NULL,
                rel_path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size INTEGER NOT NULL,
                PRIMARY KEY (snapshot_id, rel_path),
                FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_snapshots_time
            ON snapshots(timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_snapshots_branch
            ON snapshots(branch)
        """)
        # WAL table — append-only log of individual state changes between snapshots
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                rel_path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size INTEGER NOT NULL,
                branch TEXT NOT NULL DEFAULT 'main',
                snapshot_id TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_wal_unflushed
            ON wal_entries(snapshot_id) WHERE snapshot_id IS NULL
        """)
        conn.commit()
        conn.close()

    # -- File hashing --------------------------------------------------------

    @staticmethod
    def _hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _hash_file(self, path: Path) -> str:
        """Hash a file's contents (SHA-256)."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    # -- Object store --------------------------------------------------------

    def _store_object(self, data: bytes) -> str:
        """Store bytes in the content-addressed object store. Returns SHA-256."""
        sha = self._hash_bytes(data)
        obj_path = self.objects / sha[:2] / sha[2:]

        if not obj_path.exists():
            obj_path.parent.mkdir(parents=True, exist_ok=True)
            obj_path.write_bytes(data)

        return sha

    def _store_file_object(self, path: Path) -> str:
        """Store a file's contents as an object. Returns SHA-256."""
        data = path.read_bytes()
        return self._store_object(data)

    def _get_object(self, sha: str) -> Optional[bytes]:
        """Retrieve object bytes by SHA-256 hash."""
        obj_path = self.objects / sha[:2] / sha[2:]
        if obj_path.exists():
            return obj_path.read_bytes()
        return None

    # -- Write-Ahead Log -----------------------------------------------------

    def wal_append(self, rel_path: str, data: bytes) -> None:
        """Append a state change to the WAL.

        Call this for every individual state mutation (not just on snapshot).
        This ensures zero data loss between debounced snapshots.
        """
        sha = self._store_object(data)
        branch = self.get_branch()
        try:
            conn = sqlite3.connect(str(self.history_db))
            conn.execute(
                "INSERT INTO wal_entries (timestamp, rel_path, sha256, size, branch) VALUES (?, ?, ?, ?, ?)",
                (time.time(), rel_path, sha, len(data), branch),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("WAL append failed for %s: %s", rel_path, e)

    def wal_append_file(self, rel_path: str, abs_path: Path) -> None:
        """Append a file's contents to the WAL."""
        try:
            if abs_path.name == "state.db":
                # Safe copy for SQLite
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                if self._safe_copy_db(abs_path, tmp_path):
                    self.wal_append(rel_path, tmp_path.read_bytes())
                    tmp_path.unlink(missing_ok=True)
            else:
                self.wal_append(rel_path, abs_path.read_bytes())
        except Exception as e:
            logger.warning("WAL append_file failed for %s: %s", rel_path, e)

    def wal_flush(self, snapshot_id: str) -> int:
        """Mark all unflushed WAL entries as belonging to a snapshot.

        Returns count of flushed entries.
        """
        try:
            conn = sqlite3.connect(str(self.history_db))
            cur = conn.execute(
                "UPDATE wal_entries SET snapshot_id = ? WHERE snapshot_id IS NULL",
                (snapshot_id,),
            )
            count = cur.rowcount
            conn.commit()
            conn.close()
            return count
        except Exception as e:
            logger.warning("WAL flush failed: %s", e)
            return 0

    def wal_unflushed(self) -> List[Dict[str, Any]]:
        """Get all unflushed WAL entries (changes since last snapshot)."""
        try:
            conn = sqlite3.connect(str(self.history_db))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM wal_entries WHERE snapshot_id IS NULL ORDER BY id"
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def wal_replay(self) -> List[str]:
        """Replay unflushed WAL entries into the live state.

        Used on startup after a crash — restores state changes that happened
        between the last snapshot and the crash.

        Returns list of restored file paths.
        """
        entries = self.wal_unflushed()
        if not entries:
            return []

        # Deduplicate: keep only the latest version of each file
        latest: Dict[str, str] = {}  # rel_path -> sha256
        for e in entries:
            latest[e["rel_path"]] = e["sha256"]

        restored = []
        for rel_path, sha in latest.items():
            obj_path = self.objects / sha[:2] / sha[2:]
            if not obj_path.exists():
                continue
            target = self.hermes_home / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                if target.name == "state.db":
                    # Remove old WAL/SHM files FIRST — they belong to the old
                    # (possibly corrupted) DB and would corrupt the restored one.
                    for suffix in ("-wal", "-shm"):
                        old_aux = Path(str(target) + suffix)
                        old_aux.unlink(missing_ok=True)

                    tmp_path = target.parent / f".{target.name}.wal_replay"
                    shutil.copy2(obj_path, tmp_path)
                    target.unlink(missing_ok=True)
                    shutil.move(str(tmp_path), str(target))
                else:
                    shutil.copy2(obj_path, target)
                restored.append(rel_path)
            except Exception as e:
                logger.error("WAL replay failed for %s: %s", rel_path, e)

        logger.info("WAL replay: restored %d files from %d entries", len(restored), len(entries))
        return restored

    def wal_prune(self, older_than_hours: int = 72) -> int:
        """Remove flushed WAL entries older than N hours. Returns count removed."""
        cutoff = time.time() - (older_than_hours * 3600)
        try:
            conn = sqlite3.connect(str(self.history_db))
            cur = conn.execute(
                "DELETE FROM wal_entries WHERE snapshot_id IS NOT NULL AND timestamp < ?",
                (cutoff,),
            )
            count = cur.rowcount
            conn.commit()
            conn.close()
            return count
        except Exception:
            return 0

    # -- Branching -----------------------------------------------------------

    def get_branch(self) -> str:
        """Get the current branch name."""
        try:
            return self.branch_head.read_text().strip()
        except (FileNotFoundError, OSError):
            return "main"

    def create_branch(self, name: str, from_snapshot: Optional[str] = None) -> bool:
        """Create a new branch.

        Args:
            name: Branch name (alphanumeric + hyphens/underscores)
            from_snapshot: Optional snapshot to branch from (default: current HEAD)

        Returns:
            True if branch created successfully.
        """
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            logger.error("Invalid branch name: %s", name)
            return False

        branch_file = self.branches_dir / name
        if branch_file.exists():
            logger.error("Branch already exists: %s", name)
            return False

        # Determine the snapshot to point at
        if from_snapshot:
            snap_id = from_snapshot
        else:
            snap_id = self.get_head()

        # Write branch ref (points to a snapshot ID)
        branch_file.write_text(snap_id or "")
        logger.info("Created branch '%s' at %s", name, snap_id)
        return True

    def switch_branch(self, name: str) -> bool:
        """Switch to a branch. Restores state from the branch's HEAD snapshot.

        Args:
            name: Branch name to switch to.

        Returns:
            True if switched successfully.
        """
        branch_file = self.branches_dir / name
        if not branch_file.exists():
            logger.error("Branch not found: %s", name)
            return False

        snap_id = branch_file.read_text().strip()

        # Snapshot current state on current branch before switching
        current_branch = self.get_branch()
        if current_branch != name:
            self.snapshot(label=f"branch-switch-from-{current_branch}", trigger="branch_switch")

        # Switch branch
        self.branch_head.write_text(name)

        # Restore branch state if it has a snapshot
        if snap_id:
            self.restore(snap_id)

        logger.info("Switched to branch '%s' (snapshot: %s)", name, snap_id or "none")
        return True

    def delete_branch(self, name: str) -> bool:
        """Delete a branch. Cannot delete the current branch or 'main'.

        Args:
            name: Branch name to delete.

        Returns:
            True if deleted.
        """
        if name == "main":
            logger.error("Cannot delete 'main' branch")
            return False
        if name == self.get_branch():
            logger.error("Cannot delete active branch — switch first")
            return False

        branch_file = self.branches_dir / name
        if not branch_file.exists():
            logger.error("Branch not found: %s", name)
            return False

        branch_file.unlink()
        logger.info("Deleted branch '%s'", name)
        return True

    def list_branches(self) -> List[Dict[str, Any]]:
        """List all branches with their HEAD snapshot and status."""
        current = self.get_branch()
        branches = []

        # Always include main
        main_snap = ""
        main_file = self.branches_dir / "main"
        if main_file.exists():
            main_snap = main_file.read_text().strip()
        branches.append({
            "name": "main",
            "head_snapshot": main_snap,
            "is_current": current == "main",
        })

        # Other branches
        if self.branches_dir.exists():
            for bf in sorted(self.branches_dir.iterdir()):
                if bf.is_file() and bf.name != "main":
                    branches.append({
                        "name": bf.name,
                        "head_snapshot": bf.read_text().strip(),
                        "is_current": current == bf.name,
                    })

        return branches

    def update_branch_head(self, snap_id: str) -> None:
        """Update the current branch's HEAD to the given snapshot."""
        branch = self.get_branch()
        branch_file = self.branches_dir / branch
        branch_file.parent.mkdir(parents=True, exist_ok=True)
        branch_file.write_text(snap_id)

    # -- SQLite safe copy ----------------------------------------------------

    def _safe_copy_db(self, src: Path, dst: Path) -> bool:
        """Copy a SQLite database safely using the backup API.

        Handles WAL mode — produces a consistent snapshot even while
        the DB is being written to by the agent.

        Strategy:
          1. Open src read-write (needed for WAL checkpoint)
          2. PRAGMA wal_checkpoint(TRUNCATE) — flushes WAL into main DB
          3. sqlite3.Connection.backup() — consistent copy
          4. Verify backup integrity before accepting
        """
        conn = None
        backup_conn = None
        try:
            # Step 1: Open read-write (checkpoint requires write access)
            conn = sqlite3.connect(str(src), timeout=10)

            # Step 2: WAL checkpoint — flush pending writes into main DB file.
            # TRUNCATE ensures the WAL is emptied, so the .db file alone is
            # sufficient for a consistent snapshot.
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.OperationalError:
                # Checkpoint may fail if another writer holds the WAL.
                # backup() can still handle this, so continue.
                pass

            # Step 3: SQLite backup API — the ONLY safe way to copy a live DB.
            backup_conn = sqlite3.connect(str(dst))
            conn.backup(backup_conn)

            # Step 4: Verify backup integrity before accepting.
            # A corrupt backup is worse than no backup.
            integrity = backup_conn.execute("PRAGMA integrity_check").fetchone()[0]
            backup_conn.close()
            backup_conn = None
            conn.close()
            conn = None

            if integrity != "ok":
                logger.error("Backup integrity check failed for %s: %s", src, integrity)
                dst.unlink(missing_ok=True)
                return False

            return True
        except Exception as e:
            logger.error("SQLite safe copy failed for %s: %s", src, e)
            dst.unlink(missing_ok=True)
            return False
        finally:
            # Always clean up connections
            if backup_conn is not None:
                try:
                    backup_conn.close()
                except Exception:
                    pass
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    # -- Snapshot creation ---------------------------------------------------

    def _collect_state_files(self) -> List[Tuple[str, Path]]:
        """Collect all files to snapshot. Returns [(rel_path, abs_path)]."""
        files = []
        for rel_name in sorted(INCLUDE_FILES):
            abs_path = self.hermes_home / rel_name
            if abs_path.exists() and abs_path.is_file():
                # Skip WAL files
                if any(str(abs_path).endswith(p) for p in SQLITE_WAL_PATTERNS):
                    continue
                files.append((rel_name, abs_path))
        return files

    def compute_state_hash(self) -> str:
        """Compute a hash of the current state for change detection.

        Uses modification times + file sizes (fast, no I/O on large files).
        """
        parts = []
        for rel_path, abs_path in self._collect_state_files():
            try:
                stat = abs_path.stat()
                parts.append(f"{rel_path}:{stat.st_mtime_ns}:{stat.st_size}")
            except OSError:
                parts.append(f"{rel_path}:missing")
        return self._hash_bytes("|".join(parts).encode())

    def snapshot(
        self,
        label: Optional[str] = None,
        trigger: str = "manual",
    ) -> Optional[str]:
        """Create a new snapshot of the current state.

        Args:
            label: Optional human-readable label
            trigger: What triggered this snapshot (manual, memory_write, tool_result, agent_step, pre-run)

        Returns:
            Snapshot ID (timestamp-based), or None if nothing changed.
        """
        state_hash = self.compute_state_hash()

        # Skip if state hasn't changed since last snapshot
        head = self.get_head()
        if head:
            try:
                conn = sqlite3.connect(str(self.history_db))
                row = conn.execute(
                    "SELECT 1 FROM snapshot_files WHERE snapshot_id = ? LIMIT 1",
                    (head,),
                ).fetchone()
                conn.close()
                # If head exists and state hash matches, skip
                if row and state_hash == _last_state_hash:
                    logger.debug("State unchanged, skipping snapshot")
                    return None
            except Exception:
                pass

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        snap_id = f"{ts}-{label}" if label else ts

        snap_dir = self.snapshots_dir / snap_id
        snap_dir.mkdir(parents=True, exist_ok=True)

        manifest: Dict[str, str] = {}
        file_sizes: Dict[str, int] = {}
        total_size = 0
        unique_hashes: Set[str] = set()

        # Collect and store files
        for rel_path, abs_path in self._collect_state_files():
            try:
                # Special handling for SQLite databases
                if abs_path.suffix == ".db" and abs_path.name == "state.db":
                    tmp_path = snap_dir / "_tmp_db"
                    if self._safe_copy_db(abs_path, tmp_path):
                        sha = self._store_file_object(tmp_path)
                        size = tmp_path.stat().st_size
                        tmp_path.unlink(missing_ok=True)
                    else:
                        continue
                else:
                    sha = self._store_file_object(abs_path)
                    size = abs_path.stat().st_size

                manifest[rel_path] = sha
                file_sizes[rel_path] = size
                total_size += size
                unique_hashes.add(sha)

            except (OSError, PermissionError) as e:
                logger.warning("Could not snapshot %s: %s", rel_path, e)
                continue

        if not manifest:
            logger.warning("No files to snapshot")
            shutil.rmtree(snap_dir, ignore_errors=True)
            return None

        # Write manifest
        with open(snap_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Write metadata
        branch = self.get_branch()
        meta = {
            "id": snap_id,
            "timestamp": ts,
            "label": label,
            "trigger": trigger,
            "file_count": len(manifest),
            "total_size": total_size,
            "unique_objects": len(unique_hashes),
            "state_hash": state_hash,
            "branch": branch,
        }
        with open(snap_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Update HEAD
        self.head_file.write_text(snap_id)

        # Update branch HEAD
        self.update_branch_head(snap_id)

        # WAL flush — mark unflushed entries as belonging to this snapshot
        wal_count = self.wal_flush(snap_id)

        # Update history DB
        try:
            conn = sqlite3.connect(str(self.history_db))
            conn.execute(
                "INSERT OR REPLACE INTO snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (snap_id, ts, label, trigger, len(manifest), total_size,
                 len(unique_hashes), time.time(), branch),
            )
            for rel_path, sha in manifest.items():
                conn.execute(
                    "INSERT OR REPLACE INTO snapshot_files VALUES (?, ?, ?, ?)",
                    (snap_id, rel_path, sha, file_sizes.get(rel_path, 0)),
                )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("Failed to update history DB: %s", e)

        # Update module-level debounce state
        _update_state_hash(state_hash)

        logger.info(
            "Snapshot %s: %d files, %d unique objects, %s trigger, %d WAL flushed",
            snap_id, len(manifest), len(unique_hashes), trigger, wal_count,
        )
        return snap_id

    # -- Restore -------------------------------------------------------------

    def restore(self, snapshot_id: str) -> bool:
        """Restore state from a snapshot.

        Args:
            snapshot_id: The snapshot ID to restore from.

        Returns:
            True if restore succeeded.
        """
        snap_dir = self.snapshots_dir / snapshot_id
        if not snap_dir.exists():
            logger.error("Snapshot not found: %s", snapshot_id)
            return False

        manifest_path = snap_dir / "manifest.json"
        if not manifest_path.exists():
            logger.error("No manifest in snapshot: %s", snapshot_id)
            return False

        with open(manifest_path) as f:
            manifest: Dict[str, str] = json.load(f)

        restored = 0
        for rel_path, sha in manifest.items():
            obj_path = self.objects / sha[:2] / sha[2:]
            if not obj_path.exists():
                logger.error("Object not found: %s (for %s)", sha, rel_path)
                continue

            target = self.hermes_home / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)

            try:
                # For state.db, use the safe copy approach (write object to tmp, then restore)
                if target.name == "state.db":
                    # Remove old WAL/SHM files FIRST — they belong to the old
                    # (possibly corrupted) DB and would corrupt the restored one.
                    for suffix in ("-wal", "-shm"):
                        old_aux = Path(str(target) + suffix)
                        old_aux.unlink(missing_ok=True)

                    tmp_path = target.parent / f".{target.name}.snap_restore"
                    shutil.copy2(obj_path, tmp_path)
                    # Replace the live DB
                    target.unlink(missing_ok=True)
                    shutil.move(str(tmp_path), str(target))
                else:
                    shutil.copy2(obj_path, target)
                restored += 1
            except Exception as e:
                logger.error("Failed to restore %s: %s", rel_path, e)

        # Update HEAD
        self.head_file.write_text(snapshot_id)

        logger.info("Restored %d/%d files from snapshot %s", restored, len(manifest), snapshot_id)
        return restored > 0

    # -- List ----------------------------------------------------------------

    def get_head(self) -> Optional[str]:
        """Get the current HEAD snapshot ID."""
        try:
            return self.head_file.read_text().strip()
        except (FileNotFoundError, OSError):
            return None

    def list_snapshots(self, limit: int = 50, branch: Optional[str] = None) -> List[Dict[str, Any]]:
        """List recent snapshots with metadata.

        Args:
            limit: Max snapshots to return.
            branch: Filter by branch name (None = all branches).

        Returns most recent first.
        """
        try:
            conn = sqlite3.connect(str(self.history_db))
            conn.row_factory = sqlite3.Row
            if branch:
                rows = conn.execute(
                    "SELECT * FROM snapshots WHERE branch = ? ORDER BY timestamp DESC LIMIT ?",
                    (branch, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM snapshots ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            # Fallback: scan filesystem
            result = []
            if self.snapshots_dir.exists():
                for d in sorted(self.snapshots_dir.iterdir(), reverse=True):
                    if d.is_dir():
                        meta_path = d / "meta.json"
                        if meta_path.exists():
                            try:
                                with open(meta_path) as f:
                                    result.append(json.load(f))
                            except Exception:
                                pass
            return result[:limit]

    # -- Diff ----------------------------------------------------------------

    def diff(self, snap_id_a: str, snap_id_b: str) -> Dict[str, Any]:
        """Compare two snapshots. Returns changed/added/removed files."""
        def _load_manifest(sid: str) -> Dict[str, str]:
            path = self.snapshots_dir / sid / "manifest.json"
            if path.exists():
                with open(path) as f:
                    return json.load(f)
            return {}

        ma = _load_manifest(snap_id_a)
        mb = _load_manifest(snap_id_b)

        all_paths = set(ma.keys()) | set(mb.keys())
        changed = []
        added = []
        removed = []

        for p in sorted(all_paths):
            ha = ma.get(p)
            hb = mb.get(p)
            if ha and hb and ha != hb:
                changed.append(p)
            elif not ha and hb:
                added.append(p)
            elif ha and not hb:
                removed.append(p)

        return {
            "snapshot_a": snap_id_a,
            "snapshot_b": snap_id_b,
            "changed": changed,
            "added": added,
            "removed": removed,
        }

    # -- Prune ---------------------------------------------------------------

    def prune(
        self,
        keep_last: int = 100,
        keep_hourly: int = 24,
        keep_daily: int = 3,
    ) -> int:
        """Prune old snapshots. Returns count of deleted snapshots.

        Strategy:
          - Always keep the last `keep_last` snapshots
          - Keep one per hour for the last `keep_hourly` hours
          - Keep one per day for the last `keep_daily` days (default: 3)
          - NEVER prune snapshots on non-main branches (safety for upgrade experiments)
          - Delete the rest
        """
        snaps = self.list_snapshots(limit=10000)
        if len(snaps) <= keep_last:
            return 0

        keep: Set[str] = set()

        # Always keep last N
        for s in snaps[:keep_last]:
            keep.add(s["id"])

        # Keep hourly (one per hour)
        seen_hours: Set[str] = set()
        for s in snaps:
            ts = s.get("timestamp", "")
            hour_key = ts[:10] + ts[10:13]  # YYYYMMDD-HH
            if hour_key not in seen_hours and len(seen_hours) < keep_hourly:
                keep.add(s["id"])
                seen_hours.add(hour_key)

        # Keep daily (one per day)
        seen_days: Set[str] = set()
        for s in snaps:
            ts = s.get("timestamp", "")
            day_key = ts[:8]  # YYYYMMDD
            if day_key not in seen_days and len(seen_days) < keep_daily:
                keep.add(s["id"])
                seen_days.add(day_key)

        # Delete everything not in keep set
        # NEVER delete snapshots on non-main branches (safety for upgrade experiments)
        non_main_snaps: Set[str] = set()
        for s in snaps:
            if s.get("branch", "main") != "main":
                non_main_snaps.add(s["id"])

        deleted = 0
        for s in snaps:
            if s["id"] not in keep and s["id"] not in non_main_snaps:
                snap_dir = self.snapshots_dir / s["id"]
                if snap_dir.exists():
                    shutil.rmtree(snap_dir, ignore_errors=True)
                    deleted += 1

        # Clean up history DB
        if deleted:
            try:
                conn = sqlite3.connect(str(self.history_db))
                for s in snaps:
                    if s["id"] not in keep and s["id"] not in non_main_snaps:
                        conn.execute("DELETE FROM snapshot_files WHERE snapshot_id = ?", (s["id"],))
                        conn.execute("DELETE FROM snapshots WHERE id = ?", (s["id"],))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.warning("Failed to clean history DB during prune: %s", e)

        # Clean orphaned objects (optional — run occasionally)
        self._clean_orphaned_objects()

        # Also prune old WAL entries (flushed, >72h)
        wal_pruned = self.wal_prune(older_than_hours=72)

        logger.info(
            "Pruned %d snapshots (kept %d, protected %d branch snaps), %d WAL entries",
            deleted, len(keep), len(non_main_snaps), wal_pruned,
        )
        return deleted

    def _clean_orphaned_objects(self) -> int:
        """Remove objects not referenced by any snapshot. Returns count removed."""
        # Collect all referenced hashes
        referenced: Set[str] = set()
        try:
            conn = sqlite3.connect(str(self.history_db))
            rows = conn.execute("SELECT DISTINCT sha256 FROM snapshot_files").fetchall()
            conn.close()
            referenced = {r[0] for r in rows}
        except Exception:
            return 0

        removed = 0
        if self.objects.exists():
            for prefix_dir in self.objects.iterdir():
                if not prefix_dir.is_dir():
                    continue
                for obj_file in prefix_dir.iterdir():
                    full_hash = prefix_dir.name + obj_file.name
                    if full_hash not in referenced:
                        obj_file.unlink(missing_ok=True)
                        removed += 1
                        # Clean empty prefix dirs
                        try:
                            prefix_dir.rmdir()
                        except OSError:
                            pass

        return removed


# ---------------------------------------------------------------------------
# Convenience functions (module-level, no engine instance needed)
# ---------------------------------------------------------------------------

_engine: Optional[SnapshotEngine] = None
_engine_lock = threading.Lock()


def _get_engine() -> SnapshotEngine:
    """Get or create the global snapshot engine instance."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = SnapshotEngine()
    return _engine


def auto_snapshot(
    debounce_seconds: Optional[float] = None,
    label: Optional[str] = None,
    trigger: str = "auto",
) -> Optional[str]:
    """Create a debounced auto-snapshot if state has changed.

    Uses module-level state for debouncing across calls.
    Returns snapshot ID or None if skipped.
    """
    global _last_snapshot_time

    # Select debounce window based on trigger
    if debounce_seconds is None:
        debounce_seconds = (
            _DEBOUNCE_MEMORY if trigger == "memory_write" else _DEBOUNCE_DEFAULT
        )

    now = time.time()

    with _debounce_lock:
        if now - _last_snapshot_time < debounce_seconds:
            return None
        _last_snapshot_time = now

    try:
        engine = _get_engine()
        return engine.snapshot(label=label, trigger=trigger)
    except Exception as e:
        logger.warning("Auto-snapshot failed: %s", e)
        return None


def safe_run(fn, label: str = "safe-run") -> Any:
    """Execute fn() with automatic snapshot + rollback on failure.

    Creates a pre-execution snapshot. If fn() raises, restores to that snapshot.

    Usage:
        result = safe_run(lambda: risky_operation(), label="upgrade-config")
    """
    engine = _get_engine()
    snap_id = engine.snapshot(label=f"pre-{label}", trigger="pre-run")

    try:
        result = fn()
        return result
    except Exception as e:
        if snap_id:
            logger.warning("Rolling back to snapshot %s after error: %s", snap_id, e)
            engine.restore(snap_id)
        raise


def snapshot(label: Optional[str] = None, trigger: str = "manual") -> Optional[str]:
    """Create a snapshot. Shorthand for engine.snapshot()."""
    return _get_engine().snapshot(label=label, trigger=trigger)


def restore(snapshot_id: str) -> bool:
    """Restore from a snapshot. Shorthand for engine.restore()."""
    return _get_engine().restore(snapshot_id)


def list_snapshots(limit: int = 50) -> List[Dict[str, Any]]:
    """List snapshots. Shorthand for engine.list_snapshots()."""
    return _get_engine().list_snapshots(limit=limit)


def get_head() -> Optional[str]:
    """Get HEAD snapshot ID. Shorthand for engine.get_head()."""
    return _get_engine().get_head()


def diff(snap_id_a: str, snap_id_b: str) -> Dict[str, Any]:
    """Diff two snapshots. Shorthand for engine.diff()."""
    return _get_engine().diff(snap_id_a, snap_id_b)


def prune(keep_last: int = 100, keep_hourly: int = 24, keep_daily: int = 3) -> int:
    """Prune old snapshots. Shorthand for engine.prune()."""
    return _get_engine().prune(keep_last=keep_last, keep_hourly=keep_hourly, keep_daily=keep_daily)


def wal_append(rel_path: str, data: bytes) -> None:
    """Append to the WAL. Shorthand for engine.wal_append()."""
    _get_engine().wal_append(rel_path, data)


def wal_append_file(rel_path: str, abs_path: Path) -> None:
    """Append a file to the WAL. Shorthand for engine.wal_append_file()."""
    _get_engine().wal_append_file(rel_path, abs_path)


def wal_replay() -> List[str]:
    """Replay unflushed WAL. Shorthand for engine.wal_replay()."""
    return _get_engine().wal_replay()


def wal_unflushed() -> List[Dict[str, Any]]:
    """Get unflushed WAL entries. Shorthand for engine.wal_unflushed()."""
    return _get_engine().wal_unflushed()


def create_branch(name: str, from_snapshot: Optional[str] = None) -> bool:
    """Create a branch. Shorthand for engine.create_branch()."""
    return _get_engine().create_branch(name, from_snapshot=from_snapshot)


def switch_branch(name: str) -> bool:
    """Switch branch. Shorthand for engine.switch_branch()."""
    return _get_engine().switch_branch(name)


def delete_branch(name: str) -> bool:
    """Delete a branch. Shorthand for engine.delete_branch()."""
    return _get_engine().delete_branch(name)


def list_branches() -> List[Dict[str, Any]]:
    """List branches. Shorthand for engine.list_branches()."""
    return _get_engine().list_branches()


def get_branch() -> str:
    """Get current branch. Shorthand for engine.get_branch()."""
    return _get_engine().get_branch()
