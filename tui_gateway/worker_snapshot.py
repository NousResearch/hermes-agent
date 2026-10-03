"""Read-side worker recovery. Never interpret an empty non-owner registry as zero."""
from __future__ import annotations

import itertools
import threading
from pathlib import Path

from tools import worker_roster

_sequences = itertools.count(1)
_snapshot_lock = threading.Lock()


def snapshot(server, owner):
    # Order concurrent snapshots by observation, not by response delivery.
    with _snapshot_lock:
        return _snapshot(server, owner)


def _snapshot(server, owner):
    home = owner.get("profile_home")
    key = owner.get("session_key")
    if not home or not key:
        raise ValueError("worker snapshot requires a profile-bound stored session")
    # Open this profile explicitly, not the agent's possibly rebuilt/absent DB.
    from hermes_state import SessionDB
    db = SessionDB(db_path=Path(home) / "state.db")
    try:
        keys = db.get_compression_lineage(key)
    finally:
        db.close()
    if not keys:
        raise ValueError("stored session not found in owning profile")
    observations = worker_roster.local_observations(home, keys)
    # Query the existing host, never start one merely to answer a read. The
    # qualified scope crosses the pipe; no UI/runtime ID lookup in the child.
    supervisor = getattr(server, "_compute_host_supervisor", None)
    owner_available = True
    if supervisor is not None:
        try:
            reply = supervisor.worker_observations(str(Path(home).resolve()), keys)
            if reply.get("type") != "workers.ack":
                raise ValueError("owner did not acknowledge worker snapshot")
            observations.update(reply["observations"])
        except Exception:
            owner_available = False
    elif server._session_uses_compute_host(owner):
        owner_available = False
    rows = worker_roster.observe(home, keys, observations)
    return {
        "schema_version": 1, "snapshot_epoch": worker_roster.OWNER_ID,
        "snapshot_seq": next(_sequences), "session_key": key,
        "scope": "native_delegate_task", "coverage": "admitted_since_upgrade",
        "owner_available": owner_available, "workers": rows,
    }
