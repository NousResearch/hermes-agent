"""Filesystem signatures used by the TUI gateway change watcher."""

from pathlib import Path


def cron_signature(home: Path) -> int | None:
    """Return the cron ledger mtime when present."""
    try:
        return (home / "cron" / "jobs.json").stat().st_mtime_ns
    except OSError:
        return None


def sessions_signature(home: Path) -> int | None:
    """Return the newest SQLite database or WAL mtime."""
    signature = None
    for name in ("state.db", "state.db-wal"):
        try:
            mtime = (home / name).stat().st_mtime_ns
        except OSError:
            continue
        signature = mtime if signature is None else max(signature, mtime)
    return signature


def platforms_signature(home: Path) -> int | None:
    """Return the persisted gateway-state mtime when present."""
    try:
        return (home / "gateway_state.json").stat().st_mtime_ns
    except OSError:
        return None


def pairing_signature(home: Path) -> int | None:
    """Return the newest pending/approved pairing-ledger mtime."""
    signature = None
    roots = [home / "pairing", home / "platforms" / "pairing"]
    try:
        for profile_dir in (home / "profiles").iterdir():
            roots.append(profile_dir / "pairing")
            roots.append(profile_dir / "platforms" / "pairing")
    except OSError:
        pass

    for root in roots:
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.name.endswith(("-pending.json", "-approved.json")):
                continue
            try:
                mtime = entry.stat().st_mtime_ns
            except OSError:
                continue
            signature = mtime if signature is None else max(signature, mtime)
    return signature


def desktop_room_mailbox_signature(home: Path) -> int | None:
    """Return the cross-process Desktop-room command mailbox mtime."""
    root = home.parent.parent if home.parent.name == "profiles" else home
    try:
        return (root / "desktop_room_mailbox.pending").stat().st_mtime_ns
    except OSError:
        return None
