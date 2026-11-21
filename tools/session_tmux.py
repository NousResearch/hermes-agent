"""Local terminal's per-physical-session *default* tmux server.

Only this private socket is inspected; -L/-S and explicitly changed TMUX_TMPDIR
remain user-managed and are never discovered. Same-UID shell code is trusted,
not sandboxed. Sockets survive gateway restart, not machine reboot. Idle shell
classification is heuristic: shell builtins and unusual shells can be missed.
"""
from __future__ import annotations

import contextvars
import hashlib
import logging
import os
import re
import stat
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hermes_constants import get_hermes_home

_VALID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}\Z")
_PANE = re.compile(r"%[0-9]+\Z")
_SHELLS = frozenset(("sh", "bash", "zsh", "dash", "ksh", "fish", "tcsh", "csh"))


def _uid() -> int:
    return os.getuid() if hasattr(os, "getuid") else -1


def _private_dir(path: Path, create: bool = False) -> bool:
    try:
        if create:
            path.mkdir(mode=0o700, exist_ok=True)
        info = path.lstat()
        return stat.S_ISDIR(info.st_mode) and info.st_uid == _uid() and not info.st_mode & 0o077
    except OSError:
        return False


def _profile_root(*, create: bool = False) -> Path | None:
    # A deterministic, short, private root is stable across processes/restarts and
    # independent of TMPDIR (which is often long and differs between launchers).
    # Reject a squatted root rather than following it or falling back to ambient tmux.
    if os.name != "posix":
        return None
    try:
        home = os.path.normcase(str(Path(get_hermes_home()).expanduser().resolve()))
        key = hashlib.sha256(os.fsencode(home)).hexdigest()[:20]
        # no-tmp: ok — AF_UNIX sun_path must stay short and stable across gateway restarts.
        root = Path('/tmp') / f'ht-{_uid()}-{key}'
        return root if _private_dir(root, create) else None
    except (OSError, RuntimeError):
        return None


def _directory(session_id: str, *, create: bool = False) -> Path | None:
    if not isinstance(session_id, str) or not _VALID.fullmatch(session_id):
        return None
    root = _profile_root(create=create)
    if root is None:
        return None
    digest = hashlib.sha256(session_id.encode()).hexdigest()[:24]
    path = root / digest
    if not _private_dir(path, create):
        return None
    # The only reverse index is a private, validated owner file next to a
    # prospective socket. It contains no command, argv, or terminal output.
    owner = path / 'owner'
    try:
        if create and not owner.exists():
            # Publish a complete owner record atomically, even when two terminal
            # launches race for the same newly-created namespace.
            fd, temp_name = tempfile.mkstemp(prefix='.owner-', dir=path)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(session_id)
                try:
                    os.link(temp_name, owner, follow_symlinks=False)
                except FileExistsError:
                    pass
            finally:
                os.unlink(temp_name)
        info = owner.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != _uid() or info.st_mode & 0o077:
            return None
        if owner.read_text(encoding='utf-8-sig') != session_id:
            return None
    except (OSError, UnicodeError):
        return None
    return path


def _compression_root(session_id: str) -> str:
    """Resolve only a verified compression lineage, never a general parent chain.

    The reader does not create or migrate a store. Missing/unavailable persistence
    never creates a second namespace during a transient busy read.
    """
    if not isinstance(session_id, str) or not _VALID.fullmatch(session_id):
        return session_id
    home = Path(get_hermes_home())
    if not (home / 'state.db').is_file():
        return session_id
    try:
        from persistence_assembly import open_persistence_reader
        with open_persistence_reader(home, patience_s=0.25) as db:
            row = db.sessions.get(session_id)
            # The common unsplit session needs one indexed read, not a lineage walk.
            if row is None or not row.parent_session_id:
                return session_id
            if row.source == 'tool':
                root = str(db.lineage.tool_rotation_root(session_id))
                # A losing sibling is not the published continuation chosen by the chain.
                if root != session_id and str(db.lineage.tool_rotation_tip(root)) != str(db.lineage.tool_rotation_tip(session_id)):
                    return session_id
            else:
                root, seen = session_id, {session_id}
                while row and db.lineage.is_compression_child(row):
                    parent = str(row.parent_session_id)
                    if parent in seen or len(seen) >= 1000:
                        return session_id
                    seen.add(parent)
                    root, row = parent, db.sessions.get(parent)
                if root != session_id and session_id not in db.lineage.chain(root):
                    return session_id
            return root if _VALID.fullmatch(root) else session_id
    except Exception:  # health: allow BLE001 -- refuse namespace creation on a transient owner lookup failure
        logging.getLogger(__name__).debug("Cannot resolve tmux session ownership", exc_info=True)
        return ""


def session_directory(session_id: str) -> Path | None:
    """Prepare the compression-lineage-owned default-server namespace."""
    return _directory(_compression_root(session_id), create=True)


def _socket(session_id: str) -> Path | None:
    directory = _directory(_compression_root(session_id))
    if directory is None:
        return None
    socket = directory / f"tmux-{_uid()}" / "default"
    try:
        info = socket.lstat()
        parent = socket.parent.lstat()
        if (not stat.S_ISSOCK(info.st_mode) or info.st_uid != _uid()
                or not stat.S_ISDIR(parent.st_mode) or parent.st_uid != _uid()
                or parent.st_mode & 0o077):
            return None
    except OSError:
        return None
    return socket


def _run(args: list[str], timeout: float = 1.2) -> subprocess.CompletedProcess:
    return subprocess.run(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, text=True, timeout=timeout, check=False)


def _children() -> set[int] | None:
    # One bounded process-table read, never publish argv or process names from it.
    try:
        result = _run(["ps", "-A", "-o", "ppid=", "-o", "stat="], timeout=1.2)
        if result.returncode:
            return None
        return {int(parts[0]) for line in result.stdout.splitlines()
                if len(parts := line.split()) == 2 and parts[0].isdigit()
                and not parts[1].startswith("Z")}
    except (OSError, subprocess.TimeoutExpired):
        return None


def _fingerprint(socket: Path) -> str:
    info = socket.lstat()
    # Attach/detach toggles the socket mode (and thus ctime) on a live server.
    return f"{info.st_dev}:{info.st_ino}:{getattr(info, 'st_birthtime', 0)}"


def _generation(socket_identity: str, server_pid: str | int) -> str:
    return hashlib.sha256(f"{socket_identity}:{server_pid}".encode()).hexdigest()[:20]


def list_session_panes(session_id: str, *, _child_pids: set[int] | None = None, _children_known: bool = False) -> list[dict]:
    """Return process.list-shaped rows, with opaque server-generation-bound IDs.

    Idle interactive shells are reported as idle. A non-shell foreground command
    or a shell with children is running; this is best-effort, not a job monitor.
    """
    socket = _socket(session_id)
    if socket is None:
        return []
    try:
        generation = _fingerprint(socket)
        fmt = "#{pane_id}\t#{pane_pid}\t#{pane_current_command}\t#{pane_dead}\t#{session_created}\t#{pid}"
        result = _run(["tmux", "-S", str(socket), "list-panes", "-a", "-F", fmt])
        if result.returncode or generation != _fingerprint(socket):
            return []
        lines = result.stdout.splitlines()[:512]
        children = _child_pids if _children_known else _children()
        rows = []
        for line in lines:
            parts = line.split("\t")
            if len(parts) != 6:
                continue
            pane, pid, command, dead, started, server_pid = parts
            if not _PANE.fullmatch(pane) or not pid.isdigit() or not started.isdigit() or not server_pid.isdigit():
                continue
            executable = os.path.basename(command).lstrip("-")
            label = f"tmux: {executable}" if re.fullmatch(r"[A-Za-z0-9_.+-]{1,64}", executable) else "tmux pane"
            shell = executable in _SHELLS
            active = dead != "1" and (not shell or children is None or int(pid) in children)
            ts = int(started)
            rows.append({"session_id": f"tmux:{_generation(generation, server_pid)}:{pane[1:]}",
                         "command": label, "cwd": "", "pid": int(pid),
                         "owner_task_id": "", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)),
                         "uptime_seconds": max(0, int(time.time()) - ts),
                         "status": "running" if active else "idle" if dead != "1" else "exited",
                         "output_preview": "", "tmux": True, "tmux_server_pid": int(server_pid)})
        return rows
    except (OSError, subprocess.TimeoutExpired, ValueError, OverflowError):
        return []


def stop_session_pane(session_id: str, process_id: str) -> dict:
    """Stop only a currently enumerated pane in this owner's socket generation."""
    if not isinstance(process_id, str) or not re.fullmatch(r"tmux:[0-9a-f]{20}:[0-9]+", process_id):
        return {"status": "not_found"}
    row = next((row for row in list_session_panes(session_id) if row["session_id"] == process_id), None)
    if row is None:
        return {"status": "not_found"}
    socket = _socket(session_id)
    if socket is None:
        return {"status": "not_found"}
    try:
        _, generation, pane = process_id.split(":")
        if _generation(_fingerprint(socket), row['tmux_server_pid']) != generation:
            return {"status": "not_found"}
        # Check the server identity *inside* tmux before acting: replacing the
        # socket between our lstat and connect must not kill a different server.
        target = f"%{pane}"
        condition = f"#{{==:#{{pid}},{row['tmux_server_pid']}}}"
        result = _run(["tmux", "-S", str(socket), "if-shell", "-F", "-t", target,
                       condition, f"kill-pane -t {target}", "display-message -p HERMES_STALE"])
        return {"status": "killed" if result.returncode == 0 and "HERMES_STALE" not in result.stdout else "not_found", "session_id": process_id}
    except (OSError, subprocess.TimeoutExpired):
        return {"status": "not_found"}


def running_pane_owners() -> list[tuple[str, int]]:
    """Refresh only this profile's default sockets, not ambient servers or state.db.

    Directory entries without a socket are skipped without reading owner metadata
    or invoking tmux. Stale sockets and unresponsive servers contribute nothing.
    Status is heuristic; explicit -L/-S servers are outside this inventory.
    """
    root = _profile_root()
    if root is None:
        return []
    owners = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return []
    for path in entries:
        try:
            if not re.fullmatch(r'[0-9a-f]{24}', path.name) or not _private_dir(path):
                continue
            socket = path / f'tmux-{_uid()}' / 'default'
            parent = socket.parent.lstat()
            info = socket.lstat()
            if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != _uid() or parent.st_mode & 0o077
                    or not stat.S_ISSOCK(info.st_mode) or info.st_uid != _uid()):
                continue
            owner_file = path / 'owner'
            owner_info = owner_file.lstat()
            if not stat.S_ISREG(owner_info.st_mode) or owner_info.st_uid != _uid() or owner_info.st_mode & 0o077 or owner_info.st_size > 200:
                continue
            owner = owner_file.read_text(encoding='utf-8-sig')
            if _VALID.fullmatch(owner) and hashlib.sha256(owner.encode()).hexdigest()[:24] == path.name:
                owners.append(owner)
        except (OSError, UnicodeError):
            continue
    if not owners:
        return []
    children = _children()
    def count(owner: str) -> tuple[str, int]:
        rows = list_session_panes(owner, _child_pids=children, _children_known=True)
        return owner, sum(row['status'] == 'running' for row in rows)
    with ThreadPoolExecutor(max_workers=min(32, len(owners))) as pool:
        futures = [pool.submit(contextvars.copy_context().run, count, owner) for owner in owners]
        results = [future.result() for future in futures]
        active = [(owner, n) for owner, n in results if n]
    # Project durable namespace owners to the current conversation segment.
    if active and (Path(get_hermes_home()) / 'state.db').is_file():
        try:
            from persistence_assembly import open_persistence_reader
            with open_persistence_reader(Path(get_hermes_home()), patience_s=0.25) as db:
                active = [(str(db.lineage.tool_rotation_tip(db.lineage.tip(owner))), n) for owner, n in active]
        except Exception:  # health: allow BLE001 -- an unavailable store leaves the owner unprojected
            logging.getLogger(__name__).debug("Cannot project live tmux owners", exc_info=True)
    return active
