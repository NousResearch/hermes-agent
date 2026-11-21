"""Best-effort teardown of a foreground command's Git Bash process group."""

import logging
import os
from pathlib import Path
import subprocess

from hermes_cli._subprocess_compat import windows_hide_flags

logger = logging.getLogger(__name__)


def _command_pgid(output: str, winpid_depths: dict[int, int], own_winpid: int) -> int | None:
    """Select the nearest unambiguous group leader in the command's Windows tree."""
    lines = [line.split() for line in output.splitlines() if line.strip()]
    if not lines or own_winpid in winpid_depths:
        return None
    header, *body = lines
    columns = ("PID", "PPID", "PGID", "WINPID", "COMMAND")
    if any(header.count(name) != 1 for name in columns):
        return None
    indices = [header.index(name) for name in columns[:-1]]
    rows = []
    for fields in body:
        if fields[0] in ("I", "O", "S"):
            fields = fields[1:]
        if len(fields) < len(header):
            return None
        try:
            rows.append(tuple(int(fields[index]) for index in indices))
        except (IndexError, ValueError):
            return None
    # Native Windows rows (including bin/bash.exe) have no MSYS group.
    leaders = {group: winpid_depths[owner] for pid, _, group, owner in rows
               if pid == group and group > 1 and owner in winpid_depths}
    if not leaders or len({row[0] for row in rows}) != len(rows):
        return None
    nearest_depth = min(leaders.values())
    groups = {group for group, depth in leaders.items() if depth == nearest_depth}
    if len(groups) != 1:
        return None
    pgid = groups.pop()
    if any(owner == own_winpid and group == pgid for _, _, group, owner in rows):
        return None
    # Orphaned pipeline members retain their PGID after either parent tree breaks.
    return pgid


def find_msys_process_group(proc) -> tuple[Path, int] | None:
    """Snapshot a safe command group while the Windows launcher is still alive."""
    try:
        import psutil

        if proc.poll() is not None:
            return
        # bin/bash.exe is a native launcher; an exec'd or nested shell may lead
        # the MSYS group deeper in its Windows tree.
        children = psutil.Process(proc.pid).children(recursive=True)
        parents = {child.pid: child.ppid() for child in children}
        create_times = {child.pid: child.create_time() for child in children}
        winpid_depths = {proc.pid: 0}
        while parents:
            level = {pid: winpid_depths[parent] + 1 for pid, parent in parents.items() if parent in winpid_depths}
            if not level:
                break
            winpid_depths.update(level)
            parents = {pid: parent for pid, parent in parents.items() if pid not in level}
        # Popen.poll uses the original Windows process handle, so a recycled PID
        # cannot make a different process (or its children) own this command.
        if proc.poll() is not None:
            return
        # Use the command's installation without launching shell-discovery probes.
        bin_dir = Path(proc.args[0]).parent
        if bin_dir.parent.name.lower() != "usr":
            bin_dir = bin_dir.parent / "usr" / "bin"
        ps, kill = bin_dir / "ps.exe", bin_dir / "kill.exe"
        if not ps.is_file() or not kill.is_file():
            logger.debug("Terminal MSYS process group utilities are unavailable")
            return None
        options = dict(capture_output=True, encoding="utf-8", errors="replace",
                       creationflags=windows_hide_flags(), timeout=2)
        output = subprocess.run([str(ps), "-W", "-l"], check=True, **options).stdout
        # A descendant PID may have been recycled between the Windows and MSYS
        # snapshots. Only its original identity may contribute a group leader.
        for winpid, created in create_times.items():
            try:
                if psutil.Process(winpid).create_time() == created:
                    continue
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            winpid_depths.pop(winpid, None)
        pgid = _command_pgid(output, winpid_depths, os.getpid())
        if pgid is not None and proc.poll() is None:
            return kill, pgid
    except Exception:
        logger.debug("Could not resolve terminal MSYS process group", exc_info=True)
    return None


def kill_msys_process_group(group: tuple[Path, int]) -> None:
    """Reap MSYS pipeline members after taskkill has walked the live Windows tree."""
    try:
        kill, pgid = group
        subprocess.run(
            [str(kill), "-KILL", "--", f"-{pgid}"], check=True,
            capture_output=True, encoding="utf-8", errors="replace",
            creationflags=windows_hide_flags(), timeout=2,
        )
    except Exception:
        logger.debug("Could not terminate terminal MSYS process group", exc_info=True)
