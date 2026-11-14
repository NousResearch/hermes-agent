"""Whole-file read baselines for files that live on a remote backend (SSH, containers).

A host stat or hash cannot see a file on the execution target, so read_file brackets the
read with a digest taken on the backend that owns the file. The baseline counts only when
both digests match, and it is keyed by the backend session and the resolved path.
"""

from tools.file_operations import ShellFileOperations
from tools.file_tools_read_tracking import _record_remote_read


def _remote_baseline_key(file_ops: ShellFileOperations, resolved: str) -> tuple[str, str]:
    """Bind remote read knowledge to one backend session and its target path."""
    owner = getattr(file_ops.env, "_session_id", None) or str(id(file_ops.env))
    return str(owner), resolved


def remote_read_digest(file_ops, resolved: str, host_paths: bool) -> tuple[str, str | None] | None:
    """Backend digest taken before a read, or ``None`` when the file is on the host."""
    if host_paths or not isinstance(file_ops, ShellFileOperations):
        return None
    return file_ops.file_digest(resolved)


def record_remote_read(file_ops, before, task_id: str, resolved: str, *, start: int, end: int,
                       total_lines: int | None, redacted: bool) -> None:
    """Record the page as read only when the backend file did not change during the read."""
    if before is None or before[0] != "file" or file_ops.file_digest(resolved) != before:
        return
    _record_remote_read(task_id, _remote_baseline_key(file_ops, resolved), before[1],
                        start=start, end=end, total_lines=total_lines, redacted=redacted)
