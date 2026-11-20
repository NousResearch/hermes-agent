"""delegate_task ``files``: parent-named file ranges inlined into a child's first turn.

``inline`` parses ``path`` / ``path:start-end`` entries, reads them (capped) against the child's workspace, renders
the ``FILES`` block and marks the shown snapshots as read in the child's read_file tracker.
"""

from tools.delegate_files.inline import (
    FILES_HEADING, MAX_FILE_LINES, MAX_TOTAL_BYTES, coerce_task_files, mark_files_read, resolve_task_files, with_files,
)

__all__ = [
    "FILES_HEADING", "MAX_FILE_LINES", "MAX_TOTAL_BYTES", "coerce_task_files", "mark_files_read",
    "resolve_task_files", "with_files",
]
