"""Shared utility functions for hermes-agent.

This module provides common utility functions used across the Hermes Agent codebase.
Currently includes atomic file operations for safe concurrent writes.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Union

logger = logging.getLogger(__name__)


def atomic_json_write(path: Union[str, Path], data: Any, *, indent: int = 2) -> None:
    """Write JSON data to a file atomically using temp file + fsync + os.replace.

    This function ensures the target file is never left in a partially-written state.
    If the process crashes mid-write, the previous version of the file remains intact.
    Uses write-ahead logging (WAL) pattern for crash safety.

    Args:
        path: Target file path (will be created or overwritten).
        data: JSON-serializable data to write.
        indent: JSON indentation level (default: 2).

    Raises:
        OSError: If file system operations fail (e.g., permission denied, disk full).
        TypeError: If data is not JSON-serializable.

    Example:
        >>> atomic_json_write("/path/to/config.json", {"key": "value"})
        >>> atomic_json_write(Path("data.json"), [1, 2, 3], indent=4)

    Note:
        - Parent directories are created automatically if they don't exist
        - Uses ensure_ascii=False to preserve Unicode characters
        - File is synced to disk before replacing to prevent data loss
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        logger.debug("Successfully wrote JSON file atomically: %s", path)
    except Exception as e:
        logger.error("Failed to write JSON file %s: %s", path, e)
        try:
            os.unlink(tmp_path)
        except OSError as cleanup_error:
            logger.warning("Failed to clean up temp file %s: %s", tmp_path, cleanup_error)
        raise
