"""
Shared utility functions for the Hermes Agent infrastructure.

This module provides high-integrity file I/O operations. In an autonomous 
agentic environment, ensuring state persistence without corruption is 
critical, as malformed configuration or state files can lead to 
unrecoverable logic failures.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Union

import yaml


def atomic_json_write(
    path: Union[str, Path],
    data: Any,
    *,
    indent: int = 2,
    **dump_kwargs: Any
) -> None:
    """Write JSON data to a file atomically using a 'replace' pattern.

    This function mitigates the risk of file corruption during system crashes.
    By writing to a temporary file and performing an 'os.replace', the
    operation ensures that the target file is either fully updated or
    remains in its previous valid state.

    Technical Safety:
    - Uses 'os.fsync' to force the OS to flush data to physical storage.
    - Uses 'os.replace' for a POSIX-compliant atomic swap.

    NOTE: While file-system atomic, this does not implement application-level
    file locking. Concurrent writes from multiple processes (Race Conditions)
    should be managed via higher-level orchestrators.

    Args:
        path: Target file path (will be created or overwritten).
        data: JSON-serializable data to write.
        indent: JSON indentation (default 2).
        **dump_kwargs: Additional keyword args forwarded to json.dump().
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
            json.dump(
                data,
                f,
                indent=indent,
                ensure_ascii=False,
                **dump_kwargs,
            )
            f.flush()
            # Ensure the OS flushes buffers to disk before we rename
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_yaml_write(
    path: Union[str, Path],
    data: Any,
    *,
    default_flow_style: bool = False,
    sort_keys: bool = False,
    extra_content: str | None = None,
) -> None:
    """
    Write YAML data to a file atomically, mirroring the safety of 'atomic_json_write'.

    Crucial for human-readable configurations where partial writes would 
    render the YAML parser unable to recover the agent's settings.

    Args:
        path: Target file path (will be created or overwritten).
        data: YAML-serializable data to write.
        default_flow_style: YAML flow style (default False).
        sort_keys: Whether to sort dict keys (default False).
        extra_content: Optional string to append after the YAML dump 
                       (e.g., manual comments).
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
            yaml.dump(data, f, default_flow_style=default_flow_style, sort_keys=sort_keys)
            if extra_content:
                f.write(extra_content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
