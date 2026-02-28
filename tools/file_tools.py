#!/usr/bin/env python3
"""File Tools Module - LLM agent file manipulation tools."""

import json
import logging
import os
import threading
from typing import Optional
from tools.file_operations import ShellFileOperations

logger = logging.getLogger(__name__)

_file_ops_lock = threading.Lock()
_file_ops_cache: dict = {}


def _get_file_ops(task_id: str = "default") -> ShellFileOperations:
    """Get or create ShellFileOperations for a terminal environment.

    Respects the TERMINAL_ENV setting -- if the task_id doesn't have an
    environment yet, creates one using the configured backend (local, docker,
    modal, etc.) rather than always defaulting to local.

    Thread-safe: uses the same per-task creation locks as terminal_tool to
    prevent duplicate sandbox creation from concurrent tool calls.
    """
    from tools.terminal_tool import (
        _active_environments, _env_lock, _create_environment,
        _get_env_config, _last_activity, _start_cleanup_thread,
        _check_disk_usage_warning,
        _creation_locks, _creation_locks_lock,
    )
    import time

    # Fast path: check cache -- but also verify the underlying environment
    # is still alive (it may have been killed by the cleanup thread).
    with _file_ops_lock:
        cached = _file_ops_cache.get(task_id)
    if cached is not None:
        with _env_lock:
            if task_id in _active_environments:
                _last_activity[task_id] = time.time()
                return cached
            else:
                # Environment was cleaned up -- invalidate stale cache entry
                with _file_ops_lock:
                    _file_ops_cache.pop(task_id, None)

    # Need to ensure the environment exists before building file_ops.
    # Acquire per-task lock so only one thread creates the sandbox.
    with _creation_locks_lock:
        if task_id not in _creation_locks:
            _creation_locks[task_id] = threading.Lock()
        task_lock = _creation_locks[task_id]

    with task_lock:
        # Double-check: another thread may have created it while we waited
        with _env_lock:
            if task_id in _active_environments:
                _last_activity[task_id] = time.time()
                terminal_env = _active_environments[task_id]
            else:
                terminal_env = None

        if terminal_env is None:
            from tools.terminal_tool import _task_env_overrides

            config = _get_env_config()
            env_type = config["env_type"]
            overrides = _task_env_overrides.get(task_id, {})

            if env_type == "docker":
                image = overrides.get("docker_image") or config["docker_image"]
            elif env_type == "singularity":
                image = overrides.get("singularity_image") or config["singularity_image"]
            elif env_type == "modal":
                image = overrides.get("modal_image") or config["modal_image"]
            else:
                image = ""

            cwd = overrides.get("cwd") or config["cwd"]
            logger.info("Creating new %s environment for task %s...", env_type, task_id[:8])

            container_config = None
            if env_type in ("docker", "singularity", "modal"):
                container_config = {
                    "container_cpu": config.get("container_cpu", 1),
                    "container_memory": config.get("container_memory", 5120),
                    "container_disk": config.get("container_disk", 51200),
                    "container_persistent": config.get("container_persistent", True),
                }
            terminal_env = _create_environment(
                env_type=env_type,
                image=image,
                cwd=cwd,
                timeout=config["timeout"],
                container_config=container_config,
            )

            with _env_lock:
                _active_environments[task_id] = terminal_env
                _last_activity[task_id] = time.time()

            _start_cleanup_thread()
            logger.info("%s environment ready for task %s", env_type, task_id[:8])

    # Build file_ops from the (guaranteed live) environment and cache it
    file_ops = ShellFileOperations(terminal_env)
    with _file_ops_lock:
        _file_ops_cache[task_id] = file_ops
    return file_ops


def clear_file_ops_cache(task_id: str = None):
    """Clear the file operations cache."""
    with _file_ops_lock:
        if task_id:
            _file_ops_cache.pop(task_id, None)
        else:
            _file_ops_cache.clear()


def read_file_tool(path: str, offset: int = 1, limit: int = 500, task_id: str = "default") -> str:
    """Read a file with pagination and line numbers."""
    try:
        file_ops = _get_file_ops(task_id)
        result = file_ops.read_file(path, offset, limit)
        return json.dumps(result.to_dict(), ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def write_file_tool(path: str, content: str, task_id: str = "default") -> str:
    """Write content to a file."""
    try:
        file_ops = _get_file_ops(task_id)
        result = file_ops.write_file(path, content)
        return json.dumps(result.to_dict(), ensure_ascii=False)
    except Exception as e:
        print(f"[FileTools] write_file error: {type(e).__name__}: {e}", flush=True)  
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def patch_tool(mode: str = "replace", path: str = None, old_string: str = None,
               new_string: str = None, replace_all: bool = False, patch: str = None,
               task_id: str = "default") -> str:
    """Patch a file using replace mode or V4A patch format."""
    try:
        file_ops = _get_file_ops(task_id)
        
        if mode == "replace":
            if not path:
                return json.dumps({"error": "path required"})
            if old_string is None or new_string is None:
                return json.dumps({"error": "old_string and new_string required"})
            result = file_ops.patch_replace(path, old_string, new_string, replace_all)
        elif mode == "patch":
            if not patch:
                return json.dumps({"error": "patch content required"})
            result = file_ops.patch_v4a(patch)
        else:
            return json.dumps({"error": f"Unknown mode: {mode}"})
        
        return json.dumps(result.to_dict(), ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def search_tool(pattern: str, target: str = "content", path: str = ".",
                file_glob: str = None, limit: int = 50, offset: int = 0,
                output_mode: str = "content", context: int = 0,
                task_id: str = "default") -> str:
    """Search for content or files."""
    try:
        file_ops = _get_file_ops(task_id)
        result = file_ops.search(
            pattern=pattern, path=path, target=target, file_glob=file_glob,
            limit=limit, offset=offset, output_mode=output_mode, context=context
        )
        return json.dumps(result.to_dict(), ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


FILE_TOOLS = [
    {"name": "read_file", "function": read_file_tool},
    {"name": "write_file", "function": write_file_tool},
    {"name": "patch", "function": patch_tool},
    {"name": "search_files", "function": search_tool}
]


def get_file_tools():
    """Get the list of file tool definitions."""
    return FILE_TOOLS


# ---------------------------------------------------------------------------
# Schemas + Registry
# ---------------------------------------------------------------------------
from tools.registry import registry


def _check_file_reqs():
    """Lazy wrapper to avoid circular import with tools/__init__.py."""
    from tools import check_file_requirements
    return check_file_requirements()

READ_FILE_SCHEMA = {
    "name": "read_file",
    "description": "Read a text file with line numbers and pagination. Use this instead of cat/head/tail in terminal. Output format: 'LINE_NUM|CONTENT'. Suggests similar filenames if not found. Use offset and limit for large files. NOTE: Cannot read images or binary files — use vision_analyze for images.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read (absolute, relative, or ~/path)"},
            "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed, default: 1)", "default": 1, "minimum": 1},
            "limit": {"type": "integer", "description": "Maximum number of lines to read (default: 500, max: 2000)", "default": 500, "maximum": 2000}
        },
        "required": ["path"]
    }
}

WRITE_FILE_SCHEMA = {
    "name": "write_file",
    "description": "Write content to a file, completely replacing existing content. Use this instead of echo/cat heredoc in terminal. Creates parent directories automatically. OVERWRITES the entire file — use 'patch' for targeted edits.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write (will be created if it doesn't exist, overwritten if it does)"},
            "content": {"type": "string", "description": "Complete content to write to the file"}
        },
        "required": ["path", "content"]
    }
}

PATCH_SCHEMA = {
    "name": "patch",
    "description": "Targeted find-and-replace edits in files. Use this instead of sed/awk in terminal. Uses fuzzy matching (9 strategies) so minor whitespace/indentation differences won't break it. Returns a unified diff. Auto-runs syntax checks after editing.\n\nReplace mode (default): find a unique string and replace it.\nPatch mode: apply V4A multi-file patches for bulk changes.",
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["replace", "patch"], "description": "Edit mode: 'replace' for targeted find-and-replace, 'patch' for V4A multi-file patches", "default": "replace"},
            "path": {"type": "string", "description": "File path to edit (required for 'replace' mode)"},
            "old_string": {"type": "string", "description": "Text to find in the file (required for 'replace' mode). Must be unique in the file unless replace_all=true. Include enough surrounding context to ensure uniqueness."},
            "new_string": {"type": "string", "description": "Replacement text (required for 'replace' mode). Can be empty string to delete the matched text."},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences instead of requiring a unique match (default: false)", "default": False},
            "patch": {"type": "string", "description": "V4A format patch content (required for 'patch' mode). Format:\n*** Begin Patch\n*** Update File: path/to/file\n@@ context hint @@\n context line\n-removed line\n+added line\n*** End Patch"}
        },
        "required": ["mode"]
    }
}

SEARCH_FILES_SCHEMA = {
    "name": "search_files",
    "description": "Search file contents or find files by name. Use this instead of grep/rg/find/ls in terminal. Ripgrep-backed, faster than shell equivalents.\n\nContent search (target='content'): Regex search inside files. Output modes: full matches with line numbers, file paths only, or match counts.\n\nFile search (target='files'): Find files by glob pattern (e.g., '*.py', '*config*'). Also use this instead of ls — results sorted by modification time.",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern for content search, or glob pattern (e.g., '*.py') for file search"},
            "target": {"type": "string", "enum": ["content", "files"], "description": "'content' searches inside file contents, 'files' searches for files by name", "default": "content"},
            "path": {"type": "string", "description": "Directory or file to search in (default: current working directory)", "default": "."},
            "file_glob": {"type": "string", "description": "Filter files by pattern in grep mode (e.g., '*.py' to only search Python files)"},
            "limit": {"type": "integer", "description": "Maximum number of results to return (default: 50)", "default": 50},
            "offset": {"type": "integer", "description": "Skip first N results for pagination (default: 0)", "default": 0},
            "output_mode": {"type": "string", "enum": ["content", "files_only", "count"], "description": "Output format for grep mode: 'content' shows matching lines with line numbers, 'files_only' lists file paths, 'count' shows match counts per file", "default": "content"},
            "context": {"type": "integer", "description": "Number of context lines before and after each match (grep mode only)", "default": 0}
        },
        "required": ["pattern"]
    }
}


def _handle_read_file(args, **kw):
    tid = kw.get("task_id") or "default"
    return read_file_tool(path=args.get("path", ""), offset=args.get("offset", 1), limit=args.get("limit", 500), task_id=tid)


def _handle_write_file(args, **kw):
    tid = kw.get("task_id") or "default"
    return write_file_tool(path=args.get("path", ""), content=args.get("content", ""), task_id=tid)


def _handle_patch(args, **kw):
    tid = kw.get("task_id") or "default"
    return patch_tool(
        mode=args.get("mode", "replace"), path=args.get("path"),
        old_string=args.get("old_string"), new_string=args.get("new_string"),
        replace_all=args.get("replace_all", False), patch=args.get("patch"), task_id=tid)


def _handle_search_files(args, **kw):
    tid = kw.get("task_id") or "default"
    target_map = {"grep": "content", "find": "files"}
    raw_target = args.get("target", "content")
    target = target_map.get(raw_target, raw_target)
    return search_tool(
        pattern=args.get("pattern", ""), target=target, path=args.get("path", "."),
        file_glob=args.get("file_glob"), limit=args.get("limit", 50), offset=args.get("offset", 0),
        output_mode=args.get("output_mode", "content"), context=args.get("context", 0), task_id=tid)


registry.register(name="read_file", toolset="file", schema=READ_FILE_SCHEMA, handler=_handle_read_file, check_fn=_check_file_reqs)
registry.register(name="write_file", toolset="file", schema=WRITE_FILE_SCHEMA, handler=_handle_write_file, check_fn=_check_file_reqs)
registry.register(name="patch", toolset="file", schema=PATCH_SCHEMA, handler=_handle_patch, check_fn=_check_file_reqs)
registry.register(name="search_files", toolset="file", schema=SEARCH_FILES_SCHEMA, handler=_handle_search_files, check_fn=_check_file_reqs)
