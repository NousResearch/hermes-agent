#!/usr/bin/env python3
"""File Tools Module - LLM agent file manipulation tools."""

import json
import threading
from typing import Optional
from tools.file_operations import ShellFileOperations

_file_ops_lock = threading.Lock()
_file_ops_cache: dict = {}


def _get_file_ops(task_id: str = "default") -> ShellFileOperations:
    """Get or create ShellFileOperations for a terminal environment."""
    from tools.terminal_tool import _active_environments, _env_lock, _LocalEnvironment
    
    with _file_ops_lock:
        if task_id in _file_ops_cache:
            return _file_ops_cache[task_id]
        
        with _env_lock:
            if task_id not in _active_environments:
                import os
                env = _LocalEnvironment(cwd=os.getcwd(), timeout=60)
                _active_environments[task_id] = env
            terminal_env = _active_environments[task_id]
        
        file_ops = ShellFileOperations(terminal_env)
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
    {"name": "search", "function": search_tool}
]


def get_file_tools():
    """Get the list of file tool definitions."""
    return FILE_TOOLS
