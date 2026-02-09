#!/usr/bin/env python3
"""File Tools Module - LLM agent file manipulation tools."""

import json
import os
import threading
from typing import Optional
from tools.file_operations import ShellFileOperations

_file_ops_lock = threading.Lock()
_file_ops_cache: dict = {}


def _get_file_ops(task_id: str = "default") -> ShellFileOperations:
    """Get or create ShellFileOperations for a terminal environment.
    
    Respects the TERMINAL_ENV setting -- if the task_id doesn't have an
    environment yet, creates one using the configured backend (local, docker,
    modal, etc.) rather than always defaulting to local.
    """
    from tools.terminal_tool import (
        _active_environments, _env_lock, _create_environment,
        _get_env_config, _last_activity, _start_cleanup_thread,
        _check_disk_usage_warning,
    )
    import time
    
    # Fast path: check cache without heavy locks
    with _file_ops_lock:
        if task_id in _file_ops_cache:
            return _file_ops_cache[task_id]
    
    # Check if we need to create a new environment
    needs_creation = False
    with _env_lock:
        if task_id not in _active_environments:
            needs_creation = True
    
    # Create environment OUTSIDE locks so we don't block other rollouts
    # during slow Modal/Docker startup (~10s)
    if needs_creation:
        config = _get_env_config()
        env_type = config["env_type"]
        
        if env_type == "docker":
            image = config["docker_image"]
        elif env_type == "singularity":
            image = config["singularity_image"]
        elif env_type == "modal":
            image = config["modal_image"]
        else:
            image = ""
        
        cwd = config["cwd"]
        _check_disk_usage_warning()
        if not os.getenv("HERMES_QUIET"):
            print(f"[FileTools] Creating new {env_type} environment for task {task_id[:8]}...", flush=True)
        
        new_env = _create_environment(
            env_type=env_type,
            image=image,
            cwd=cwd,
            timeout=config["timeout"],
        )
        
        # Store under lock (brief) -- do NOT call _start_cleanup_thread inside
        # the lock because it also acquires _env_lock (non-reentrant = deadlock)
        created = False
        with _env_lock:
            if task_id not in _active_environments:
                _active_environments[task_id] = new_env
                created = True
            else:
                try:
                    if hasattr(new_env, 'stop'):
                        new_env.stop()
                except Exception:
                    pass
        
        if created:
            _start_cleanup_thread()
            if not os.getenv("HERMES_QUIET"):
                print(f"[FileTools] {env_type} environment ready for task {task_id[:8]}", flush=True)
    
    # Now get the environment and build file_ops
    with _env_lock:
        _last_activity[task_id] = time.time()
        terminal_env = _active_environments[task_id]
    
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
    {"name": "search", "function": search_tool}
]


def get_file_tools():
    """Get the list of file tool definitions."""
    return FILE_TOOLS
