#!/usr/bin/env python3
"""
Unified Tools — 6 high-level tools with DIRECT internal implementations.

Each unified tool performs its work WITHOUT going through the old individual
handler chain.  Results are LRU-cached so identical parameter sets never
execute twice in the same session.

Reduction chain (filesystem example):
  Before (33 individual):  LLM → read_file(handler) → read_file(handler) → ...
  Before (6 unified, old):  LLM → filesystem(handler) → _h_read → _h_read → ...
  After  (6 unified, new):  LLM → filesystem(direct) → ShellFileOperations × N
"""

import functools
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict

from tools.registry import registry

logger = logging.getLogger(__name__)

# ===========================================================================
# LRU result cache — avoids redundant tool executions
# ===========================================================================

class _ResultCache:
    """Simple LRU cache keyed on (tool_name, frozenset(kwargs.items()))."""
    
    def __init__(self, maxsize: int = 128):
        self._cache: Dict[str, Any] = {}
        self._order: list = []
        self._maxsize = maxsize
    
    def _make_key(self, tool_name: str, data: dict) -> str:
        """Build a normalized cache key."""
        # Normalize data for consistent keying
        normalized = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
        return f"{tool_name}::{normalized}"
    
    def get(self, tool_name: str, data: dict) -> Any:
        key = self._make_key(tool_name, data)
        return self._cache.get(key)
    
    def put(self, tool_name: str, data: dict, result: Any) -> None:
        key = self._make_key(tool_name, data)
        if key in self._cache:
            # Move to end (most recently used)
            self._order.remove(key)
            self._order.append(key)
            self._cache[key] = result
            return
        if len(self._order) >= self._maxsize:
            # Evict least recently used
            oldest = self._order.pop(0)
            self._cache.pop(oldest, None)
        self._order.append(key)
        self._cache[key] = result
    
    def clear(self) -> None:
        self._cache.clear()
        self._order.clear()


_RESULT_CACHE = _ResultCache(maxsize=256)


def _cached(tool_name: str, data: dict, runner) -> Any:
    """Return cached result if available, else compute and cache."""
    cached = _RESULT_CACHE.get(tool_name, data)
    if cached is not None:
        return cached
    result = runner()
    _RESULT_CACHE.put(tool_name, data, result)
    return result


# ===========================================================================
# Direct low-level imports — no handler wrappers
# ===========================================================================

# File operations — direct implementations
from tools.file_operations import ShellFileOperations, ReadResult, WriteResult
from tools.binary_extensions import has_binary_extension
from agent.file_safety import get_read_block_error
from agent.redact import redact_sensitive_text

_FILE_OPS: ShellFileOperations = None  # lazy init

def _ensure_file_ops():
    global _FILE_OPS
    if _FILE_OPS is None:
        from tools.file_operations import ShellFileOperations
        from tools.environments.local import LocalEnvironment
        _FILE_OPS = ShellFileOperations(LocalEnvironment())


def _read_file_direct(path: str, offset: int = 1, limit: int = 500) -> dict:
    """Read a single file directly — bypasses all handler wrappers."""
    _ensure_file_ops()
    
    # Safety checks
    block_error = get_read_block_error(path)
    if block_error:
        return {"error": block_error, "status": "blocked"}
    
    if has_binary_extension(path):
        return {"error": f"Cannot read binary file: {path}", "status": "binary"}
    
    try:
        result = _FILE_OPS.read_file(path, offset=offset, limit=limit)
    except Exception as e:
        return {"error": str(e), "status": "error"}
    
    if result.error:
        return {"error": result.error, "status": "error"}
    
    return {
        "status": "ok",
        "content": result.content,
        "file_size": result.file_size,
        "total_lines": result.total_lines,
    }


def _read_multiple_files(paths: list, offset: int = 1, limit: int = 500) -> dict:
    """Read multiple files in one function call — NO per-file handler dispatch."""
    results = {}
    for p in paths:
        results[p] = _read_file_direct(p, offset=offset, limit=limit)
    return {"status": "ok", "files": results}


def _search_files_direct(pattern: str, path: str = ".", target: str = "content",
                          file_glob: str = None, limit: int = 50,
                          output_mode: str = "content") -> dict:
    """Search files directly using ripgrep/os.walk — bypasses handlers."""
    import subprocess
    
    search_path = Path(path).expanduser().resolve()
    if not search_path.exists():
        return {"error": f"Path does not exist: {path}", "status": "error"}
    
    if target == "content":
        # Use ripgrep for content search
        cmd = ["rg", "--no-heading", "--line-number", "-i", pattern, str(search_path)]
        if file_glob:
            cmd.extend(["-g", file_glob])
        if limit:
            cmd.extend(["--max-count", str(limit)])
        
        try:
            rg_result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if rg_result.returncode not in (0, 1):  # 1 = no matches
                rg_result.check_returncode()
        except subprocess.TimeoutExpired:
            return {"error": "Search timed out", "status": "error"}
        except subprocess.CalledProcessError as e:
            return {"error": str(e), "status": "error"}
        
        lines = rg_result.stdout.strip().split("\n") if rg_result.stdout.strip() else []
        
        if output_mode == "files_only":
            files = set()
            for line in lines:
                parts = line.split(":", 1)
                if parts:
                    files.add(parts[0])
            return {"status": "ok", "matches": sorted(files)[:limit]}
        elif output_mode == "count":
            files = {}
            for line in lines:
                fname = line.split(":", 1)[0] if ":" in line else line
                files[fname] = files.get(fname, 0) + 1
            return {"status": "ok", "counts": files}
        else:
            matches = []
            for line in lines[:limit]:
                # ripgrep format: path:linenum:content
                m = re.match(r'^(.+?):(\d+):(.*)', line)
                if m:
                    matches.append({"path": m.group(1), "line": int(m.group(2)), "content": m.group(3)})
                elif ":" in line:
                    parts = line.split(":", 1)
                    matches.append({"path": parts[0], "content": parts[1]})
                else:
                    matches.append({"content": line})
            return {"status": "ok", "matches": matches}
    
    elif target == "files":
        # Use os.walk for filename search
        matches = []
        for root, dirs, files in os.walk(search_path):
            # Skip hidden dirs unless pattern starts with dot
            dirs[:] = [d for d in dirs if not d.startswith(".") or pattern.startswith(".")]
            for f in files:
                if re.search(pattern, f, re.IGNORECASE):
                    full = os.path.join(root, f)
                    matches.append(full)
                    if len(matches) >= limit:
                        break
            if len(matches) >= limit:
                break
        return {"status": "ok", "matches": matches[:limit]}
    
    return {"error": f"Unknown target: {target}", "status": "error"}


# ===========================================================================
# Lazy imports for non-filesystem tools (can't easily bypass handlers)
# ===========================================================================

_H_IMPORTED = False

def _ensure_handlers():
    global _H_IMPORTED
    if _H_IMPORTED:
        return
    _H_IMPORTED = True
    
    # Web — use the real tool (backend selection is complex)
    import tools.web_tools as _wt
    globals()["_web_search"] = _wt.web_search_tool
    globals()["_web_extract"] = _wt.web_extract_tool
    
    # Code — use the real tools
    import tools.code_execution_tool as _ce
    import tools.terminal_tool as _te
    import tools.delegate_tool as _de
    globals()["_exec_code"] = _ce.execute_code
    globals()["_term"] = _te.terminal_tool
    globals()["_proc"] = _te.process_tool
    globals()["_del_task"] = _de.delegate_task_tool
    
    # System — use the real tools (memory/todo are agent-loop-intercepted)
    import tools.memory_tool as _me
    import tools.todo_tool as _td
    import tools.clarify_tool as _cl
    import tools.send_message_tool as _sm
    import tools.cron_tool as _cr
    globals()["_mem"] = _me.memory_tool
    globals()["_todo_fn"] = _td.todo_tool
    globals()["_clarify_fn"] = _cl.clarify_tool
    globals()["_send_msg"] = _sm.send_message_tool
    globals()["_cron"] = _cr.cronjob_tool
    
    # Media — use the real tools
    import tools.vision_tools as _vi
    import tools.image_generation_tool as _ig
    import tools.tts_tool as _tt
    globals()["_vision"] = _vi.vision_analyze_tool
    globals()["_img_gen"] = _ig.image_generate_tool
    globals()["_tts_fn"] = _tt.text_to_speech_tool
    
    # Browser — use the real tools
    import tools.browser_tool as _br
    globals()["_bnav"] = _br.browser_navigate
    globals()["_bclick"] = _br.browser_click
    globals()["_btype"] = _br.browser_type
    globals()["_bscroll"] = _br.browser_scroll
    globals()["_bsnap"] = _br.browser_snapshot
    globals()["_bback"] = _br.browser_back
    globals()["_bpress"] = _br.browser_press
    globals()["_bimgs"] = _br.browser_get_images
    globals()["_bvision"] = _br.browser_vision
    globals()["_bconsole"] = _br.browser_console


def _ensure_json(obj):
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except (json.JSONDecodeError, TypeError):
            return {"raw": obj[:500]}
    return obj


# ===========================================================================
# 1. FILESYSTEM — DIRECT implementations (no handler wrappers)
# ===========================================================================

_FS_SCHEMA = {
    "name": "filesystem",
    "description": (
        "Unified filesystem. Read/write/patch/search files. "
        "Batch multiple files: data={'operations': [{'type': 'read', "
        "'paths': ['a.py', 'b.py']}]}. "
        "Types: read, write, patch, search, grep_and_read."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "data": {"type": "object", "properties": {
                "type": {"type": "string"},
                "paths": {"type": "array", "items": {"type": "string"}},
                "path": {"type": "string"},
                "content": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
                "pattern": {"type": "string"},
                "target": {"type": "string"},
                "file_glob": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
                "operations": {"type": "array", "items": {"type": "object"}},
            }, "additionalProperties": True},
        },
        "required": ["data"],
    },
}


def _handle_filesystem(args, **kw):
    data = args.get("data", {})
    task_id = kw.get("task_id") or os.environ.get("HERMES_TASK_ID", "default")
    
    # Single operation
    if isinstance(data, dict) and "type" in data:
        return json.dumps(_run_fs_op(data, task_id), ensure_ascii=False, default=str)
    
    # Batch operations
    ops = data.get("operations", [])
    if not ops:
        return json.dumps({"status": "error", "error": "No operations"}, ensure_ascii=False)
    
    results = {}
    for op in ops:
        oid = op.get("id", str(len(results)))
        results[oid] = _run_fs_op(op, task_id)
    return json.dumps({"status": "ok", "results": results}, ensure_ascii=False, default=str)


def _run_fs_op(op: dict, task_id: str) -> dict:
    t = op["type"]
    
    if t == "read":
        paths = op.get("paths", [op.get("path", "")])
        if isinstance(paths, str):
            paths = [paths]
        offset = op.get("offset", 1)
        limit = op.get("limit", 500)
        # DIRECT: read all files in one batch call — no handler dispatch
        return _cached("fs.read", {"paths": tuple(paths), "offset": offset, "limit": limit},
                       lambda: _read_multiple_files(paths, offset, limit))
    
    elif t == "write":
        _ensure_file_ops()
        path = op.get("path", "")
        content = op.get("content", "")
        try:
            result = _FILE_OPS.write_file(path, content)
        except Exception as e:
            return {"status": "error", "error": str(e)}
        if result.error:
            return {"status": "error", "error": result.error}
        return {"status": "ok", "path": path}
    
    elif t == "patch":
        _ensure_file_ops()
        path = op.get("path", "")
        old = op.get("old", "")
        new = op.get("new", "")
        try:
            result = _FILE_OPS.patch_replace(path, old, new, replace_all=op.get("replace_all", False))
        except Exception as e:
            return {"status": "error", "error": str(e)}
        if result.error:
            return {"status": "error", "error": result.error}
        return {"status": "ok", "path": path, "diff": result.diff}
    
    elif t == "search":
        # DIRECT: search via ripgrep — no handler dispatch
        return _cached("fs.search", {k: op.get(k) for k in ("pattern", "path", "target", "file_glob", "limit", "output_mode") if k in op},
                       lambda: _search_files_direct(
                           pattern=op.get("pattern", ""),
                           path=op.get("path", "."),
                           target=op.get("target", "content"),
                           file_glob=op.get("file_glob"),
                           limit=op.get("limit", 50),
                           output_mode=op.get("output_mode", "content"),
                       ))
    
    elif t == "grep_and_read":
        # DIRECT: search once, read matched files — no handler dispatch
        pattern = op.get("pattern", "")
        path = op.get("path", ".")
        fg = op.get("file_glob")
        limit = op.get("limit", 50)
        
        search_result = _search_files_direct(pattern, path, "content", fg, limit, "files_only")
        matched = search_result.get("matches", [])[:5]
        
        contents = {}
        for fp in matched:
            r = _read_file_direct(fp, 1, 100)
            contents[fp] = r.get("content", r.get("error", "?"))
        
        return {"status": "ok", "pattern": pattern, "matched": matched, "contents": contents}
    
    return {"status": "error", "error": f"Unknown fs op: {t}"}


# ===========================================================================
# 2. RESEARCH — web search + extract (cached, handler-based)
# ===========================================================================

_RESEARCH_SCHEMA = {
    "name": "research",
    "description": (
        "Unified web research. Search, extract, deep-dive. "
        "Batch: data={'operations': [{'type':'search','query':'...'}, "
        "{'type':'extract','urls':['...']}]}. "
        "Results are cached per query."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "data": {"type": "object", "additionalProperties": True},
        },
        "required": ["data"],
    },
}


def _handle_research(args, **kw):
    data = args.get("data", {})
    
    if isinstance(data, dict) and "type" in data:
        return json.dumps(_run_research_op(data), ensure_ascii=False, default=str)
    
    ops = data.get("operations", [])
    if not ops:
        return json.dumps({"status": "error", "error": "No operations"}, ensure_ascii=False)
    
    results = {}
    for op in ops:
        oid = op.get("id", str(len(results)))
        results[oid] = _run_research_op(op)
    return json.dumps({"status": "ok", "results": results}, ensure_ascii=False, default=str)


def _run_research_op(op: dict) -> dict:
    _ensure_handlers()
    t = op["type"]
    
    if t == "search":
        query = op.get("query", "")
        limit = op.get("limit", 5)
        return _cached("research.search", {"query": query, "limit": limit},
                       lambda: {"status": "ok", "type": "search", "query": query,
                                "result": _ensure_json(_web_search(query=query, limit=limit))})
    
    elif t == "extract":
        urls = op.get("urls", [])
        return _cached("research.extract", {"urls": tuple(urls)},
                       lambda: {"status": "ok", "type": "extract",
                                "result": _ensure_json(_web_extract(urls))})
    
    elif t == "deep_dive":
        q = op.get("query", "")
        limit = op.get("limit", 5)
        # Search
        sr = _ensure_json(_web_search(query=q, limit=limit))
        urls = []
        if isinstance(sr, dict):
            rl = sr.get("data", {}).get("web", [])
            urls = [r.get("url", "") for r in rl if r.get("url")][:3]
        # Extract top results
        er = {}
        if urls:
            er = _ensure_json(_web_extract(urls))
        return {"status": "ok", "type": "deep_dive", "query": q, "search": sr, "extracted": er}
    
    return {"status": "error", "error": f"Unknown research op: {t}"}


# ===========================================================================
# 3. CODE — execute_code + terminal + delegate
# ===========================================================================

_CODE_SCHEMA = {
    "name": "code",
    "description": (
        "Unified code execution. Types: python (execute_code), shell (terminal), "
        "process (process management), delegate (subagent). "
        "Single operation per call. Prefer python for multi-step logic."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "data": {"type": "object", "additionalProperties": True},
        },
        "required": ["data"],
    },
}


def _handle_code(args, **kw):
    data = args.get("data", {})
    t = data.get("type", "")
    _ensure_handlers()
    
    if t == "python":
        r = _exec_code(code=data.get("code", ""))
        return json.dumps({"status": "ok", "type": "python", "result": _ensure_json(r)}, ensure_ascii=False, default=str)
    elif t == "shell":
        r = _term(command=data.get("command", ""), timeout=data.get("timeout", 180), workdir=data.get("workdir"))
        _RESULT_CACHE.clear()  # Shell commands can change filesystem state
        return json.dumps({"status": "ok", "type": "shell", "result": _ensure_json(r)}, ensure_ascii=False, default=str)
    elif t == "process":
        r = _proc(**data)
        return json.dumps({"status": "ok", "type": "process", "result": _ensure_json(r)}, ensure_ascii=False, default=str)
    elif t == "delegate":
        r = _del_task(goal=data.get("goal", ""), context=data.get("context", ""), toolsets=data.get("toolsets"))
        return json.dumps({"status": "ok", "type": "delegate", "result": _ensure_json(r)}, ensure_ascii=False, default=str)
    return json.dumps({"status": "error", "error": f"Unknown code op: {t}"}, ensure_ascii=False)


# ===========================================================================
# 4. SYSTEM — memory / todo / cron / clarify / send / session / skill
# ===========================================================================

_SYSTEM_SCHEMA = {
    "name": "system",
    "description": (
        "Unified system management. Types: memory, todo, cron, clarify, "
        "send_message, session_search, skill. "
        "Batch: data={'operations': [{'type':'memory',...}, {'type':'todo',...}]}."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "data": {"type": "object", "additionalProperties": True},
        },
        "required": ["data"],
    },
}


def _handle_system(args, **kw):
    data = args.get("data", {})
    ops = []
    if isinstance(data, dict) and "type" in data:
        ops = [data]
    else:
        ops = data.get("operations", [])
    if not ops:
        return json.dumps({"status": "error", "error": "No operations"}, ensure_ascii=False)
    _ensure_handlers()
    results = {}
    for op in ops:
        oid = op.get("id", str(len(results)))
        t = op["type"]
        try:
            if t == "memory":
                from tools.memory_tool import memory_tool as _mt
                r = _mt(action=op.get("action", "add"), target=op.get("target", "memory"),
                         content=op.get("content", ""), old_text=op.get("old_text"))
                results[oid] = {"status": "ok", "type": "memory", "action": op.get("action"), "result": _ensure_json(r)}
            elif t == "todo":
                r = _todo_fn(items=op.get("items"), merge=op.get("merge", False))
                results[oid] = {"status": "ok", "type": "todo", "result": _ensure_json(r)}
            elif t == "cron":
                r = _cron(**{k: v for k, v in op.items() if k not in ("type", "id")})
                results[oid] = {"status": "ok", "type": "cron", "action": op.get("action"), "result": _ensure_json(r)}
            elif t == "clarify":
                r = _clarify_fn(question=op.get("question", ""), choices=op.get("choices"))
                results[oid] = {"status": "ok", "type": "clarify", "result": _ensure_json(r)}
            elif t == "send_message":
                r = _send_msg(target=op.get("target", ""), message=op.get("message", ""))
                results[oid] = {"status": "ok", "type": "send_message", "result": _ensure_json(r)}
            elif t == "session_search":
                from tools.session_search_tool import session_search_tool as _sst
                r = _sst(query=op.get("query", ""), limit=op.get("limit", 3))
                results[oid] = {"status": "ok", "type": "session_search", "result": _ensure_json(r)}
            elif t == "skill":
                from tools.skill_tools import skill_tool as _sk
                r = _sk(action=op.get("action", "list"), name=op.get("skill_name"), content=op.get("content"))
                results[oid] = {"status": "ok", "type": "skill", "action": op.get("action"), "result": _ensure_json(r)}
            else:
                results[oid] = {"status": "error", "error": f"Unknown system op: {t}"}
        except Exception as e:
            results[oid] = {"status": "error", "error": str(e)}
    return json.dumps({"status": "ok", "results": results}, ensure_ascii=False, default=str)


# ===========================================================================
# 5. MEDIA — vision / generate_image / tts
# ===========================================================================

_MEDIA_SCHEMA = {
    "name": "media",
    "description": (
        "Unified media. Types: vision (image analysis), generate_image, tts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "data": {"type": "object", "additionalProperties": True},
        },
        "required": ["data"],
    },
}


def _handle_media(args, **kw):
    data = args.get("data", {})
    t = data.get("type", "")
    _ensure_handlers()
    if t == "vision":
        r = _vision(image_url=data.get("image_url", ""), question=data.get("question", ""))
        return json.dumps({"status": "ok", "type": "vision", "result": _ensure_json(r)}, ensure_ascii=False, default=str)
    elif t == "generate_image":
        r = _img_gen(prompt=data.get("prompt", ""), aspect_ratio=data.get("aspect_ratio", "landscape"))
        return json.dumps({"status": "ok", "type": "generate_image", "result": _ensure_json(r)}, ensure_ascii=False, default=str)
    elif t == "tts":
        r = _tts_fn(text=data.get("text", ""))
        return json.dumps({"status": "ok", "type": "tts", "result": _ensure_json(r)}, ensure_ascii=False, default=str)
    return json.dumps({"status": "error", "error": f"Unknown media op: {t}"}, ensure_ascii=False)


# ===========================================================================
# 6. BROWSER — all browser automation
# ===========================================================================

_BROWSER_SCHEMA = {
    "name": "browser",
    "description": (
        "Unified browser automation. Types: navigate, click, type, scroll, "
        "snapshot, back, press, get_images, screenshot, console. "
        "Batch via data={'operations': [...]}. Empty data returns snapshot."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "data": {"type": "object", "additionalProperties": True},
        },
        "required": ["data"],
    },
}


def _handle_browser(args, **kw):
    data = args.get("data", {})
    ops = []
    if isinstance(data, dict) and "type" in data:
        ops = [data]
    else:
        ops = data.get("operations", [])
    _ensure_handlers()
    if not ops:
        r = _bsnap(full=False)
        return json.dumps({"status": "ok", "type": "snapshot", "result": _ensure_json(r)}, ensure_ascii=False, default=str)
    results = {}
    for op in ops:
        oid = op.get("id", str(len(results)))
        t = op["type"]
        try:
            if t == "navigate":
                r = _bnav(url=op.get("url", ""))
            elif t == "click":
                r = _bclick(ref=op.get("ref", ""))
            elif t == "type":
                r = _btype(ref=op.get("ref", ""), text=op.get("text", ""))
            elif t == "scroll":
                r = _bscroll(direction=op.get("direction", "down"))
            elif t == "snapshot":
                r = _bsnap(full=op.get("full", False))
            elif t == "back":
                r = _bback()
            elif t == "press":
                r = _bpress(key=op.get("key", ""))
            elif t == "get_images":
                r = _bimgs()
            elif t == "screenshot":
                r = _bvision(question=op.get("question", ""), annotate=op.get("annotate", False))
            elif t == "console":
                r = _bconsole()
            else:
                results[oid] = {"status": "error", "error": f"Unknown browser op: {t}"}
                continue
            results[oid] = {"status": "ok", "type": t, "result": _ensure_json(r)}
        except Exception as e:
            results[oid] = {"status": "error", "error": str(e)}
    return json.dumps({"status": "ok", "results": results}, ensure_ascii=False, default=str)


# ===========================================================================
# Register all 6 unified tools
# ===========================================================================

_TOOL_CONFIGS = [
    ("filesystem", "Unified filesystem. Read/write/patch/search files. Batch multiple files in one call. Types: read, write, patch, search, grep_and_read.", "\U0001f4c1", _handle_filesystem),
    ("research", "Unified web research. Search, extract, deep-dive. Batch operations in one call. Types: search, extract, deep_dive.", "\U0001f50d", _handle_research),
    ("code", "Unified code execution. Types: python (execute_code), shell (terminal), process (process management), delegate (subagent). Prefer python for multi-step logic.", "\u26a1", _handle_code),
    ("system", "Unified system management. Types: memory, todo, cron, clarify, send_message, session_search, skill. Batch via operations.", "\u2699\ufe0f", _handle_system),
    ("media", "Unified media. Types: vision (image analysis), generate_image, tts (text-to-speech).", "\U0001f3a8", _handle_media),
    ("browser", "Unified browser automation. Types: navigate, click, type, scroll, snapshot, back, press, get_images, screenshot, console. Empty data returns snapshot.", "\U0001f310", _handle_browser),
]

_OP_TYPES = {
    "filesystem": ["read", "write", "patch", "search", "grep_and_read"],
    "research": ["search", "extract", "deep_dive"],
    "code": ["python", "shell", "process", "delegate"],
    "system": ["memory", "todo", "cron", "clarify", "send_message", "session_search", "skill"],
    "media": ["vision", "generate_image", "tts"],
    "browser": ["navigate", "click", "type", "scroll", "snapshot", "back", "press", "get_images", "screenshot", "console"],
}

for _name, _desc, _emoji, _handler in _TOOL_CONFIGS:
    registry.register(
        name=_name,
        toolset="unified",
        schema={
            "name": _name,
            "description": _desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": _OP_TYPES[_name]},
                                },
                                "required": ["type"],
                                "additionalProperties": True,
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "operations": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "type": {"type": "string", "enum": _OP_TYPES[_name]},
                                                "id": {"type": "string"},
                                            },
                                            "required": ["type"],
                                            "additionalProperties": True,
                                        },
                                    }
                                },
                                "required": ["operations"],
                            },
                        ]
                    }
                },
                "required": ["data"],
            },
        },
        handler=_handler,
        check_fn=None,
        emoji=_emoji,
        max_result_size_chars=200_000,
    )
