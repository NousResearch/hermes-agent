#!/usr/bin/env python3
"""
Unified Tools — 6 high-level tools replacing 33 individual ones.

Each unified tool accepts a structured ``data`` parameter that encodes one
or more operations.  The handler dispatches each operation to the existing
tool implementation internally, then aggregates the results.

This dramatically reduces:
  - Tool selection overhead for the LLM (33 to 6 choices)
  - Turn count (N operations in 1 call instead of N calls)
  - Context window waste (fewer tool-call/result round-trip messages)

Design:
  Each tool takes ``data`` (dict or list[dict]).  A single operation dict
  has a ``type`` key that selects the sub-operation.  Independent operations
  are run in parallel where possible; dependent ones are sequenced.

Operations always return a dict with at least ``status: ok | error``
and operation-specific result keys.
"""

import json
import logging
import os
from typing import Any, Dict

from tools.registry import registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy import of existing tool handlers
# ---------------------------------------------------------------------------

_IMPORTED = False

def _ensure_handlers():
    global _IMPORTED
    if _IMPORTED:
        return
    _IMPORTED = True
    _lazy_import()


def _lazy_import():
    # File
    import tools.file_tools as _ft
    globals()["_h_read"] = _ft._handle_read_file
    globals()["_h_write"] = _ft._handle_write_file
    globals()["_h_patch"] = _ft._handle_patch
    globals()["_h_search"] = _ft._handle_search_files
    # Web
    import tools.web_tools as _wt
    globals()["_web_search"] = _wt.web_search_tool
    globals()["_web_extract"] = _wt.web_extract_tool
    # Code
    import tools.code_execution_tool as _ce
    import tools.terminal_tool as _te
    import tools.delegate_tool as _de
    globals()["_exec_code"] = _ce.execute_code_tool
    globals()["_term"] = _te.terminal_tool
    globals()["_proc"] = _te.process_tool
    globals()["_del_task"] = _de.delegate_task_tool
    # System
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
    # Media
    import tools.vision_tools as _vi
    import tools.image_generation_tool as _ig
    import tools.tts_tool as _tt
    globals()["_vision"] = _vi.vision_analyze_tool
    globals()["_img_gen"] = _ig.image_generate_tool
    globals()["_tts_fn"] = _tt.text_to_speech_tool
    # Browser
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


# ---------------------------------------------------------------------------
# Utility: batch executor
# ---------------------------------------------------------------------------

def _call_handler(handler, args, **kw):
    try:
        return handler(args, **kw)
    except Exception as e:
        logger.exception("Unified tool handler error")
        return {"status": "error", "error": str(e)}


def _ensure_json(obj):
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except (json.JSONDecodeError, TypeError):
            return {"raw": obj[:500]}
    return obj

def _run_research(op):
    _ensure_handlers()
    t = op["type"]
    if t == "search":
        r = _web_search(query=op.get("query", ""), limit=op.get("limit", 5))
        return {"status": "ok", "type": "search", "query": op.get("query"), "result": _ensure_json(r)}
    elif t == "extract":
        r = _web_extract(op.get("urls", []))
        return {"status": "ok", "type": "extract", "result": _ensure_json(r)}
    elif t == "deep_dive":
        q = op.get("query", "")
        sr = _web_search(query=q, limit=op.get("limit", 5))
        sd = _ensure_json(sr)
        urls = []
        if isinstance(sd, dict):
            results_list = sd.get("data", {}).get("web", [])
            urls = [r.get("url", "") for r in results_list if r.get("url")][:3]
        er = {}
        if urls:
            er = _ensure_json(_web_extract(urls))
        return {"status": "ok", "type": "deep_dive", "query": q, "search": sd, "extracted": er}
    return {"status": "error", "error": "Unknown research op: " + t}

def _run_fs(op, task_id="default"):
    _ensure_handlers()
    t = op["type"]
    if t == "read":
        paths = op.get("paths", [op.get("path", "")])
        if isinstance(paths, str):
            paths = [paths]
        files = {}
        for p in paths:
            r = _h_read({"path": p, "offset": op.get("offset", 1), "limit": op.get("limit", 500)}, task_id=task_id)
            files[p] = _ensure_json(r)
        return {"status": "ok", "type": "read", "files": files}
    elif t == "write":
        r = _h_write({"path": op.get("path", ""), "content": op.get("content", "")}, task_id=task_id)
        return {"status": "ok", "type": "write", "path": op.get("path"), "result": _ensure_json(r)}
    elif t == "patch":
        r = _h_patch({"path": op.get("path", ""), "old_string": op.get("old", ""), "new_string": op.get("new", ""), "replace_all": op.get("replace_all", False)}, task_id=task_id)
        return {"status": "ok", "type": "patch", "path": op.get("path"), "result": _ensure_json(r)}
    elif t == "search":
        r = _h_search({"pattern": op.get("pattern", ""), "target": op.get("target", "content"),
                        "path": op.get("path", "."), "file_glob": op.get("file_glob"),
                        "limit": op.get("limit", 50), "output_mode": op.get("output_mode", "content")}, task_id=task_id)
        return {"status": "ok", "type": "search", "pattern": op.get("pattern"), "result": _ensure_json(r)}
    elif t == "grep_and_read":
        pattern = op.get("pattern", "")
        path = op.get("path", ".")
        fg = op.get("file_glob")
        sr = _h_search({"pattern": pattern, "target": "content", "path": path,
                         "file_glob": fg, "limit": op.get("limit", 50), "output_mode": "files_only"}, task_id=task_id)
        sd = _ensure_json(sr)
        matched = []
        if isinstance(sd, dict):
            matches = sd.get("matches", sd.get("data", {}).get("matches", []))
            if isinstance(matches, list):
                matched = [m["path"] if isinstance(m, dict) and "path" in m else str(m) for m in matches[:5]]
        contents = {}
        for fp_ in matched:
            r = _h_read({"path": fp_, "offset": 1, "limit": 100}, task_id=task_id)
            contents[fp_] = _ensure_json(r)
        return {"status": "ok", "type": "grep_and_read", "pattern": pattern, "matched": matched, "contents": contents}
    return {"status": "error", "error": "Unknown fs op: " + t}

def _handle_research(args, **kw):
    data = args.get("data", {})
    if isinstance(data, dict) and "type" in data:
        return json.dumps(_run_research(data), ensure_ascii=False, default=str)
    ops = data.get("operations", [])
    if not ops:
        return json.dumps({"status": "error", "error": "No operations specified"}, ensure_ascii=False)
    results = {}
    for op in ops:
        oid = op.get("id", str(len(results)))
        results[oid] = _run_research(op)
    return json.dumps({"status": "ok", "results": results}, ensure_ascii=False, default=str)


def _handle_filesystem(args, **kw):
    data = args.get("data", {})
    task_id = kw.get("task_id") or os.environ.get("HERMES_TASK_ID", "default")
    if isinstance(data, dict) and "type" in data:
        return json.dumps(_run_fs(data, task_id), ensure_ascii=False, default=str)
    ops = data.get("operations", [])
    if not ops:
        return json.dumps({"status": "error", "error": "No operations specified"}, ensure_ascii=False)
    results = {}
    for op in ops:
        oid = op.get("id", str(len(results)))
        results[oid] = _run_fs(op, task_id)
    return json.dumps({"status": "ok", "results": results}, ensure_ascii=False, default=str)


def _handle_code(args, **kw):
    data = args.get("data", {})
    t = data.get("type", "")
    _ensure_handlers()
    if t == "python":
        r = _exec_code(code=data.get("code", ""))
        return json.dumps({"status": "ok", "type": "python", "result": _ensure_json(r)}, ensure_ascii=False, default=str)
    elif t == "shell":
        cmd = data.get("command", "")
        r = _term(command=cmd, timeout=data.get("timeout", 180), workdir=data.get("workdir"))
        return json.dumps({"status": "ok", "type": "shell", "result": _ensure_json(r)}, ensure_ascii=False, default=str)
    elif t == "process":
        r = _proc(**data)
        return json.dumps({"status": "ok", "type": "process", "result": _ensure_json(r)}, ensure_ascii=False, default=str)
    elif t == "delegate":
        r = _del_task(goal=data.get("goal", ""), context=data.get("context", ""),
                      toolsets=data.get("toolsets"))
        return json.dumps({"status": "ok", "type": "delegate", "result": _ensure_json(r)}, ensure_ascii=False, default=str)
    return json.dumps({"status": "error", "error": "Unknown code op: " + t}, ensure_ascii=False)


def _handle_system(args, **kw):
    data = args.get("data", {})
    ops = []
    if isinstance(data, dict) and "type" in data:
        ops = [data]
    else:
        ops = data.get("operations", [])
    if not ops:
        return json.dumps({"status": "error", "error": "No operations specified"}, ensure_ascii=False)
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
                results[oid] = {"status": "error", "error": "Unknown system op: " + t}
        except Exception as e:
            results[oid] = {"status": "error", "error": str(e)}
    return json.dumps({"status": "ok", "results": results}, ensure_ascii=False, default=str)


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
    return json.dumps({"status": "error", "error": "Unknown media op: " + t}, ensure_ascii=False)


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
                results[oid] = {"status": "error", "error": "Unknown browser op: " + t}
                continue
            results[oid] = {"status": "ok", "type": t, "result": _ensure_json(r)}
        except Exception as e:
            results[oid] = {"status": "error", "error": str(e)}
    return json.dumps({"status": "ok", "results": results}, ensure_ascii=False, default=str)

# ===========================================================================
# Register all 6 unified tools
# ===========================================================================

_TOOL_DESCS = {
    "research": (
        "Unified web research. Search, extract, and deep-dive in one call. "
        "Batch independent queries via {operations: [...]}. "
        "Operation types: search, extract, deep_dive."
    ),
    "filesystem": (
        "Unified filesystem operations. Read, write, patch, search, grep_and_read. "
        "Batch via {operations: [...]}. "
        "Read multiple files: {type: 'read', paths: ['a.py', 'b.py']}."
    ),
    "code": (
        "Unified code execution. Types: python (execute_code), shell (terminal), "
        "process, delegate. Single operation per call. "
        "Prefer python for complex multi-step logic."
    ),
    "system": (
        "Unified system management. Types: memory, todo, cron, clarify, "
        "send_message, session_search, skill. Batch via {operations: [...]}."
    ),
    "media": (
        "Unified media. Types: vision (image analysis), generate_image, tts (text-to-speech)."
    ),
    "browser": (
        "Unified browser automation. Types: navigate, click, type, scroll, "
        "snapshot, back, press, get_images, screenshot, console. "
        "Batch via {operations: [...]}. Empty data returns snapshot."
    ),
}

_TOOL_EMOJIS = {
    "research": "\U0001f50d",
    "filesystem": "\U0001f4c1",
    "code": "\u26a1",
    "system": "\u2699\ufe0f",
    "media": "\U0001f3a8",
    "browser": "\U0001f310",
}

_TOOL_OP_TYPES = {
    "research": ["search", "extract", "deep_dive"],
    "filesystem": ["read", "write", "patch", "search", "grep_and_read"],
    "code": ["python", "shell", "process", "delegate"],
    "system": ["memory", "todo", "cron", "clarify", "send_message", "session_search", "skill"],
    "media": ["vision", "generate_image", "tts"],
    "browser": [
        "navigate", "click", "type", "scroll", "snapshot",
        "back", "press", "get_images", "screenshot", "console",
    ],
}

_HANDLERS_MAP = {
    "research": _handle_research,
    "filesystem": _handle_filesystem,
    "code": _handle_code,
    "system": _handle_system,
    "media": _handle_media,
    "browser": _handle_browser,
}

for _name, _desc in _TOOL_DESCS.items():
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
                                    "type": {"type": "string", "enum": _TOOL_OP_TYPES[_name]},
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
                                                "type": {"type": "string", "enum": _TOOL_OP_TYPES[_name]},
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
        handler=_HANDLERS_MAP[_name],
        check_fn=None,
        emoji=_TOOL_EMOJIS[_name],
        max_result_size_chars=200_000,
    )
