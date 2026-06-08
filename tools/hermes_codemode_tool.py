"""Hermes Codemode: lazy tool schemas + programmable tool orchestration.

This is a local, Hermes-owned analogue of Cloudflare Codemode.  It exposes a
small control-plane surface (status/schema/execute) instead of requiring every
underlying tool schema to be model-visible on every turn.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from hermes_constants import get_hermes_home
from tools.registry import registry

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
CODEMODE_TOOL_NAMES = frozenset({
    "hermes_codemode_status",
    "hermes_codemode_schema",
    "hermes_codemode_execute",
})

# Conservative read-only set for plan mode. Anything not listed here requires
# mode="apply" because it may mutate local state, remote state, user-facing
# messages, browser state, or long-running process state.
READ_ONLY_TOOLS = frozenset({
    "read_file",
    "search_files",
    "web_search",
    "web_extract",
    "skills_list",
    "skill_view",
    "session_search",
    "vision_analyze",
    "browser_snapshot",
    "browser_get_images",
    "browser_console",
    "ha_list_entities",
    "ha_get_state",
    "ha_list_services",
})

_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "Exception": Exception,
    "ValueError": ValueError,
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


def _tool_name(td: Dict[str, Any]) -> str:
    return str((td.get("function") or {}).get("name") or "")


def _tool_description(td: Dict[str, Any]) -> str:
    return str((td.get("function") or {}).get("description") or "")


def _tool_parameters(td: Dict[str, Any]) -> Dict[str, Any]:
    params = (td.get("function") or {}).get("parameters") or {}
    return params if isinstance(params, dict) else {}


def _toolset_for(name: str) -> str:
    try:
        return registry.get_toolset_for_tool(name) or "unknown"
    except Exception:
        return "unknown"


def _is_mutating(name: str) -> bool:
    return name not in READ_ONLY_TOOLS


def _keywords_for(name: str, description: str, toolset: str) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for token in _tokenize(f"{name.replace('_', ' ')} {description} {toolset}"):
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out[:24]


def _input_fields(parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
    properties = parameters.get("properties") or {}
    if not isinstance(properties, dict):
        return []
    required = set(parameters.get("required") or [])
    fields: List[Dict[str, Any]] = []
    for name, spec in properties.items():
        if not isinstance(spec, dict):
            spec = {}
        field_type = spec.get("type") or "any"
        if isinstance(field_type, list):
            field_type = "|".join(str(x) for x in field_type)
        field = {
            "name": name,
            "type": str(field_type),
            "required": name in required,
        }
        desc = spec.get("description")
        if desc:
            field["description"] = str(desc)[:300]
        enum = spec.get("enum")
        if enum:
            field["enum"] = enum
        fields.append(field)
    return fields


def _compact_definition(td: Dict[str, Any], *, include_schema: bool) -> Dict[str, Any]:
    fn = td.get("function") or {}
    name = str(fn.get("name") or "")
    description = str(fn.get("description") or "")
    parameters = _tool_parameters(td)
    toolset = _toolset_for(name)
    definition = {
        "name": name,
        "description": description[:900],
        "product": toolset,
        "toolset": toolset,
        "mutating": _is_mutating(name),
        "required": list(parameters.get("required") or []),
        "aliases": [name.replace("_", " ")],
        "keywords": _keywords_for(name, description, toolset),
        "inputFields": _input_fields(parameters),
    }
    if include_schema:
        definition["inputSchema"] = parameters
    return definition


def _scored_matches(defs: Iterable[Dict[str, Any]], query: str, limit: int) -> List[Dict[str, Any]]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for td in defs:
        name = _tool_name(td)
        desc = _tool_description(td)
        toolset = _toolset_for(name)
        text = f"{name.replace('_', ' ')} {desc} {toolset}"
        tokens = _tokenize(text)
        token_set = set(tokens)
        score = 0.0
        matched: List[str] = []
        raw = text.lower()
        for token in query_tokens:
            if token in token_set:
                score += 4.0 if token in _tokenize(name) else 2.0
                matched.append(token)
            elif len(token) >= 4 and token in raw:
                score += 1.0
                matched.append(token)
        if query.lower().strip() in raw:
            score += 4.0
        if score > 0:
            item = _compact_definition(td, include_schema=False)
            item["score"] = round(score, 3)
            item["matchedTokens"] = sorted(set(matched))
            scored.append((score, item))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["name"]))
    return [item for _, item in scored[:limit]]


def _method_scope_toolsets(enabled_toolsets: Optional[List[str]]) -> Optional[List[str]]:
    """Resolve visible Codemode bridge toolsets to their hidden backing catalog.

    A session can opt into the lean ``hermes-codemode-cli`` surface, where the
    model sees only three Codemode tools but Codemode itself can lazily describe
    and call the ordinary ``hermes-cli`` methods. Other restricted toolsets keep
    their restriction exactly as passed.
    """
    if not enabled_toolsets:
        return enabled_toolsets
    normalized = [str(item) for item in enabled_toolsets]
    if "hermes-codemode-cli" in normalized or "hermes_codemode" in normalized:
        return ["hermes-cli"]
    return enabled_toolsets


def _available_tool_defs(
    *,
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    # Import lazily to avoid a discovery-time cycle: model_tools imports all
    # self-registering tools, including this module.
    from model_tools import get_tool_definitions

    defs = get_tool_definitions(
        enabled_toolsets=_method_scope_toolsets(enabled_toolsets),
        disabled_toolsets=disabled_toolsets,
        quiet_mode=True,
        skip_tool_search_assembly=True,
    ) or []
    return [td for td in defs if _tool_name(td) and _tool_name(td) not in CODEMODE_TOOL_NAMES]


def _available_method_names(**scope: Any) -> frozenset[str]:
    return frozenset(_tool_name(td) for td in _available_tool_defs(**scope))


def hermes_codemode_status(
    *,
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
) -> str:
    defs = _available_tool_defs(enabled_toolsets=enabled_toolsets, disabled_toolsets=disabled_toolsets)
    methods = sorted(_tool_name(td) for td in defs)
    log_path = _audit_log_path()
    return json.dumps({
        "schemaVersion": SCHEMA_VERSION,
        "methodCount": len(methods),
        "methodsPreview": methods[:20],
        "methodsRemaining": max(0, len(methods) - 20),
        "planMode": {
            "readOnlyMethods": sorted(m for m in methods if not _is_mutating(m)),
            "default": True,
        },
        "applyMode": {
            "mutationsAllowed": True,
            "note": "Mutating methods require mode='apply'; underlying Hermes hooks, approvals, and policies still run.",
        },
        "auditLog": str(log_path),
        "lookup": "Use hermes_codemode_schema(query=...) or methods=[...] for exact input schemas.",
    }, ensure_ascii=False)


def hermes_codemode_schema(
    query: Optional[str] = None,
    methods: Optional[List[str]] = None,
    limit: int = 8,
    include_schema: Optional[bool] = None,
    *,
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
) -> str:
    defs = _available_tool_defs(enabled_toolsets=enabled_toolsets, disabled_toolsets=disabled_toolsets)
    by_name = {_tool_name(td): td for td in defs}
    requested = [str(m).strip() for m in (methods or []) if str(m).strip()]
    include_exact_schema = True if include_schema is None else bool(include_schema)

    definitions: List[Dict[str, Any]] = []
    missing: List[str] = []
    for name in requested:
        td = by_name.get(name)
        if td is None:
            missing.append(name)
            continue
        definitions.append(_compact_definition(td, include_schema=include_exact_schema))

    matches: List[Dict[str, Any]] = []
    if query and not requested:
        matches = _scored_matches(defs, str(query), max(1, min(int(limit or 8), 25)))

    return json.dumps({
        "schemaVersion": SCHEMA_VERSION,
        "toolCount": len(defs),
        "query": query,
        "requestedMethods": requested,
        "matches": matches,
        "definitions": definitions,
        "missing": missing,
        "note": (
            "Search matches omit full inputSchema to keep context small. "
            "Request exact methods to get inputSchema."
        ),
    }, ensure_ascii=False)


class CodemodeError(RuntimeError):
    pass


class CodemodeAPI:
    def __init__(
        self,
        *,
        mode: str,
        allowed_methods: frozenset[str],
        task_id: Optional[str],
        enabled_toolsets: Optional[List[str]],
        disabled_toolsets: Optional[List[str]],
        call_log: List[Dict[str, Any]],
    ) -> None:
        self._mode = mode
        self._allowed_methods = allowed_methods
        self._task_id = task_id
        self._enabled_toolsets = enabled_toolsets
        self._disabled_toolsets = disabled_toolsets
        self._call_log = call_log

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        def _caller(arguments: Optional[Dict[str, Any]] = None):
            return self.call(name, arguments or {})

        return _caller

    def call(self, name: str, arguments: Optional[Dict[str, Any]] = None):
        name = str(name).strip()
        if not name:
            raise CodemodeError("method name is required")
        if name in CODEMODE_TOOL_NAMES:
            raise CodemodeError(f"{name} cannot call itself through codemode")
        if name not in self._allowed_methods:
            raise CodemodeError(f"{name} is not available in this codemode session")
        if self._mode == "plan" and _is_mutating(name):
            raise CodemodeError(f"{name} is not allowed in plan mode; rerun with mode='apply' if you intend to mutate state")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise CodemodeError("codemode method arguments must be an object/dict")

        started = time.monotonic()
        result_text = _dispatch_tool_call(
            name,
            arguments,
            task_id=self._task_id,
            enabled_toolsets=self._enabled_toolsets,
            disabled_toolsets=self._disabled_toolsets,
        )
        duration_ms = int((time.monotonic() - started) * 1000)
        self._call_log.append({
            "method": name,
            "mutating": _is_mutating(name),
            "durationMs": duration_ms,
        })
        try:
            return json.loads(result_text)
        except Exception:
            return result_text


def _dispatch_tool_call(function_name: str, function_args: Dict[str, Any], **kwargs: Any) -> str:
    from model_tools import handle_function_call

    return handle_function_call(function_name, function_args, **kwargs)


def _run_user_code(code: str, codemode: CodemodeAPI) -> Any:
    local_vars: Dict[str, Any] = {}
    global_vars = {
        "__builtins__": _SAFE_BUILTINS,
        "json": json,
        "math": math,
        "re": re,
    }
    stripped = (code or "").strip()
    if not stripped:
        raise CodemodeError("code is required")

    if stripped.startswith("lambda"):
        fn = eval(stripped, global_vars, local_vars)  # noqa: S307 - restricted globals, local control-plane DSL
    else:
        exec(stripped, global_vars, local_vars)  # noqa: S102 - intentional codemode execution with restricted globals
        fn = local_vars.get("main") or global_vars.get("main")
        if fn is None and "result" in local_vars:
            return local_vars["result"]
    if not callable(fn):
        raise CodemodeError("code must define callable main(codemode) or evaluate to a lambda")

    sig = inspect.signature(fn)
    if len(sig.parameters) == 0:
        return fn()
    return fn(codemode)


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _audit_log_path() -> Path:
    return get_hermes_home() / "logs" / "hermes-codemode.jsonl"


def _write_audit(entry: Dict[str, Any]) -> None:
    try:
        path = _audit_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.debug("failed to write hermes codemode audit log: %s", exc)


def hermes_codemode_execute(
    code: str,
    mode: str = "plan",
    *,
    task_id: Optional[str] = None,
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
) -> str:
    normalized_mode = str(mode or "plan").strip().lower()
    if normalized_mode not in {"plan", "apply"}:
        return json.dumps({"ok": False, "error": "mode must be 'plan' or 'apply'"}, ensure_ascii=False)

    allowed = _available_method_names(enabled_toolsets=enabled_toolsets, disabled_toolsets=disabled_toolsets)
    call_log: List[Dict[str, Any]] = []
    api = CodemodeAPI(
        mode=normalized_mode,
        allowed_methods=allowed,
        task_id=task_id,
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
        call_log=call_log,
    )
    started = time.monotonic()
    audit_base = {
        "ts": time.time(),
        "mode": normalized_mode,
        "taskId": task_id or "",
        "codePreview": (code or "")[:500],
    }
    try:
        result = _run_user_code(code, api)
        payload = {
            "ok": True,
            "mode": normalized_mode,
            "durationMs": int((time.monotonic() - started) * 1000),
            "calls": call_log,
            "result": _jsonable(result),
        }
        _write_audit({**audit_base, "ok": True, "calls": call_log})
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        payload = {
            "ok": False,
            "mode": normalized_mode,
            "durationMs": int((time.monotonic() - started) * 1000),
            "calls": call_log,
            "error": str(exc),
            "errorType": type(exc).__name__,
        }
        _write_audit({**audit_base, "ok": False, "calls": call_log, "error": str(exc)})
        return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool schemas and registry
# ---------------------------------------------------------------------------

HERMES_CODEMODE_STATUS_SCHEMA = {
    "name": "hermes_codemode_status",
    "description": "Show Hermes Codemode status: available method count, preview, safety modes, and audit log path.",
    "parameters": {"type": "object", "properties": {}},
}

HERMES_CODEMODE_SCHEMA_SCHEMA = {
    "name": "hermes_codemode_schema",
    "description": (
        "Search or fetch exact schemas for Hermes Codemode methods. Use query for compact matches; "
        "use methods=[...] to get exact input schemas just-in-time."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keywords describing the method you need."},
            "methods": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exact method names to describe, e.g. ['read_file', 'terminal'].",
            },
            "limit": {"type": "integer", "description": "Max search matches. Default 8."},
            "include_schema": {"type": "boolean", "description": "Include full inputSchema for exact methods. Default true."},
        },
    },
}

HERMES_CODEMODE_EXECUTE_SCHEMA = {
    "name": "hermes_codemode_execute",
    "description": (
        "Run a small Python codemode program against Hermes tools via a codemode object. "
        "Use mode='plan' for read-only inspection; mode='apply' permits mutating methods while preserving underlying Hermes hooks and approvals. "
        "Code must define main(codemode) or be a lambda. Example: def main(codemode): return codemode.read_file({'path':'README.md'})."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code defining main(codemode) or a lambda codemode: ..."},
            "mode": {"type": "string", "enum": ["plan", "apply"], "description": "plan is read-only; apply permits mutations."},
        },
        "required": ["code"],
    },
}

registry.register(
    name="hermes_codemode_status",
    toolset="hermes_codemode",
    schema=HERMES_CODEMODE_STATUS_SCHEMA,
    handler=lambda args, **kw: hermes_codemode_status(
        enabled_toolsets=kw.get("enabled_toolsets"),
        disabled_toolsets=kw.get("disabled_toolsets"),
    ),
    emoji="🧩",
)

registry.register(
    name="hermes_codemode_schema",
    toolset="hermes_codemode",
    schema=HERMES_CODEMODE_SCHEMA_SCHEMA,
    handler=lambda args, **kw: hermes_codemode_schema(
        query=args.get("query"),
        methods=args.get("methods"),
        limit=args.get("limit", 8),
        include_schema=args.get("include_schema"),
        enabled_toolsets=kw.get("enabled_toolsets"),
        disabled_toolsets=kw.get("disabled_toolsets"),
    ),
    emoji="🧩",
)

registry.register(
    name="hermes_codemode_execute",
    toolset="hermes_codemode",
    schema=HERMES_CODEMODE_EXECUTE_SCHEMA,
    handler=lambda args, **kw: hermes_codemode_execute(
        code=args.get("code", ""),
        mode=args.get("mode", "plan"),
        task_id=kw.get("task_id"),
        enabled_toolsets=kw.get("enabled_toolsets"),
        disabled_toolsets=kw.get("disabled_toolsets"),
    ),
    emoji="🧩",
    max_result_size_chars=100_000,
)
