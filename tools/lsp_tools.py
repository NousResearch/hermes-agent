"""LSP query tools — code intelligence for agent-driven development.

These tools let the agent ask a running language server about source code:

- ``lsp_go_to_definition`` — find where a symbol is defined
- ``lsp_find_references`` — find all usages of a symbol
- ``lsp_hover`` — get type information and documentation
- ``lsp_document_symbols`` — list all symbols in a file
- ``lsp_workspace_symbols`` — search symbols across the project

Unlike the post-write diagnostics that are automatically injected into
``write_file``/``patch`` results, these are **explicit agent-callable tools**
for asking code-intelligence questions about files the agent hasn't
just edited or about the project structure in general.

When the LSP service is not active (disabled in config, no git workspace,
or no matching server), the tools return a clear error message explaining
why.  This mirrors how ``lsp_diagnostics`` in ``file_operations.py``
silently returns empty when LSP is unavailable — the tools need to be
more explicit because the agent *asked* for the information.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger("tools.lsp_tools")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_lsp_service():
    """Best-effort: return the LSP service singleton, or None."""
    try:
        from agent.lsp import get_service
        svc = get_service()
        if svc is not None and svc.is_active():
            return svc
    except Exception:  # noqa: BLE001
        pass
    return None


def _lsp_unavailable_reason(path: str = "") -> str:
    """Return a human-readable explanation of why LSP is unavailable."""
    import os

    reasons = []
    try:
        cfg_path = os.path.join(
            os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
            "config.yaml",
        )
        import yaml
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            lsp_cfg = cfg.get("lsp", {})
            if not lsp_cfg.get("enabled", True):
                reasons.append("LSP is disabled in config (lsp.enabled: false)")
            else:
                reasons.append("LSP is enabled but either no git workspace is detected or no matching language server is installed")
        else:
            reasons.append("No config.yaml found")
    except Exception:  # noqa: BLE001
        reasons.append("Could not read config")

    if not reasons:
        reasons.append("LSP service is not active")

    # Check if there's at least a git workspace
    try:
        git_root = (
            subprocess_check_output(
                ["git", "rev-parse", "--show-toplevel"],
                timeout=5,
            )
            if False  # skip in import
            else ""
        )
        if git_root:
            reasons.append(f"Detected git workspace at {git_root.strip()}")
        else:
            reasons.append("No git workspace detected (LSP requires a git repository)")
    except Exception:  # noqa: BLE001
        pass

    return "; ".join(reasons)


def _run_query(method: str, params: dict, file_path: str = "") -> Optional[Any]:
    """Run an LSP query synchronously and return the raw result, or None.

    This is the synchronous bridge: it obtains the LSPService singleton,
    calls ``query_lsp_sync``, and returns the result.  On any failure,
    returns None.
    """
    svc = _get_lsp_service()
    if svc is None:
        return None
    try:
        return svc.query_lsp_sync(file_path or params.get("textDocument", {}).get("uri", ""), method, params)
    except Exception as e:  # noqa: BLE001
        logger.debug("LSP query %s failed: %s", method, e)
        return None


def _format_hover(hover_result: Optional[dict]) -> str:
    """Format a Hover result into readable text.

    Hover returns (contents, range) where ``contents`` is a
    ``MarkupContent | MarkedString | MarkedString[]``.
    """
    if hover_result is None:
        return "(no hover information)"
    contents = hover_result.get("contents")
    if contents is None:
        return "(no hover information)"

    lines: list[str] = []

    if isinstance(contents, str):
        lines.append(contents)
    elif isinstance(contents, list):
        for item in contents:
            line = _format_marked_string(item)
            if line:
                lines.append(line)
    elif isinstance(contents, dict):
        kind = contents.get("kind", "")
        value = contents.get("value", "")
        if kind == "markdown":
            lines.append(value)
        else:
            lines.append(f"{value}")

    rng = hover_result.get("range")
    if rng:
        start = rng.get("start", {})
        end = rng.get("end", {})
        lines.append(
            f"(range: line {start.get('line',0)+1}:{start.get('character',0)+1}"
            f" — line {end.get('line',0)+1}:{end.get('character',0)+1})"
        )

    return "\n".join(lines) if lines else "(no hover information)"


def _format_marked_string(item: Any) -> str:
    """Format a single MarkedString (string or {language, value})."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        lang = item.get("language", "")
        value = item.get("value", "")
        if lang and value:
            return f"[{lang}]\n{value}"
        return value or ""
    return str(item) if item else ""


def _format_locations(locations: list[dict], label: str) -> str:
    """Format a list of LSP Location objects into readable output.

    Each Location is ``{uri, range: {start, end}}``.
    """
    if not locations:
        return f"(no {label})"
    lines: list[str] = [f"{label} ({len(locations)}):"]
    for loc in locations:
        from agent.lsp.client import uri_to_path

        uri = loc.get("uri", "")
        rng = loc.get("range", {})
        start = rng.get("start", {})
        end = rng.get("end", {})
        path = uri_to_path(uri) if uri else "?"
        lines.append(
            f"  {path}:{start.get('line',0)+1}:{start.get('character',0)+1}"
            f" — line {end.get('line',0)+1}:{end.get('character',0)+1}"
        )
    return "\n".join(lines)


def _format_find_references(locations: list[dict]) -> str:
    """Format find-references results into readable output."""
    return _format_locations(locations, "References")


def _format_go_to_definition(locations: list[dict]) -> str:
    """Format go-to-definition results into readable output."""
    return _format_locations(locations, "Definitions")


def _format_symbols(symbols: list[dict]) -> str:
    """Format a list of DocumentSymbol / SymbolInformation into readable output."""
    if not symbols:
        return "(no symbols found)"
    parts: list[str] = [f"Symbols ({len(symbols)}):"]
    for sym in symbols:
        parts.append(_format_single_symbol(sym, ""))
    return "\n".join(parts)


def _format_single_symbol(sym: dict, indent: str) -> str:
    """Format one symbol node, recursing into children."""
    from agent.lsp.client import _format_symbol, uri_to_path

    name = sym.get("name", "?")
    kind_val = sym.get("kind", 0)
    _SYMBOL_KIND_NAMES: dict[int, str] = {
        1: "File", 2: "Module", 3: "Namespace", 4: "Package",
        5: "Class", 6: "Method", 7: "Property", 8: "Field",
        9: "Constructor", 10: "Enum", 11: "Interface",
        12: "Function", 13: "Variable", 14: "Constant",
        15: "String", 16: "Number", 17: "Boolean", 18: "Array",
        19: "Object", 20: "Key", 21: "Null", 22: "EnumMember",
        23: "Struct", 24: "Event", 25: "Operator", 26: "TypeParameter",
    }
    kind = _SYMBOL_KIND_NAMES.get(kind_val, f"kind={kind_val}")
    detail = sym.get("detail", "")
    detail_str = f" — {detail}" if detail else ""

    children = sym.get("children")
    if children is not None:
        # DocumentSymbol tree node
        entries = [f"{indent}{kind:20s}  {name}{detail_str}"]
        for child in children:
            entries.append(_format_single_symbol(child, indent + "  "))
        return "\n".join(entries)
    else:
        # SymbolInformation — has location
        loc = sym.get("location", {})
        if loc:
            loc_uri = loc.get("uri", "")
            rng = loc.get("range", {})
            start = rng.get("start", {})
            path = uri_to_path(loc_uri) if loc_uri else "?"
            return f"{path}:{start.get('line',0)+1}:{start.get('character',0)+1}  {kind:20s}  {name}{detail_str}"
        return f"{indent}{kind:20s}  {name}{detail_str}"


# ------------------------------------------------------------------
# Tool handlers
# ------------------------------------------------------------------

def _handle_go_to_definition(args: dict, **kw) -> str:
    """Return definition locations for the symbol at (path, line, character)."""
    path = (args.get("path") or "").strip()
    line = int(args.get("line", 0))
    character = int(args.get("character", 0))

    if not path:
        return json.dumps({"error": "path is required", "result": ""})

    svc = _get_lsp_service()
    if svc is None:
        return json.dumps({
            "error": "LSP not available",
            "reason": _lsp_unavailable_reason(path),
            "result": "",
        })

    result = svc.query_lsp_sync(
        path,
        "textDocument/definition",
        {
            "textDocument": {"uri": "file://" + __import__("os").path.abspath(path)},
            "position": {"line": line, "character": character},
        },
    )
    if result is None:
        return json.dumps({"error": "", "result": "(no definition found)"})

    locations = result if isinstance(result, list) else [result]
    return json.dumps({"error": "", "result": _format_go_to_definition(locations)})


def _handle_find_references(args: dict, **kw) -> str:
    """Return reference locations for the symbol at (path, line, character)."""
    path = (args.get("path") or "").strip()
    line = int(args.get("line", 0))
    character = int(args.get("character", 0))
    include_declaration = bool(args.get("include_declaration", False))

    if not path:
        return json.dumps({"error": "path is required", "result": ""})

    svc = _get_lsp_service()
    if svc is None:
        return json.dumps({
            "error": "LSP not available",
            "reason": _lsp_unavailable_reason(path),
            "result": "",
        })

    result = svc.query_lsp_sync(
        path,
        "textDocument/references",
        {
            "textDocument": {"uri": "file://" + __import__("os").path.abspath(path)},
            "position": {"line": line, "character": character},
            "context": {"includeDeclaration": include_declaration},
        },
    )
    if result is None:
        locs: list = []
    else:
        locs = result if isinstance(result, list) else []
    return json.dumps({"error": "", "result": _format_find_references(locs)})


def _handle_hover(args: dict, **kw) -> str:
    """Return type info and documentation for the symbol at (path, line, character)."""
    path = (args.get("path") or "").strip()
    line = int(args.get("line", 0))
    character = int(args.get("character", 0))

    if not path:
        return json.dumps({"error": "path is required", "result": ""})

    svc = _get_lsp_service()
    if svc is None:
        return json.dumps({
            "error": "LSP not available",
            "reason": _lsp_unavailable_reason(path),
            "result": "",
        })

    result = svc.query_lsp_sync(
        path,
        "textDocument/hover",
        {
            "textDocument": {"uri": "file://" + __import__("os").path.abspath(path)},
            "position": {"line": line, "character": character},
        },
    )
    return json.dumps({"error": "", "result": _format_hover(result)})


def _handle_document_symbols(args: dict, **kw) -> str:
    """Return all symbols defined in a file."""
    path = (args.get("path") or "").strip()

    if not path:
        return json.dumps({"error": "path is required", "result": ""})

    svc = _get_lsp_service()
    if svc is None:
        return json.dumps({
            "error": "LSP not available",
            "reason": _lsp_unavailable_reason(path),
            "result": "",
        })

    result = svc.query_lsp_sync(
        path,
        "textDocument/documentSymbol",
        {
            "textDocument": {"uri": "file://" + __import__("os").path.abspath(path)},
        },
    )
    symbols = result if isinstance(result, list) else []
    return json.dumps({"error": "", "result": _format_symbols(symbols)})


def _handle_workspace_symbols(args: dict, **kw) -> str:
    """Search for symbols matching ``query`` across the project."""
    query = (args.get("query") or "").strip()

    if not query:
        return json.dumps({"error": "query is required", "result": ""})

    svc = _get_lsp_service()
    if svc is None:
        return json.dumps({
            "error": "LSP not available",
            "reason": _lsp_unavailable_reason(),
            "result": "",
        })

    # workspace/symbol doesn't need a file path; pass cwd for workspace detection
    cwd = __import__("os").getcwd()
    result = svc.query_lsp_sync(
        cwd,
        "workspace/symbol",
        {"query": query},
    )
    symbols = result if isinstance(result, list) else []
    return json.dumps({"error": "", "result": _format_symbols(symbols)})


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------

LSP_GO_TO_DEFINITION_SCHEMA = {
    "name": "lsp_go_to_definition",
    "description": (
        "Find where a symbol is defined.  Provide a file path and a line/column "
        "position (0-based) to locate a symbol's definition.  Returns one or more "
        "file locations in ``path:line:col`` format."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the source file containing the symbol.",
            },
            "line": {
                "type": "integer",
                "description": "0-based line number of the symbol.",
            },
            "character": {
                "type": "integer",
                "description": "0-based character offset on the line.",
            },
        },
        "required": ["path", "line", "character"],
    },
}

LSP_FIND_REFERENCES_SCHEMA = {
    "name": "lsp_find_references",
    "description": (
        "Find all usages of a symbol.  Provide a file path and a line/column "
        "position (0-based) to locate a symbol's definition.  Returns all "
        "reference locations in ``path:line:col`` format."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the source file containing the symbol.",
            },
            "line": {
                "type": "integer",
                "description": "0-based line number of the symbol.",
            },
            "character": {
                "type": "integer",
                "description": "0-based character offset on the line.",
            },
            "include_declaration": {
                "type": "boolean",
                "description": "Whether to include the declaration site in results (default false).",
            },
        },
        "required": ["path", "line", "character"],
    },
}

LSP_HOVER_SCHEMA = {
    "name": "lsp_hover",
    "description": (
        "Get type information and documentation for the symbol at a given "
        "position.  Provide a file path and a line/column position (0-based). "
        "Returns the type signature and documentation if available."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the source file containing the symbol.",
            },
            "line": {
                "type": "integer",
                "description": "0-based line number of the symbol.",
            },
            "character": {
                "type": "integer",
                "description": "0-based character offset on the line.",
            },
        },
        "required": ["path", "line", "character"],
    },
}

LSP_DOCUMENT_SYMBOLS_SCHEMA = {
    "name": "lsp_document_symbols",
    "description": (
        "List all symbols (classes, methods, functions, variables, etc.) defined "
        "in a source file.  Provide an absolute file path.  Returns a structured "
        "listing with indentation for nested symbols."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the source file.",
            },
        },
        "required": ["path"],
    },
}

LSP_WORKSPACE_SYMBOLS_SCHEMA = {
    "name": "lsp_workspace_symbols",
    "description": (
        "Search for symbols matching a query string across the entire project. "
        "Useful when you know the name (or part of it) of a class, function, "
        "or variable but not which file it lives in.  Returns matching symbols "
        "with their file locations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to match against symbol names.",
            },
        },
        "required": ["query"],
    },
}


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

from tools.registry import registry  # noqa: E402

registry.register(
    name="lsp_go_to_definition",
    toolset="coding",
    schema=LSP_GO_TO_DEFINITION_SCHEMA,
    handler=_handle_go_to_definition,
    emoji="🎯",
)

registry.register(
    name="lsp_find_references",
    toolset="coding",
    schema=LSP_FIND_REFERENCES_SCHEMA,
    handler=_handle_find_references,
    emoji="🔗",
)

registry.register(
    name="lsp_hover",
    toolset="coding",
    schema=LSP_HOVER_SCHEMA,
    handler=_handle_hover,
    emoji="ℹ️",
)

registry.register(
    name="lsp_document_symbols",
    toolset="coding",
    schema=LSP_DOCUMENT_SYMBOLS_SCHEMA,
    handler=_handle_document_symbols,
    emoji="📋",
)

registry.register(
    name="lsp_workspace_symbols",
    toolset="coding",
    schema=LSP_WORKSPACE_SYMBOLS_SCHEMA,
    handler=_handle_workspace_symbols,
    emoji="🔍",
)