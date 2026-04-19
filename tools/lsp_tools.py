#!/usr/bin/env python3
"""Local-first LSP tools for repo/code grounding via document symbols, definitions, and diagnostics.

These LSP helpers stay local-first and host-dependent, but their primary product
framing is the canonical `repo-code-knowledge` surface: deeper semantic grounding
inside a local codebase when AST-only structure is not enough.
"""

from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)


_LANGUAGE_IDS = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascriptreact",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescriptreact",
    ".go": "go",
    ".rs": "rust",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "shellscript",
    ".bash": "shellscript",
}

_AUTO_DETECT_COMMANDS = {
    "python": [["pylsp"], ["pyright-langserver", "--stdio"], ["jedi-language-server"]],
    "javascript": [["typescript-language-server", "--stdio"], ["vtsls", "--stdio"]],
    "javascriptreact": [["typescript-language-server", "--stdio"], ["vtsls", "--stdio"]],
    "typescript": [["typescript-language-server", "--stdio"], ["vtsls", "--stdio"]],
    "typescriptreact": [["typescript-language-server", "--stdio"], ["vtsls", "--stdio"]],
    "go": [["gopls"]],
    "rust": [["rust-analyzer"]],
    "json": [["vscode-json-language-server", "--stdio"]],
    "yaml": [["yaml-language-server", "--stdio"]],
    "shellscript": [["bash-language-server", "start"]],
}

_SYMBOL_KINDS = {
    1: "File",
    2: "Module",
    3: "Namespace",
    4: "Package",
    5: "Class",
    6: "Method",
    7: "Property",
    8: "Field",
    9: "Constructor",
    10: "Enum",
    11: "Interface",
    12: "Function",
    13: "Variable",
    14: "Constant",
    15: "String",
    16: "Number",
    17: "Boolean",
    18: "Array",
    19: "Object",
    20: "Key",
    21: "Null",
    22: "EnumMember",
    23: "Struct",
    24: "Event",
    25: "Operator",
    26: "TypeParameter",
}

_DIAGNOSTIC_SEVERITIES = {
    1: "Error",
    2: "Warning",
    3: "Information",
    4: "Hint",
}


class LspProtocolError(RuntimeError):
    """Raised when an LSP server returns malformed or error responses."""


class _LspSession:
    def __init__(self, command: list[str], cwd: str):
        self.command = command
        self.process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._next_id = 1

    def _send(self, payload: dict[str, Any]) -> None:
        if not self.process.stdin:
            raise LspProtocolError("LSP server stdin is unavailable")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self.process.stdin.write(header + body)
        self.process.stdin.flush()

    def _read_message(self) -> dict[str, Any]:
        if not self.process.stdout:
            raise LspProtocolError("LSP server stdout is unavailable")

        content_length = None
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise LspProtocolError("LSP server closed stdout before a response was received")
            if line in (b"\r\n", b"\n"):
                break
            key, _, value = line.decode("utf-8", errors="replace").partition(":")
            if key.lower() == "content-length":
                try:
                    content_length = int(value.strip())
                except ValueError as exc:
                    raise LspProtocolError("LSP response had an invalid Content-Length header") from exc

        if content_length is None:
            raise LspProtocolError("LSP response missing Content-Length header")

        body = self.process.stdout.read(content_length)
        if len(body) != content_length:
            raise LspProtocolError("LSP response body was truncated")

        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise LspProtocolError("LSP response body was not valid JSON") from exc

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method: str, params: dict[str, Any]) -> Any:
        request_id = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})

        while True:
            message = self._read_message()
            if message.get("id") != request_id:
                continue
            if "error" in message:
                error = message["error"] or {}
                code = error.get("code", "unknown")
                detail = error.get("message", "Unknown LSP error")
                raise LspProtocolError(f"LSP request failed ({method}, code={code}): {detail}")
            return message.get("result")

    def shutdown(self) -> None:
        try:
            self.request("shutdown", {})
        except Exception:
            logger.debug("Best-effort LSP shutdown failed", exc_info=True)
        try:
            self.notify("exit", {})
        except Exception:
            logger.debug("Best-effort LSP exit failed", exc_info=True)
        try:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=1)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                logger.debug("Best-effort LSP kill failed", exc_info=True)


LSP_DOCUMENT_SYMBOLS_SCHEMA = {
    "name": "lsp_document_symbols",
    "description": "Get document symbols for a source file using a local Language Server Protocol server. Supports explicit server_command or local auto-detection by language. This is a built-in repo/code knowledge primitive for semantic local source grounding.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the source file"},
            "language": {"type": "string", "description": "Optional LSP language id override (for example: python, typescript, rust)"},
            "workspace_root": {"type": "string", "description": "Optional workspace root directory passed to the language server"},
            "server_command": {"type": "string", "description": "Optional explicit server command, for example 'pylsp' or 'typescript-language-server --stdio'"},
        },
        "required": ["path"],
    },
}

LSP_DEFINITION_SCHEMA = {
    "name": "lsp_definition",
    "description": "Resolve definitions at a zero-based line/character position using a local LSP server. This is a built-in repo/code knowledge primitive for semantic local source grounding.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the source file"},
            "line": {"type": "integer", "description": "Zero-based line number"},
            "character": {"type": "integer", "description": "Zero-based character offset on the line"},
            "language": {"type": "string", "description": "Optional LSP language id override"},
            "workspace_root": {"type": "string", "description": "Optional workspace root directory"},
            "server_command": {"type": "string", "description": "Optional explicit server command"},
        },
        "required": ["path", "line", "character"],
    },
}

LSP_DIAGNOSTICS_SCHEMA = {
    "name": "lsp_diagnostics",
    "description": "Request file diagnostics from a local LSP server using textDocument/diagnostic. This is a built-in repo/code knowledge primitive for semantic local source grounding.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the source file"},
            "language": {"type": "string", "description": "Optional LSP language id override"},
            "workspace_root": {"type": "string", "description": "Optional workspace root directory"},
            "server_command": {"type": "string", "description": "Optional explicit server command"},
        },
        "required": ["path"],
    },
}


def get_host_lsp_capabilities() -> dict[str, Any]:
    auto_detect_commands: dict[str, list[str]] = {}
    for language, candidates in sorted(_AUTO_DETECT_COMMANDS.items()):
        available_commands = []
        for candidate in candidates:
            binary = candidate[0] if candidate else None
            if binary and shutil.which(binary):
                available_commands.append(" ".join(candidate))
        if available_commands:
            auto_detect_commands[language] = available_commands

    return {
        "available": bool(auto_detect_commands),
        "auto_detect_languages": list(auto_detect_commands.keys()),
        "auto_detect_commands": auto_detect_commands,
    }


def get_lsp_host_capability_description() -> str:
    capabilities = get_host_lsp_capabilities()
    languages = capabilities["auto_detect_languages"]
    if languages:
        return (
            f"Auto-detect currently supports: {', '.join(languages)}. "
            "Use server_command to target any other installed local LSP server for repo/code grounding."
        )
    return "Auto-detect is unavailable on this host. Use server_command to target an installed local LSP server."


def _check_lsp_requirements() -> bool:
    return get_host_lsp_capabilities()["available"]


def _detect_language(path: Path, language: str | None) -> str:
    if language:
        return language
    return _LANGUAGE_IDS.get(path.suffix.lower(), "plaintext")


def _normalize_command(server_command: str | list[str] | None, language: str) -> tuple[list[str] | None, list[str]]:
    if server_command:
        if isinstance(server_command, str):
            command = shlex.split(server_command)
        else:
            command = list(server_command)
        if not command:
            raise ValueError("server_command must not be empty")
        return command, []

    candidates = _AUTO_DETECT_COMMANDS.get(language, [])
    for candidate in candidates:
        if shutil.which(candidate[0]):
            return candidate, [" ".join(entry) for entry in candidates]
    return None, [" ".join(entry) for entry in candidates]


def _path_to_uri(path: Path) -> str:
    return path.resolve().as_uri()


def _uri_to_path(uri: str | None) -> str | None:
    if not uri:
        return None
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return uri
    return unquote(parsed.path)


def _normalize_range(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not value:
        return None
    return {
        "start": value.get("start"),
        "end": value.get("end"),
    }


def _normalize_symbol(item: dict[str, Any]) -> dict[str, Any]:
    if "location" in item:
        location = item.get("location") or {}
        return {
            "name": item.get("name"),
            "detail": item.get("detail"),
            "kind": item.get("kind"),
            "kind_name": _SYMBOL_KINDS.get(item.get("kind"), "Unknown"),
            "container_name": item.get("containerName"),
            "uri": location.get("uri"),
            "path": _uri_to_path(location.get("uri")),
            "range": _normalize_range(location.get("range")),
        }

    return {
        "name": item.get("name"),
        "detail": item.get("detail"),
        "kind": item.get("kind"),
        "kind_name": _SYMBOL_KINDS.get(item.get("kind"), "Unknown"),
        "range": _normalize_range(item.get("range")),
        "selection_range": _normalize_range(item.get("selectionRange")),
        "children": [_normalize_symbol(child) for child in item.get("children", [])],
    }


def _normalize_locations(result: Any) -> list[dict[str, Any]]:
    if result is None:
        return []
    if isinstance(result, dict):
        items = [result]
    else:
        items = list(result)

    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if "targetUri" in item:
            uri = item.get("targetUri")
            normalized.append(
                {
                    "uri": uri,
                    "path": _uri_to_path(uri),
                    "target_range": _normalize_range(item.get("targetRange")),
                    "target_selection_range": _normalize_range(item.get("targetSelectionRange")),
                    "origin_selection_range": _normalize_range(item.get("originSelectionRange")),
                }
            )
            continue

        uri = item.get("uri")
        normalized.append(
            {
                "uri": uri,
                "path": _uri_to_path(uri),
                "range": _normalize_range(item.get("range")),
            }
        )
    return normalized


def _normalize_diagnostics(result: Any) -> list[dict[str, Any]]:
    if result is None:
        return []
    items = result.get("items", []) if isinstance(result, dict) else result
    normalized = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        severity = item.get("severity")
        normalized.append(
            {
                "message": item.get("message"),
                "severity": severity,
                "severity_name": _DIAGNOSTIC_SEVERITIES.get(severity, "Unknown"),
                "code": item.get("code"),
                "source": item.get("source"),
                "range": _normalize_range(item.get("range")),
                "tags": item.get("tags", []),
            }
        )
    return normalized


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _build_initialize_params(workspace_root: Path) -> dict[str, Any]:
    root_uri = _path_to_uri(workspace_root)
    return {
        "processId": None,
        "clientInfo": {"name": "hermes-agent", "version": "phase-3"},
        "rootUri": root_uri,
        "workspaceFolders": [{"uri": root_uri, "name": workspace_root.name or str(workspace_root)}],
        "capabilities": {
            "textDocument": {
                "definition": {"dynamicRegistration": False},
                "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                "diagnostic": {"dynamicRegistration": False},
            }
        },
    }


def _run_lsp_request(
    *,
    path: str,
    requested_method: str,
    request_params: dict[str, Any],
    language: str | None = None,
    workspace_root: str | None = None,
    server_command: str | list[str] | None = None,
) -> str:
    try:
        source_path = Path(path).expanduser()
        if not source_path.exists():
            return tool_error(f"File not found: {path}", path=str(source_path), requested_method=requested_method)
        source_path = source_path.resolve()
        language_id = _detect_language(source_path, language)
        command, candidates = _normalize_command(server_command, language_id)
        if not command:
            return tool_error(
                f"No LSP server found for language '{language_id}'",
                path=str(source_path),
                language=language_id,
                requested_method=requested_method,
                candidate_commands=candidates,
            )

        workspace_path = Path(workspace_root).expanduser().resolve() if workspace_root else source_path.parent
        uri = _path_to_uri(source_path)
        session = _LspSession(command, cwd=str(workspace_path))
        try:
            session.request("initialize", _build_initialize_params(workspace_path))
            session.notify("initialized", {})
            session.notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": language_id,
                        "version": 1,
                        "text": _read_source(source_path),
                    }
                },
            )
            raw_result = session.request(requested_method, request_params)
        finally:
            session.shutdown()

        payload = {
            "ok": True,
            "path": str(source_path),
            "language": language_id,
            "requested_method": requested_method,
            "server_command": command,
            "workspace_root": str(workspace_path),
        }
        if requested_method == "textDocument/documentSymbol":
            payload["symbols"] = [_normalize_symbol(item) for item in raw_result or []]
        elif requested_method == "textDocument/definition":
            payload["definitions"] = _normalize_locations(raw_result)
        elif requested_method == "textDocument/diagnostic":
            payload["diagnostics"] = _normalize_diagnostics(raw_result)
        else:
            payload["result"] = raw_result
        return tool_result(payload)
    except Exception as exc:
        logger.debug("LSP request failed", exc_info=True)
        return tool_error(str(exc), path=path, requested_method=requested_method)


def lsp_document_symbols_tool(
    *,
    path: str,
    language: str | None = None,
    workspace_root: str | None = None,
    server_command: str | None = None,
) -> str:
    return _run_lsp_request(
        path=path,
        language=language,
        workspace_root=workspace_root,
        server_command=server_command,
        requested_method="textDocument/documentSymbol",
        request_params={"textDocument": {"uri": _path_to_uri(Path(path).expanduser().resolve())}},
    )


def lsp_definition_tool(
    *,
    path: str,
    line: int,
    character: int,
    language: str | None = None,
    workspace_root: str | None = None,
    server_command: str | None = None,
) -> str:
    return _run_lsp_request(
        path=path,
        language=language,
        workspace_root=workspace_root,
        server_command=server_command,
        requested_method="textDocument/definition",
        request_params={
            "textDocument": {"uri": _path_to_uri(Path(path).expanduser().resolve())},
            "position": {"line": line, "character": character},
        },
    )


def lsp_diagnostics_tool(
    *,
    path: str,
    language: str | None = None,
    workspace_root: str | None = None,
    server_command: str | None = None,
) -> str:
    return _run_lsp_request(
        path=path,
        language=language,
        workspace_root=workspace_root,
        server_command=server_command,
        requested_method="textDocument/diagnostic",
        request_params={"textDocument": {"uri": _path_to_uri(Path(path).expanduser().resolve())}},
    )


LSP_TOOLS = [
    {"name": "lsp_document_symbols", "function": lsp_document_symbols_tool},
    {"name": "lsp_definition", "function": lsp_definition_tool},
    {"name": "lsp_diagnostics", "function": lsp_diagnostics_tool},
]


def _handle_lsp_document_symbols(args, **_kwargs):
    return lsp_document_symbols_tool(
        path=args.get("path", ""),
        language=args.get("language"),
        workspace_root=args.get("workspace_root"),
        server_command=args.get("server_command"),
    )


def _handle_lsp_definition(args, **_kwargs):
    return lsp_definition_tool(
        path=args.get("path", ""),
        line=args.get("line", 0),
        character=args.get("character", 0),
        language=args.get("language"),
        workspace_root=args.get("workspace_root"),
        server_command=args.get("server_command"),
    )


def _handle_lsp_diagnostics(args, **_kwargs):
    return lsp_diagnostics_tool(
        path=args.get("path", ""),
        language=args.get("language"),
        workspace_root=args.get("workspace_root"),
        server_command=args.get("server_command"),
    )


registry.register(
    name="lsp_document_symbols",
    toolset="code_intel",
    schema=LSP_DOCUMENT_SYMBOLS_SCHEMA,
    handler=_handle_lsp_document_symbols,
    check_fn=_check_lsp_requirements,
    emoji="🛰️",
)
registry.register(
    name="lsp_definition",
    toolset="code_intel",
    schema=LSP_DEFINITION_SCHEMA,
    handler=_handle_lsp_definition,
    check_fn=_check_lsp_requirements,
    emoji="📍",
)
registry.register(
    name="lsp_diagnostics",
    toolset="code_intel",
    schema=LSP_DIAGNOSTICS_SCHEMA,
    handler=_handle_lsp_diagnostics,
    check_fn=_check_lsp_requirements,
    emoji="🚨",
)
