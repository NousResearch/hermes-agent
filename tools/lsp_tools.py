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

    def __init__(
        self,
        message: str,
        *,
        method: str | None = None,
        code: int | str | None = None,
        detail: str | None = None,
    ):
        super().__init__(message)
        self.method = method
        self.code = code
        self.detail = detail or message


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
                raise LspProtocolError(
                    f"LSP request failed ({method}, code={code}): {detail}",
                    method=method,
                    code=code,
                    detail=detail,
                )
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

LSP_PREPARE_RENAME_SCHEMA = {
    "name": "lsp_prepare_rename",
    "description": "Validate whether a symbol can be renamed at a zero-based line/character position using a local LSP server. Returns the validated rename range and placeholder when the server supports prepareRename. This is a built-in repo/code knowledge primitive for semantic local source grounding.",
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

LSP_REFERENCES_SCHEMA = {
    "name": "lsp_references",
    "description": "Find references at a zero-based line/character position using a local LSP server. Returns structured locations suitable for rename/reference verification. This is a built-in repo/code knowledge primitive for semantic local source grounding.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the source file"},
            "line": {"type": "integer", "description": "Zero-based line number"},
            "character": {"type": "integer", "description": "Zero-based character offset on the line"},
            "include_declaration": {"type": "boolean", "description": "Whether declaration locations should be included", "default": True},
            "language": {"type": "string", "description": "Optional LSP language id override"},
            "workspace_root": {"type": "string", "description": "Optional workspace root directory"},
            "server_command": {"type": "string", "description": "Optional explicit server command"},
        },
        "required": ["path", "line", "character"],
    },
}

LSP_RENAME_SCHEMA = {
    "name": "lsp_rename",
    "description": "Rename a symbol at a zero-based line/character position using a local LSP server. Hermes validates prepareRename first, then returns a structured workspace edit preview instead of mutating files directly. This is a built-in repo/code knowledge primitive for semantic local source grounding.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the source file"},
            "line": {"type": "integer", "description": "Zero-based line number"},
            "character": {"type": "integer", "description": "Zero-based character offset on the line"},
            "new_name": {"type": "string", "description": "New identifier name to request from the LSP server"},
            "language": {"type": "string", "description": "Optional LSP language id override"},
            "workspace_root": {"type": "string", "description": "Optional workspace root directory"},
            "server_command": {"type": "string", "description": "Optional explicit server command"},
        },
        "required": ["path", "line", "character", "new_name"],
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


def _offset_for_position(text: str, line: int, character: int) -> int:
    if line < 0 or character < 0:
        raise ValueError("LSP position must use non-negative line and character values")
    lines = text.splitlines(keepends=True)
    if text:
        logical_line_count = len(lines)
    else:
        lines = [""]
        logical_line_count = 1
    if line >= logical_line_count:
        raise ValueError(f"LSP position line {line} is outside the document")
    prefix = sum(len(lines[index]) for index in range(min(line, len(lines))))
    current_line = lines[line] if line < len(lines) else ""
    line_text = current_line.rstrip("\r\n")
    if character > len(line_text):
        raise ValueError(f"LSP position character {character} is outside line {line}")
    return prefix + character


def _apply_text_edits(text: str, edits: list[dict[str, Any]]) -> str:
    replacements = []
    for edit in edits:
        edit_range = edit.get("range") or {}
        start = edit_range.get("start") or {}
        end = edit_range.get("end") or {}
        start_offset = _offset_for_position(text, start.get("line", 0), start.get("character", 0))
        end_offset = _offset_for_position(text, end.get("line", 0), end.get("character", 0))
        if end_offset < start_offset:
            raise ValueError("LSP edit range end precedes start")
        replacements.append((start_offset, end_offset, edit.get("new_text", "")))

    updated = text
    for start_offset, end_offset, new_text in sorted(replacements, key=lambda item: (item[0], item[1]), reverse=True):
        updated = updated[:start_offset] + new_text + updated[end_offset:]
    return updated


def _normalize_prepare_rename_result(result: Any) -> dict[str, Any]:
    if result in (None, False):
        return {"can_rename": False}
    if isinstance(result, dict) and "range" in result:
        return {
            "can_rename": True,
            "range": _normalize_range(result.get("range")),
            "placeholder": result.get("placeholder"),
            "default_behavior": result.get("defaultBehavior"),
        }
    return {
        "can_rename": True,
        "range": _normalize_range(result),
        "placeholder": None,
        "default_behavior": None,
    }


def _normalize_text_edit(edit: dict[str, Any]) -> dict[str, Any]:
    return {
        "range": _normalize_range(edit.get("range")),
        "new_text": edit.get("newText", ""),
    }


def _normalize_workspace_edit(result: Any) -> dict[str, Any]:
    workspace_edit = result if isinstance(result, dict) else {}
    changes = []
    preview_by_path: dict[str, str] = {}
    touched_paths: set[str] = set()
    edit_count = 0
    resource_operations = []

    for uri, edits in (workspace_edit.get("changes") or {}).items():
        path = _uri_to_path(uri)
        normalized_edits = [_normalize_text_edit(edit) for edit in edits or [] if isinstance(edit, dict)]
        edit_count += len(normalized_edits)
        if path:
            touched_paths.add(path)
            source_path = Path(path)
            if source_path.exists():
                preview_by_path[path] = _apply_text_edits(_read_source(source_path), normalized_edits)
        changes.append({"uri": uri, "path": path, "edits": normalized_edits})

    document_changes = []
    for change in workspace_edit.get("documentChanges") or []:
        if not isinstance(change, dict):
            continue
        kind = change.get("kind")
        if kind:
            resource_operations.append(change)
            document_changes.append({"kind": kind, **{k: v for k, v in change.items() if k != "kind"}})
            continue

        text_document = change.get("textDocument") or {}
        uri = text_document.get("uri")
        path = _uri_to_path(uri)
        normalized_edits = [_normalize_text_edit(edit) for edit in change.get("edits", []) if isinstance(edit, dict)]
        edit_count += len(normalized_edits)
        if path:
            touched_paths.add(path)
            source_path = Path(path)
            if source_path.exists():
                preview_by_path[path] = _apply_text_edits(_read_source(source_path), normalized_edits)
        document_changes.append(
            {
                "kind": "textDocumentEdit",
                "uri": uri,
                "path": path,
                "version": text_document.get("version"),
                "edits": normalized_edits,
            }
        )

    return {
        "changes": changes,
        "document_changes": document_changes,
        "resource_operations": resource_operations,
        "touched_paths": sorted(touched_paths),
        "edit_count": edit_count,
        "resource_operation_count": len(resource_operations),
        "preview_by_path": preview_by_path,
    }


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
                "references": {"dynamicRegistration": False},
                "rename": {"dynamicRegistration": False, "prepareSupport": True},
            }
        },
    }


def _build_missing_server_error(
    *,
    source_path: Path,
    language_id: str,
    requested_method: str,
    candidates: list[str],
) -> str:
    if not candidates:
        return tool_error(
            f"LSP auto-detect does not support language '{language_id}'",
            path=str(source_path),
            language=language_id,
            requested_method=requested_method,
            failure_kind="unsupported_language",
            supported_languages=sorted(_AUTO_DETECT_COMMANDS.keys()),
            candidate_commands=[],
        )
    return tool_error(
        f"No LSP server found for language '{language_id}'",
        path=str(source_path),
        language=language_id,
        requested_method=requested_method,
        failure_kind="missing_server",
        candidate_commands=candidates,
    )


def _is_unsupported_capability_error(error: LspProtocolError) -> bool:
    detail = (error.detail or str(error)).lower()
    if error.code == -32601:
        return True
    return any(
        needle in detail
        for needle in (
            "method not found",
            "unsupported method",
            "unsupported capability",
            "does not support",
            "not supported",
        )
    )


def _build_protocol_error(
    error: LspProtocolError,
    *,
    path: str,
    requested_method: str,
    language: str | None = None,
    server_command: list[str] | None = None,
    workspace_root: str | None = None,
) -> str:
    extra: dict[str, Any] = {
        "path": path,
        "requested_method": requested_method,
        "lsp_error_message": error.detail,
    }
    if language is not None:
        extra["language"] = language
    if server_command is not None:
        extra["server_command"] = server_command
    if workspace_root is not None:
        extra["workspace_root"] = workspace_root
    if error.code is not None:
        extra["lsp_error_code"] = error.code
    if _is_unsupported_capability_error(error):
        extra["failure_kind"] = "unsupported_capability"
    return tool_error(str(error), **extra)


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
            return _build_missing_server_error(
                source_path=source_path,
                language_id=language_id,
                requested_method=requested_method,
                candidates=candidates,
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
        elif requested_method == "textDocument/references":
            payload["references"] = _normalize_locations(raw_result)
            payload["reference_count"] = len(payload["references"])
        elif requested_method == "textDocument/prepareRename":
            payload["prepare_rename"] = _normalize_prepare_rename_result(raw_result)
            payload["can_rename"] = payload["prepare_rename"]["can_rename"]
        else:
            payload["result"] = raw_result
        return tool_result(payload)
    except LspProtocolError as exc:
        logger.debug("LSP request failed", exc_info=True)
        return _build_protocol_error(
            exc,
            path=path,
            requested_method=requested_method,
            language=language_id if "language_id" in locals() else None,
            server_command=command if "command" in locals() else None,
            workspace_root=str(workspace_path) if "workspace_path" in locals() else None,
        )
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


def lsp_prepare_rename_tool(
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
        requested_method="textDocument/prepareRename",
        request_params={
            "textDocument": {"uri": _path_to_uri(Path(path).expanduser().resolve())},
            "position": {"line": line, "character": character},
        },
    )


def lsp_references_tool(
    *,
    path: str,
    line: int,
    character: int,
    include_declaration: bool = True,
    language: str | None = None,
    workspace_root: str | None = None,
    server_command: str | None = None,
) -> str:
    return _run_lsp_request(
        path=path,
        language=language,
        workspace_root=workspace_root,
        server_command=server_command,
        requested_method="textDocument/references",
        request_params={
            "textDocument": {"uri": _path_to_uri(Path(path).expanduser().resolve())},
            "position": {"line": line, "character": character},
            "context": {"includeDeclaration": include_declaration},
        },
    )


def lsp_rename_tool(
    *,
    path: str,
    line: int,
    character: int,
    new_name: str,
    language: str | None = None,
    workspace_root: str | None = None,
    server_command: str | None = None,
) -> str:
    requested_method = "textDocument/rename"
    try:
        source_path = Path(path).expanduser()
        if not source_path.exists():
            return tool_error(f"File not found: {path}", path=str(source_path), requested_method=requested_method)
        source_path = source_path.resolve()
        language_id = _detect_language(source_path, language)
        command, candidates = _normalize_command(server_command, language_id)
        if not command:
            return _build_missing_server_error(
                source_path=source_path,
                language_id=language_id,
                requested_method=requested_method,
                candidates=candidates,
            )

        workspace_path = Path(workspace_root).expanduser().resolve() if workspace_root else source_path.parent
        uri = _path_to_uri(source_path)
        position = {"line": line, "character": character}
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
            prepare_raw_result = session.request(
                "textDocument/prepareRename",
                {"textDocument": {"uri": uri}, "position": position},
            )
            prepare_result = _normalize_prepare_rename_result(prepare_raw_result)
            if not prepare_result["can_rename"]:
                return tool_error(
                    "LSP server reported that the selected symbol cannot be renamed",
                    path=str(source_path),
                    language=language_id,
                    requested_method=requested_method,
                    failure_kind="prepare_rename_rejected",
                    prepare_rename=prepare_result,
                    server_command=command,
                    workspace_root=str(workspace_path),
                )
            rename_raw_result = session.request(
                requested_method,
                {"textDocument": {"uri": uri}, "position": position, "newName": new_name},
            )
        finally:
            session.shutdown()

        normalized_edit = _normalize_workspace_edit(rename_raw_result)
        return tool_result(
            {
                "ok": True,
                "path": str(source_path),
                "language": language_id,
                "requested_method": requested_method,
                "server_command": command,
                "workspace_root": str(workspace_path),
                "new_name": new_name,
                "prepare_rename": prepare_result,
                "workspace_edit": normalized_edit,
            }
        )
    except LspProtocolError as exc:
        logger.debug("LSP rename request failed", exc_info=True)
        return _build_protocol_error(
            exc,
            path=path,
            requested_method=requested_method,
            language=language_id if "language_id" in locals() else None,
            server_command=command if "command" in locals() else None,
            workspace_root=str(workspace_path) if "workspace_path" in locals() else None,
        )
    except Exception as exc:
        logger.debug("LSP rename request failed", exc_info=True)
        return tool_error(str(exc), path=path, requested_method=requested_method)


LSP_TOOLS = [
    {"name": "lsp_document_symbols", "function": lsp_document_symbols_tool},
    {"name": "lsp_definition", "function": lsp_definition_tool},
    {"name": "lsp_diagnostics", "function": lsp_diagnostics_tool},
    {"name": "lsp_prepare_rename", "function": lsp_prepare_rename_tool},
    {"name": "lsp_references", "function": lsp_references_tool},
    {"name": "lsp_rename", "function": lsp_rename_tool},
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


def _handle_lsp_prepare_rename(args, **_kwargs):
    return lsp_prepare_rename_tool(
        path=args.get("path", ""),
        line=args.get("line", 0),
        character=args.get("character", 0),
        language=args.get("language"),
        workspace_root=args.get("workspace_root"),
        server_command=args.get("server_command"),
    )


def _handle_lsp_references(args, **_kwargs):
    return lsp_references_tool(
        path=args.get("path", ""),
        line=args.get("line", 0),
        character=args.get("character", 0),
        include_declaration=args.get("include_declaration", True),
        language=args.get("language"),
        workspace_root=args.get("workspace_root"),
        server_command=args.get("server_command"),
    )


def _handle_lsp_rename(args, **_kwargs):
    return lsp_rename_tool(
        path=args.get("path", ""),
        line=args.get("line", 0),
        character=args.get("character", 0),
        new_name=args.get("new_name", ""),
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
registry.register(
    name="lsp_prepare_rename",
    toolset="code_intel",
    schema=LSP_PREPARE_RENAME_SCHEMA,
    handler=_handle_lsp_prepare_rename,
    check_fn=_check_lsp_requirements,
    emoji="✏️",
)
registry.register(
    name="lsp_references",
    toolset="code_intel",
    schema=LSP_REFERENCES_SCHEMA,
    handler=_handle_lsp_references,
    check_fn=_check_lsp_requirements,
    emoji="🔎",
)
registry.register(
    name="lsp_rename",
    toolset="code_intel",
    schema=LSP_RENAME_SCHEMA,
    handler=_handle_lsp_rename,
    check_fn=_check_lsp_requirements,
    emoji="📝",
)
