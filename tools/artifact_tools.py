"""Retrieve sections from raw artifacts saved by artifact-aware compaction."""

import json

from agent.artifact_compaction import DEFAULT_RETRIEVE_MAX_CHARS, retrieve_artifact_section
from tools.registry import registry


ARTIFACT_RETRIEVE_SCHEMA = {
    "name": "artifact_retrieve",
    "description": "Load a targeted section, line range, or regex match from a raw artifact saved under ~/.hermes/artifacts by artifact-aware context compaction. Use this instead of reading the full artifact when a summary card needs details.",
    "parameters": {
        "type": "object",
        "properties": {
            "artifact_id": {"type": "string", "description": "Artifact id from the summary card (first 16 sha256 chars)."},
            "sha256": {"type": "string", "description": "Full sha256 fingerprint from the summary card."},
            "path": {"type": "string", "description": "Artifact path from the summary card."},
            "section": {"type": "string", "description": "Optional detected section/symbol name or substring."},
            "symbol_name": {"type": "string", "description": "Optional source-code symbol name to retrieve (module, subroutine, function, type, interface)."},
            "module_name": {"type": "string", "description": "Optional Fortran module name to retrieve."},
            "subroutine_name": {"type": "string", "description": "Optional Fortran subroutine/function name to retrieve."},
            "line_start": {"type": "integer", "description": "Optional 1-indexed start line."},
            "line_end": {"type": "integer", "description": "Optional 1-indexed end line."},
            "regex": {"type": "string", "description": "Optional regex; returns a small context window around the first match."},
            "context_lines": {"type": "integer", "description": "Context lines for regex retrieval.", "default": 20},
            "max_chars": {"type": "integer", "description": "Maximum characters to return.", "default": DEFAULT_RETRIEVE_MAX_CHARS},
        },
        "anyOf": [
            {"required": ["artifact_id"]},
            {"required": ["sha256"]},
            {"required": ["path"]},
        ],
    },
}

# Backwards-compatible schema name for callers/tests that already discovered it.
RETRIEVE_ARTIFACT_SECTION_SCHEMA = {
    **ARTIFACT_RETRIEVE_SCHEMA,
    "name": "retrieve_artifact_section",
}


def _configured_default_max_chars() -> int:
    try:
        from hermes_cli.config import load_config

        cfg = load_config().get("compression", {}) or {}
        return int(cfg.get("artifact_retrieve_max_chars") or DEFAULT_RETRIEVE_MAX_CHARS)
    except Exception:
        return DEFAULT_RETRIEVE_MAX_CHARS


def _handler(args, **kwargs):
    identifier = args.get("artifact_id") or args.get("sha256") or args.get("path") or ""
    section = args.get("section")
    if not section:
        if args.get("symbol_name"):
            section = args.get("symbol_name")
        elif args.get("module_name"):
            section = f"module {args.get('module_name')}"
        elif args.get("subroutine_name"):
            section = args.get("subroutine_name")
    return json.dumps(retrieve_artifact_section(
        identifier,
        section=section,
        line_start=args.get("line_start"),
        line_end=args.get("line_end"),
        regex=args.get("regex"),
        context_lines=int(args.get("context_lines") or 20),
        max_chars=int(args.get("max_chars") or _configured_default_max_chars()),
    ), ensure_ascii=False)


registry.register(
    name="artifact_retrieve",
    toolset="file",
    schema=ARTIFACT_RETRIEVE_SCHEMA,
    handler=_handler,
)

registry.register(
    name="retrieve_artifact_section",
    toolset="file",
    schema=RETRIEVE_ARTIFACT_SECTION_SCHEMA,
    handler=_handler,
)
