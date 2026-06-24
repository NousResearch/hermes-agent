"""Read-only GBrain tools for local project/company memory recall."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.registry import registry


_DEFAULT_TIMEOUT_SECONDS = 20
_DEFAULT_MAX_CHARS = 8000
_MAX_LIMIT = 20
_USER_BIN_SUBDIRS = (".bun/bin", ".local/bin", "bin")


def _json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


def _coerce_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        coerced = default
    return max(minimum, min(coerced, maximum))


def _candidate_homes() -> List[str]:
    homes: List[str] = []
    for raw in (
        os.environ.get("HOME"),
        os.environ.get("HERMES_REAL_HOME"),
        os.environ.get("USERPROFILE"),
    ):
        if raw and raw not in homes:
            homes.append(raw)
    return homes


def _path_with_user_bins() -> str:
    parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    seen = set(parts)
    for home in _candidate_homes():
        for rel in _USER_BIN_SUBDIRS:
            candidate = str(Path(home).expanduser() / rel)
            if candidate not in seen and os.path.isdir(candidate):
                parts.append(candidate)
                seen.add(candidate)
    return os.pathsep.join(parts)


def _resolve_gbrain_bin() -> Optional[str]:
    for env_name in ("HERMES_GBRAIN_BIN", "GBRAIN_BIN"):
        raw = os.environ.get(env_name)
        if raw:
            expanded = os.path.expanduser(os.path.expandvars(raw))
            if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
                return expanded

    search_path = _path_with_user_bins()
    found = shutil.which("gbrain", path=search_path)
    if found:
        return found

    for home in _candidate_homes():
        candidate = Path(home).expanduser() / ".bun" / "bin" / "gbrain"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    return None


def check_gbrain_requirements() -> bool:
    return _resolve_gbrain_bin() is not None


def _run_gbrain(args: List[str], *, timeout: int = _DEFAULT_TIMEOUT_SECONDS) -> Dict[str, Any]:
    binary = _resolve_gbrain_bin()
    if not binary:
        return {
            "ok": False,
            "error": "gbrain_unavailable",
            "fix": "Install GBrain or set HERMES_GBRAIN_BIN/GBRAIN_BIN to the gbrain executable.",
        }

    env = os.environ.copy()
    env["PATH"] = _path_with_user_bins()
    env.setdefault("GBRAIN_NO_ONBOARD_NUDGE", "1")

    try:
        completed = subprocess.run(
            [binary, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": "gbrain_timeout",
            "fix": "Retry a narrower query or check whether another GBrain/PGLite process is holding the local database lock.",
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc) or exc.__class__.__name__}

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if completed.returncode != 0:
        detail = stderr or stdout or f"gbrain exited {completed.returncode}"
        result: Dict[str, Any] = {
            "ok": False,
            "error": "gbrain_failed",
            "exit_code": completed.returncode,
            "detail": detail,
        }
        if "PGLite lock" in detail or "Timed out waiting" in detail:
            result["fix"] = "Stop the other GBrain process or retry after the PGLite lock is released."
        return result

    return {"ok": True, "output": stdout}


def _truncate_output(result: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    output = str(result.get("output") or "")
    if output and len(output) > max_chars:
        result = dict(result)
        result["output"] = output[: max_chars - 1] + "…"
        result["truncated"] = True
        result["original_chars"] = len(output)
    return result


def gbrain_search(query: str, limit: int = 5, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
    """Search local GBrain memory. Results are context; verify source-of-truth."""
    query = str(query or "").strip()
    if not query:
        return _json({"ok": False, "error": "query is required"})

    limit = _coerce_int(limit, 5, 1, _MAX_LIMIT)
    max_chars = _coerce_int(max_chars, _DEFAULT_MAX_CHARS, 1000, 20000)
    result = _run_gbrain(["search", query, "--limit", str(limit)])
    result = _truncate_output(result, max_chars)
    result["note"] = "Brain output is context, not source-of-truth. Verify with DB/API/log/code/GitHub before concluding."
    return _json(result)


def gbrain_get(slug: str, fuzzy: bool = True, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
    """Read a GBrain page by slug. Results are context; verify source-of-truth."""
    slug = str(slug or "").strip()
    if not slug:
        return _json({"ok": False, "error": "slug is required"})

    max_chars = _coerce_int(max_chars, _DEFAULT_MAX_CHARS, 1000, 30000)
    args = ["get", slug]
    if fuzzy:
        args.append("--fuzzy")
    result = _run_gbrain(args)
    result = _truncate_output(result, max_chars)
    result["note"] = "Brain output is context, not source-of-truth. Verify with DB/API/log/code/GitHub before concluding."
    return _json(result)


GBRAIN_SEARCH_SCHEMA = {
    "name": "gbrain_search",
    "description": (
        "Search local GBrain project/company memory before Vucar/Hermes debugging, "
        "design, PR, or incident work. Use this to avoid starting from zero. "
        "Treat results as context only; verify current source-of-truth before concluding."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search terms, including concrete repo/module/customer/incident names when known.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results to return. Default 5, max 20.",
                "default": 5,
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum output characters. Default 8000.",
                "default": _DEFAULT_MAX_CHARS,
            },
        },
        "required": ["query"],
    },
}


GBRAIN_GET_SCHEMA = {
    "name": "gbrain_get",
    "description": (
        "Read a local GBrain page by slug after gbrain_search returns a relevant page. "
        "Treat the page as context only; verify current source-of-truth before concluding."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "GBrain page slug, for example systems/hermes-runtime.",
            },
            "fuzzy": {
                "type": "boolean",
                "description": "Allow fuzzy slug resolution. Default true.",
                "default": True,
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum output characters. Default 8000.",
                "default": _DEFAULT_MAX_CHARS,
            },
        },
        "required": ["slug"],
    },
}


registry.register(
    name="gbrain_search",
    toolset="gbrain",
    schema=GBRAIN_SEARCH_SCHEMA,
    handler=lambda args, **kw: gbrain_search(
        query=args.get("query", ""),
        limit=args.get("limit", 5),
        max_chars=args.get("max_chars", _DEFAULT_MAX_CHARS),
    ),
    check_fn=check_gbrain_requirements,
    emoji="",
)

registry.register(
    name="gbrain_get",
    toolset="gbrain",
    schema=GBRAIN_GET_SCHEMA,
    handler=lambda args, **kw: gbrain_get(
        slug=args.get("slug", ""),
        fuzzy=args.get("fuzzy", True),
        max_chars=args.get("max_chars", _DEFAULT_MAX_CHARS),
    ),
    check_fn=check_gbrain_requirements,
    emoji="",
)
