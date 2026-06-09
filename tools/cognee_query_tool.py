#!/usr/bin/env python3
"""Read-only cognee-lab query tool.

This tool intentionally exposes only explicit retrieval from the isolated
``cognee-lab`` proof-of-concept directory. It does not ingest data, reset state,
or write into Hermes memory.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from tools.registry import registry


LAB_SUBDIR = "cognee_lab"
QUERY_SCRIPT = Path("scripts") / "query.py"
DEFAULT_TIMEOUT_SECONDS = 180
MAX_QUESTION_CHARS = 2000
MAX_RESULT_CHARS = 20000
ALLOWED_SEARCH_TYPES = {"CHUNKS", "RAG_COMPLETION", "GRAPH_COMPLETION"}
TRAILING_SOURCE_PUNCTUATION = ".,;:)]}>\"'"


def _normalize_source_filename(name: Any) -> str:
    return str(name or "").strip().rstrip(TRAILING_SOURCE_PUNCTUATION)


def _normalize_source_file_list(values: Any) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    if not isinstance(values, list):
        return normalized
    for value in values:
        name = _normalize_source_filename(value)
        if name and name not in seen:
            seen.add(name)
            normalized.append(name)
    return normalized


def _normalize_source_fields(value: Any) -> Any:
    """Normalize structured source filename fields in a parsed envelope in-place."""
    if not isinstance(value, dict):
        return value

    if isinstance(value.get("source_files"), list):
        value["source_files"] = _normalize_source_file_list(value.get("source_files"))

    if isinstance(value.get("sources"), list):
        normalized_sources = []
        seen_sources: set[str] = set()
        for source in value["sources"]:
            if not isinstance(source, dict):
                continue
            source = dict(source)
            source["file"] = _normalize_source_filename(source.get("file"))
            if source["file"] and source["file"] not in seen_sources:
                seen_sources.add(source["file"])
                normalized_sources.append(source)
        value["sources"] = normalized_sources

    if isinstance(value.get("result_items"), list):
        normalized_items = []
        for item in value["result_items"]:
            if not isinstance(item, dict):
                normalized_items.append(item)
                continue
            item = dict(item)
            if isinstance(item.get("source_files"), list):
                item["source_files"] = _normalize_source_file_list(item.get("source_files"))
            normalized_items.append(item)
        value["result_items"] = normalized_items

    if "source_count" in value and isinstance(value.get("source_files"), list):
        value["source_count"] = len(value["source_files"])
    return value


def _candidate_lab_roots() -> list[Path]:
    """Return possible profile-local cognee lab roots, most specific first."""
    homes: list[Path] = []
    home = get_hermes_home().expanduser()
    homes.append(home)

    # If the current process is not already running under the cognee-lab
    # profile, allow this tool to find the isolated lab profile without
    # touching default memory/state. This keeps storage explicit and fixed to
    # the lab directory rather than the active default profile.
    if home.name != "cognee-lab":
        homes.append(home / "profiles" / "cognee-lab")
        # Standard root for this deployment. Kept as a final fallback so the
        # tool remains usable from the default profile during lab evaluation,
        # while still resolving storage to the isolated lab profile.
        homes.append(Path.home() / ".hermes" / "profiles" / "cognee-lab")

    roots: list[Path] = []
    seen: set[Path] = set()
    for h in homes:
        root = (h / LAB_SUBDIR).expanduser()
        try:
            key = root.resolve()
        except Exception:
            key = root
        if key not in seen:
            seen.add(key)
            roots.append(root)
    return roots


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _resolve_lab_root() -> Path | None:
    for root in _candidate_lab_roots():
        if _path_exists(root / QUERY_SCRIPT) and _path_exists(root / ".venv" / "bin" / "python"):
            return root
    return None


def check_cognee_query_requirements() -> bool:
    return _resolve_lab_root() is not None


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return text


def _load_source_envelope_helpers(lab_root: Path):
    helper = lab_root / "scripts" / "source_envelope.py"
    if not helper.exists():
        return None
    try:
        digest = hashlib.sha256(str(helper.resolve()).encode("utf-8")).hexdigest()[:12]
    except Exception:
        digest = "unknown"
    module_name = f"_cognee_lab_source_envelope_{digest}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, helper)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return (
            module.extract_source_files,
            module.extract_sources,
            module.flatten_result_items,
        )
    except Exception:
        return None


def _fallback_iter_text_values(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _fallback_iter_text_values(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _fallback_iter_text_values(item)


def _fallback_source_counts(value: Any) -> dict[str, int]:
    import re

    counts: dict[str, int] = {}
    for text in _fallback_iter_text_values(value):
        for match in re.finditer(r"SOURCE_FILE:\s*([A-Za-z0-9_.-]+)", text):
            name = _normalize_source_filename(match.group(1))
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


def _fallback_extract_source_files(value: Any) -> list[str]:
    return list(_fallback_source_counts(value).keys())


def _fallback_flatten_result_items(value: Any, path: str = "result") -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if isinstance(value, str):
        text = value.strip()
        if text:
            items.append({
                "path": path,
                "source_files": _fallback_extract_source_files(text),
                "text_preview": text[:1200],
                "chars": len(text),
            })
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            items.extend(_fallback_flatten_result_items(item, f"{path}[{idx}]"))
    elif isinstance(value, dict):
        for key, item in value.items():
            items.extend(_fallback_flatten_result_items(item, f"{path}.{key}"))
    return items


def _fallback_extract_sources(value: Any) -> list[dict[str, Any]]:
    counts = _fallback_source_counts(value)
    items = _fallback_flatten_result_items(value)
    first_paths: dict[str, str] = {}
    previews: dict[str, str] = {}
    for item in items:
        for source in item.get("source_files", []):
            first_paths.setdefault(source, item.get("path", "result"))
            previews.setdefault(source, item.get("text_preview", ""))
    return [
        {
            "file": source,
            "mentions": counts[source],
            "first_path": first_paths.get(source, "result"),
            "text_preview": previews.get(source, "")[:600],
        }
        for source in counts
    ]


def _get_source_helpers(lab_root: Path | None = None):
    """Return source extraction helpers from lab helper module when available.

    Loading by explicit file path avoids mutating global sys.path or accidentally
    importing an unrelated `source_envelope` module from the Hermes process.
    """
    if lab_root is not None:
        helpers = _load_source_envelope_helpers(lab_root)
        if helpers is not None:
            return helpers
    return _fallback_extract_source_files, _fallback_extract_sources, _fallback_flatten_result_items


def _extract_source_files(value: Any) -> list[str]:
    extract_source_files, _, _ = _get_source_helpers(_resolve_lab_root())
    return extract_source_files(value)


def _extract_sources(value: Any) -> list[dict[str, Any]]:
    _, extract_sources, _ = _get_source_helpers(_resolve_lab_root())
    return extract_sources(value)


def _flatten_result_items(value: Any, path: str = "result") -> list[dict[str, Any]]:
    _, _, flatten_result_items = _get_source_helpers(_resolve_lab_root())
    return flatten_result_items(value, path)


def _truncate_text(text: str, limit: int = MAX_RESULT_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n...[truncated]", True


def _run_query_script(
    lab_root: Path,
    question: str,
    search_type: str,
    top_k: int,
    answer: bool,
) -> tuple[subprocess.CompletedProcess[str], Any]:
    py = lab_root / ".venv" / "bin" / "python"
    cmd = [
        str(py),
        str(lab_root / QUERY_SCRIPT),
        question,
        "--search-type",
        search_type,
        "--top-k",
        str(top_k),
        "--envelope",
    ]
    if answer:
        cmd.append("--answer")

    proc = subprocess.run(
        cmd,
        cwd=str(lab_root),
        text=True,
        capture_output=True,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    parsed = _normalize_source_fields(_safe_json_loads(proc.stdout or ""))
    return proc, parsed


def _extract_answer_text(parsed: Any) -> str:
    if not isinstance(parsed, dict):
        return ""
    results = parsed.get("results")
    if isinstance(results, list):
        for item in results:
            if isinstance(item, str) and item.strip():
                return item.strip()
    if isinstance(results, str):
        return results.strip()
    items = parsed.get("result_items")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and str(item.get("text_preview", "")).strip():
                return str(item.get("text_preview", "")).strip()
    return ""


def cognee_query(
    question: str,
    search_type: str = "GRAPH_COMPLETION",
    answer: bool = False,
    top_k: int = 5,
    include_raw: bool = True,
    answer_with_sources: bool = False,
) -> str:
    """Query the isolated cognee-lab PoC read-only and return JSON."""
    lab_root = _resolve_lab_root()
    if lab_root is None:
        return json.dumps({
            "success": False,
            "error": "cognee-lab PoC not available; expected cognee_lab/scripts/query.py and .venv/bin/python under the cognee-lab profile",
            "checked_roots": [str(p) for p in _candidate_lab_roots()],
        }, ensure_ascii=False)

    question = (question or "").strip()
    if not question:
        return json.dumps({"success": False, "error": "question is required"}, ensure_ascii=False)
    if len(question) > MAX_QUESTION_CHARS:
        return json.dumps({
            "success": False,
            "error": f"question too long: {len(question)} chars > {MAX_QUESTION_CHARS}",
        }, ensure_ascii=False)

    search_type = (search_type or "GRAPH_COMPLETION").strip().upper()
    if search_type not in ALLOWED_SEARCH_TYPES:
        return json.dumps({
            "success": False,
            "error": f"unsupported search_type {search_type!r}; allowed: {sorted(ALLOWED_SEARCH_TYPES)}",
        }, ensure_ascii=False)

    try:
        top_k_int = int(top_k)
    except Exception:
        top_k_int = 5
    top_k_int = max(1, min(top_k_int, 20))

    proc, parsed = _run_query_script(
        lab_root=lab_root,
        question=question,
        search_type=search_type,
        top_k=top_k_int,
        answer=answer,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    answer_parsed = parsed
    source_parsed = parsed
    source_proc: subprocess.CompletedProcess[str] | None = None

    if answer_with_sources and answer:
        source_proc, source_parsed = _run_query_script(
            lab_root=lab_root,
            question=question,
            search_type="CHUNKS",
            top_k=top_k_int,
            answer=False,
        )
        source_stderr = source_proc.stderr or ""
        if source_stderr.strip():
            stderr = (stderr + "\n" + source_stderr).strip()
        if source_proc.returncode == 0:
            parsed = {
                "ok": proc.returncode == 0,
                "query_meta": {
                    "question": question,
                    "answer_with_sources": True,
                    "answer_search_type": search_type,
                    "source_search_type": "CHUNKS",
                    "answer_mode": True,
                    "top_k": top_k_int,
                },
                "answer": _extract_answer_text(answer_parsed),
                "answer_results": answer_parsed.get("results") if isinstance(answer_parsed, dict) else answer_parsed,
                "source_results": source_parsed.get("results") if isinstance(source_parsed, dict) else source_parsed,
                "source_files": source_parsed.get("source_files", []) if isinstance(source_parsed, dict) else [],
                "sources": source_parsed.get("sources", []) if isinstance(source_parsed, dict) else [],
                "source_count": source_parsed.get("source_count", 0) if isinstance(source_parsed, dict) else 0,
                "result_items": source_parsed.get("result_items", []) if isinstance(source_parsed, dict) else [],
                "answer_raw": answer_parsed,
                "source_raw": source_parsed,
            }
    rendered = json.dumps(parsed, ensure_ascii=False, indent=2) if not isinstance(parsed, str) else parsed
    truncated_rendered, truncated = _truncate_text(rendered)

    extract_source_files, extract_sources, flatten_result_items = _get_source_helpers(lab_root)
    result_items = parsed.get("result_items", []) if isinstance(parsed, dict) else flatten_result_items(parsed)
    source_files = parsed.get("source_files", []) if isinstance(parsed, dict) else extract_source_files(parsed)
    sources = parsed.get("sources", []) if isinstance(parsed, dict) else extract_sources(parsed)
    source_count = parsed.get("source_count", len(source_files)) if isinstance(parsed, dict) else len(source_files)

    result: dict[str, Any] = {
        "success": proc.returncode == 0,
        "lab_root": str(lab_root),
        "question": question,
        "search_type": search_type,
        "answer_mode": bool(answer),
        "top_k": top_k_int,
        "source_files": source_files,
        "sources": sources,
        "source_count": source_count,
        "result_items": result_items[:20],
        "answer_with_sources": bool(answer_with_sources and answer),
        "result_text": truncated_rendered,
        "truncated": truncated,
        "returncode": proc.returncode,
    }
    if answer_with_sources and answer:
        result["answer_text"] = _extract_answer_text(answer_parsed)
        if source_proc is not None:
            result["source_returncode"] = source_proc.returncode
    if include_raw and not truncated:
        result["raw"] = parsed
    if stderr.strip():
        result["stderr_preview"] = stderr[-4000:]
    if proc.returncode != 0 or (source_proc is not None and source_proc.returncode != 0):
        result["error"] = "query.py returned non-zero exit status"

    return json.dumps(result, ensure_ascii=False, indent=2)


COGNEE_QUERY_SCHEMA = {
    "name": "cognee_query",
    "description": (
        "Read-only query against the isolated cognee-lab PoC corpus. "
        "Use only for explicit retrieval from /root/.hermes/profiles/cognee-lab/cognee_lab. "
        "This tool never ingests, resets, writes memory, or modifies the default profile. "
        "Returns retrieved text, structured source objects, per-result items, and source filenames when present."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Question to ask the isolated cognee-lab retrieval layer.",
            },
            "search_type": {
                "type": "string",
                "enum": sorted(ALLOWED_SEARCH_TYPES),
                "description": "Cognee SearchType to use. Defaults to GRAPH_COMPLETION.",
                "default": "GRAPH_COMPLETION",
            },
            "answer": {
                "type": "boolean",
                "description": "If true, ask cognee for an answer; if false, return retrieved context only.",
                "default": False,
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum retrieval items, clamped to 1..20. Defaults to 5.",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
            },
            "answer_with_sources": {
                "type": "boolean",
                "description": "If true together with answer=true, also run CHUNKS retrieval and attach source_files/sources from source-bearing context.",
                "default": False,
            },
            "include_raw": {
                "type": "boolean",
                "description": "Include parsed raw JSON when output is not truncated. Defaults to true.",
                "default": True,
            },
        },
        "required": ["question"],
    },
}


registry.register(
    name="cognee_query",
    toolset="cognee",
    schema=COGNEE_QUERY_SCHEMA,
    handler=lambda args, **kw: cognee_query(
        question=args.get("question", ""),
        search_type=args.get("search_type", "GRAPH_COMPLETION"),
        answer=args.get("answer", False),
        top_k=args.get("top_k", 5),
        include_raw=args.get("include_raw", True),
        answer_with_sources=args.get("answer_with_sources", False),
    ),
    check_fn=check_cognee_query_requirements,
    emoji="🧠",
    max_result_size_chars=MAX_RESULT_CHARS + 4000,
)
