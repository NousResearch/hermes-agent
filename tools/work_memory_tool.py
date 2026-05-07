"""Profile-scoped structured work memory tool.

This is intentionally separate from the small prompt-injected ``memory`` tool.
It stores richer, queryable work intelligence on disk under the active Hermes
profile, so Slack/work bots can maintain durable project context without
stuffing raw channel history into the system prompt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from hermes_constants import get_hermes_home
from tools.registry import registry
from utils import atomic_replace

try:  # pragma: no cover - platform dependent
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


KINDS = {
    "person",
    "project",
    "decision",
    "open_loop",
    "risk",
    "glossary",
    "process",
    "note",
}
STATUSES = {"open", "watching", "closed", "stale", "unknown"}
CONFIDENCE = {"low", "medium", "high"}
DEFAULT_LIMIT = 20
MAX_LIMIT = 100


WORK_MEMORY_SCHEMA = {
    "name": "work_memory",
    "description": (
        "Structured, profile-scoped work memory for project intelligence. Use it to store and query distilled "
        "work facts from Slack or other sources: people, projects, decisions, open loops, risks, glossary terms, "
        "processes, and notes. Store concise curated facts with source provenance; do not dump raw Slack history."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "query", "list", "update", "summary", "export_markdown"],
                "description": "Operation to perform.",
            },
            "id": {"type": "string", "description": "Existing work memory item id for update."},
            "kind": {
                "type": "string",
                "enum": sorted(KINDS),
                "description": "Structured memory type.",
            },
            "title": {"type": "string", "description": "Short canonical title."},
            "content": {"type": "string", "description": "Concise distilled fact, decision, risk, or loop. No raw dumps."},
            "project": {"type": "string", "description": "Project/team/domain key, e.g. sports_product."},
            "owner": {"type": "string", "description": "Owner/person/team if known."},
            "status": {
                "type": "string",
                "enum": sorted(STATUSES),
                "description": "Open-loop/risk status. Defaults to open for risks/open_loop, unknown otherwise.",
            },
            "confidence": {
                "type": "string",
                "enum": sorted(CONFIDENCE),
                "description": "Confidence in the extracted fact. Default medium.",
            },
            "due_date": {"type": "string", "description": "Due date or target date, preferably YYYY-MM-DD."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Small set of query tags."},
            "source_channel": {"type": "string", "description": "Source channel/name/id, e.g. #sports_product."},
            "source_ts": {"type": "string", "description": "Slack timestamp or external source timestamp."},
            "source_url": {"type": "string", "description": "Optional source permalink/URL."},
            "query": {"type": "string", "description": "Case-insensitive text query over title/content/tags/project/owner."},
            "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "description": "Max returned items."},
            "note": {"type": "string", "description": "Update note to append to an existing item."},
        },
        "required": ["action"],
    },
}


def _store_dir() -> Path:
    return get_hermes_home() / "work_memory"


def _items_path() -> Path:
    return _store_dir() / "items.jsonl"


def _markdown_dir() -> Path:
    return _store_dir() / "markdown"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_ok(**payload: Any) -> str:
    return json.dumps({"ok": True, **payload}, ensure_ascii=False)


def _json_error(error: str, **payload: Any) -> str:
    return json.dumps({"ok": False, "error": error, **payload}, ensure_ascii=False)


def _coerce_tags(tags: Any) -> List[str]:
    if not tags:
        return []
    if isinstance(tags, str):
        values = re.split(r"[,#]", tags)
    elif isinstance(tags, Iterable):
        values = [str(v) for v in tags]
    else:
        values = [str(tags)]
    out: List[str] = []
    for value in values:
        tag = re.sub(r"\s+", "_", value.strip().lower().lstrip("#"))
        if tag and tag not in out:
            out.append(tag[:64])
    return out[:12]


def _clean_key(value: str) -> str:
    value = (value or "").strip()
    value = re.sub(r"\s+", "_", value.lower())
    value = re.sub(r"[^a-z0-9_.#:-]+", "", value)
    return value[:96]


def _normal_kind(kind: str) -> str:
    value = _clean_key(kind or "note")
    return value if value in KINDS else "note"


def _normal_status(status: str, kind: str) -> str:
    value = _clean_key(status or "")
    if value in STATUSES:
        return value
    return "open" if kind in {"risk", "open_loop"} else "unknown"


def _normal_confidence(confidence: str) -> str:
    value = _clean_key(confidence or "medium")
    return value if value in CONFIDENCE else "medium"


def _item_id(kind: str, title: str, project: str, source_channel: str = "", source_ts: str = "") -> str:
    if source_channel and source_ts:
        raw = f"{kind}|{project}|{source_channel}|{source_ts}"
    else:
        raw = f"{kind}|{project}|{title.strip().lower()}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"wm_{digest}"


def _lock_path() -> Path:
    return _items_path().with_suffix(".jsonl.lock")


class _FileLock:
    def __enter__(self):
        _store_dir().mkdir(parents=True, exist_ok=True)
        self._fd = open(_lock_path(), "a+", encoding="utf-8")
        if fcntl is not None:
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        if fcntl is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        self._fd.close()
        return False


def _load_items() -> List[Dict[str, Any]]:
    path = _items_path()
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if isinstance(item, dict) and item.get("id"):
                items.append(item)
        except json.JSONDecodeError:
            continue
    return items


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
        atomic_replace(tmp_name, path)
    finally:
        tmp = Path(tmp_name)
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _save_items(items: List[Dict[str, Any]]) -> None:
    _store_dir().mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in items)
    _atomic_write_text(_items_path(), text)


def _write_project_markdown(project: str, items: Optional[List[Dict[str, Any]]] = None) -> Path:
    if items is None:
        items = _filter_items(_load_items(), project=project)
    markdown = _render_markdown(items, project=project)
    _markdown_dir().mkdir(parents=True, exist_ok=True)
    path = _markdown_dir() / f"{_clean_key(project) or 'all'}.md"
    _atomic_write_text(path, markdown)
    return path


def _score(item: Dict[str, Any], terms: List[str]) -> int:
    if not terms:
        return 1
    haystack = " ".join(
        str(item.get(key, "")) for key in ["title", "content", "project", "owner", "status", "kind"]
    ).lower()
    haystack += " " + " ".join(item.get("tags") or [])
    return sum(1 for term in terms if term in haystack)


def _filter_items(
    items: List[Dict[str, Any]],
    *,
    query: str = "",
    kind: str = "",
    project: str = "",
    status: str = "",
    tags: Any = None,
) -> List[Dict[str, Any]]:
    kind = _clean_key(kind)
    project = _clean_key(project)
    status = _clean_key(status)
    tag_filters = set(_coerce_tags(tags))
    terms = [t for t in re.split(r"\s+", (query or "").lower().strip()) if t]

    filtered: List[Tuple[int, Dict[str, Any]]] = []
    for item in items:
        if kind and item.get("kind") != kind:
            continue
        if project and _clean_key(str(item.get("project", ""))) != project:
            continue
        if status and item.get("status") != status:
            continue
        item_tags = set(item.get("tags") or [])
        if tag_filters and not tag_filters.issubset(item_tags):
            continue
        score = _score(item, terms)
        if terms and score <= 0:
            continue
        filtered.append((score, item))
    filtered.sort(key=lambda pair: (pair[0], pair[1].get("updated_at", "")), reverse=True)
    return [item for _, item in filtered]


def _add_action(
    *,
    kind: str = "note",
    title: str = "",
    content: str = "",
    project: str = "",
    owner: str = "",
    status: str = "",
    confidence: str = "medium",
    due_date: str = "",
    tags: Any = None,
    source_channel: str = "",
    source_ts: str = "",
    source_url: str = "",
) -> str:
    title = (title or "").strip()
    content = (content or "").strip()
    if not title:
        return _json_error("title is required")
    if not content:
        return _json_error("content is required")

    normal_kind = _normal_kind(kind)
    project_key = _clean_key(project)
    now = _now_iso()
    item_id = _item_id(normal_kind, title, project_key, source_channel, source_ts)
    new_item = {
        "id": item_id,
        "kind": normal_kind,
        "title": title[:180],
        "content": content[:4000],
        "project": project_key,
        "owner": (owner or "").strip()[:120],
        "status": _normal_status(status, normal_kind),
        "confidence": _normal_confidence(confidence),
        "due_date": (due_date or "").strip()[:40],
        "tags": _coerce_tags(tags),
        "source": {
            "channel": (source_channel or "").strip()[:160],
            "ts": (source_ts or "").strip()[:80],
            "url": (source_url or "").strip()[:400],
        },
        "notes": [],
        "created_at": now,
        "updated_at": now,
    }

    with _FileLock():
        items = _load_items()
        for idx, item in enumerate(items):
            if item.get("id") == item_id:
                merged = {**item, **new_item, "created_at": item.get("created_at", now), "notes": item.get("notes", [])}
                items[idx] = merged
                _save_items(items)
                if project_key:
                    _write_project_markdown(project_key, _filter_items(items, project=project_key))
                return _json_ok(item=merged, updated=True, path=str(_items_path()))
        items.append(new_item)
        _save_items(items)
        if project_key:
            _write_project_markdown(project_key, _filter_items(items, project=project_key))
    return _json_ok(item=new_item, updated=False, path=str(_items_path()))


def _list_or_query_action(
    *,
    query: str = "",
    kind: str = "",
    project: str = "",
    status: str = "",
    tags: Any = None,
    limit: int = DEFAULT_LIMIT,
) -> str:
    items = _filter_items(_load_items(), query=query, kind=kind, project=project, status=status, tags=tags)
    max_items = max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))
    return _json_ok(items=items[:max_items], count=len(items[:max_items]), total=len(items), path=str(_items_path()))


def _update_action(*, id: str = "", status: str = "", note: str = "", owner: str = "", due_date: str = "", tags: Any = None) -> str:
    item_id = (id or "").strip()
    if not item_id:
        return _json_error("id is required for update")
    now = _now_iso()
    with _FileLock():
        items = _load_items()
        for idx, item in enumerate(items):
            if item.get("id") != item_id:
                continue
            if status:
                item["status"] = _normal_status(status, item.get("kind", "note"))
            if owner:
                item["owner"] = owner.strip()[:120]
            if due_date:
                item["due_date"] = due_date.strip()[:40]
            if tags is not None:
                item["tags"] = _coerce_tags(tags)
            if note:
                item.setdefault("notes", []).append({"at": now, "text": note.strip()[:1000]})
            item["updated_at"] = now
            items[idx] = item
            _save_items(items)
            project = item.get("project") or ""
            if project:
                _write_project_markdown(project, _filter_items(items, project=project))
            return _json_ok(item=item, path=str(_items_path()))
    return _json_error(f"work memory item not found: {item_id}")


def _summary_action(*, project: str = "", limit: int = DEFAULT_LIMIT) -> str:
    project_key = _clean_key(project)
    items = _filter_items(_load_items(), project=project_key) if project_key else _load_items()
    items.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    counts = Counter(item.get("kind", "note") for item in items)
    open_items = [item for item in items if item.get("status") in {"open", "watching"}]
    decisions = [item for item in items if item.get("kind") == "decision"]
    max_items = max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))
    return _json_ok(
        project=project_key,
        total=len(items),
        counts_by_kind=dict(counts),
        open_items_count=len(open_items),
        open_items=open_items[:max_items],
        recent_decisions=decisions[:max_items],
        path=str(_items_path()),
    )


def _heading(kind: str) -> str:
    return {
        "decision": "Decisions",
        "open_loop": "Open Loops",
        "risk": "Risks",
        "person": "People",
        "project": "Projects",
        "glossary": "Glossary",
        "process": "Processes",
        "note": "Notes",
    }.get(kind, kind.replace("_", " ").title())


def _render_markdown(items: List[Dict[str, Any]], *, project: str = "") -> str:
    title = f"# Work Memory: {project}" if project else "# Work Memory"
    lines = [title, "", f"Updated: {_now_iso()}", ""]
    for kind in ["project", "decision", "open_loop", "risk", "person", "glossary", "process", "note"]:
        section = [item for item in items if item.get("kind") == kind]
        if not section:
            continue
        lines.extend([f"## {_heading(kind)}", ""])
        for item in section:
            meta = []
            if item.get("status") and item.get("status") != "unknown":
                meta.append(f"status: {item['status']}")
            if item.get("owner"):
                meta.append(f"owner: {item['owner']}")
            if item.get("due_date"):
                meta.append(f"due: {item['due_date']}")
            if item.get("confidence"):
                meta.append(f"confidence: {item['confidence']}")
            source = item.get("source") or {}
            if source.get("channel"):
                source_text = source.get("channel")
                if source.get("ts"):
                    source_text += f" @ {source['ts']}"
                meta.append(f"source: {source_text}")
            tags = item.get("tags") or []
            if tags:
                meta.append("tags: " + ", ".join(tags))
            suffix = f" ({'; '.join(meta)})" if meta else ""
            lines.append(f"- **{item.get('title', 'Untitled')}**{suffix}")
            content = item.get("content") or ""
            if content:
                lines.append(f"  - {content}")
            notes = item.get("notes") or []
            for note in notes[-3:]:
                lines.append(f"  - note {note.get('at', '')}: {note.get('text', '')}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _export_markdown_action(*, project: str = "") -> str:
    project_key = _clean_key(project)
    items = _filter_items(_load_items(), project=project_key) if project_key else _load_items()
    markdown = _render_markdown(items, project=project_key)
    path = None
    if project_key:
        path = _write_project_markdown(project_key, items)
    return _json_ok(markdown=markdown, path=str(path) if path else None, count=len(items))


def work_memory_handler(
    action: str,
    id: str = "",
    kind: str = "",
    title: str = "",
    content: str = "",
    project: str = "",
    owner: str = "",
    status: str = "",
    confidence: str = "medium",
    due_date: str = "",
    tags: Any = None,
    source_channel: str = "",
    source_ts: str = "",
    source_url: str = "",
    query: str = "",
    limit: int = DEFAULT_LIMIT,
    note: str = "",
) -> str:
    """Handle a structured work-memory operation."""
    action = (action or "").strip()
    try:
        if action == "add":
            return _add_action(
                kind=kind,
                title=title,
                content=content,
                project=project,
                owner=owner,
                status=status,
                confidence=confidence,
                due_date=due_date,
                tags=tags,
                source_channel=source_channel,
                source_ts=source_ts,
                source_url=source_url,
            )
        if action in {"query", "list"}:
            return _list_or_query_action(query=query, kind=kind, project=project, status=status, tags=tags, limit=limit)
        if action == "update":
            return _update_action(id=id, status=status, note=note, owner=owner, due_date=due_date, tags=tags)
        if action == "summary":
            return _summary_action(project=project, limit=limit)
        if action == "export_markdown":
            return _export_markdown_action(project=project)
        return _json_error("Unknown action. Use add, query, list, update, summary, or export_markdown.")
    except Exception as exc:  # defensive: tool should fail gracefully in agent loops
        return _json_error(f"work_memory failed: {exc}")


_HANDLER_DEFAULTS = {
    "action": "",
    "id": "",
    "kind": "",
    "title": "",
    "content": "",
    "project": "",
    "owner": "",
    "status": "",
    "confidence": "medium",
    "due_date": "",
    "tags": None,
    "source_channel": "",
    "source_ts": "",
    "source_url": "",
    "query": "",
    "limit": DEFAULT_LIMIT,
    "note": "",
}


registry.register(
    name="work_memory",
    toolset="work_memory",
    schema=WORK_MEMORY_SCHEMA,
    handler=lambda args, **kw: work_memory_handler(
        **{key: args.get(key, default) for key, default in _HANDLER_DEFAULTS.items()}
    ),
    emoji="🧠",
    max_result_size_chars=60000,
)
