"""Native Aion structured observation tools.

Minimal Phase 8 surface: save and search evidence-linked observations in the
active Hermes profile. This intentionally does not write persistent memory,
read raw source files, or inject observations into prompts.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from tools.registry import registry

ALLOWED_TYPES = {
    "decision",
    "bugfix",
    "discovery",
    "procedure",
    "client_fact",
    "person_fact",
    "risk",
    "verification",
    "handoff",
}
ALLOWED_CONFIDENCE = {"low", "medium", "high"}
ALLOWED_STATUS = {"active", "stale", "superseded", "archived"}
MAX_LIMIT = 25
DEFAULT_LIMIT = 10
MAX_LIST_ITEM_LENGTH = 512
MAX_LIST_SERIALIZED_LENGTH = 2000
MAX_SEARCH_TEXT_LENGTH = 240
MAX_TEXT_LENGTHS = {
    "title": 240,
    "narrative": 4000,
    "lesson_learned": 2000,
    "project": 200,
    "client": 200,
    "domain": 120,
    "source_session_id": 200,
    "stale_after": 120,
}

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    source_session_id TEXT,
    project TEXT,
    client TEXT,
    domain TEXT,
    type TEXT NOT NULL CHECK (type IN (
        'decision', 'bugfix', 'discovery', 'procedure', 'client_fact',
        'person_fact', 'risk', 'verification', 'handoff'
    )),
    title TEXT NOT NULL,
    narrative TEXT NOT NULL,
    lesson_learned TEXT,
    evidence_paths TEXT,
    tags TEXT,
    importance INTEGER NOT NULL DEFAULT 3 CHECK (importance BETWEEN 1 AND 5),
    confidence TEXT NOT NULL DEFAULT 'medium' CHECK (confidence IN ('low', 'medium', 'high')),
    stale_after TEXT,
    superseded_by INTEGER REFERENCES observations(id) ON DELETE SET NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    injection_count INTEGER NOT NULL DEFAULT 0,
    use_feedback TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'stale', 'superseded', 'archived'))
);

CREATE INDEX IF NOT EXISTS idx_observations_type ON observations(type);
CREATE INDEX IF NOT EXISTS idx_observations_project ON observations(project);
CREATE INDEX IF NOT EXISTS idx_observations_client ON observations(client);
CREATE INDEX IF NOT EXISTS idx_observations_domain ON observations(domain);
CREATE INDEX IF NOT EXISTS idx_observations_tags ON observations(tags);
CREATE INDEX IF NOT EXISTS idx_observations_status ON observations(status);
CREATE INDEX IF NOT EXISTS idx_observations_importance ON observations(importance);
CREATE INDEX IF NOT EXISTS idx_observations_created_at ON observations(created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS observation_fts USING fts5(
    title,
    narrative,
    lesson_learned,
    evidence_paths,
    tags,
    content='observations',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS observations_ai AFTER INSERT ON observations BEGIN
    INSERT INTO observation_fts(rowid, title, narrative, lesson_learned, evidence_paths, tags)
    VALUES (new.id, new.title, new.narrative, new.lesson_learned, new.evidence_paths, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS observations_ad AFTER DELETE ON observations BEGIN
    INSERT INTO observation_fts(observation_fts, rowid, title, narrative, lesson_learned, evidence_paths, tags)
    VALUES('delete', old.id, old.title, old.narrative, old.lesson_learned, old.evidence_paths, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS observations_au AFTER UPDATE ON observations BEGIN
    INSERT INTO observation_fts(observation_fts, rowid, title, narrative, lesson_learned, evidence_paths, tags)
    VALUES('delete', old.id, old.title, old.narrative, old.lesson_learned, old.evidence_paths, old.tags);
    INSERT INTO observation_fts(rowid, title, narrative, lesson_learned, evidence_paths, tags)
    VALUES (new.id, new.title, new.narrative, new.lesson_learned, new.evidence_paths, new.tags);
END;

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT OR REPLACE INTO schema_meta(key, value) VALUES
    ('schema_name', 'aion_observations'),
    ('schema_version', '1.0.0'),
    ('created_for', 'Aion structured observation memory native tool');
"""


def _json_error(message: str, **extra: Any) -> str:
    payload = {"status": "error", "error": message}
    payload.update(extra)
    return json.dumps(payload, indent=2)


def _json_ok(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2)


def _db_path() -> Path:
    return get_hermes_home().expanduser().absolute() / "aion" / "observations" / "aion_observations.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    return conn


def _connect_existing() -> sqlite3.Connection | None:
    path = _db_path()
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _coerce_optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    max_len = MAX_TEXT_LENGTHS.get(field)
    if max_len and len(text) > max_len:
        raise ValueError(f"{field} exceeds {max_len} characters")
    return text


def _coerce_required_string(value: Any, field: str) -> str:
    text = _coerce_optional_string(value, field)
    if text is None:
        raise ValueError(f"{field} is required")
    return text


def _coerce_list_text(value: Any, field: str) -> tuple[str | None, list[str]]:
    if value is None or value == "":
        return None, []
    if isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[\n,]", value) if part.strip()]
    elif isinstance(value, (list, tuple)):
        parts = [str(part).strip() for part in value if str(part).strip()]
    else:
        raise ValueError(f"{field} must be a string or list of strings")
    for part in parts:
        if len(part) > MAX_LIST_ITEM_LENGTH:
            raise ValueError(f"{field} item exceeds {MAX_LIST_ITEM_LENGTH} characters")
    normalized = json.dumps(parts, ensure_ascii=False) if parts else None
    if normalized is not None and len(normalized) > MAX_LIST_SERIALIZED_LENGTH:
        raise ValueError(f"{field} exceeds {MAX_LIST_SERIALIZED_LENGTH} serialized characters")
    return normalized, parts


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _row_to_observation(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "source_session_id": row["source_session_id"],
        "project": row["project"],
        "client": row["client"],
        "domain": row["domain"],
        "type": row["type"],
        "title": row["title"],
        "narrative": row["narrative"],
        "lesson_learned": row["lesson_learned"],
        "evidence_paths": json.loads(row["evidence_paths"] or "[]"),
        "tags": json.loads(row["tags"] or "[]"),
        "importance": row["importance"],
        "confidence": row["confidence"],
        "stale_after": row["stale_after"],
        "access_count": row["access_count"],
        "injection_count": row["injection_count"],
        "status": row["status"],
    }


def _fts_query(query: str) -> str:
    terms = re.findall(r"\b\w[\w.-]*\b", query, flags=re.UNICODE)
    return " OR ".join(f'"{term.replace(chr(34), chr(34) + chr(34))}"' for term in terms[:12])


def _string_schema(field: str) -> dict[str, Any]:
    return {"type": "string", "maxLength": MAX_TEXT_LENGTHS[field]}


def _list_schema(description: str) -> dict[str, Any]:
    return {
        "description": description,
        "oneOf": [
            {"type": "string", "maxLength": MAX_LIST_SERIALIZED_LENGTH},
            {"type": "array", "items": {"type": "string", "maxLength": MAX_LIST_ITEM_LENGTH}},
        ],
        "items": {"type": "string", "maxLength": MAX_LIST_ITEM_LENGTH},
    }


def observation_save(args: dict[str, Any], **_: Any) -> str:
    """Save one structured observation in the active Hermes profile."""
    try:
        obs_type = str(args.get("type", "")).strip()
        if obs_type not in ALLOWED_TYPES:
            return _json_error(f"invalid type: {obs_type!r}", allowed_types=sorted(ALLOWED_TYPES))

        confidence = str(args.get("confidence", "medium")).strip() or "medium"
        if confidence not in ALLOWED_CONFIDENCE:
            return _json_error(f"invalid confidence: {confidence!r}", allowed_confidence=sorted(ALLOWED_CONFIDENCE))

        importance = int(args.get("importance", 3))
        if importance < 1 or importance > 5:
            return _json_error("importance must be between 1 and 5")

        title = _coerce_required_string(args.get("title"), "title")
        narrative = _coerce_required_string(args.get("narrative"), "narrative")
        lesson_learned = _coerce_optional_string(args.get("lesson_learned"), "lesson_learned")
        project = _coerce_optional_string(args.get("project"), "project")
        client = _coerce_optional_string(args.get("client"), "client")
        domain = _coerce_optional_string(args.get("domain"), "domain")
        stale_after = _coerce_optional_string(args.get("stale_after"), "stale_after")
        source_session_id = _coerce_optional_string(args.get("source_session_id"), "source_session_id")
        evidence_paths, evidence_items = _coerce_list_text(args.get("evidence_paths"), "evidence_paths")
        tags_json, ignored_tags = _coerce_list_text(args.get("tags"), "tags")

        warnings: list[str] = []
        if confidence == "high" and not evidence_items:
            return _json_error("evidence_paths is required when confidence is high")
        if not evidence_items:
            warnings.append("No evidence_paths supplied; observation should remain low-confidence unless independently verified.")
        if confidence != "low" and not evidence_items:
            confidence = "low"
            warnings.append("confidence downgraded to low because evidence_paths was empty")

        now = _now()
        with _connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO observations (
                    created_at, updated_at, source_session_id, project, client, domain,
                    type, title, narrative, lesson_learned, evidence_paths, tags,
                    importance, confidence, stale_after, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                """,
                (
                    now,
                    now,
                    source_session_id,
                    project,
                    client,
                    domain,
                    obs_type,
                    title,
                    narrative,
                    lesson_learned,
                    evidence_paths,
                    tags_json,
                    importance,
                    confidence,
                    stale_after,
                ),
            )
            if cursor.lastrowid is None:
                raise sqlite3.Error("observation insert did not return an id")
            obs_id = int(cursor.lastrowid)
            row = conn.execute("SELECT * FROM observations WHERE id = ?", (obs_id,)).fetchone()
            if row is None:
                raise sqlite3.Error("saved observation could not be reloaded")

        return _json_ok(
            {
                "status": "saved",
                "observation": _row_to_observation(row),
                "warnings": warnings,
                "db_path": str(_db_path()),
            }
        )
    except (ValueError, TypeError, sqlite3.Error) as exc:
        return _json_error(str(exc), db_path=str(_db_path()))


def observation_search(args: dict[str, Any], **_: Any) -> str:
    """Search active Hermes profile observations."""
    try:
        limit = int(args.get("limit", DEFAULT_LIMIT))
        if limit < 1 or limit > MAX_LIMIT:
            return _json_error(f"limit must be between 1 and {MAX_LIMIT}")

        status = str(args.get("status", "active")).strip() or "active"
        if status not in ALLOWED_STATUS:
            return _json_error(f"invalid status: {status!r}", allowed_status=sorted(ALLOWED_STATUS))

        obs_type = args.get("type")
        if obs_type is not None:
            obs_type = str(obs_type).strip()
            if obs_type and obs_type not in ALLOWED_TYPES:
                return _json_error(f"invalid type: {obs_type!r}", allowed_types=sorted(ALLOWED_TYPES))

        clauses = ["o.status = ?"]
        params: list[Any] = [status]
        query = str(args.get("query", "")).strip()
        if len(query) > MAX_SEARCH_TEXT_LENGTH:
            return _json_error(f"query exceeds {MAX_SEARCH_TEXT_LENGTH} characters")
        fts = _fts_query(query) if query else ""
        if query and not fts:
            return _json_error("query must include at least one search token")
        if fts:
            clauses.append("o.id IN (SELECT rowid FROM observation_fts WHERE observation_fts MATCH ?)")
            params.append(fts)
        if obs_type:
            clauses.append("o.type = ?")
            params.append(obs_type)
        for field in ("project", "client", "domain"):
            value = args.get(field)
            if value is not None and str(value).strip():
                clauses.append(f"o.{field} = ?")
                params.append(str(value).strip())

        sql = f"""
            SELECT o.*
            FROM observations o
            WHERE {' AND '.join(clauses)}
            ORDER BY o.importance DESC, o.created_at DESC, o.id DESC
            LIMIT ?
        """
        params.append(limit)

        conn = _connect_existing()
        if conn is None:
            return _json_ok({"status": "ok", "count": 0, "observations": [], "db_path": str(_db_path())})
        with conn:
            rows = conn.execute(sql, params).fetchall()

        observations = [_row_to_observation(row) for row in rows]
        return _json_ok({"status": "ok", "count": len(observations), "observations": observations, "db_path": str(_db_path())})
    except (ValueError, TypeError, sqlite3.Error) as exc:
        return _json_error(str(exc), db_path=str(_db_path()))


OBSERVATION_SAVE_SCHEMA = {
    "description": "Save one evidence-linked structured observation in the active Hermes profile.",
    "parameters": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": sorted(ALLOWED_TYPES)},
            "title": _string_schema("title"),
            "narrative": _string_schema("narrative"),
            "lesson_learned": _string_schema("lesson_learned"),
            "evidence_paths": _list_schema("String or list of evidence path/reference strings."),
            "tags": _list_schema("String or list of tag strings."),
            "importance": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
            "confidence": {"type": "string", "enum": sorted(ALLOWED_CONFIDENCE), "default": "medium"},
            "stale_after": _string_schema("stale_after"),
            "project": _string_schema("project"),
            "client": _string_schema("client"),
            "domain": _string_schema("domain"),
            "source_session_id": _string_schema("source_session_id"),
        },
        "required": ["type", "title", "narrative"],
    },
}

OBSERVATION_SEARCH_SCHEMA = {
    "description": "Search structured observations in the active Hermes profile without reading raw source files.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "maxLength": MAX_SEARCH_TEXT_LENGTH},
            "type": {"type": "string", "enum": sorted(ALLOWED_TYPES)},
            "project": _string_schema("project"),
            "client": _string_schema("client"),
            "domain": _string_schema("domain"),
            "status": {"type": "string", "enum": sorted(ALLOWED_STATUS), "default": "active"},
            "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "default": DEFAULT_LIMIT},
        },
    },
}

registry.register(
    name="observation_save",
    toolset="skills",
    schema=OBSERVATION_SAVE_SCHEMA,
    handler=lambda args, **kwargs: observation_save(args, **kwargs),
    description=OBSERVATION_SAVE_SCHEMA["description"],
    emoji="🧠",
)

registry.register(
    name="observation_search",
    toolset="skills",
    schema=OBSERVATION_SEARCH_SCHEMA,
    handler=lambda args, **kwargs: observation_search(args, **kwargs),
    description=OBSERVATION_SEARCH_SCHEMA["description"],
    emoji="🔎",
)
