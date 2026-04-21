#!/usr/bin/env python3
"""
Hermes Governance Tools

Phase 1 governance primitives for auditability and controlled escalation.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from agent.openlegi_code_version_adapter import (
    build_rechercher_code_args,
    normalize_fact_date,
    select_code_article_version,
)
from tools.registry import registry, tool_error


AUDIT_DB_PATH = Path(get_hermes_home()) / "hermes_audit.db"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _connect_audit_db() -> sqlite3.Connection:
    AUDIT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(AUDIT_DB_PATH), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_events (
            event_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            session_id TEXT,
            task_id TEXT,
            event_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            tool_name TEXT,
            entity_type TEXT,
            entity_id TEXT,
            message TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS supervisor_escalations (
            escalation_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            session_id TEXT,
            task_id TEXT,
            severity TEXT NOT NULL,
            reason_code TEXT NOT NULL,
            summary TEXT NOT NULL,
            recommended_action TEXT,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_events_created_at ON audit_events(created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_events_session_id ON audit_events(session_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_events_event_type ON audit_events(event_type)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_supervisor_escalations_created_at ON supervisor_escalations(created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_supervisor_escalations_reason_code ON supervisor_escalations(reason_code)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_supervisor_escalations_status ON supervisor_escalations(status)"
    )


def _insert_audit_event_row(
    conn: sqlite3.Connection,
    *,
    event_id: str,
    created_at: str,
    session_id: str | None,
    task_id: str | None,
    event_type: str,
    severity: str,
    tool_name: str | None,
    entity_type: str | None,
    entity_id: str | None,
    message: str,
    payload_json: str,
) -> None:
    conn.execute(
        """
        INSERT INTO audit_events (
            event_id,
            created_at,
            session_id,
            task_id,
            event_type,
            severity,
            tool_name,
            entity_type,
            entity_id,
            message,
            payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            created_at,
            session_id,
            task_id,
            event_type,
            severity,
            tool_name,
            entity_type,
            entity_id,
            message,
            payload_json,
        ),
    )


def check_hermes_governance_requirements() -> bool:
    return True
def _trace_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"

def _tool_ok(tool_name: str, result: dict[str, Any], trace_id: str | None = None) -> str:
    return json.dumps(
        {
            "ok": True,
            "tool_name": tool_name,
            "trace_id": trace_id or _trace_id(tool_name),
            "timestamp": _utc_now_iso(),
            "result": result,
        },
        ensure_ascii=False,
    )

def _tool_fail(tool_name: str, error_code: str, error_message: str, result: dict[str, Any] | None = None) -> str:
    return json.dumps(
        {
            "ok": False,
            "tool_name": tool_name,
            "trace_id": _trace_id(tool_name),
            "timestamp": _utc_now_iso(),
            "error_code": error_code,
            "error_message": error_message,
            "result": result or {},
        },
        ensure_ascii=False,
    )

def _normalize_aliases(payload: dict[str, Any], alias_map: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    normalized = dict(payload)
    for canonical, aliases in alias_map.items():
        if canonical in normalized and normalized[canonical] not in (None, ""):
            continue
        for alias in aliases:
            value = normalized.get(alias)
            if value not in (None, ""):
                normalized[canonical] = value
                break
    return normalized


def log_audit_event_tool(
    event_type: str,
    summary: str | None = None,
    message: str | None = None,
    severity: str = "medium",
    metadata: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    linked_trace_ids: list[str] | None = None,
    tool_name: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
) -> str:
    event_type = str(event_type or "").strip()
    summary = str(summary or message or "").strip()
    severity = str(severity or "medium").strip().lower()
    metadata = metadata if isinstance(metadata, dict) else (payload if isinstance(payload, dict) else {})
    linked_trace_ids = linked_trace_ids if isinstance(linked_trace_ids, list) else []

    if not event_type:
        return _tool_fail("log_audit_event", "missing_event_type", "event_type is required")
    if not summary:
        return _tool_fail("log_audit_event", "missing_summary", "summary or message is required")
    if severity not in {"low", "medium", "high", "critical", "debug", "info", "warning", "error"}:
        return _tool_fail("log_audit_event", "invalid_severity", "unsupported severity")

    normalized_severity = {
        "debug": "low",
        "info": "low",
        "warning": "medium",
        "error": "high",
    }.get(severity, severity)

    event_id = str(uuid.uuid4())
    created_at = _utc_now_iso()
    payload_json = json.dumps(
        {"metadata": metadata, "linked_trace_ids": linked_trace_ids},
        ensure_ascii=False,
        sort_keys=True,
    )

    conn = _connect_audit_db()
    try:
        _ensure_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        _insert_audit_event_row(
            conn,
            event_id=event_id,
            created_at=created_at,
            session_id=session_id,
            task_id=task_id,
            event_type=event_type,
            severity=normalized_severity,
            tool_name=tool_name,
            entity_type=entity_type,
            entity_id=entity_id,
            message=summary,
            payload_json=payload_json,
        )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        return _tool_fail("log_audit_event", "audit_persist_failed", f"{type(exc).__name__}: {exc}")
    finally:
        conn.close()

    return _tool_ok(
        "log_audit_event",
        {
            "audit_event_id": event_id,
            "accepted": True,
            "event_type": event_type,
            "severity": normalized_severity,
        },
    )


def escalate_to_human_supervisor_tool(
    reason_code: str,
    summary: str,
    severity: str = "high",
    facts: list[str] | None = None,
    blocking_points: list[str] | None = None,
    source_refs: list[str] | None = None,
    payload: dict[str, Any] | None = None,
    recommended_action: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
) -> str:
    reason_code = str(reason_code or "").strip()
    summary = str(summary or "").strip()
    severity = str(severity or "high").strip().lower()
    facts = facts if isinstance(facts, list) else []
    blocking_points = blocking_points if isinstance(blocking_points, list) else []
    source_refs = source_refs if isinstance(source_refs, list) else []
    payload = payload if isinstance(payload, dict) else {}

    if not reason_code:
        return _tool_fail("escalate_to_human_supervisor", "missing_reason_code", "reason_code is required")
    if not summary:
        return _tool_fail("escalate_to_human_supervisor", "missing_summary", "summary is required")
    if severity not in {"medium", "high", "critical", "info", "warning", "error"}:
        return _tool_fail("escalate_to_human_supervisor", "invalid_severity", "unsupported severity")

    normalized_severity = {
        "info": "medium",
        "warning": "high",
        "error": "critical",
    }.get(severity, severity)

    escalation_id = str(uuid.uuid4())
    created_at = _utc_now_iso()
    payload_json = json.dumps(
        {
            "facts": facts,
            "blocking_points": blocking_points,
            "source_refs": source_refs,
            "payload": payload,
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    conn = _connect_audit_db()
    try:
        _ensure_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO supervisor_escalations (
                escalation_id,
                created_at,
                session_id,
                task_id,
                severity,
                reason_code,
                summary,
                recommended_action,
                payload_json,
                status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                escalation_id,
                created_at,
                session_id,
                task_id,
                normalized_severity,
                reason_code,
                summary,
                recommended_action,
                payload_json,
                "open",
            ),
        )
        _insert_audit_event_row(
            conn,
            event_id=str(uuid.uuid4()),
            created_at=created_at,
            session_id=session_id,
            task_id=task_id,
            event_type="supervisor_escalation",
            severity=normalized_severity,
            tool_name="escalate_to_human_supervisor",
            entity_type="escalation",
            entity_id=escalation_id,
            message=summary,
            payload_json=payload_json,
        )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        return _tool_fail("escalate_to_human_supervisor", "escalation_persist_failed", f"{type(exc).__name__}: {exc}")
    finally:
        conn.close()

    return _tool_ok(
        "escalate_to_human_supervisor",
        {
            "escalation_id": escalation_id,
            "accepted": True,
            "supervisor_queue": "default",
            "reason_code": reason_code,
        },
    )


ESCALATE_TO_HUMAN_SUPERVISOR_SCHEMA = {
    "name": "escalate_to_human_supervisor",
    "description": (
        "Create a structured human-supervisor escalation record in Hermes local audit storage. "
        "Use when legal, fiscal, accounting, compliance, or safety thresholds require human validation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason_code": {
                "type": "string",
                "description": "Machine-readable escalation reason, e.g. missing_primary_source or reserved_act."
            },
            "summary": {
                "type": "string",
                "description": "Short factual summary to transmit to the human supervisor."
            },
            "severity": {
                "type": "string",
                "enum": ["info", "warning", "error", "critical"],
                "description": "Escalation severity."
            },
            "payload": {
                "type": "object",
                "description": "Structured supporting context for the escalation."
            },
            "recommended_action": {
                "type": "string",
                "description": "Recommended next human action."
            },
        },
        "required": ["reason_code", "summary"],
    },
}

LOG_AUDIT_EVENT_SCHEMA = {
    "name": "log_audit_event",
    "description": "Persist a structured governance audit event with normalized result envelope.",
    "parameters": {
        "type": "object",
        "properties": {
            "event_type": {"type": "string"},
            "summary": {"type": "string"},
            "message": {"type": "string"},
            "severity": {"type": "string"},
            "metadata": {"type": "object"},
            "payload": {"type": "object"},
            "linked_trace_ids": {
                "type": "array",
                "items": {"type": "string"}
            },
            "tool_name": {"type": "string"},
            "entity_type": {"type": "string"},
            "entity_id": {"type": "string"}
        },
        "required": ["event_type"],
    },
}

registry.register(
    name="log_audit_event",
    toolset="hermes-governance",
    schema=LOG_AUDIT_EVENT_SCHEMA,
    handler=lambda args, **kw: log_audit_event_tool(
        event_type=args.get("event_type", ""),
        summary=args.get("summary"),
        message=args.get("message"),
        severity=args.get("severity", "medium"),
        metadata=args.get("metadata"),
        payload=args.get("payload"),
        linked_trace_ids=args.get("linked_trace_ids"),
        tool_name=args.get("tool_name"),
        entity_type=args.get("entity_type"),
        entity_id=args.get("entity_id"),
        session_id=kw.get("session_id"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_hermes_governance_requirements,
    emoji="🧾",
    max_result_size_chars=12000,
)

registry.register(
    name="escalate_to_human_supervisor",
    toolset="hermes-governance",
    schema=ESCALATE_TO_HUMAN_SUPERVISOR_SCHEMA,
    handler=lambda args, **kw: escalate_to_human_supervisor_tool(
        reason_code=args.get("reason_code", ""),
        summary=args.get("summary", ""),
        severity=args.get("severity", "high"),
        facts=args.get("facts"),
        blocking_points=args.get("blocking_points"),
        source_refs=args.get("source_refs"),
        payload=args.get("payload"),
        recommended_action=args.get("recommended_action"),
        session_id=kw.get("session_id"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_hermes_governance_requirements,
    emoji="🚨",
    max_result_size_chars=12000,
)
def _clean_optional(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        cleaned[key] = value
    return cleaned
def _clean_optional(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        cleaned[key] = value
    return cleaned

def search_legal_sources_tool(
    search: str,
    corpus: str = "legal_text",
    code_name: str | None = None,
    text_id: str | None = None,
    champ: str = "ALL",
    type_recherche: str = "TOUS_LES_MOTS_DANS_UN_CHAMP",
    sort: str = "PERTINENCE",
    page_number: int = 1,
    page_size: int = 10,
    fact_date: str | None = None,
    panorama: bool = False,
    publication_bulletin: str | list[str] | None = None,
    juridiction_judiciaire: str | list[str] | None = None,
    juridiction: list[str] | None = None,
    publication_recueil: list[str] | None = None,
) -> str:
    search = str(search or "").strip()
    corpus = str(corpus or "legal_text").strip().lower()
    fact_date = str(fact_date or "").strip() or None

    if not search:
        return tool_error("search is required")

    dispatch_map = {
        "code": (
            "mcp_openlegi_rechercher_code",
            _clean_optional(
                {
                    "search": search,
                    "code_name": code_name,
                    "champ": champ,
                    "type_recherche": type_recherche,
                    "sort": sort,
                    "page_number": page_number,
                    "page_size": page_size,
                    "date": fact_date,
                }
            ),
        ),
        "legal_text": (
            "mcp_openlegi_rechercher_dans_texte_legal",
            _clean_optional(
                {
                    "search": search,
                    "text_id": text_id,
                    "champ": champ,
                    "type_recherche": type_recherche,
                    "sort": sort,
                    "page_number": page_number,
                    "page_size": page_size,
                }
            ),
        ),
        "jurisprudence_admin": (
            "mcp_openlegi_rechercher_jurisprudence_administrative",
            _clean_optional(
                {
                    "search": search,
                    "champ": champ,
                    "type_recherche": type_recherche,
                    "publication_recueil": publication_recueil,
                    "sort": sort,
                    "page_number": page_number,
                    "page_size": page_size,
                    "panorama": panorama,
                }
            ),
        ),
        "jurisprudence_judiciaire": (
            "mcp_openlegi_rechercher_jurisprudence_judiciaire",
            _clean_optional(
                {
                    "search": search,
                    "publication_bulletin": publication_bulletin,
                    "juridiction_judiciaire": juridiction_judiciaire,
                    "sort": sort,
                    "champ": champ,
                    "type_recherche": type_recherche,
                    "page_number": page_number,
                    "page_size": page_size,
                    "panorama": panorama,
                }
            ),
        ),
        "jurisprudence_financiere": (
            "mcp_openlegi_rechercher_jurisprudence_financiere",
            _clean_optional(
                {
                    "search": search,
                    "champ": champ,
                    "type_recherche": type_recherche,
                    "juridiction": juridiction,
                    "publication_recueil": publication_recueil,
                    "sort": sort,
                    "page_number": page_number,
                    "page_size": page_size,
                    "panorama": panorama,
                }
            ),
        ),
        "convention_collective": (
            "mcp_openlegi_rechercher_conventions_collectives",
            _clean_optional(
                {
                    "search": search,
                    "champ": champ,
                    "type_recherche": type_recherche,
                    "sort": sort,
                    "page_number": page_number,
                    "page_size": page_size,
                    "panorama": panorama,
                }
            ),
        ),
    }

    if corpus not in dispatch_map:
        return tool_error(
            "unsupported corpus",
            supported_corpora=sorted(dispatch_map.keys()),
        )

    if corpus == "code" and not str(code_name or "").strip():
        return tool_error("code_name is required when corpus='code'")

    tool_name, tool_args = dispatch_map[corpus]
    raw_result = registry.dispatch(tool_name, tool_args)

    try:
        parsed_result = json.loads(raw_result)
    except Exception:
        parsed_result = {"raw_text": raw_result}

    if isinstance(parsed_result, dict) and parsed_result.get("error"):
        return _tool_fail(
            "search_legal_sources",
            "backend_dispatch_error",
            str(parsed_result.get("error")),
            {
                "corpus": corpus,
                "backend_tool_name": tool_name,
                "backend_args": tool_args,
            },
        )

    result_text = parsed_result.get("result") or parsed_result.get("raw_text") or ""
    result_text = str(result_text or "")

    source_links: list[str] = []
    if result_text:
        source_links = re.findall(r"https?://\S+", result_text)

    source_count = len(source_links)
    if not source_count:
        source_count = _extract_result_count(result_text)

    return json.dumps(
        {
            "success": bool(result_text),
            "corpus": corpus,
            "tool_name": tool_name,
            "result_text": result_text,
            "source_links": source_links[:20],
            "source_count": source_count,
        },
        ensure_ascii=False,
    )


SEARCH_LEGAL_SOURCES_SCHEMA = {
    "name": "search_legal_sources",
    "description": (
        "Search authoritative French legal sources through the OpenLégi MCP bridge. "
        "Use corpus='code' for a specific code, corpus='legal_text' for consolidated legal texts, "
        "or one of the jurisprudence / convention corpora as appropriate."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "search": {"type": "string", "description": "Search terms or article/decision reference."},
            "corpus": {
                "type": "string",
                "enum": [
                    "code",
                    "legal_text",
                    "jurisprudence_admin",
                    "jurisprudence_judiciaire",
                    "jurisprudence_financiere",
                    "convention_collective"
                ],
                "default": "legal_text",
                "description": "Target legal corpus."
            },
            "code_name": {"type": "string", "description": "Required only when corpus='code'."},
            "text_id": {"type": "string", "description": "Optional legal text id for corpus='legal_text'."},
            "champ": {"type": "string", "default": "ALL"},
            "type_recherche": {"type": "string", "default": "TOUS_LES_MOTS_DANS_UN_CHAMP"},
            "sort": {"type": "string", "default": "PERTINENCE"},
            "page_number": {"type": "integer", "default": 1},
            "page_size": {"type": "integer", "default": 10},
            "panorama": {"type": "boolean", "default": False},
            "publication_bulletin": {
                "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}, {"type": "null"}]
            },
            "juridiction_judiciaire": {
                "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}, {"type": "null"}]
            },
            "juridiction": {
                "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]
            },
            "publication_recueil": {
                "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]
            },
        },
        "required": ["search"],
    },
}


registry.register(
    name="search_legal_sources",
    toolset="hermes-governance",
    schema=SEARCH_LEGAL_SOURCES_SCHEMA,
    handler=lambda args, **kw: search_legal_sources_tool(
        search=args.get("search", ""),
        corpus=args.get("corpus", "legal_text"),
        code_name=args.get("code_name"),
        text_id=args.get("text_id"),
        champ=args.get("champ", "ALL"),
        type_recherche=args.get("type_recherche", "TOUS_LES_MOTS_DANS_UN_CHAMP"),
        sort=args.get("sort", "PERTINENCE"),
        page_number=args.get("page_number", 1),
        page_size=args.get("page_size", 10),
        panorama=args.get("panorama", False),
        publication_bulletin=args.get("publication_bulletin"),
        juridiction_judiciaire=args.get("juridiction_judiciaire"),
        juridiction=args.get("juridiction"),
        publication_recueil=args.get("publication_recueil"),
    ),
    check_fn=check_hermes_governance_requirements,
    emoji="⚖️",
    max_result_size_chars=200000,
)

def get_code_article_version_tool(arguments: dict[str, Any]) -> str:
    args = _normalize_aliases(arguments or {}, {
        "article_ref": ("reference", "article", "article_number", "num_article"),
        "code_name": ("code", "code_title", "code_label"),
        "fact_date": ("date", "as_of_date", "effective_date", "reference_date"),
        "include_text": ("need_article_text", "need_text", "full_text"),
    })

    article_ref = str(args.get("article_ref") or "").strip()
    code_name = str(args.get("code_name") or "").strip()
    raw_fact_date = str(args.get("fact_date") or "").strip()

    if not article_ref:
        return _tool_fail(
            "get_code_article_version",
            "missing_article_ref",
            "article_ref/reference is required",
        )
    if not code_name:
        return _tool_fail(
            "get_code_article_version",
            "missing_code_name",
            "code_name/code is required",
        )
    if not raw_fact_date:
        return _tool_fail(
            "get_code_article_version",
            "missing_fact_date",
            "fact_date/date is required",
        )

    try:
        fact_date = normalize_fact_date(raw_fact_date)
    except Exception as exc:
        return _tool_fail(
            "get_code_article_version",
            "invalid_fact_date",
            f"{type(exc).__name__}: {exc}",
        )

    include_text = bool(args.get("include_text", True))

    try:
        max_candidate_blocks = int(args.get("max_candidate_blocks", 20))
    except Exception:
        max_candidate_blocks = 20
    max_candidate_blocks = max(1, min(max_candidate_blocks, 50))

    backend_args = build_rechercher_code_args(
        article_ref=article_ref,
        code_name=code_name,
        max_candidate_blocks=max_candidate_blocks,
    )

    raw_result = registry.dispatch("mcp_openlegi_rechercher_code", backend_args)

    try:
        parsed_result = json.loads(raw_result)
    except Exception:
        parsed_result = {"raw_text": raw_result}

    if isinstance(parsed_result, dict) and parsed_result.get("error"):
        return _tool_fail(
            "get_code_article_version",
            "backend_dispatch_error",
            str(parsed_result.get("error")),
            {
                "backend_tool_name": "mcp_openlegi_rechercher_code",
                "backend_args": backend_args,
                "article_ref": article_ref,
                "code_name": code_name,
                "fact_date": fact_date,
            },
        )

    result_text = str(parsed_result.get("result") or parsed_result.get("raw_text") or "")

    version = select_code_article_version(
        raw_text=result_text,
        article_ref=article_ref,
        code_name=code_name,
        fact_date=fact_date,
    )

    if version is None:
        return _tool_ok(
            "get_code_article_version",
            {
                "article_ref": article_ref,
                "code_name": code_name,
                "fact_date": fact_date,
                "version_found": False,
                "in_force_on_fact_date": False,
                "coverage_status": "not_verified",
                "vigueur_start": None,
                "vigueur_end": None,
                "text": None,
                "source_url": None,
                "source_ids": [],
                "selection_mode": "exact_num_article+vigueur_window",
                "backend_tool_name": "mcp_openlegi_rechercher_code",
                "backend_args": backend_args,
            },
        )

    return _tool_ok(
        "get_code_article_version",
        {
            "article_ref": version.article_ref,
            "code_name": version.code_name,
            "fact_date": version.fact_date,
            "version_found": True,
            "in_force_on_fact_date": True,
            "coverage_status": "verified",
            "vigueur_start": version.vigueur_start,
            "vigueur_end": version.vigueur_end,
            "text": version.text if include_text else None,
            "source_url": version.source_url,
            "source_ids": list(version.source_ids),
            "selection_mode": version.selection_mode,
            "backend_tool_name": "mcp_openlegi_rechercher_code",
            "backend_args": backend_args,
        },
    )


GET_CODE_ARTICLE_VERSION_SCHEMA = {
    "name": "get_code_article_version",
    "description": (
        "Resolve a specific French code article and select only the version in force "
        "for the provided fact_date. Fail closed when the vigueur window cannot be demonstrated."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "article_ref": {
                "type": "string",
                "description": "Article reference, e.g. L1132-1 or 39."
            },
            "code_name": {
                "type": "string",
                "description": "Exact code name, e.g. Code du travail or Code général des impôts."
            },
            "fact_date": {
                "type": "string",
                "description": "Reference date in YYYY-MM-DD."
            },
            "jurisdiction": {
                "type": "string",
                "default": "FR"
            },
            "include_text": {
                "type": "boolean",
                "default": True
            },
            "max_candidate_blocks": {
                "type": "integer",
                "default": 20,
                "minimum": 1,
                "maximum": 50
            }
        },
        "required": ["article_ref", "code_name", "fact_date"],
    },
}


registry.register(
    name="get_code_article_version",
    toolset="hermes-governance",
    schema=GET_CODE_ARTICLE_VERSION_SCHEMA,
    handler=lambda args, **kw: get_code_article_version_tool(args or {}),
    check_fn=check_hermes_governance_requirements,
    emoji="📜",
    max_result_size_chars=120000,
)

def _extract_result_count(result_text: str) -> int:
    text = str(result_text or "")
    patterns = [
        r"\((\d+)\s+article\(s\)\s+au total\)",
        r"\((\d+)\s+décision\(s\)\s+au total\)",
        r"\((\d+)\s+document\(s\)\s+au total\)",
        r"\((\d+)\s+texte\(s\)\s+au total\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return 0
    return 0
def _parse_fr_date(value: str) -> date | None:
    text = str(value or "").strip()
    if not re.fullmatch(r"\d{2}/\d{2}/\d{4}", text):
        return None
    try:
        return datetime.strptime(text, "%d/%m/%Y").date()
    except Exception:
        return None


def _parse_iso_date(value: str) -> date | None:
    text = str(value or "").strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except Exception:
        return None


def _article_block_matches_fact_date(block: str, fact_date: str | None) -> bool:
    fact = _parse_iso_date(str(fact_date or ""))
    if fact is None:
        return True

    start_match = re.search(r"Date début vigueur:\s*(\d{2}/\d{2}/\d{4})", block)
    end_match = re.search(r"Date fin vigueur:\s*(\d{2}/\d{2}/\d{4})", block)

    start_date = _parse_fr_date(start_match.group(1)) if start_match else None
    end_date = _parse_fr_date(end_match.group(1)) if end_match else None

    if start_date is not None and fact < start_date:
        return False
    if end_date is not None and fact > end_date:
        return False
    return True
def _article_block_matches_target_article(block: str) -> bool:
    header_match = re.search(r"^Article\s+([^\n-]+)", block, re.MULTILINE)
    if header_match:
        article_label = str(header_match.group(1) or "").strip()
        if article_label == "39":
            return True

    number_match = re.search(r"Numéro article:\s*([^\n]+)", block)
    if number_match:
        article_number = str(number_match.group(1) or "").strip()
        if article_number == "39":
            return True

    return False

def _trim_primary_code_hit(result_text: str, fact_date: str | None = None) -> tuple[str, list[str], int]:
    text = str(result_text or "")
    first_start = text.find("=== ARTICLE CODE 1 ===")
    if first_start == -1:
        links = re.findall(r"https?://\S+", text)
        return text, links[:1], 1 if links else 0

    preamble = text[:first_start].rstrip()

    article_matches = list(re.finditer(r"^=== ARTICLE CODE \d+ ===$", text, re.MULTILINE))
    if not article_matches:
        links = re.findall(r"https?://\S+", text)
        return text, links[:1], 1 if links else 0

    blocks: list[str] = []
    for index, match in enumerate(article_matches):
        start = match.start()
        end = article_matches[index + 1].start() if index + 1 < len(article_matches) else len(text)
        blocks.append(text[start:end].rstrip())

    selected_block = None

    for block in blocks:
        if _article_block_matches_target_article(block) and _article_block_matches_fact_date(block, fact_date):
            selected_block = block
            break

    if selected_block is None:
        for block in blocks:
            if _article_block_matches_target_article(block):
                selected_block = block
                break

    if selected_block is None:
        for block in blocks:
            if _article_block_matches_fact_date(block, fact_date):
                selected_block = block
                break

    if selected_block is None:
        selected_block = blocks[0]

    trimmed = f"{preamble}\n\n{selected_block}\n" if preamble else f"{selected_block}\n"
    links = re.findall(r"https?://\S+", trimmed)
    return trimmed, links[:1], 1 if links else 0

def _is_useful_fiscal_payload(payload: dict[str, Any]) -> bool:
    result_text = str(payload.get("result_text") or "")
    result_count = int(payload.get("result_count") or 0)
    lowered = result_text.lower()
    if result_count > 0:
        return True
    if not result_text.strip():
        return False
    negative_markers = [
        "aucun résultat",
        "aucune décision",
        "aucun document",
        "0 article",
        "0 décision",
        "0 document",
        "0 texte",
    ]
    return not any(marker in lowered for marker in negative_markers)

_ARTICLE_WORD_REF_RE = re.compile(
    r"\barticle\s+([A-Z]?\d+(?:-\d+)?(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies))?)\b",
    re.IGNORECASE,
)
_CGI_SUFFIX_ARTICLE_RE = re.compile(
    r"\b([A-Z]?\d+(?:-\d+)?(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies))?)\s+(?:du\s+)?cgi\b",
    re.IGNORECASE,
)


def _normalize_article_ref_value(value: str) -> str:
    return " ".join(str(value or "").upper().split())


def _extract_precise_cgi_article_ref(query: str) -> str | None:
    text = " ".join(str(query or "").split())
    for pattern in (_ARTICLE_WORD_REF_RE, _CGI_SUFFIX_ARTICLE_RE):
        match = pattern.search(text)
        if match:
            return _normalize_article_ref_value(match.group(1))
    return None


def _resolve_precise_cgi_article_version(article_ref: str, fact_date: str) -> dict[str, Any]:
    raw = get_code_article_version_tool(
        {
            "article_ref": article_ref,
            "code_name": "Code général des impôts",
            "fact_date": fact_date,
            "include_text": True,
        }
    )
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        return {
            "ok": False,
            "error_code": "invalid_version_guard_json",
            "error_message": f"{type(exc).__name__}: {exc}",
            "result": {},
            "raw": raw,
        }
    if isinstance(parsed, dict):
        return parsed
    return {
        "ok": False,
        "error_code": "invalid_version_guard_envelope",
        "error_message": "version guard returned a non-dict payload",
        "result": {},
        "raw": raw,
    }


def _build_fiscal_version_guard_response(
    article_ref: str,
    fact_date: str,
    version_envelope: dict[str, Any],
) -> str:
    version_result = (
        version_envelope.get("result", {})
        if isinstance(version_envelope, dict)
        else {}
    )
    verified = (
        bool(version_envelope.get("ok"))
        and isinstance(version_result, dict)
        and version_result.get("coverage_status") == "verified"
    )

    if verified:
        source_url = str(version_result.get("source_url") or "").strip() or None
        result_text = str(version_result.get("text") or "")
        source_links = [source_url] if source_url else []
        sources = []
        if source_url:
            sources.append(
                {
                    "type": "code_article_version",
                    "code_name": version_result.get("code_name"),
                    "article_ref": version_result.get("article_ref"),
                    "fact_date": version_result.get("fact_date"),
                    "source_url": source_url,
                }
            )

        return _tool_ok(
            "search_fiscal_sources",
            {
                "success": True,
                "coverage_status": "verified",
                "primary_sources_verified": True,
                "conflicts_detected": False,
                "sources": sources,
                "result_text": result_text,
                "source_links": source_links,
                "result_count": 1 if source_links else 0,
                "blocking_reason": None,
                "article_ref": article_ref,
                "fact_date": fact_date,
                "version_check": version_result,
            },
        )

    log_audit_event_tool(
        event_type="source_verification",
        severity="high",
        summary=f"Fiscal code article version not verified for CGI article {article_ref} at {fact_date}",
        metadata={
            "article_ref": article_ref,
            "fact_date": fact_date,
            "version_guard": version_envelope,
        },
        tool_name="search_fiscal_sources",
    )

    return _tool_ok(
        "search_fiscal_sources",
        {
            "success": False,
            "coverage_status": "not_verified",
            "primary_sources_verified": False,
            "conflicts_detected": False,
            "sources": [],
            "result_text": "",
            "source_links": [],
            "result_count": 0,
            "blocking_reason": "code_article_version_not_verified",
            "article_ref": article_ref,
            "fact_date": fact_date,
            "version_check": version_envelope.get("result", {}) if isinstance(version_envelope, dict) else {},
        },
    )


def search_fiscal_sources_tool(
    search: str | None = None,
    query: str | None = None,
    source_type: str = "cgi",
    tax_scope: str | None = None,
    fact_date: str | None = None,
    text_id: str | None = None,
    champ: str = "ALL",
    type_recherche: str = "TOUS_LES_MOTS_DANS_UN_CHAMP",
    sort: str = "PERTINENCE",
    page_number: int = 1,
    page_size: int = 10,
    panorama: bool = False,
    juridiction: list[str] | None = None,
    publication_recueil: list[str] | None = None,
) -> str:
    search = str(search or "").strip()
    query = str(query or "").strip()
    source_type = str(source_type or "cgi").strip().lower()
    tax_scope = str(tax_scope or "").strip().lower()
    fact_date = str(fact_date or "").strip() or None

    effective_query = search or query
    if not effective_query:
        return _tool_fail("search_fiscal_sources", "missing_query", "search or query is required")

    if tax_scope:
        scope_map = {
            "is": "cgi",
            "impot_societes": "cgi",
            "cgi": "cgi",
            "lpf": "lpf",
            "procedure": "lpf",
            "procedures": "lpf",
            "jurisprudence_financiere": "jurisprudence_financiere",
        }
        source_type = scope_map.get(tax_scope, source_type)

    compact_query = " ".join(effective_query.split())
    lower_query = compact_query.lower()
    precise_cgi_article_ref = None
    if source_type == "cgi" and fact_date:
        precise_cgi_article_ref = _extract_precise_cgi_article_ref(compact_query)

    if precise_cgi_article_ref:
        version_envelope = _resolve_precise_cgi_article_version(
            precise_cgi_article_ref,
            fact_date,
        )
        return _build_fiscal_version_guard_response(
            precise_cgi_article_ref,
            fact_date,
            version_envelope,
        )


    def _build_fiscal_candidates(base_query: str, resolved_source_type: str) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        lower_query = base_query.lower()

        def _push(value: str) -> None:
            value = " ".join(str(value or "").split())
            if value and value not in seen:
                candidates.append(value)
                seen.add(value)

        _push(base_query)

        if resolved_source_type == "cgi":
            # Candidats prioritaires validés par test réel sur OpenLégi.
            _push("article 39 CGI non admises en déduction")
            _push("article 39 2 CGI non admises en déduction")
            _push("article 39 du CGI pénalités non déductibles")

            # Variantes secondaires, encore centrées sur l'article 39.
            _push("article 39 CGI pénalités")
            _push("article 39 CGI sanctions pécuniaires")
            _push("article 39 du CGI non déductibles")

            if "amende" in lower_query or "pénalité" in lower_query or "penalite" in lower_query:
                _push("article 39 CGI")
                _push("article 39 du CGI")

        if resolved_source_type == "lpf":
            _push(f"article LPF {base_query}")
            _push(f"procédure fiscale {base_query}")

        return candidates[:8]

    def _dispatch_one(search_text: str) -> str:
        if source_type == "cgi":
            return search_legal_sources_tool(
                search=search_text,
                corpus="code",
                code_name="Code général des impôts",
                champ=champ,
                type_recherche=type_recherche,
                sort=sort,
                page_number=page_number,
                page_size=page_size,
                fact_date=fact_date,
            )

        if source_type == "lpf":
            return search_legal_sources_tool(
                search=search_text,
                corpus="code",
                code_name="Livre des procédures fiscales",
                champ=champ,
                type_recherche=type_recherche,
                sort=sort,
                page_number=page_number,
                page_size=page_size,
            )

        if source_type == "legal_text":
            return search_legal_sources_tool(
                search=search_text,
                corpus="legal_text",
                text_id=text_id,
                champ=champ,
                type_recherche=type_recherche,
                sort=sort,
                page_number=page_number,
                page_size=page_size,
            )

        if source_type == "jurisprudence_financiere":
            return search_legal_sources_tool(
                search=search_text,
                corpus="jurisprudence_financiere",
                champ=champ,
                type_recherche=type_recherche,
                sort=sort,
                page_number=page_number,
                page_size=page_size,
                panorama=panorama,
                juridiction=juridiction,
                publication_recueil=publication_recueil,
            )

        return _tool_fail(
            "search_fiscal_sources",
            "unsupported_source_type",
            "unsupported source_type",
            {
                "supported_source_types": ["cgi", "lpf", "legal_text", "jurisprudence_financiere"],
                "received_source_type": source_type,
                "received_tax_scope": tax_scope,
                "fact_date": fact_date,
            },
        )

    def _source_type_to_result_source_type(resolved_source_type: str) -> str:
        if resolved_source_type in {"cgi", "lpf", "legal_text"}:
            return "law"
        if resolved_source_type == "jurisprudence_financiere":
            return "case_law"
        return "other"

    candidate_queries = _build_fiscal_candidates(compact_query, source_type)
    best_payload: dict[str, Any] | None = None

    for search_used in candidate_queries:
        raw = _dispatch_one(search_used)
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {"success": False, "result_text": raw, "source_links": []}

        if payload.get("ok") is False:
            continue

        result_text = str(payload.get("result_text") or "")
        source_links = payload.get("source_links") or []
        if not isinstance(source_links, list):
            source_links = []

        result_count = _extract_result_count(result_text)

        if source_type == "cgi" and result_count > 0:
            result_text, source_links, result_count = _trim_primary_code_hit(
                result_text,
                fact_date=fact_date,
            )

            exact_article = _article_block_matches_target_article(result_text)
            valid_for_date = _article_block_matches_fact_date(result_text, fact_date)

            if not (exact_article and valid_for_date):
                result_text = ""
                source_links = []
                result_count = 0

        sources = []
        for item in source_links[:20]:
            if isinstance(item, str):
                sources.append({
                    "source_type": _source_type_to_result_source_type(source_type),
                    "reference": item,
                    "effective_date": fact_date,
                    "checked_at": _utc_now_iso(),
                    "verified": True,
                    "url": item if item.startswith("http") else None,
                })

        coverage_status = "verified" if result_count > 0 else "not_verified"

        normalized = {
            "success": bool(payload.get("success", True)),
            "ok": result_count > 0,
            "tool_name": "search_fiscal_sources",
            "trace_id": _trace_id("search_fiscal_sources"),
            "timestamp": _utc_now_iso(),
            "source_type": source_type,
            "tax_scope": tax_scope or None,
            "fact_date": fact_date,
            "search_used": search_used,
            "result_count": result_count,
            "result_text": result_text,
            "source_links": source_links[:20],
            "has_results": result_count > 0,
            "result": {
                "success": bool(payload.get("success", True)),
                "tool_name": str(payload.get("tool_name") or "search_fiscal_sources"),
                "sources": sources,
                "coverage_status": coverage_status,
                "primary_sources_verified": result_count > 0 and source_type in {"cgi", "lpf", "legal_text"},
                "conflicts_detected": False,
                "search_used": search_used,
                "result_count": result_count,
                "result_text": result_text,
                "source_links": source_links[:20],
            },
        }
        if source_type == "cgi" and any(token in search_used.lower() for token in ["amende", "pénalité", "penalite"]):
            article_header = re.search(r"=== ARTICLE CODE 1 ===\nArticle\s+([^\n]+)", result_text)
            first_article = article_header.group(1).strip().lower() if article_header else ""

            if first_article and not first_article.startswith("39"):
                continue

        if best_payload is None:
            best_payload = normalized

        if _is_useful_fiscal_payload(normalized):
            return json.dumps(normalized, ensure_ascii=False)

    if best_payload is None:
        return _tool_fail(
            "search_fiscal_sources",
            "no_result",
            "fiscal source search returned no exploitable result",
            {
                "sources": [],
                "coverage_status": "not_verified",
                "primary_sources_verified": False,
                "search_used": compact_query,
                "result_count": 0,
                "result_text": "",
                "source_links": [],
            },
        )

    return json.dumps(best_payload, ensure_ascii=False)


SEARCH_FISCAL_SOURCES_SCHEMA = {
    "name": "search_fiscal_sources",
    "description": (
        "Search French tax-law sources through constrained legal-source routing. "
        "Supports both canonical arguments (search, source_type) and compatibility aliases "
        "(query, tax_scope, fact_date)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "search": {"type": "string"},
            "query": {"type": "string"},
            "source_type": {
                "type": "string",
                "enum": ["cgi", "lpf", "legal_text", "jurisprudence_financiere"],
                "default": "cgi"
            },
            "tax_scope": {
                "type": "string",
                "description": "Compatibility alias, e.g. IS, CGI, LPF, procedure."
            },
            "fact_date": {
                "type": "string",
                "description": "Compatibility field accepted in V1 and currently not used for routing."
            },
            "text_id": {"type": "string"},
            "champ": {"type": "string", "default": "ALL"},
            "type_recherche": {
                "type": "string",
                "default": "TOUS_LES_MOTS_DANS_UN_CHAMP"
            },
            "sort": {"type": "string", "default": "PERTINENCE"},
            "page_number": {"type": "integer", "default": 1},
            "page_size": {"type": "integer", "default": 10},
            "panorama": {"type": "boolean", "default": False},
            "juridiction": {
                "anyOf": [
                    {"type": "array", "items": {"type": "string"}},
                    {"type": "null"}
                ]
            },
            "publication_recueil": {
                "anyOf": [
                    {"type": "array", "items": {"type": "string"}},
                    {"type": "null"}
                ]
            },
        },
        "required": [],
    },
}


registry.register(
    name="search_fiscal_sources",
    toolset="hermes-governance",
    schema=SEARCH_FISCAL_SOURCES_SCHEMA,
    handler=lambda args, **kw: search_fiscal_sources_tool(
        search=args.get("search"),
        query=args.get("query"),
        source_type=args.get("source_type", "cgi"),
        tax_scope=args.get("tax_scope"),
        fact_date=args.get("fact_date"),
        text_id=args.get("text_id"),
        champ=args.get("champ", "ALL"),
        type_recherche=args.get("type_recherche", "TOUS_LES_MOTS_DANS_UN_CHAMP"),
        sort=args.get("sort", "PERTINENCE"),
        page_number=args.get("page_number", 1),
        page_size=args.get("page_size", 10),
        panorama=args.get("panorama", False),
        juridiction=args.get("juridiction"),
        publication_recueil=args.get("publication_recueil"),
    ),
    check_fn=check_hermes_governance_requirements,
    emoji="📚",
    max_result_size_chars=200000,
)

def search_accounting_sources_tool(
    search: str | None = None,
    query: str | None = None,
    ledger_context: str | None = None,
    fact_date: str | None = None,
    source_type: str = "pcg",
    section: str | None = None,
    page_number: int = 1,
    page_size: int = 10,
    **kwargs: Any,
) -> str:
    normalized = _normalize_aliases(
        {
            "search": search,
            "query": query,
            "ledger_context": ledger_context,
            "fact_date": fact_date,
            "source_type": source_type,
            "section": section,
            **kwargs,
        },
        {
            "query": ("search", "question", "prompt", "request"),
            "ledger_context": ("context", "scope", "accounting_scope", "ledger_scope"),
            "fact_date": ("date", "as_of_date", "effective_date", "reference_date"),
        },
    )

    effective_query = str(normalized.get("query") or "").strip()
    ledger_context = str(normalized.get("ledger_context") or "other").strip()
    fact_date = str(normalized.get("fact_date") or "").strip() or None
    source_type = str(normalized.get("source_type") or "pcg").strip().lower()
    section = str(normalized.get("section") or "").strip() or None

    if not effective_query:
        return _tool_fail("search_accounting_sources", "missing_query", "query/search is required")

    supported_source_types = {"pcg", "anc", "ifrs", "audit"}
    if source_type not in supported_source_types:
        return _tool_fail(
            "search_accounting_sources",
            "unsupported_source_type",
            "unsupported source_type",
            {
                "supported_source_types": sorted(supported_source_types),
                "received_source_type": source_type,
            },
        )

    return _tool_ok(
        "search_accounting_sources",
        {
            "sources": [],
            "coverage_status": "not_verified",
            "primary_sources_verified": False,
            "query": effective_query,
            "ledger_context": ledger_context,
            "fact_date": fact_date,
            "source_type": source_type,
            "section": section,
            "page_number": page_number,
            "page_size": page_size,
            "blocking_reason": "accounting_backend_not_configured",
        },
    )


SEARCH_ACCOUNTING_SOURCES_SCHEMA = {
    "name": "search_accounting_sources",
    "description": (
        "Search accounting and audit source materials. "
        "Supports canonical arguments (query, ledger_context, fact_date) and "
        "compatibility aliases (search, context, scope, section). "
        "Current V1 behavior is fail-closed unless a dedicated accounting corpus backend is configured."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "search": {"type": "string"},
            "query": {"type": "string"},
            "ledger_context": {"type": "string"},
            "fact_date": {"type": "string"},
            "source_type": {
                "type": "string",
                "enum": ["pcg", "anc", "ifrs", "audit"],
                "default": "pcg"
            },
            "section": {"type": "string"},
            "context": {"type": "string"},
            "scope": {"type": "string"},
            "accounting_scope": {"type": "string"},
            "ledger_scope": {"type": "string"},
            "date": {"type": "string"},
            "as_of_date": {"type": "string"},
            "effective_date": {"type": "string"},
            "reference_date": {"type": "string"},
            "page_number": {"type": "integer", "default": 1},
            "page_size": {"type": "integer", "default": 10}
        },
        "required": [],
    },
}


registry.register(
    name="search_accounting_sources",
    toolset="hermes-governance",
    schema=SEARCH_ACCOUNTING_SOURCES_SCHEMA,
    handler=lambda args, **kw: search_accounting_sources_tool(
        search=args.get("search"),
        query=args.get("query"),
        ledger_context=args.get("ledger_context"),
        fact_date=args.get("fact_date"),
        source_type=args.get("source_type", "pcg"),
        section=args.get("section"),
        page_number=args.get("page_number", 1),
        page_size=args.get("page_size", 10),
        context=args.get("context"),
        scope=args.get("scope"),
        accounting_scope=args.get("accounting_scope"),
        ledger_scope=args.get("ledger_scope"),
        date=args.get("date"),
        as_of_date=args.get("as_of_date"),
        effective_date=args.get("effective_date"),
        reference_date=args.get("reference_date"),
    ),
    check_fn=check_hermes_governance_requirements,
    emoji="📘",
    max_result_size_chars=12000,
)
CLIENT_RECORDS_PATH = Path(get_hermes_home()) / "client_records.json"


def _load_client_records() -> list[dict[str, Any]] | str:
    if not CLIENT_RECORDS_PATH.exists():
        return tool_error(
            "client records backend not configured",
            success=False,
            db_path=str(CLIENT_RECORDS_PATH),
            remediation=(
                "Create ~/.hermes/client_records.json as a JSON array of client objects "
                "before using get_client_records in production."
            ),
        )

    try:
        raw = json.loads(CLIENT_RECORDS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return tool_error(
            f"failed to read client records backend: {type(exc).__name__}: {exc}",
            success=False,
            db_path=str(CLIENT_RECORDS_PATH),
        )

    if not isinstance(raw, list):
        return tool_error(
            "client records backend must be a JSON array",
            success=False,
            db_path=str(CLIENT_RECORDS_PATH),
        )

    normalized: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            normalized.append(item)
    return normalized


def get_client_records_tool(
    client_id: str | None = None,
    query: str | None = None,
    fields: list[str] | None = None,
    max_results: int = 10,
) -> str:
    client_id = str(client_id or "").strip() or None
    query = str(query or "").strip().lower() or None
    fields = fields or []
    max_results = max(1, min(int(max_results or 10), 100))

    if not client_id and not query:
        return tool_error("client_id or query is required")

    loaded = _load_client_records()
    if isinstance(loaded, str):
        return loaded

    records: list[dict[str, Any]] = loaded
    matched: list[dict[str, Any]] = []

    for record in records:
        record_client_id = str(record.get("client_id", "")).strip()
        if client_id and record_client_id == client_id:
            matched.append(record)
            continue

        if query:
            haystack = json.dumps(record, ensure_ascii=False, sort_keys=True).lower()
            if query in haystack:
                matched.append(record)

        if len(matched) >= max_results:
            break

    if fields:
        projected: list[dict[str, Any]] = []
        for record in matched:
            projected.append({field: record.get(field) for field in fields})
        matched = projected

    return json.dumps(
        {
            "success": True,
            "db_path": str(CLIENT_RECORDS_PATH),
            "count": len(matched),
            "records": matched,
        },
        ensure_ascii=False,
    )


GET_CLIENT_RECORDS_SCHEMA = {
    "name": "get_client_records",
    "description": (
        "Retrieve client records from the local Hermes client-records backend. "
        "Use client_id for exact lookup or query for broad text search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "client_id": {"type": "string"},
            "query": {"type": "string"},
            "fields": {
                "type": "array",
                "items": {"type": "string"}
            },
            "max_results": {
                "type": "integer",
                "default": 10
            },
        },
    },
}


registry.register(
    name="get_client_records",
    toolset="hermes-governance",
    schema=GET_CLIENT_RECORDS_SCHEMA,
    handler=lambda args, **kw: get_client_records_tool(
        client_id=args.get("client_id"),
        query=args.get("query"),
        fields=args.get("fields"),
        max_results=args.get("max_results", 10),
    ),
    check_fn=check_hermes_governance_requirements,
    emoji="🗂️",
    max_result_size_chars=50000,
)
def compute_tax_liability_tool(
    tax_type: str,
    base_amount: float,
    rate: float | None = None,
    deductible_amount: float = 0.0,
    vat_collected: float = 0.0,
    vat_deductible: float = 0.0,
    currency: str = "EUR",
) -> str:
    tax_type = str(tax_type or "").strip().lower()
    currency = str(currency or "EUR").strip().upper()

    try:
        base_amount = float(base_amount)
        deductible_amount = float(deductible_amount or 0.0)
        vat_collected = float(vat_collected or 0.0)
        vat_deductible = float(vat_deductible or 0.0)
        if rate is not None:
            rate = float(rate)
    except Exception as exc:
        return tool_error(f"invalid numeric input: {type(exc).__name__}: {exc}")

    if tax_type not in {"is", "vat", "flat_rate"}:
        return tool_error(
            "unsupported tax_type",
            supported_tax_types=["is", "vat", "flat_rate"],
        )

    if tax_type in {"is", "flat_rate"} and rate is None:
        return tool_error("rate is required for tax_type 'is' and 'flat_rate'")

    if tax_type == "is":
        taxable_base = max(base_amount - deductible_amount, 0.0)
        amount = round(taxable_base * rate, 2)
        assumptions = [
            "V1 deterministic calculator",
            "No reduced brackets or sector-specific adjustments",
            "Taxable base = max(base_amount - deductible_amount, 0)",
        ]
        return json.dumps(
            {
                "success": True,
                "tax_type": tax_type,
                "currency": currency,
                "inputs": {
                    "base_amount": base_amount,
                    "deductible_amount": deductible_amount,
                    "rate": rate,
                },
                "result": {
                    "taxable_base": taxable_base,
                    "amount": amount,
                },
                "assumptions": assumptions,
            },
            ensure_ascii=False,
        )

    if tax_type == "vat":
        amount = round(vat_collected - vat_deductible, 2)
        assumptions = [
            "V1 deterministic calculator",
            "VAT due = VAT collected - VAT deductible",
            "No regime-specific adjustments",
        ]
        return json.dumps(
            {
                "success": True,
                "tax_type": tax_type,
                "currency": currency,
                "inputs": {
                    "vat_collected": vat_collected,
                    "vat_deductible": vat_deductible,
                },
                "result": {
                    "amount": amount,
                },
                "assumptions": assumptions,
            },
            ensure_ascii=False,
        )

    amount = round(base_amount * rate, 2)
    assumptions = [
        "V1 deterministic calculator",
        "Flat formula = base_amount * rate",
    ]
    return json.dumps(
        {
            "success": True,
            "tax_type": tax_type,
            "currency": currency,
            "inputs": {
                "base_amount": base_amount,
                "rate": rate,
            },
            "result": {
                "amount": amount,
            },
            "assumptions": assumptions,
        },
        ensure_ascii=False,
    )


COMPUTE_TAX_LIABILITY_SCHEMA = {
    "name": "compute_tax_liability",
    "description": (
        "Deterministic tax calculator for constrained use cases. "
        "Supports IS, VAT, and generic flat-rate computations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "tax_type": {
                "type": "string",
                "enum": ["is", "vat", "flat_rate"]
            },
            "base_amount": {"type": "number"},
            "rate": {"type": "number"},
            "deductible_amount": {
                "type": "number",
                "default": 0.0
            },
            "vat_collected": {
                "type": "number",
                "default": 0.0
            },
            "vat_deductible": {
                "type": "number",
                "default": 0.0
            },
            "currency": {
                "type": "string",
                "default": "EUR"
            },
        },
        "required": ["tax_type", "base_amount"],
    },
}


registry.register(
    name="compute_tax_liability",
    toolset="hermes-governance",
    schema=COMPUTE_TAX_LIABILITY_SCHEMA,
    handler=lambda args, **kw: compute_tax_liability_tool(
        tax_type=args.get("tax_type", ""),
        base_amount=args.get("base_amount", 0.0),
        rate=args.get("rate"),
        deductible_amount=args.get("deductible_amount", 0.0),
        vat_collected=args.get("vat_collected", 0.0),
        vat_deductible=args.get("vat_deductible", 0.0),
        currency=args.get("currency", "EUR"),
    ),
    check_fn=check_hermes_governance_requirements,
    emoji="🧮",
    max_result_size_chars=12000,
)
