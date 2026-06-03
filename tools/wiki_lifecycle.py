#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class WikiLifecycleStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sources (
              source_id TEXT PRIMARY KEY,
              source_type TEXT NOT NULL,
              title TEXT NOT NULL,
              path TEXT NOT NULL,
              author TEXT,
              created_at TEXT,
              ingested_at TEXT NOT NULL,
              checksum TEXT,
              freshness_weight REAL NOT NULL DEFAULT 0.5
            );
            CREATE TABLE IF NOT EXISTS claims (
              claim_id TEXT PRIMARY KEY,
              subject TEXT NOT NULL,
              predicate TEXT NOT NULL,
              object_value TEXT NOT NULL,
              claim_text TEXT NOT NULL,
              domain_tag TEXT NOT NULL,
              first_seen_at TEXT NOT NULL,
              last_confirmed_at TEXT,
              last_accessed_at TEXT,
              status TEXT NOT NULL,
              confidence REAL NOT NULL,
              volatility_class TEXT NOT NULL,
              page_path TEXT NOT NULL,
              superseded_by TEXT,
              archived INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS claim_evidence (
              id TEXT PRIMARY KEY,
              claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
              source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
              evidence_quote TEXT,
              evidence_strength REAL NOT NULL,
              extracted_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS claim_supersession (
              id TEXT PRIMARY KEY,
              old_claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
              new_claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
              reason TEXT NOT NULL,
              decided_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS claim_access_log (
              id TEXT PRIMARY KEY,
              claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
              access_type TEXT NOT NULL,
              accessed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS page_map (
              page_path TEXT PRIMARY KEY,
              page_type TEXT NOT NULL,
              owner TEXT,
              updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_claims_lookup ON claims(subject, predicate, status, confidence);
            CREATE INDEX IF NOT EXISTS idx_claim_evidence_claim_id ON claim_evidence(claim_id);
            CREATE INDEX IF NOT EXISTS idx_claim_access_claim_id ON claim_access_log(claim_id);
            """
        )
        self.conn.commit()

    def add_source(
        self,
        *,
        source_type: str,
        title: str,
        path: str,
        author: str | None = None,
        created_at: str | None = None,
        checksum: str | None = None,
        freshness_weight: float = 0.5,
    ) -> str:
        source_id = "src_" + hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]
        self.conn.execute(
            """
            INSERT OR REPLACE INTO sources
            (source_id, source_type, title, path, author, created_at, ingested_at, checksum, freshness_weight)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (source_id, source_type, title, path, author, created_at, _now(), checksum, freshness_weight),
        )
        self.conn.commit()
        return source_id

    def add_claim(
        self,
        *,
        subject: str,
        predicate: str,
        object_value: str,
        claim_text: str,
        domain_tag: str,
        volatility_class: str,
        page_path: str,
        status: str = "active",
        confidence: float = 0.8,
    ) -> str:
        key = "|".join([subject, predicate, object_value, claim_text])
        claim_id = "clm_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        now = _now()
        self.conn.execute(
            """
            INSERT INTO claims
            (claim_id, subject, predicate, object_value, claim_text, domain_tag, first_seen_at,
             last_confirmed_at, last_accessed_at, status, confidence, volatility_class, page_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(claim_id) DO UPDATE SET
              last_confirmed_at=excluded.last_confirmed_at,
              last_accessed_at=excluded.last_accessed_at,
              status=excluded.status,
              confidence=excluded.confidence,
              page_path=excluded.page_path
            """,
            (
                claim_id,
                subject,
                predicate,
                object_value,
                claim_text,
                domain_tag,
                now,
                now,
                now,
                status,
                confidence,
                volatility_class,
                page_path,
            ),
        )
        self.conn.commit()
        return claim_id

    def add_evidence(
        self,
        *,
        claim_id: str,
        source_id: str,
        evidence_quote: str,
        evidence_strength: float,
    ) -> str:
        evidence_id = _id("ev")
        self.conn.execute(
            """
            INSERT INTO claim_evidence
            (id, claim_id, source_id, evidence_quote, evidence_strength, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (evidence_id, claim_id, source_id, evidence_quote, evidence_strength, _now()),
        )
        self.conn.commit()
        return evidence_id

    def lint(self, now: datetime | None = None) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        rows = self.conn.execute(
            """
            SELECT claim_id, subject, predicate, claim_text, status, confidence, volatility_class,
                   page_path, superseded_by, archived, last_confirmed_at
            FROM claims
            ORDER BY claim_id
            """
        ).fetchall()
        for row in rows:
            claim_id = row["claim_id"]
            evidence_count = self.conn.execute(
                "SELECT COUNT(*) FROM claim_evidence WHERE claim_id = ?", (claim_id,)
            ).fetchone()[0]
            if not row["page_path"]:
                issues.append(self._issue("high", "missing_page_path", claim_id, "Claim has no page_path."))
            if evidence_count == 0:
                issues.append(self._issue("medium", "missing_evidence", claim_id, "Claim has no linked evidence."))
            if row["confidence"] < 0 or row["confidence"] > 1:
                issues.append(self._issue("high", "invalid_confidence", claim_id, "Confidence must be between 0 and 1."))
            if row["status"] not in {"active", "active-high", "draft", "weak", "stale", "superseded", "archived"}:
                issues.append(self._issue("medium", "unknown_status", claim_id, f"Unknown status: {row['status']}"))
            if row["status"] == "superseded" and not row["superseded_by"]:
                issues.append(self._issue("medium", "missing_supersession", claim_id, "Superseded claim has no superseded_by value."))
            if row["archived"] and row["status"] not in {"archived", "superseded"}:
                issues.append(self._issue("low", "archived_active_claim", claim_id, "Archived claim is still marked active."))
        return issues

    def _issue(self, severity: str, kind: str, claim_id: str | None, message: str) -> dict[str, Any]:
        return {"severity": severity, "kind": kind, "claim_id": claim_id, "message": message}

    def export_snapshot(self) -> dict[str, Any]:
        tables = ["sources", "claims", "claim_evidence", "claim_supersession", "claim_access_log", "page_map"]
        payload: dict[str, Any] = {"exported_at": _now(), "db": str(self.db_path), "tables": {}}
        for table in tables:
            rows = self.conn.execute(f"SELECT * FROM {table}").fetchall()
            payload["tables"][table] = [dict(row) for row in rows]
        return payload

    def recompute(self) -> dict[str, Any]:
        rows = self.conn.execute(
            "SELECT page_path, domain_tag FROM claims WHERE page_path IS NOT NULL AND page_path != ''"
        ).fetchall()
        updated = 0
        now = _now()
        for row in rows:
            cur = self.conn.execute(
                """
                INSERT INTO page_map (page_path, page_type, owner, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(page_path) DO UPDATE SET
                  page_type=excluded.page_type,
                  owner=excluded.owner,
                  updated_at=excluded.updated_at
                """,
                (row["page_path"], "claim_page", row["domain_tag"], now),
            )
            updated += cur.rowcount if cur.rowcount is not None else 0
        self.conn.commit()
        return {"ok": True, "updated": updated, "page_count": len(rows), "updated_at": now}
