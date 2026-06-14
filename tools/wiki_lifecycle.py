from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _status_from_confidence(confidence: float) -> str:
    c = float(confidence)
    if c >= 0.80:
        return "active-high"
    if c >= 0.60:
        return "active"
    if c >= 0.40:
        return "weak"
    if c >= 0.25:
        return "stale"
    return "archived"


def init_wiki_structure(wiki_dir: str, domain: str = "nasa-upotreba") -> dict[str, Any]:
    root = Path(wiki_dir)
    dirs = [
        "protocols", "state", "memory",
        "wiki/concepts", "wiki/systems", "wiki/decisions", "wiki/incidents", "wiki/playbooks", "wiki/comparisons", "wiki/audits",
        "raw/specs", "raw/upstream", "raw/audits", "raw/research", "raw/transcripts", "raw/screenshots", "raw/assets",
        "docs/architecture", "docs/runbooks", "docs/integrations", "docs/deployment", "docs/archive",
        "work/plans", "work/tasks", "work/investigations", "work/reviews", "work/cleanup",
        "templates", "scripts/ingest", "scripts/compile", "scripts/lint", "scripts/verify", "scripts/export",
        "exports",
    ]
    files = {
        "wiki/index.md": "# Wiki Index\n\n",
        "wiki/log.md": "# Wiki Log\n\n",
        "templates/lesson-template.md": "# Lesson\n",
        "templates/decision-template.md": "# Decision\n",
        "templates/incident-template.md": "# Incident\n",
        "templates/playbook-template.md": "# Playbook\n",
        "templates/task-template.md": "# Task\n",
        "state/domain.txt": f"{domain}\n",
    }

    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        p = root / rel
        if not p.exists():
            p.write_text(content, encoding="utf-8")

    db_path = root / "wiki_lifecycle.db"
    store = WikiLifecycleStore(str(db_path))
    store.close()
    return {"ok": True, "wiki_root": str(root), "db": str(db_path)}


@dataclass
class _ClaimCalc:
    confidence: float
    status: str


class WikiLifecycleStore:
    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        self.conn.close()

    def _ensure_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sources (
              source_id TEXT PRIMARY KEY,
              source_type TEXT NOT NULL,
              title TEXT NOT NULL,
              path TEXT,
              freshness_weight REAL NOT NULL DEFAULT 0.5,
              created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS claims (
              claim_id TEXT PRIMARY KEY,
              subject TEXT NOT NULL,
              predicate TEXT NOT NULL,
              object_value TEXT NOT NULL,
              claim_text TEXT NOT NULL,
              domain_tag TEXT,
              volatility_class TEXT NOT NULL DEFAULT 'medium',
              page_path TEXT,
              confidence REAL NOT NULL DEFAULT 0.5,
              status TEXT NOT NULL DEFAULT 'weak',
              first_seen_at TEXT NOT NULL,
              last_confirmed_at TEXT,
              superseded_by TEXT,
              archived INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS claim_evidence (
              evidence_id TEXT PRIMARY KEY,
              claim_id TEXT NOT NULL,
              source_id TEXT NOT NULL,
              evidence_quote TEXT,
              evidence_strength REAL NOT NULL DEFAULT 0.7,
              created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS claim_supersession (
              id TEXT PRIMARY KEY,
              old_claim_id TEXT NOT NULL,
              new_claim_id TEXT NOT NULL,
              reason TEXT,
              created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS claim_access_log (
              id TEXT PRIMARY KEY,
              claim_id TEXT NOT NULL,
              query TEXT,
              accessed_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS page_map (
              page_path TEXT NOT NULL,
              claim_id TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (page_path, claim_id)
            )
            """
        )
        # Legacy DB compatibility: some older schemas missed newer columns.
        self._ensure_column("sources", "path", "TEXT")
        self._ensure_column("sources", "freshness_weight", "REAL NOT NULL DEFAULT 0.5")

        self._ensure_column("claims", "domain_tag", "TEXT")
        self._ensure_column("claims", "volatility_class", "TEXT NOT NULL DEFAULT 'medium'")
        self._ensure_column("claims", "page_path", "TEXT")
        self._ensure_column("claims", "superseded_by", "TEXT")
        self._ensure_column("claims", "archived", "INTEGER NOT NULL DEFAULT 0")

        self.conn.commit()

    def _table_columns(self, table: str) -> set[str]:
        rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(r[1]) for r in rows}

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        cols = self._table_columns(table)
        if column not in cols:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def add_source(self, source_type: str, title: str, path: str = "", freshness_weight: float = 0.5) -> str:
        source_id = self._new_id("src")
        self.conn.execute(
            "INSERT INTO sources(source_id, source_type, title, path, freshness_weight, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (source_id, source_type, title, path, _clamp01(freshness_weight), _utc_now_iso()),
        )
        self.conn.commit()
        return source_id

    def add_claim(
        self,
        subject: str,
        predicate: str,
        object_value: str,
        claim_text: str,
        domain_tag: str = "ops",
        volatility_class: str = "medium",
        page_path: str = "",
    ) -> str:
        claim_id = self._new_id("clm")
        now = _utc_now_iso()
        status = _status_from_confidence(0.55)
        self.conn.execute(
            """
            INSERT INTO claims(
              claim_id, subject, predicate, object_value, claim_text, domain_tag, volatility_class,
              page_path, confidence, status, first_seen_at, last_confirmed_at, superseded_by, archived
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0)
            """,
            (claim_id, subject, predicate, object_value, claim_text, domain_tag, volatility_class, page_path, 0.55, status, now, now),
        )
        if page_path:
            self.conn.execute(
                "INSERT OR REPLACE INTO page_map(page_path, claim_id, updated_at) VALUES (?, ?, ?)",
                (page_path, claim_id, now),
            )
        self.conn.commit()
        return claim_id

    def add_evidence(self, claim_id: str, source_id: str, evidence_quote: str, evidence_strength: float = 0.8) -> str:
        evidence_id = self._new_id("evd")
        now = _utc_now_iso()
        strength = _clamp01(evidence_strength)
        self.conn.execute(
            """
            INSERT INTO claim_evidence(evidence_id, claim_id, source_id, evidence_quote, evidence_strength, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (evidence_id, claim_id, source_id, evidence_quote, strength, now),
        )
        row = self.conn.execute("SELECT confidence FROM claims WHERE claim_id = ?", (claim_id,)).fetchone()
        if row:
            bumped = _clamp01(float(row["confidence"]) + (0.08 * strength))
            self.conn.execute(
                "UPDATE claims SET confidence = ?, status = ?, last_confirmed_at = ? WHERE claim_id = ?",
                (bumped, _status_from_confidence(bumped), now, claim_id),
            )
        self.conn.commit()
        return evidence_id

    def supersede_claim(self, old_claim_id: str, new_claim_id: str, reason: str = "") -> str:
        sid = self._new_id("sup")
        now = _utc_now_iso()
        self.conn.execute(
            "INSERT INTO claim_supersession(id, old_claim_id, new_claim_id, reason, created_at) VALUES (?, ?, ?, ?, ?)",
            (sid, old_claim_id, new_claim_id, reason, now),
        )
        self.conn.execute(
            "UPDATE claims SET superseded_by = ?, archived = 1, status = 'archived' WHERE claim_id = ?",
            (new_claim_id, old_claim_id),
        )
        self.conn.commit()
        return sid

    def _compute_claim_confidence(self, claim: sqlite3.Row, now_dt: datetime) -> _ClaimCalc:
        evidence_rows = self.conn.execute(
            "SELECT evidence_strength FROM claim_evidence WHERE claim_id = ?",
            (claim["claim_id"],),
        ).fetchall()
        cnt = len(evidence_rows)
        avg_strength = sum(float(r["evidence_strength"]) for r in evidence_rows) / cnt if cnt else 0.0

        base = 0.35
        if cnt:
            base += min(0.45, avg_strength * 0.45 + min(cnt, 5) * 0.04)

        last = claim["last_confirmed_at"] or claim["first_seen_at"]
        last_dt = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
        age_days = max(0.0, (now_dt - last_dt).total_seconds() / 86400.0)

        vol = (claim["volatility_class"] or "medium").lower()
        half_life = {"high": 14.0, "medium": 45.0, "low": 120.0}.get(vol, 45.0)
        decay = min(0.35, (age_days / half_life) * 0.35)

        confidence = _clamp01(base - decay)
        if cnt == 0:
            confidence = min(confidence, 0.30)

        status = _status_from_confidence(confidence)
        return _ClaimCalc(confidence=confidence, status=status)

    def recompute_all_confidence(self, now: datetime | None = None) -> int:
        now_dt = now or datetime.now(timezone.utc)
        rows = self.conn.execute("SELECT * FROM claims WHERE archived = 0").fetchall()
        updated = 0
        for r in rows:
            calc = self._compute_claim_confidence(r, now_dt)
            old_conf = float(r["confidence"])
            old_status = str(r["status"])
            if abs(old_conf - calc.confidence) > 1e-9 or old_status != calc.status:
                self.conn.execute(
                    "UPDATE claims SET confidence = ?, status = ? WHERE claim_id = ?",
                    (calc.confidence, calc.status, r["claim_id"]),
                )
                updated += 1
        self.conn.commit()
        return updated

    def query_claims(self, query: str, min_confidence: float = 0.6) -> list[dict[str, Any]]:
        q = f"%{query.strip()}%"
        rows = self.conn.execute(
            """
            SELECT * FROM claims
            WHERE archived = 0
              AND confidence >= ?
              AND (
                subject LIKE ? OR predicate LIKE ? OR object_value LIKE ? OR claim_text LIKE ?
              )
            ORDER BY confidence DESC, COALESCE(last_confirmed_at, first_seen_at) DESC
            """,
            (float(min_confidence), q, q, q, q),
        ).fetchall()
        now = _utc_now_iso()
        for r in rows:
            self.conn.execute(
                "INSERT INTO claim_access_log(id, claim_id, query, accessed_at) VALUES (?, ?, ?, ?)",
                (self._new_id("acc"), r["claim_id"], query, now),
            )
        self.conn.commit()
        return [dict(r) for r in rows]

    def lint(self, now: datetime | None = None) -> list[dict[str, Any]]:
        now_dt = now or datetime.now(timezone.utc)
        rows = self.conn.execute("SELECT * FROM claims").fetchall()
        issues: list[dict[str, Any]] = []
        for r in rows:
            cid = r["claim_id"]
            cnt = self.conn.execute(
                "SELECT COUNT(*) AS c FROM claim_evidence WHERE claim_id = ?",
                (cid,),
            ).fetchone()["c"]
            if int(cnt) == 0:
                issues.append({
                    "kind": "missing_evidence",
                    "severity": "high",
                    "claim_id": cid,
                    "message": "Claim nema evidence zapise.",
                })

            last = r["last_confirmed_at"] or r["first_seen_at"]
            last_dt = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
            age_days = (now_dt - last_dt).total_seconds() / 86400.0
            vol = (r["volatility_class"] or "medium").lower()
            stale_days = {"high": 21, "medium": 60, "low": 180}.get(vol, 60)
            if age_days > stale_days:
                issues.append({
                    "kind": "stale_claim",
                    "severity": "medium",
                    "claim_id": cid,
                    "message": f"Claim nije potvrđen {int(age_days)} dana (limit {stale_days}).",
                })

            conf = float(r["confidence"])
            if conf < 0.40 and int(r["archived"]) == 0:
                issues.append({
                    "kind": "low_confidence_active",
                    "severity": "medium",
                    "claim_id": cid,
                    "message": f"Claim je aktivan s niskim confidence ({conf:.2f}).",
                })

        return issues

    def export_snapshot(self) -> dict[str, Any]:
        def fetch(table: str) -> list[dict[str, Any]]:
            return [dict(r) for r in self.conn.execute(f"SELECT * FROM {table}").fetchall()]

        payload = {
            "exported_at": _utc_now_iso(),
            "db_path": self.db_path,
            "sources": fetch("sources"),
            "claims": fetch("claims"),
            "claim_evidence": fetch("claim_evidence"),
            "claim_supersession": fetch("claim_supersession"),
            "claim_access_log": fetch("claim_access_log"),
            "page_map": fetch("page_map"),
        }
        return payload
