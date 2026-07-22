from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

STATE_SYNTHETIC = "synthetic"
STATE_APPROVED = "approved"
STATE_VALIDATED = "validated"
_STATES = {STATE_SYNTHETIC, STATE_APPROVED, STATE_VALIDATED, "quarantine", "archived"}

VALID_STAGES = {"Teaser In Market", "LOI Submitted", "Due Diligence", "IC Review"}
VALID_RISKS = {"Watch", "Conditional", "Hard Stop"}
VALID_TIERS = {"T0", "T1", "T2", "T3", "T4"}

_TEAMS = ["M&A", "Growth", "Infrastructure", "Real Estate", "Credit"]
_SECTORS = ["Industrial", "Business Services", "Healthcare", "Technology", "Energy"]
_OWNERS = ["Grant", "Malcolm", "Naomi", "Victor", "Evelyn"]


def _rand_suffix(n: int = 6) -> str:
    return secrets.token_hex(n)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_content(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


_DOC_EMPTY_SHA256 = sha256_content(b"")


@dataclass(frozen=True)
class Deal:
    id: str = field(default_factory=lambda: "BG-D-" + _rand_suffix())
    name: str = ""
    stage: str = "Teaser In Market"
    risk: str = "Watch"
    sector: str = ""
    owner: str = "Grant"
    estimated_value: int = 0
    status: str = "open"
    tier: str = "T1"
    state: str = STATE_SYNTHETIC
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)


@dataclass(frozen=True)
class Document:
    id: str = field(default_factory=lambda: "BG-DOC-" + _rand_suffix())
    deal_id: str = ""
    title: str = ""
    extension: str = "pdf"
    sha256: str = field(default=_DOC_EMPTY_SHA256)
    tier: str = "T1"
    state: str = STATE_SYNTHETIC
    stored_path: str = ""
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)


@dataclass(frozen=True)
class Contact:
    id: str = field(default_factory=lambda: "BG-C-" + _rand_suffix())
    name: str = ""
    role: str = ""
    email: str = ""
    phone_e164: str = ""
    tier: str = "T1"
    state: str = STATE_SYNTHETIC
    created_at: str = field(default_factory=_now_iso)


@dataclass(frozen=True)
class Task:
    id: str = field(default_factory=lambda: "BG-T-" + _rand_suffix())
    deal_id: str = ""
    assignee: str = ""
    status: str = "open"
    priority: str = "medium"
    due_at: str = ""
    source: str = "synthetic-seed"
    tier: str = "T1"
    state: str = STATE_SYNTHETIC
    created_at: str = field(default_factory=_now_iso)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_deal(deal: Deal) -> Dict[str, Any]:
    errors = []
    if not deal.name:
        errors.append("name required")
    if deal.stage not in VALID_STAGES:
        errors.append("invalid stage")
    if deal.risk not in VALID_RISKS:
        errors.append("invalid risk")
    if deal.tier not in VALID_TIERS:
        errors.append("invalid tier")
    if not deal.owner:
        errors.append("owner required")

    env = {"deal_id": deal.id, "state": deal.state}
    if errors:
        return {
            "success": False,
            "error": {
                "code": "invalid_deal",
                "details": errors,
                "actionable_remediation": "Correct field values",
                **env,
            },
        }
    return {"success": True, "data": env}


def validate_contact(contact: Contact) -> Dict[str, Any]:
    errors = []
    if not contact.name:
        errors.append("name required")
    if contact.tier not in VALID_TIERS:
        errors.append("invalid tier")
    if contact.state not in _STATES:
        errors.append("invalid state")

    env = {"contact_id": contact.id, "state": contact.state}
    if errors:
        return {
            "success": False,
            "error": {"code": "invalid_contact", "details": errors, **env},
        }
    return {"success": True, "data": env}


# ---------------------------------------------------------------------------
# Synthetic pack generation
# ---------------------------------------------------------------------------

def _rand_item(seq: List[str]) -> str:
    return secrets.choice(seq)


def _next_tier(index: int) -> str:
    return f"T{index % 5}"


def build_synthetic_deal(index: int = 0, *, sector: Optional[str] = None, owner: Optional[str] = None) -> Deal:
    sector = sector or _rand_item(_SECTORS)
    owner = owner or _rand_item(_OWNERS)
    tier = _next_tier(index)
    return Deal(
        name=f"{_rand_suffix(4).capitalize()} {sector}",
        stage=_rand_item(list(VALID_STAGES)),
        risk=_rand_item(list(VALID_RISKS)),
        sector=sector,
        owner=owner,
        tier=tier,
        estimated_value=(index + 1) * 1_000_000 + secrets.randbelow(950_000),
    )


def build_synthetic_document(deal_id: str, title: str, *, extension: str = "pdf", content: Optional[bytes] = None) -> Document:
    data = content or title.encode("utf-8")
    return Document(
        deal_id=deal_id,
        title=title,
        extension=extension,
        sha256=sha256_content(data),
    )


def build_synthetic_contact(name: Optional[str] = None, *, tier: Optional[str] = None) -> Contact:
    name = name or f"Contact {_rand_suffix(3)}"
    tier = tier or _rand_item(list(VALID_TIERS))
    return Contact(
        name=name,
        role=_rand_item(["Selling Partner", "Buyer", "Advisor", "Lender"]),
        email=f"{name.lower().replace(' ', '.')}@example.test",
        phone_e164=f"+1555000{secrets.randbelow(10000):04d}",
        tier=tier,
    )


def build_synthetic_task(deal_id: str, assignee: str) -> Task:
    return Task(
        deal_id=deal_id,
        assignee=assignee,
        status=_rand_item(["open", "pending", "blocked"]),
        priority=_rand_item(["low", "medium", "high"]),
        due_at=_now_iso(),
    )


def make_pack(count: int = 5, *, seed_owner: Optional[str] = None) -> Dict[str, Any]:
    if count < 1:
        raise ValueError("count must be >= 1")

    deals: List[Deal] = []
    documents: List[Document] = []
    contacts: List[Contact] = []
    tasks: List[Task] = []

    for i in range(count):
        deal = build_synthetic_deal(i, owner=seed_owner)
        deals.append(deal)

        documents.append(build_synthetic_document(deal.id, f"{deal.name} - CIM"))
        documents.append(build_synthetic_document(deal.id, f"{deal.name} - NDA", extension="docx"))

        contacts.append(build_synthetic_contact())
        tasks.append(build_synthetic_task(deal.id, deal.owner))

    pack: Dict[str, Any] = {
        "deals": [d.__dict__ for d in deals],
        "documents": [d.__dict__ for d in documents],
        "contacts": [c.__dict__ for c in contacts],
        "tasks": [t.__dict__ for t in tasks],
        "meta": {
            "count": count,
            "generated_at": _now_iso(),
            "state": STATE_SYNTHETIC,
            "owner": seed_owner or _rand_item(_OWNERS),
        },
    }
    return pack


# ---------------------------------------------------------------------------
# Quarantine / approval workflow
# ---------------------------------------------------------------------------

def approve_pack(pack: Dict[str, Any], approved_by: str = "arthur", note: Optional[str] = None) -> Dict[str, Any]:
    if pack.get("meta", {}).get("state") != STATE_SYNTHETIC:
        raise ValueError("pack is not in synthetic state")

    now = _now_iso()
    audit = {
        "action": "approve",
        "by": approved_by,
        "at": now,
    }
    if note:
        audit["note"] = note

    for section in ("deals", "documents", "contacts", "tasks"):
        for item in pack.get(section, []):
            item["state"] = STATE_APPROVED
            item["updated_at"] = now
            item["approved_by"] = approved_by

    pack.setdefault("audit", []).append(audit)
    pack["meta"]["state"] = STATE_APPROVED
    pack["meta"]["approved_at"] = now
    pack["meta"]["approved_by"] = approved_by
    return pack


# ---------------------------------------------------------------------------
# Docs/manifest helpers
# ---------------------------------------------------------------------------

def manifest_markdown(pack: Dict[str, Any]) -> str:
    lines = [
        "| ID | Type | State | Owner/Tier |",
        "|----|------|-------|------------|",
    ]
    for d in pack.get("deals", []):
        lines.append(f"| {d['id']} | deal | {d['state']} | {d.get('owner','')} / {d.get('tier','')} |")
    for doc in pack.get("documents", []):
        lines.append(f"| {doc['id']} | document | {doc['state']} | {doc.get('tier','')} |")
    for c in pack.get("contacts", []):
        lines.append(f"| {c['id']} | contact | {c['state']} | {c.get('role','')} / {c.get('tier','')} |")
    return "\n".join(lines)


def to_json(pack: Dict[str, Any], indent: int = 2) -> str:
    return json.dumps(pack, indent=indent)
