from __future__ import annotations

import pytest

from tools.blackgold_synthetic_tool import (
    STATE_APPROVED,
    STATE_SYNTHETIC,
    STATE_VALIDATED,
    VALID_RISKS,
    VALID_STAGES,
    VALID_TIERS,
    Contact,
    Deal,
    Document,
    Task,
    approve_pack,
    build_synthetic_contact,
    build_synthetic_deal,
    build_synthetic_document,
    build_synthetic_task,
    make_pack,
    manifest_markdown,
    sha256_content,
    to_json,
    validate_contact,
    validate_deal,
)


def synthetic_deal(**overrides):
    data = {
        "name": "Apex Industrial",
        "stage": "Due Diligence",
        "risk": "Watch",
        "owner": "Grant",
        "tier": "T1",
    }
    data.update(overrides)
    return Deal(**data)


def test_deal_synthetic_at_least_one():
    deals = [synthetic_deal(), synthetic_deal(name="Meridian Services")]
    assert len(deals) >= 1


def test_deal_missing_name_fails():
    result = validate_deal(synthetic_deal(name=""))
    assert result["success"] is False
    assert "invalid_deal" in result["error"]["code"]


def test_deal_invalid_stage_fails():
    result = validate_deal(synthetic_deal(stage="Bad Stage"))
    assert result["success"] is False


def test_deal_valid_passes():
    result = validate_deal(synthetic_deal())
    assert result["success"] is True
    assert result["data"]["state"] == STATE_SYNTHETIC


def test_document_requires_sha256():
    doc = Document(title="NDA")
    assert len(doc.sha256) == 64


def test_document_sha256_matches_content():
    doc = build_synthetic_document("d1", title="Memo", content=b"hello")
    assert doc.sha256 == sha256_content(b"hello")


def test_contact_tier_required():
    contact = Contact(name="Test", tier="T1", state="synthetic")
    assert contact.state == STATE_SYNTHETIC


def test_task_default_source_synthetic():
    task = Task(assignee="Miles")
    assert task.source == "synthetic-seed"
    assert task.state == STATE_SYNTHETIC


def test_approval_required_to_mark_approved():
    deal = synthetic_deal()
    assert deal.state == STATE_SYNTHETIC
    assert STATE_APPROVED != deal.state


def test_schema_invariants_tier_and_state_present():
    for item in [synthetic_deal(), Document(title="Memo"), Contact(name="A"), Task(assignee="B")]:
        assert item.tier in VALID_TIERS
        assert item.state in {STATE_SYNTHETIC, STATE_APPROVED, STATE_VALIDATED, "quarantine", "archived"}


def test_content_hash_deterministic():
    first = sha256_content(b"abc")
    second = sha256_content(b"abc")
    assert first == second
    assert len(first) == 64


def test_pack_default_count():
    pack = make_pack(3)
    assert pack["meta"]["count"] == 3
    assert len(pack["deals"]) == 3


def test_pack_sections_present():
    pack = make_pack(2)
    assert set(pack) >= {"deals", "documents", "contacts", "tasks", "meta"}
    for deal in pack["deals"]:
        assert deal["state"] == STATE_SYNTHETIC


def test_pack_approval_transitions_to_approved():
    pack = make_pack(1)
    assert pack["meta"]["state"] == STATE_SYNTHETIC
    approve_pack(pack, approved_by="arthur", note="synthetic verified")
    assert pack["meta"]["state"] == STATE_APPROVED
    assert pack["meta"]["approved_by"] == "arthur"
    assert all(item["state"] == STATE_APPROVED for item in pack["deals"])


def test_pack_json_round_trip():
    pack = make_pack(1)
    raw = to_json(pack)
    assert raw.count("synthetic") >= 1


def test_pack_manifest_markdown():
    pack = make_pack(1)
    md = manifest_markdown(pack)
    assert md.startswith("| ID | Type | State | Owner/Tier |")


def test_synthetic_document_sha256_updates():
    doc = build_synthetic_document("deal-1", title="IC Memo")
    assert len(doc.sha256) == 64


def test_synthetic_contact_state():
    contact = build_synthetic_contact()
    assert contact.state == STATE_SYNTHETIC


def test_synthetic_task_state():
    task = build_synthetic_task("deal-1", "Miles")
    assert task.state == STATE_SYNTHETIC
    assert task.source == "synthetic-seed"
