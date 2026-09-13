from __future__ import annotations

from types import SimpleNamespace

from tools.delegate_tool_child_run import _SchemaOutcome, _build_result_entry


def _child(receipts=None):
    child = SimpleNamespace(
        model="test/model",
        session_prompt_tokens=10,
        session_completion_tokens=5,
        session_estimated_cost_usd=0.0,
        session_cost_status="exact",
        _delegate_role="leaf",
        _delegate_purpose="research_evidence",
    )
    if receipts is not None:
        child._delegate_runtime_receipts = receipts
    return child


def _result(summary: str):
    return {
        "final_response": summary,
        "completed": True,
        "interrupted": False,
        "api_calls": 1,
        "messages": [],
    }


def _receipt(receipt_id: str):
    return {
        "receipt_id": receipt_id,
        "child_session_id": "child-session",
        "child_subagent_id": "sa-0-test",
        "tool_call_id": "call-1",
        "tool_name": "read_file",
        "status": "ok",
        "effect_disposition": "none",
        "input_summary": {"argument_keys": ["path"], "targets": {"path": "C:/work/item.txt"}},
        "output_sha256": "a" * 64,
    }


def test_result_entry_promotes_only_runtime_ledger_citations():
    receipt_id = "dr_1234567890abcdef12345678"
    entry = _build_result_entry(
        _child([_receipt(receipt_id)]),
        _result(f"Verified the file [{receipt_id}]."),
        0, 1.0, _SchemaOutcome(None, None, [], 0),
    )
    assert entry["provenance_status"] == "verified_citations"
    assert entry["purpose"] == "research_evidence"
    assert entry["cited_receipt_ids"] == [receipt_id]
    assert entry["runtime_receipts"][0]["output_sha256"] == "a" * 64
    assert entry["fabricated_receipt_ids"] == []


def test_result_entry_rejects_forged_receipt_citation():
    real_id = "dr_1234567890abcdef12345678"
    fake_id = "dr_000000000000000000000000"
    entry = _build_result_entry(
        _child([_receipt(real_id)]), _result(f"Verified [{fake_id}]."),
        0, 1.0, _SchemaOutcome(None, None, [], 0),
    )
    assert entry["provenance_status"] == "missing_citation"
    assert entry["cited_receipt_ids"] == []
    assert entry["fabricated_receipt_ids"] == [fake_id]


def test_reasoning_only_child_has_no_runtime_evidence():
    entry = _build_result_entry(
        _child(), _result("Reasoning-only conclusion; no tools used."),
        0, 1.0, _SchemaOutcome(None, None, [], 0),
    )
    assert entry["provenance_status"] == "no_runtime_evidence"
    assert entry["runtime_receipts"] == []
    assert entry["cited_receipt_ids"] == []
