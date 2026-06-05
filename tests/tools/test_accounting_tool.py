from __future__ import annotations

import json

import toolsets
from tools import accounting_tool


def _loads(result: str) -> dict:
    return json.loads(result)


def test_accounting_toolset_registered():
    tools = toolsets.resolve_toolset("accounting")
    assert "accounting_status" in tools
    assert "accounting_receipt_create" in tools
    assert "accounting_journal_entry_create" in tools
    assert "accounting_export_create" in tools


def test_journal_totals_and_balance_validation(monkeypatch):
    assert accounting_tool._journal_totals([
        {"debit": 400, "credit": 0},
        {"debit": 0, "credit": 400},
    ]) == (400.0, 400.0)
    result = _loads(accounting_tool._handle_journal_entry_create({
        "description": "Unbalanced",
        "lines": [
            {"account_id": "expense", "debit": 400},
            {"account_id": "cash", "credit": 300},
        ],
    }))
    assert result["error"]
    assert "balance" in result["error"]


def test_receipt_create_requires_concept_and_positive_amount():
    result = _loads(accounting_tool._handle_receipt_create({"concept": "", "amount": 0}))
    assert result["error"]
    assert "concept" in result["error"]
