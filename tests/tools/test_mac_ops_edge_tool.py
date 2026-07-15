from __future__ import annotations

import json

from gateway.mac_ops_edge_client import MacOpsEdgeClientError
from tools import mac_ops_edge_tool
from tools.registry import registry


class _Client:
    def __init__(self) -> None:
        self.calls = []

    def submit_readonly(self, **kwargs):
        self.calls.append(("submit", kwargs))
        return {"state": "queued", "result": {"issue_iid": 17}}

    def read_task(self, **kwargs):
        self.calls.append(("read", kwargs))
        return {"state": "completed", "result": {"notes": []}}


def test_mac_ops_toolset_is_service_gated_and_not_part_of_core_surface() -> None:
    submit = registry.get_entry("mac_ops_readonly_submit")
    read = registry.get_entry("mac_ops_task_read")

    assert submit is not None and submit.toolset == "mac_ops"
    assert read is not None and read.toolset == "mac_ops"
    assert submit.check_fn is mac_ops_edge_tool.mac_ops_edge_configured
    assert read.check_fn is mac_ops_edge_tool.mac_ops_edge_configured


def test_submit_passes_exact_model_authored_contract_without_interpreting(
    monkeypatch,
) -> None:
    client = _Client()
    monkeypatch.setattr(mac_ops_edge_tool, "privileged_mac_ops_edge_client", lambda: client)
    contract = "Objective\nA\nAllowed scope\nB\nForbidden actions\nC\nSecrets handling\nD\nVerification\nE\nExpected report\nF"

    result = json.loads(
        mac_ops_edge_tool._submit(
            {
                "title": "Selected browser read",
                "task_class": "readonly.browser",
                "contract": contract,
                "idempotency_key": "case:17:bitrix",
            }
        )
    )

    assert result["state"] == "queued"
    assert client.calls == [
        (
            "submit",
            {
                "title": "Selected browser read",
                "task_class": "readonly.browser",
                "contract": contract,
                "idempotency_key": "case:17:bitrix",
            },
        )
    ]


def test_uncertain_submit_tells_model_to_reconcile_same_key(monkeypatch) -> None:
    class _Uncertain:
        def submit_readonly(self, **_kwargs):
            raise MacOpsEdgeClientError("transport_failed", dispatch_uncertain=True)

    monkeypatch.setattr(
        mac_ops_edge_tool, "privileged_mac_ops_edge_client", lambda: _Uncertain()
    )

    result = mac_ops_edge_tool._submit(
        {
            "title": "Read",
            "task_class": "readonly.browser",
            "contract": "contract",
            "idempotency_key": "case:uncertain",
        }
    )

    assert "transport_failed" in result
    assert "same idempotency key" in result


def test_read_returns_external_evidence_without_content_classification(monkeypatch) -> None:
    client = _Client()
    monkeypatch.setattr(mac_ops_edge_tool, "privileged_mac_ops_edge_client", lambda: client)

    result = json.loads(
        mac_ops_edge_tool._read(
            {"issue_iid": 17, "idempotency_key": "case:17:observe"}
        )
    )

    assert result["state"] == "completed"
    assert client.calls == [
        ("read", {"issue_iid": 17, "idempotency_key": "case:17:observe"})
    ]
