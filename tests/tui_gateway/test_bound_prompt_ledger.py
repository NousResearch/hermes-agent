"""Bound input retains the original engine when the canonical ledger owns work."""
import pytest

from hermes_state_runtime import (
    admit_session_input, begin_runtime_epoch, claim_session_input, list_session_admissions,
)
from tui_gateway import owner_forward, server
from tests.tui_gateway import test_running_conditional_activation as running
from tests.tui_gateway.test_bound_operations import invoke
from tests.tui_gateway.test_conditional_activation import Peer, activate


@pytest.fixture
def built(monkeypatch, tmp_path):
    yield from running.built.__wrapped__(monkeypatch, tmp_path)


@pytest.mark.parametrize("started", [False, True])
def test_bound_prompt_refuses_ledger_owned_work_without_forwarding(built, monkeypatch, started):
    _creator, binding, record, _agent, db = built
    peer = Peer()
    assert "result" in activate(peer, binding)
    monkeypatch.setattr(server, "_get_db", lambda: db)
    key = binding["stored_session_id"]
    epoch = begin_runtime_epoch(db, instance_id="gateway")
    admit_session_input(db, epoch=epoch, principal_id="owner", session_id=key,
                        request_id="existing", payload={"text": "existing work"})
    if started:
        claim_session_input(db, epoch=epoch, session_id=key)
    before = list_session_admissions(db, session_id=key)
    forwarded = []

    def forward(*args, **kwargs):
        forwarded.append((args, kwargs))
        return {"admission_id": "new-owner-admission"}

    monkeypatch.setattr(owner_forward, "forward_prompt", forward)
    response = invoke(peer, binding, "prompt.submit", text="deliberate original-engine input")
    assert response.get("error", {}).get("code") == 4090, response
    assert response["error"]["data"]["reason"] == "SESSION_NOT_OWNED"
    assert not forwarded
    assert list_session_admissions(db, session_id=key) == before
    assert not record["running"] and db.get_messages_as_conversation(key) == []
