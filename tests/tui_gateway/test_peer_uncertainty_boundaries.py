"""Inert producer/consumer regressions: no database, worker, RPC or control execution."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.platforms import api_server_authority_runs as authority_runs
from gateway.platforms import api_server_runs as runs
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError


def _error(message, **kwargs):
    return {"error": {"message": message, **kwargs}}


def _peer(monkeypatch, projection):
    now = [0.0]
    client = PeerRunsHTTPClient(base_url="http://127.0.0.1:12345", api_key="", clock=lambda: now[0])
    record = dict(run_id="run-fixture", session_id="logical-session", room_id="room", target_profile="named",
                  task_id="task", execution_generation=3)
    client._runs[("task", 3)] = record
    client.bind_observation(task_id="task", execution_generation=3)
    request = Mock(side_effect=lambda *a, **kw: projection())
    monkeypatch.setattr(client, "_request", request)
    return client, record, now, request


def _observation(client, name):
    return getattr(client, name)(room_id="room", profile="named", session_id="logical-session", grant="fixture-grant")


def test_dead_owner_reads_preserve_unacknowledged_bytes(monkeypatch, tmp_path):
    from gateway import hosted_room_artifacts
    fixture = tmp_path / "ambiguous-output.bin"
    fixture.write_bytes(b"unacknowledged ambiguous output\x00")
    original = fixture.read_bytes()
    outbox = Mock()
    monkeypatch.setattr(hosted_room_artifacts, "RoomArtifactOutbox", Mock(return_value=outbox))
    monkeypatch.setattr(hosted_room_artifacts.RoomArtifactScope, "from_mapping", Mock(return_value=object()))
    monkeypatch.setattr(authority_runs, "run_projection", Mock(return_value=None))
    monkeypatch.setattr(runs, "_owner_alive", Mock(return_value=False))
    stored = dict(run_id="run-fixture", status="running", output="retained partial output", room_artifact_scope={"fixture": True})
    store = Mock()
    store.status_for_run.side_effect = lambda *a, **kw: {"status": dict(stored), "owner_pid": 123, "owner_started": 456}
    store.update_status.side_effect = lambda rid, status: stored.update(status)
    adapter = SimpleNamespace(_run_statuses={}, _run_idempotency_ids=set(), _run_owners={},
                              _run_idempotency_scope=lambda req: "scope", _run_idempotency_store=store)
    for _ in range(3):
        status = runs._durable_run_status(adapter, {}, "run-fixture")
        assert outbox.discard_durably.call_count == 0
        assert fixture.read_bytes() == original
        assert status["status"] == status["execution_state"] == "unknown"
        assert status["settled"] is False
        assert status["output"] == "retained partial output"
        adapter._run_statuses.clear()  # repeat the durable read, not just the memory cache


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_state", ["interrupted", "failed", "cancelled", "unknown"])
async def test_uncertainty_never_authorizes_terminal_stop_cleanup(monkeypatch, wire_state):
    from gateway import hosted_room_artifacts
    outbox = Mock()
    monkeypatch.setattr(hosted_room_artifacts, "RoomArtifactOutbox", Mock(return_value=outbox))
    monkeypatch.setattr(hosted_room_artifacts.RoomArtifactScope, "from_mapping", Mock(return_value=object()))
    status = dict(status=wire_state, execution_state="unknown", settled=False, room_artifact_scope={"fixture": True})
    monkeypatch.setattr(runs, "_load_owned_run", Mock(return_value=("run-fixture", status, None, None, None)))
    response = await runs._handle_stop_run(SimpleNamespace(gateway_runner=None), object(),
                                          _api_server=SimpleNamespace(_openai_error=_error))
    assert outbox.discard_durably.call_count == 0
    assert response.status == 409
    assert json.loads(response.text)["error"]["code"] == "unknown_execution"
    # Automatic retention classification is also a reader of the discriminator.
    # Bind only that method with a mock cursor: no store/DB or deletion executes.
    from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
    conn = Mock()
    conn.execute.return_value.fetchall.return_value = [("scope", "key", json.dumps(status))]
    store = SimpleNamespace(_conn=conn, ACKNOWLEDGED_RETENTION_SECONDS=1, RETENTION_SECONDS=1)
    RunIdempotencyStore._prune_stale_terminal_locked(store, 100.0)
    assert not any(call.args[0].startswith("DELETE") for call in conn.execute.call_args_list)


@pytest.mark.parametrize("outcome", ["completed", "failed", "interrupted"])
def test_canonical_unknown_can_later_be_observed_terminal(monkeypatch, outcome):
    row = dict(status="unknown", outcome=None, target_session_id="target-session", admission_id="admission", generation=17)
    authority = SimpleNamespace(db=object(), sessions={})
    monkeypatch.setattr(authority_runs, "run_admission", Mock(return_value=(authority, row)))
    monkeypatch.setattr(authority_runs, "admission_result", Mock(side_effect=lambda *a: (
        {"result": {"final_response": "final fixture"}, "usage": {}} if row["status"] == "terminal" else None)))
    client, record, now, request = _peer(monkeypatch, lambda: authority_runs.run_projection(None, "run-fixture"))
    assert _observation(client, "history") == []
    info = _observation(client, "status")
    assert info["status"] == info["execution_state"] == "unknown"
    assert info["settled"] is False and info["active"] is True
    assert info["admission_id"] == "admission"
    assert info["target_execution_generation"] == 17
    assert info["execution_generation"] == record["execution_generation"] == 3
    assert not client._terminal_receipts
    row.update(status="terminal", outcome=outcome)
    now[0] = 10.0
    history = _observation(client, "history")
    assert history[0]["status"] == ("settled" if outcome == "completed" else "failed")
    assert history[0]["admission_id"] == "admission"
    assert history[0]["target_execution_generation"] == 17
    assert history[0]["execution_generation"] == 3
    assert client._terminal_receipts == {("task", 3)}
    assert request.call_count == 2


@pytest.mark.parametrize("mismatch", [
    "run_id", "admission_id", "execution_generation", "bool_generation", "lost_discriminator",
    "lost_admission", "cached_admission", "cached_generation", "pending_generation", "grant_refresh",
    "first_lost_admission"])
def test_stale_canonical_projection_cannot_settle(monkeypatch, mismatch):
    wire = dict(run_id="run-fixture", status="running", execution_state="started", settled=False,
                admission_id="admission", execution_generation=17, pending_controls=[])
    client, record, now, _ = _peer(monkeypatch, lambda: dict(wire))
    if mismatch == "first_lost_admission":
        wire.pop("admission_id")
        with pytest.raises(PeerRunsHTTPError):
            _observation(client, "status")
        assert not client._terminal_receipts
        return
    assert _observation(client, "history") == []
    wire.update(status="completed", execution_state="terminal", settled=True)
    if mismatch.startswith("cached_"):
        assert _observation(client, "history") == []  # active backoff still applies
        now[0] = 10.0
        assert _observation(client, "history")[0]["status"] == "settled"
        if mismatch == "cached_admission":
            record["admission_id"] = "stale"
        else:
            record["target_execution_generation"] = 18
        with pytest.raises(PeerRunsHTTPError):
            _observation(client, "history")
        return
    if mismatch == "lost_discriminator":
        wire.pop("execution_state")
        wire.pop("settled")
    elif mismatch == "lost_admission":
        wire.pop("admission_id")
    elif mismatch == "pending_generation":
        wire["pending_controls"] = [{"kind": "approval", "prompt_id": "prompt", "execution_generation": 18}]
    elif mismatch == "grant_refresh":
        wire["admission_id"] = "stale"
        client._status_cache["run-fixture"].update(
            grant_sha256="retired", error=PeerRunsHTTPError(
                "expired", status_code=403, error_code="invalid_room_grant"))
    elif mismatch == "bool_generation":
        wire["execution_generation"] = True
    else:
        wire[mismatch] = 18 if mismatch == "execution_generation" else "stale"
    now[0] = 10.0
    with pytest.raises(PeerRunsHTTPError):
        _observation(client, "history")
    assert not client._terminal_receipts
    wire.update(run_id="run-fixture", admission_id="admission", execution_generation=17,
                execution_state="terminal", settled=True, pending_controls=[])
    now[0] = 20.0
    assert _observation(client, "history")[0]["status"] == "settled"


def test_pending_approval_keeps_target_identity_and_refuses_control(monkeypatch):
    prompt = dict(kind="approval", prompt_id="prompt", execution_generation=17, command="fixture", choices=["once", "deny"])
    row = dict(status="started", outcome=None, target_session_id="target-session", admission_id="admission", generation=17)
    authority = SimpleNamespace(db=object(), sessions={"target-session": SimpleNamespace(controls=SimpleNamespace(snapshot=Mock(return_value=[prompt])))})
    monkeypatch.setattr(authority_runs, "run_admission", Mock(return_value=(authority, row)))
    monkeypatch.setattr(authority_runs, "admission_result", Mock(return_value=None))
    client, _, now, request = _peer(monkeypatch, lambda: authority_runs.run_projection(None, "run-fixture"))
    info = _observation(client, "status")
    assert info["pending_controls"] == [prompt]
    assert info["approval"]["request_id"] == "prompt"
    assert info["approval"]["target_execution_generation"] == 17
    assert info["approval"]["admission_id"] == "admission"
    assert info["approval"]["control_supported"] is False
    assert info["execution_generation"] == 3
    with pytest.raises(PeerRunsHTTPError) as exc:
        client.approve_receipt(task_id="task", execution_generation=3, request_id="prompt", choice="once", grant="fixture-grant")
    assert exc.value.error_code == "canonical_room_peer_unsupported"
    assert all(call.kwargs.get("method", "GET") == "GET" for call in request.call_args_list)
    row['status'] = 'unknown'
    now[0] = 10.0
    unknown = _observation(client, "status")
    assert unknown['last_observed_pending_controls'] == [prompt]
    assert unknown['settled'] is False and unknown['active'] is True
    assert not client._terminal_receipts


@pytest.mark.parametrize("state", ["completed", "failed", "cancelled", "interrupted"])
def test_legacy_positive_terminal_behavior_remains(monkeypatch, state):
    from gateway.platforms.api_server_run_idempotency import run_status_is_terminal
    assert run_status_is_terminal({"status": state}) is True
    client, _, _, _ = _peer(monkeypatch, lambda: {"run_id": "run-fixture", "status": state, "output": "legacy fixture"})
    history = _observation(client, "history")
    assert history[0]["status"] == ("settled" if state == "completed" else "failed")
    assert client._terminal_receipts == {("task", 3)}


@pytest.mark.parametrize("mismatch", [None, "run_id", "execution_generation"])
def test_control_result_cannot_turn_unknown_into_terminal(monkeypatch, mismatch):
    client, record, _, request = _peer(monkeypatch, lambda: {})
    request.side_effect = None
    result = dict(run_id="run-fixture", status="interrupted", execution_state="unknown", settled=False,
                  admission_id="admission", execution_generation=17, pending_controls=[])
    if mismatch:
        record.update(admission_id="admission", target_execution_generation=17)
        result[mismatch] = "stale" if mismatch == "run_id" else 18
    # Bind only this real consumer; the POST boundary is a mock, not a control action.
    monkeypatch.setattr(client, "_post_run_action", Mock(return_value=result))
    if mismatch:
        with pytest.raises(PeerRunsHTTPError):
            client.stop_receipt(task_id="task", execution_generation=3, grant="fixture-grant")
    else:
        observed = client.stop_receipt(task_id="task", execution_generation=3, grant="fixture-grant")
        assert observed["status"] == "unknown" and observed["settled"] is False
        assert observed["target_execution_generation"] == 17
    assert not client._terminal_receipts
