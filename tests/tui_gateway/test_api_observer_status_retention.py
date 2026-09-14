"""Observer completion cannot settle canonical work; all runtime edges are inert."""
import asyncio
import copy
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.platforms import api_server_runs as runs
from gateway.platforms import api_server_authority_runs as authority_runs
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore, run_status_is_terminal
from gateway import session_api_turn
from hermes_state_runtime import RuntimeStoreError


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["unknown", "observer_cancelled", "projection_unavailable", "completed"])
async def test_wrapper_persists_only_canonical_settlement(monkeypatch, outcome):
    events, persisted = [], []
    row = {"admission_id": "admission", "target_session_id": "session", "generation": 17, "status": "started"}
    projection = {"run_id": "run", "status": "running", "execution_state": "started", "settled": False,
                  "admission_id": "admission", "session_id": "session", "execution_generation": 17,
                  "pending_controls": [], "output": "", "usage": {}}
    store = Mock()
    store.update_status.side_effect = lambda _run_id, value: persisted.append(copy.deepcopy(value))
    adapter = SimpleNamespace(_run_statuses={}, _run_idempotency_ids={"run"}, _run_idempotency_store=store,
                              _run_streams={"run": object()}, _stopping_run_ids=set(),
                              _make_run_event_callback=Mock(return_value=Mock()), _profile_scope=lambda _: nullcontext())
    adapter._set_run_status = lambda run_id, status, **fields: runs._set_run_status(adapter, run_id, status, **fields)
    launch = SimpleNamespace(run_id="run", admission=(object(), SimpleNamespace(session_id="session"), row),
                             request_profile="named", put_event=events.append, approval_session_key="run")
    api = SimpleNamespace(_redact_api_error_text=str, _ProviderAuthResolutionError=type("AuthError", (Exception,), {}))
    reader = Mock(side_effect=lambda *_: copy.deepcopy(projection))
    monkeypatch.setattr(authority_runs, "run_projection", reader)
    monkeypatch.setattr(session_api_turn, "observe_api_controls", lambda *_: nullcontext())
    monkeypatch.setattr(runs, "_unregister_approval_notify", Mock())
    monkeypatch.setattr(runs, "_retire_live_run", Mock())

    async def observed(*_args, **_kwargs):
        if outcome == "completed":
            projection.update(status="completed", execution_state="terminal", settled=True, output="canonical result")
            return {"final_response": "canonical result"}, {}
        if outcome == "observer_cancelled":
            raise asyncio.CancelledError()
        if outcome == "projection_unavailable":
            reader.side_effect = RuntimeStoreError("storage_unavailable")
        else:
            projection.update(status="unknown", execution_state="unknown", settled=False)
        raise RuntimeStoreError("unknown_execution")

    monkeypatch.setattr(session_api_turn, "observe_api_turn", AsyncMock(side_effect=observed))
    if outcome == "observer_cancelled":
        with pytest.raises(asyncio.CancelledError):
            await runs._execute_run(adapter, launch, _api_server=api)
    else:
        await runs._execute_run(adapter, launch, _api_server=api)
    assert persisted
    final = adapter._run_statuses["run"]
    assert final["admission_id"] == "admission" and final["execution_generation"] == 17
    assert all(value.get("admission_id") == "admission" for value in persisted)
    assert persisted[-1] == final
    if outcome == "completed":
        assert final["status"] == "completed" and run_status_is_terminal(final)
        assert final["output"] == "canonical result"
    else:
        assert final["status"] == ("running" if outcome == "observer_cancelled" else "unknown")
        assert final["settled"] is False and not run_status_is_terminal(final)
        assert not any(event and event.get("event") in {"run.failed", "run.cancelled", "run.completed"} for event in events)
        conn = Mock()
        import json
        conn.execute.return_value.fetchall.return_value = [("scope", "key", json.dumps(final))]
        retention = SimpleNamespace(_conn=conn, ACKNOWLEDGED_RETENTION_SECONDS=1, RETENTION_SECONDS=1)
        RunIdempotencyStore._prune_stale_terminal_locked(retention, 100)
        assert not any(call.args[0].startswith("DELETE") for call in conn.execute.call_args_list)


def test_cold_get_reconciles_replay_status_even_before_memory_binding(monkeypatch):
    observed = {"run_id": "run", "status": "unknown", "execution_state": "unknown", "settled": False,
                "admission_id": "admission", "session_id": "session", "execution_generation": 17}
    monkeypatch.setattr(authority_runs, "run_projection", Mock(return_value=observed))
    store = Mock()
    adapter = SimpleNamespace(_run_statuses={}, _run_idempotency_ids=set(), _run_idempotency_store=store)
    adapter._set_run_status = lambda run_id, status, **fields: runs._set_run_status(adapter, run_id, status, **fields)
    assert runs._durable_run_status(adapter, object(), "run") == observed
    store.update_status.assert_called_once()
    assert store.update_status.call_args.args[1]["settled"] is False
