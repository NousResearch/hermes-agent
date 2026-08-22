from __future__ import annotations

import contextlib
import hashlib
import json
import threading
import types

import pytest

from tui_gateway import server


class FakeDB:
    def __init__(self, root="root-1"):
        self.root = root

    def get_conversation_root(self, _session_id):
        return self.root


class CapturingThread:
    starts = []

    def __init__(self, *, target, daemon):
        self.target = target
        self.daemon = daemon

    def start(self):
        self.starts.append(self.target)

    def is_alive(self):
        return True


def session(tmp_path, *, running=False):
    return {
        "agent": types.SimpleNamespace(),
        "agent_ready": None,
        "agent_error": None,
        "attached_images": [],
        "cwd": str(tmp_path),
        "history": [],
        "history_lock": threading.RLock(),
        "history_version": 0,
        "inflight_turn": None,
        "last_active": 0,
        "profile_home": str(tmp_path),
        "running": running,
        "session_key": "stored-1",
        "transport": None,
    }


@pytest.fixture
def exact_env(tmp_path, monkeypatch):
    sid = "runtime-1"
    record = session(tmp_path)
    server._sessions[sid] = record
    CapturingThread.starts = []
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_args: None)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda *_args: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda *_args: None)
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {})
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_args: False)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server.threading, "Thread", CapturingThread)

    @contextlib.contextmanager
    def fake_db(_session):
        yield FakeDB()

    monkeypatch.setattr(server, "_session_db", fake_db)
    yield sid, record, tmp_path
    server._sessions.pop(sid, None)


def exact_params(sid="runtime-1", **overrides):
    source_text = overrides.pop("source_text", "  Unicode Ω\nsecond line  ")
    attachments = overrides.pop("attachments", [])
    context_text = overrides.pop("context_text", source_text)
    source_digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    context_digest = hashlib.sha256(context_text.encode("utf-8")).hexdigest()
    attachment_bytes = json.dumps(attachments, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    attachment_manifest_digest = hashlib.sha256(attachment_bytes).hexdigest()
    material = json.dumps(
        {"text": source_text, "contextText": context_text, "attachments": attachments},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    exact = {
        "submission_id": "submit_exact_00000001",
        "connection_id": "connection-a",
        "profile": "worker",
        "stored_session_id": "stored-1",
        "lineage_root_id": "root-1",
        "payload_digest": hashlib.sha256(material).hexdigest(),
        "source_digest": source_digest,
        "context_digest": context_digest,
        "attachment_manifest_digest": attachment_manifest_digest,
        "attachment_count": len(attachments),
        "source_text": source_text,
        "context_text": context_text,
        "attachments": attachments,
    }
    exact.update(overrides.pop("exact", {}))
    return {"session_id": sid, "text": context_text, "exact_submission": exact, **overrides}


def call(method, params):
    return server._methods[method]("request-1", params)


def receipt_params(**overrides):
    request = exact_params()
    exact = request["exact_submission"]
    params = {
        "session_id": request["session_id"],
        "submission_id": exact["submission_id"],
        "connection_id": exact["connection_id"],
        "profile": exact["profile"],
        "stored_session_id": exact["stored_session_id"],
        "lineage_root_id": exact["lineage_root_id"],
        "payload_digest": exact["payload_digest"],
        "source_digest": exact["source_digest"],
        "context_digest": exact["context_digest"],
        "attachment_manifest_digest": exact["attachment_manifest_digest"],
        "attachment_count": exact["attachment_count"],
    }
    params.update(overrides)
    return params


def test_exact_submit_atomically_admits_receipt_and_turn_before_dispatch(exact_env):
    _sid, record, home = exact_env
    from tui_gateway.exact_admission import get_exact_receipt, read_exact_turn_marker

    response = call("prompt.submit", exact_params())

    assert "error" not in response, response
    receipt = response["result"]["receipt"]
    assert receipt["state"] == "durably_accepted"
    assert receipt["connection_id"] == "connection-a"
    assert receipt["profile"] == "worker"
    assert response["result"]["idempotent_replay"] is False
    assert get_exact_receipt(home, "submit_exact_00000001") == receipt
    marker = read_exact_turn_marker(home, "stored-1")
    assert marker["prompt"] == "  Unicode Ω\nsecond line  "
    assert marker["persist_user_text"] == "  Unicode Ω\nsecond line  "
    assert CapturingThread.starts, "dispatch starts only after durable admission"
    assert record["running"] is True


def test_exact_submit_receipt_binds_source_context_and_complete_attachment_manifest(exact_env):
    attachment = {
        "id": "file-1",
        "occurrenceId": "occ-1",
        "sourceId": "file-1",
        "kind": "file",
        "name": "notes.txt",
        "mediaType": "text/plain",
        "refText": "@file:attachments/notes.txt",
        "runtimeSessionId": "runtime-1",
        "sha256": "a" * 64,
        "size": 5,
        "storedName": "notes.txt",
        "order": 0,
    }
    source = "  exact Ω source  "
    context = f"{attachment['refText']}\n\n{source}"

    response = call("prompt.submit", exact_params(source_text=source, context_text=context, attachments=[attachment]))

    assert "error" not in response, response
    receipt = response["result"]["receipt"]
    assert receipt["attachment_count"] == 1
    assert receipt["source_digest"] == hashlib.sha256(source.encode()).hexdigest()
    assert receipt["context_digest"] == hashlib.sha256(context.encode()).hexdigest()
    assert receipt["attachment_manifest_digest"] == hashlib.sha256(
        json.dumps([attachment], ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def test_exact_submit_preserves_source_and_atomic_admission_through_compute_host(exact_env, monkeypatch):
    _sid, _record, _home = exact_env
    captured = {}
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_args: True)

    def submit_compute(rid, sid, session, text, **kwargs):
        captured.update({"rid": rid, "sid": sid, "text": text, **kwargs})
        return server._ok(rid, {"status": "streaming", "turn_isolation": True})

    monkeypatch.setattr(server, "_submit_prompt_to_compute_host", submit_compute)
    source = "  exact compute source  "
    context = "@file:attachments/notes.txt\n\n" + source

    response = call("prompt.submit", exact_params(source_text=source, context_text=context))

    assert "error" not in response, response
    assert captured["text"] == context
    assert captured["persist_user_text"] == source
    assert captured["exact_admitted"] is True
    assert captured["exact_submission_id"] == "submit_exact_00000001"
    assert response["result"]["receipt"]["state"] == "durably_accepted"


def test_exact_submit_replay_is_idempotent_even_while_session_is_busy(exact_env):
    _sid, record, _home = exact_env
    first = call("prompt.submit", exact_params())
    starts = len(CapturingThread.starts)
    assert record["running"] is True

    replay = call("prompt.submit", exact_params())

    assert replay["result"]["receipt"] == first["result"]["receipt"]
    assert replay["result"]["idempotent_replay"] is True
    assert len(CapturingThread.starts) == starts
    assert record.get("queued_prompt") is None


def test_exact_submit_conflict_rejects_without_dispatch_or_overwrite(exact_env):
    _sid, record, _home = exact_env
    call("prompt.submit", exact_params())
    starts = len(CapturingThread.starts)

    conflict = call("prompt.submit", exact_params(source_text="changed", context_text="changed"))

    assert conflict["error"]["code"] == 4091
    assert len(CapturingThread.starts) == starts
    assert record.get("queued_prompt") is None


def test_exact_submit_busy_rejects_before_native_redirect_queue_or_receipt(exact_env, monkeypatch):
    _sid, record, home = exact_env
    from tui_gateway.exact_admission import get_exact_receipt

    record["running"] = True
    busy_handler = pytest.fail
    monkeypatch.setattr(server, "_handle_busy_submit", busy_handler)

    response = call("prompt.submit", exact_params())

    assert response["error"]["code"] == 4009
    assert get_exact_receipt(home, "submit_exact_00000001")["state"] == "rejected"
    assert record.get("queued_prompt") is None
    assert record.get("inflight_turn") is None
    assert not CapturingThread.starts


@pytest.mark.parametrize(
    "exact",
    [
        {"stored_session_id": "wrong"},
        {"lineage_root_id": "wrong"},
        {"payload_digest": "0" * 64},
        {"source_digest": "1" * 64},
        {"context_digest": "2" * 64},
        {"attachment_manifest_digest": "3" * 64},
        {"attachment_count": 3},
        {"submission_id": "../escape"},
        {"unexpected": True},
    ],
)
def test_exact_submit_rejects_stale_malformed_or_incompletely_bound_requests(exact_env, exact):
    _sid, record, _home = exact_env

    response = call("prompt.submit", exact_params(exact=exact))

    assert response["error"]["code"] in {4004, 4017, 4018}
    assert record["running"] is False
    assert record.get("inflight_turn") is None
    assert not CapturingThread.starts


def test_receipt_query_is_profile_scoped_closed_and_non_secret(exact_env):
    call("prompt.submit", exact_params())

    response = call("prompt.receipt", receipt_params())

    assert response["result"]["receipt"]["state"] == "durably_accepted"
    assert "path" not in str(response["result"]).lower()
    assert "token" not in str(response["result"]).lower()


@pytest.mark.parametrize(
    "field,value",
    [
        ("connection_id", "connection-b"),
        ("profile", "other-worker"),
        ("session_id", "runtime-2"),
        ("stored_session_id", "stored-2"),
        ("lineage_root_id", "root-2"),
        ("payload_digest", "0" * 64),
        ("source_digest", "1" * 64),
        ("context_digest", "2" * 64),
        ("attachment_manifest_digest", "3" * 64),
        ("attachment_count", 1),
    ],
)
def test_receipt_query_rejects_every_target_and_payload_binding_mismatch(exact_env, field, value):
    call("prompt.submit", exact_params())

    response = call("prompt.receipt", receipt_params(**{field: value}))

    assert response["error"]["code"] in {4001, 4018, 4091}


def test_receipt_query_rejects_partial_lookup_identity(exact_env):
    call("prompt.submit", exact_params())

    response = call("prompt.receipt", {"session_id": "runtime-1", "submission_id": "submit_exact_00000001"})

    assert response["error"]["code"] == 4004


@pytest.mark.parametrize("shape", ["subclass", "partial", "extra", "mistyped"])
def test_receipt_query_rejects_non_closed_or_mistyped_stored_receipts(exact_env, monkeypatch, shape):
    call("prompt.submit", exact_params())
    from tui_gateway import exact_admission

    valid = exact_admission.get_exact_receipt(
        exact_env[2], "submit_exact_00000001"
    )
    assert valid is not None

    if shape == "subclass":
        class HostileReceipt(dict):
            calls = 0

            def get(self, key, default=None):
                type(self).calls += 1
                return super().get(key, default)

        hostile = HostileReceipt(valid)
    elif shape == "partial":
        hostile = {key: value for key, value in valid.items() if key != "source_digest"}
    elif shape == "extra":
        hostile = {**valid, "extra": True}
    else:
        hostile = {**valid, "version": "1"}

    monkeypatch.setattr(exact_admission, "get_exact_receipt", lambda *_args: hostile)

    response = call("prompt.receipt", receipt_params())

    assert response["error"]["code"] == 4004
    if shape == "subclass":
        assert HostileReceipt.calls == 0


def test_session_identity_is_closed_lineage_aware_busy_and_versioned(exact_env):
    _sid, record, _home = exact_env

    idle = call("session.identity", {"session_id": "runtime-1"})
    assert idle["result"] == {
        "runtime_session_id": "runtime-1",
        "stored_session_id": "stored-1",
        "lineage_root_id": "root-1",
        "busy": False,
        "capabilities": {"exact_submission": 1, "attachment_relay": 1},
    }
    record["running"] = True
    assert call("session.identity", {"session_id": "runtime-1"})["result"]["busy"] is True
