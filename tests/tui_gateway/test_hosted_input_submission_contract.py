"""Inert producer/adapter contracts: no workers, model, sockets or control loop.

Only the attempt's submission segment runs. Lease, wait, settlement and retirement
edges are replaced before invocation; real adapters call fake admission handlers.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import threading
from contextlib import nullcontext
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from gateway import hosted_room_driver as state
from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from gateway.hosted_room_peer import attachment_manifest_digest
from gateway.session_contract import AdmissionReceipt, Principal
from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
from gateway.session_hosted_transport import HostedRoomOwnerRPC
from tui_gateway.hosted_room_driver import HostedRoomBinding, HostedRoomRuntime
from tui_gateway.hosted_room_peer_transport import PeerHostedRoomTransport, PeerMemberRoute
from tui_gateway.hosted_room_server_rpc import HostedRoomServerRPC, HostedRoomSessionError


BINDING = HostedRoomBinding("room", "gateway-a", 1)
TASK = state.TaskIdentity("room", "task", "thread", "turn")
PROMPT = "Review the original prompt.\nKeep this unchanged."


def _manifest(data=b"notes"):
    return [{"event_id": "event-1", "attachment_id": "att_" + "1" * 32,
             "kind": "file", "name": "notes.txt", "mime": "text/plain", "size": len(data)}]


def _attempt(monkeypatch, tmp_path, transport, *, legacy=False, manifests=None, loader=None,
             task_status="queued", prior_generation=1):
    """Invoke the real producer with every control/lifecycle edge inert."""
    runtime = HostedRoomRuntime(
        db_path=tmp_path / "unused.db", rooms=[], turn_lock=lambda _profile: nullcontext(),
        rpc=transport if legacy else object(), transport_resolver=lambda *_: transport,
        attachment_loader=loader)
    runtime._resolve_or_create = lambda *_: {"session_id": "session"}
    runtime._wait_for_terminal = Mock(return_value=None)
    runtime._on_terminal = Mock(side_effect=AssertionError("terminal processing is held"))
    runtime._settle_failure_if_current = Mock()
    runtime._mark_ambiguous = Mock()
    runtime._defer_unavailable_route = Mock(return_value=1)
    monkeypatch.setattr(state, "require_active_lease", Mock())
    monkeypatch.setattr(state, "requeue_not_admitted_task", Mock(return_value=None))
    monkeypatch.setattr(state, "settle_task", Mock(side_effect=AssertionError("settlement is held")))
    attempt = state.TaskAttempt(TASK, state.DriverLease("room", "gateway-a", 1, "process", 1, 999), 2, 0)
    payload = {"target_profile": "ops", "target_member_id": "member-ops", "prompt": PROMPT}
    if manifests is not None:
        payload["attachments"] = manifests
    runtime._execute_attempt(BINDING, {"identity": TASK, "status": task_status,
        "execution_generation": prior_generation, "payload": payload}, attempt)
    return runtime


def _legacy(monkeypatch, tmp_path, *, rejection=False):
    monkeypatch.setattr("gateway.hosted_rooms.local_authority_gateway_id", lambda: "home-install")
    calls = []
    images = ["previous-image"]
    target = tmp_path / "attachments"
    target.mkdir()
    upload = target / "notes.txt"

    from tui_gateway import methods_prompt
    monkeypatch.setattr(methods_prompt, "_err", lambda rid, code, message: {
        "id": rid, "error": {"code": code, "message": message}}, raising=False)
    def prompt(rid, params):
        calls.append(("submit", params))
        invalid = methods_prompt._hosted_submit_error(rid, {"source": "bot_room"},
            params["_hosted_task"], params["_hosted_terminal_callback"])
        if invalid is not None:
            return invalid
        return {"id": rid, **({"error": {"code": 4121, "message": "not admitted"}}
                             if rejection else {"result": {"status": "accepted"}})}

    def attach(rid, params):
        calls.append(("attach", params))
        upload.write_bytes(base64.b64decode(params["data_url"].split(",", 1)[1]))
        images.append("attempt-image")
        return {"id": rid, "result": {"attached": True, "uploaded": True,
            "path": str(upload), "ref_text": "@file:attachments/notes.txt"}}

    server = SimpleNamespace(_methods={"prompt.submit": prompt, "file.attach": attach},
        _sessions={"session": {"history_lock": threading.Lock(), "attached_images": images}},
        _sessions_lock=threading.Lock(), _session_home_dir=lambda *_: target)
    rpc = HostedRoomServerRPC(server)
    rpc.retire_idle = Mock()  # Never release a real session or worker.
    return rpc, calls, upload


def test_legacy_driver_derives_complete_bound_proof_without_member_keyword(monkeypatch, tmp_path):
    rpc, calls, _ = _legacy(monkeypatch, tmp_path)
    scopes = []
    original = rpc.submit
    def submit(**params):
        scopes.append(rpc._artifact_scopes[(TASK.task_id, 2)])
        return original(**params)
    rpc.submit = submit
    runtime = _attempt(monkeypatch, tmp_path, rpc, legacy=True)
    assert [kind for kind, _ in calls] == ["submit"], runtime._last_error
    proof = calls[0][1]
    assert proof["text"] == PROMPT
    assert proof["session_id"] == "session"
    # The actual prompt owner accepts exactly the legacy six-field proof. The
    # producer consumes the stronger internal scope, not output-era envelope fields.
    assert proof["_hosted_task"] == {**asdict(TASK), "execution_generation": 2, "member_id": "member-ops"}
    assert scopes == [(TASK, "session", dict(room_id="room", task_id="task", execution_generation=2,
        member_id="member-ops", target_profile="ops", home_install_id="home-install",
        target_install_id="home-install", authority_gateway_id="gateway-a", authority_epoch=1))]
    assert rpc._artifact_scopes == {}
    assert callable(proof["_hosted_terminal_callback"])
    state.requeue_not_admitted_task.assert_not_called()
    runtime._mark_ambiguous.assert_not_called()
    runtime._settle_failure_if_current.assert_not_called()


@pytest.mark.parametrize("wrong", ["member", "profile", "room"])
def test_legacy_wrong_assertion_or_scope_refuses_before_handler(monkeypatch, tmp_path, wrong):
    rpc, calls, _ = _legacy(monkeypatch, tmp_path)
    rpc.bind_artifact_scope(task=TASK, execution_generation=2, member_id="member-ops",
        profile="ops", authority_gateway_id="gateway-a", authority_epoch=1)
    with pytest.raises(HostedRoomSessionError) as error:
        rpc.submit(profile="other" if wrong == "profile" else "ops", session_id="session",
            prompt=PROMPT, source="bot_room", task=replace(TASK, room_id="other") if wrong == "room" else TASK,
            execution_generation=2, on_terminal=Mock(),
            member_id="other" if wrong == "member" else "member-ops")
    assert error.value.code == 4120
    assert error.value.not_admitted is True
    assert calls == []


@pytest.mark.parametrize("rejection", [False, True])
def test_legacy_stages_once_and_only_proven_nonadmission_removes_bytes(monkeypatch, tmp_path, rejection):
    rpc, calls, upload = _legacy(monkeypatch, tmp_path, rejection=rejection)
    manifests = _manifest()
    loader = Mock(return_value=[(manifests[0], b"notes")])
    commit = Mock(wraps=rpc.commit_attachment_staging)
    rollback = Mock(wraps=rpc.rollback_attachment_staging)
    rpc.commit_attachment_staging, rpc.rollback_attachment_staging = commit, rollback
    runtime = _attempt(monkeypatch, tmp_path, rpc, legacy=True, manifests=manifests, loader=loader)
    assert [kind for kind, _ in calls] == ["attach", "submit"], runtime._last_error
    loader.assert_called_once()
    assert calls[1][1]["text"] == PROMPT + (
        "\n\nAttached files staged in your session workspace:\nnotes.txt: @file:attachments/notes.txt")
    assert rpc._attachment_attempts == {}
    assert rpc._staged_attachments == {}
    if rejection:
        rollback.assert_called_once()
        commit.assert_not_called()
        assert not upload.exists()
        assert rpc.server._sessions["session"]["attached_images"] == ["previous-image"]
        state.requeue_not_admitted_task.assert_called_once()
        _attempt(monkeypatch, tmp_path, rpc, legacy=True)
        assert [kind for kind, _ in calls] == ["attach", "submit", "submit"]
        assert calls[-1][1]["text"] == PROMPT
        assert rpc.server._sessions["session"]["attached_images"] == ["previous-image"]
    else:
        commit.assert_called_once()
        rollback.assert_not_called()
        assert upload.read_bytes() == b"notes"
    runtime._mark_ambiguous.assert_not_called()


@pytest.mark.parametrize("kind", ["local", "cross-profile"])
def test_canonical_real_submit_binding_preserves_prompt_manifest_without_legacy_staging(monkeypatch, tmp_path, kind):
    captured = []
    manifests = _manifest()
    target_home, source_home = tmp_path / "target", tmp_path / "source"
    if kind == "local":
        principal = Principal("owner", str(target_home), frozenset({"session:submit"}), "private")
        rpc = HostedRoomAuthorityRPC(SimpleNamespace(profile_id=str(target_home)), None,
            room_id="room", member_id="member-ops", profile="ops", principal=principal,
            authorize=lambda *_: True)
        rpc.ref = replace(rpc.ref, session_id="session")

        from gateway.hosted_room_input_preparation import PreparedHostedInput
        handle = object()
        def prepare(owner, *, request_id, prompt, attachments):
            assert owner is rpc
            assert request_id.startswith("hosted:")
            captured.append({"prompt": prompt, "attachments": attachments})
            return PreparedHostedInput({"text": prompt}, handle)
        prepared = Mock(side_effect=prepare)
        monkeypatch.setattr("gateway.hosted_room_input_preparation.prepare_hosted_input", prepared)
        async def admission(owner, submission, *, _input_custody):
            assert owner is principal
            assert _input_custody is handle
            assert submission.payload == {"text": PROMPT}
            return AdmissionReceipt("admission", rpc.ref, 1, "queued", None, 1, None)
        rpc.authority.submit = admission
        rpc.authority.waiters = {}
        rpc._rows = lambda: []
        async def inline(function, *args, **kwargs):
            return function(*args, **kwargs)
        monkeypatch.setattr("gateway.session_hosted_rpc.asyncio.to_thread", inline)
        def call(operation, **params):
            async def dispatch():
                rpc.loop = asyncio.get_running_loop()
                return await rpc._dispatch_owned(operation, params)
            return asyncio.run(dispatch())
        rpc._call = call
    else:
        rpc = HostedRoomOwnerRPC(home=target_home, source_home=source_home,
            room_id="room", member_id="member-ops", profile="ops")
        rpc._monitor = SimpleNamespace(is_alive=lambda: True)  # No watch thread.

        def owner_request(home, operation, payload):
            assert home == target_home and operation == "hosted-producer"
            assert payload["source_home"] == str(source_home)
            assert payload["selector"] == {"room_id": "room", "member_id": "member-ops", "profile": "ops"}
            assert payload["operation"] == "submit"
            captured.append(payload["params"])
            return {"admission_id": "admission", "status": "queued"}

        monkeypatch.setattr("gateway.session_hosted_transport.owner_request", owner_request)
    loader = Mock(side_effect=AssertionError("canonical input must not use legacy loader"))
    staging = [Mock(side_effect=AssertionError("canonical input must not use session staging")) for _ in range(4)]
    for name, mock in zip(("begin_attachment_staging", "stage_attachment", "commit_attachment_staging", "rollback_attachment_staging"), staging):
        setattr(rpc, name, mock)
    runtime = _attempt(monkeypatch, tmp_path, rpc, manifests=manifests, loader=loader)
    assert len(captured) == 1, runtime._last_error
    assert captured[0]["prompt"] == PROMPT
    assert captured[0]["attachments"] is manifests
    assert "member_id" not in captured[0]
    loader.assert_not_called()
    for mock in staging:
        mock.assert_not_called()
    if kind == "local":
        prepared.assert_called_once()
    runtime._mark_ambiguous.assert_not_called()
    runtime._settle_failure_if_current.assert_not_called()


def _peer(tmp_path, monkeypatch):
    store = HostedRoomAttachmentStore(tmp_path / "source.db")
    manifests = []
    for event, name, data in (("event-1", "notes.txt", b"notes"), ("event-2", "followup.txt", b"next")):
        raw = store.put(room_id="room", upload_id=event, kind="file", name=name,
            mime="text/plain", data=data)
        item = {key: raw[key] for key in ("attachment_id", "kind", "name", "mime", "size")}
        store.commit_message(room_id="room", event_id=event, manifest=[item],
            recipient_member_ids=["member-ops"], viewer_access=False)
        manifests.append({**item, "event_id": event})
    read = Mock(wraps=store.read)
    monkeypatch.setattr(store, "read", read)
    client = SimpleNamespace(stage_attachments=Mock(return_value={"complete": True}),
        dispatch=Mock(return_value={"status": "accepted"}))
    route = PeerMemberRoute(home_install_id="home-install", member_id="member-ops",
        target_install_id="peer-install", target_profile="ops", capability_digest="a" * 64,
        execution_policy_digest="b" * 64, cancellation_scope_id="cancel", trace_id="trace", grant="test-grant",
        attachments=True)
    rpc = PeerHostedRoomTransport(binding=BINDING, route=route, client=client, task_id=TASK.task_id,
        execution_generation=2, source_event_seq=1, attachment_store=store)
    return rpc, client, read, manifests


def test_peer_driver_prepares_one_recipient_bound_batch(monkeypatch, tmp_path):
    rpc, client, read, manifests = _peer(tmp_path, monkeypatch)
    loader = Mock(side_effect=AssertionError("peer must resolve event-bound bytes itself"))
    runtime = _attempt(monkeypatch, tmp_path, rpc, manifests=manifests, loader=loader)
    assert client.dispatch.call_count == 1, runtime._last_error
    loader.assert_not_called()
    assert read.call_args_list == [call(room_id="room", event_id=item["event_id"],
        attachment_id=item["attachment_id"], recipient_member_id="member-ops") for item in manifests]
    client.stage_attachments.assert_called_once()
    staged = client.stage_attachments.call_args.kwargs
    dispatched = client.dispatch.call_args.kwargs
    assert staged["dispatch"] == dispatched["dispatch"]
    assert staged["grant"] == dispatched["grant"] == "test-grant"
    assert staged["attachments"] == [
        {key: value for key, value in item.items() if key != "event_id"} |
        {"sha256": hashlib.sha256(data).hexdigest(), "data": data}
        for item, data in zip(manifests, (b"notes", b"next"))]
    signed = [{key: value for key, value in item.items() if key != "data"} for item in staged["attachments"]]
    assert dispatched["dispatch"]["attachment_manifest_digest"] == attachment_manifest_digest(signed)
    assert dispatched["dispatch"]["prompt"] == PROMPT
    runtime._mark_ambiguous.assert_not_called()


@pytest.mark.parametrize("field,value", [("event_id", "other-event"), ("name", "other.txt"),
    ("mime", "application/octet-stream"), ("kind", "pdf"), ("size", 6), ("recipient", "other-member")])
def test_peer_rejects_event_metadata_or_recipient_mismatch_before_upload(monkeypatch, tmp_path, field, value):
    rpc, client, _, manifests = _peer(tmp_path, monkeypatch)
    if field == "recipient":
        rpc.route = replace(rpc.route, member_id=value)
    else:
        manifests[1][field] = value  # A valid first member must not trigger a partial upload.
    with pytest.raises(ValueError):
        rpc.submit(profile="ops", session_id="session", prompt=PROMPT, source="bot_room",
            task=TASK, execution_generation=2, on_terminal=Mock(), attachments=manifests)
    client.stage_attachments.assert_not_called()
    client.dispatch.assert_not_called()


def test_service_supplies_its_source_store_to_peer_transport(monkeypatch, tmp_path):
    from tui_gateway.hosted_room_service import HostedRoomService
    rpc, client, _, manifests = _peer(tmp_path, monkeypatch)
    service = object.__new__(HostedRoomService)  # No initialization/pruning/coordinator.
    service.attachments = rpc.attachment_store
    service.db_path = tmp_path / "source.db"
    service.peer_routes = {("room", "member-ops"): rpc.route}
    service.peer_clients = {("room", "member-ops"): client}
    service._hydrate_persisted_peer_route = lambda *_: None
    service._tracked_peer_client = lambda *_, **__: client
    service._refresh_peer_attachment_catalog = lambda *args: rpc.route
    service._recover_peer_admission = Mock()  # Held recovery is not exercised.
    monkeypatch.setattr("gateway.hosted_room_link_records.room_link_retirement_started", lambda *_args, **_kw: False)
    result = service._resolve_member_transport(BINDING, {"identity": TASK, "status": "queued",
        "execution_generation": 1, "payload": {"target_profile": "ops", "target_member_id": "member-ops",
        "source_event_seq": 1, "prompt": PROMPT, "attachments": manifests}})
    assert result.attachment_store is service.attachments


@pytest.mark.parametrize("coordinate", ["session_id", "thread_id", "turn_id"])
def test_legacy_submission_cannot_change_its_bound_task_or_session(monkeypatch, tmp_path, coordinate):
    rpc, calls, _ = _legacy(monkeypatch, tmp_path)
    submit = rpc.submit
    def changed(**params):
        if coordinate == "session_id":
            params[coordinate] = "foreign-session"
        else:
            params["task"] = replace(params["task"], **{coordinate: "foreign"})
        return submit(**params)
    rpc.submit = changed
    runtime = _attempt(monkeypatch, tmp_path, rpc, legacy=True)
    assert calls == []
    state.requeue_not_admitted_task.assert_called_once()
    runtime._mark_ambiguous.assert_not_called()


def test_peer_upload_failure_carries_phase_evidence_not_blanket_nonadmission(monkeypatch, tmp_path):
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
    rpc, client, _, manifests = _peer(tmp_path, monkeypatch)
    failure = PeerRunsHTTPError("test upload unavailable", retryable=True, ambiguous=True,
        status_code=503, error_code="test_upload_error")
    client.stage_attachments.side_effect = failure
    with pytest.raises(PeerRunsHTTPError) as result:
        rpc.submit(profile="ops", session_id="session", prompt=PROMPT, source="bot_room",
            task=TASK, execution_generation=2, on_terminal=Mock(), attachments=manifests)
    assert result.value.dispatch_not_attempted is True
    assert result.value.not_admitted is False
    assert result.value.retryable is True
    assert result.value.status_code == 503
    assert result.value.error_code == "test_upload_error"
    assert failure.ambiguous is True  # The transport does not rewrite prior evidence.
    client.dispatch.assert_not_called()


@pytest.mark.parametrize("submitted,not_admitted,retained", [
    (False, False, False), (True, True, False), (True, False, True)])
def test_legacy_staging_disposition_preserves_ambiguous_bytes(monkeypatch, tmp_path, submitted, not_admitted, retained):
    rpc, _, upload = _legacy(monkeypatch, tmp_path)
    rpc.begin_attachment_staging(profile="ops", session_id="session", source="bot_room", execution_generation=2)
    rpc.stage_attachment(profile="ops", session_id="session", source="bot_room",
        execution_generation=2, attachment=_manifest()[0], data=b"notes")
    runtime = HostedRoomRuntime(db_path=tmp_path / "unused.db", rooms=[],
        turn_lock=lambda _: nullcontext(), rpc=rpc)
    runtime._finish_attachment_staging_after_error(transport=rpc, profile="ops", session_id="session",
        execution_generation=2, submit_attempted=submitted, not_admitted=not_admitted)
    assert upload.exists() is retained
    assert rpc._attachment_attempts == {} and rpc._staged_attachments == {}
    if not retained:
        assert rpc.server._sessions["session"]["attached_images"] == ["previous-image"]


@pytest.mark.parametrize("owner", ["gateway-a", "other-gateway"])
def test_restored_authority_helper_retains_exact_ownership_check(monkeypatch, owner):
    from gateway import hosted_rooms
    from tui_gateway.hosted_room_service import HostedRoomService
    service = object.__new__(HostedRoomService)
    service._room = lambda _: {"authority_gateway_id": owner, "authority_epoch": 4}
    monkeypatch.setattr(hosted_rooms, "local_authority_gateway_id", lambda: "gateway-a")
    if owner == "gateway-a":
        assert service._owned_authority("room") == (owner, 4)
    else:
        with pytest.raises(hosted_rooms.AuthorityConflictError):
            service._owned_authority("room")


@pytest.mark.parametrize("changed", [None, "event_id", "name", "size", "mime", "kind"])
def test_legacy_loader_uses_the_exact_event_bound_metadata(monkeypatch, tmp_path, changed):
    from gateway import hosted_rooms
    from tui_gateway.hosted_room_service import HostedRoomService
    rpc, _, _, manifests = _peer(tmp_path, monkeypatch)
    hosted_rooms.create_room(tmp_path / "source.db", room_id="room", name="Room",
        authority_gateway_id="gateway-a", members=[dict(member_id="member-ops", profile="ops", handle="ops")])
    service = object.__new__(HostedRoomService)
    service.db_path, service.attachments = tmp_path / "source.db", rpc.attachment_store
    if changed:
        manifests[1][changed] = 999 if changed == "size" else "wrong"
    task = {"payload": {"target_profile": "ops", "target_member_id": "member-ops", "attachments": manifests}}
    if changed:
        with pytest.raises((ValueError, RuntimeError)):
            list(service._load_task_attachments(BINDING, task))
    else:
        expected = [{key: value for key, value in item.items() if key != "event_id"} for item in manifests]
        assert list(service._load_task_attachments(BINDING, task)) == list(zip(expected, (b"notes", b"next")))


@pytest.mark.parametrize("status", ["stopping", "running", "queued"])
def test_service_receipt_only_and_fresh_generation_never_stage(monkeypatch, tmp_path, status):
    from tui_gateway.hosted_room_service import HostedRoomService
    rpc, client, _, manifests = _peer(tmp_path, monkeypatch)
    service = object.__new__(HostedRoomService)
    service.db_path = tmp_path / "source.db"
    service._load_task_attachments = Mock(side_effect=AssertionError("receipt-only/fresh work must not restage"))
    client.recover_dispatch = Mock()
    monkeypatch.setattr("gateway.hosted_room_link_records.room_link_retirement_started", lambda *a, **k: False)
    task = {"identity": TASK, "status": status, "execution_generation": 2,
            "payload": {"prompt": PROMPT, "source_event_seq": 1, "attachments": manifests}}
    service._recover_peer_admission(BINDING, task, rpc.route, client)
    client.stage_attachments.assert_not_called()
    service._load_task_attachments.assert_not_called()
    if status == "stopping":
        client.recover_dispatch.assert_called_once()
        assert client.recover_dispatch.call_args.kwargs["receipt_only"] is True
    else:
        client.recover_dispatch.assert_not_called()


@pytest.mark.parametrize('status,generation,fresh', [('queued', 1, True), ('queued', 2, False),
                                                     ('running', 1, False)])
def test_permanent_upload_failure_only_settles_fenced_fresh_generation(monkeypatch, tmp_path, status, generation, fresh):
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
    rpc, client, _, manifests = _peer(tmp_path, monkeypatch)
    client.stage_attachments.side_effect = PeerRunsHTTPError('upload too large', status_code=413, retryable=False)
    client.discard_attachments = Mock()
    runtime = _attempt(monkeypatch, tmp_path, rpc, manifests=manifests,
                       task_status=status, prior_generation=generation)
    client.dispatch.assert_not_called()
    client.discard_attachments.assert_not_called()
    runtime._on_terminal.assert_not_called()
    state.requeue_not_admitted_task.assert_not_called()
    runtime._defer_unavailable_route.assert_not_called()
    if fresh:
        runtime._settle_failure_if_current.assert_called_once()
        runtime._mark_ambiguous.assert_not_called()
    else:
        runtime._settle_failure_if_current.assert_not_called()
        runtime._mark_ambiguous.assert_called_once()
