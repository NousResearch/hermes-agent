"""Inert RoomLink routing/refusal boundaries; all consequential dependencies mocked."""
import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway import hosted_room_peer as peer
from gateway import hosted_room_execution_policy as policy
from gateway import session_api_turn
from gateway.platforms import api_server_room_grants as grants
from gateway.platforms import api_server_room_dispatch as dispatches
from gateway.platforms import api_server_room_attachments as attachments
from gateway.platforms import api_server_runs as runs
from hermes_state_runtime import RuntimeStoreError


def _error(message, **kwargs):
    return {"error": {"message": message, **kwargs}}


@pytest.fixture
def boundary(monkeypatch):
    current = ContextVar("fixture-profile", default="default")
    visited = []
    mode = ["canonical"]
    authority = SimpleNamespace(db=Mock(), profile_id="fixture", epoch=1, _require_admission_open=Mock())

    @contextmanager
    def scope(profile):
        visited.append(profile)
        token = current.set(profile)
        try:
            yield
        finally:
            current.reset(token)

    def active():
        # A global-only lookup sees no owner. Only the named target is canonical.
        return authority if current.get() == "named" and mode[0] == "canonical" else None

    runner = SimpleNamespace(session_authority=None, session_authorities=SimpleNamespace(active=active))
    adapter = SimpleNamespace(gateway_runner=runner, _profile_scope=scope,
        _room_grant_token=Mock(return_value="fixture-grant"), _room_grant_secret=Mock(return_value=b"fixture"),
        _room_grant_claims=Mock(return_value={"target_profile": "named", "target_install_id": "target", "grant_id": "grant", "permissions": []}),
        _ensure_session_db_async=AsyncMock(return_value=authority.db), _ensure_session_db=Mock(return_value=authority.db),
        _ensure_hosted_member_session=AsyncMock(return_value="hidden-session"),
        _parse_session_key_header=Mock(return_value=(None, None)),
        _run_idempotency_store=Mock(), _activate_admitted_request=Mock())
    value = dict(version=1, target_profile="named", enabled_toolsets=["bot_room"], approval_mode="manual", max_iterations=2)
    value["policy_digest"] = policy._policy_digest(value)
    monkeypatch.setattr(policy, "execution_policy_mapping", Mock(return_value=value))
    monkeypatch.setattr(peer, "local_room_link_endpoint", Mock(return_value={"available": False, "reason": "not_configured"}))
    monkeypatch.setattr(attachments, "roomlink_attachments_available", Mock(return_value=True))
    from gateway import hosted_rooms
    monkeypatch.setattr(hosted_rooms, "local_authority_gateway_id", Mock(return_value="target"))
    monkeypatch.setattr(grants, "_effective_room_profile", Mock(return_value="named"))
    # This leaf was imported by value in the staging module.
    monkeypatch.setattr(attachments, "_effective_room_profile", Mock(return_value="named"))
    _, catalog = grants._local_room_catalog(adapter, "named", "target")
    dispatch = peer.HostedMemberDispatch(
        protocol_version=peer.PROTOCOL_VERSION, room_id="room", home_install_id="home", authority_gateway_id="home",
        authority_epoch=1, member_id="member", target_install_id="target", target_profile="named",
        task_id="task", execution_generation=3, source_event_seq=1, cancellation_scope_id="cancel",
        prompt="fixture prompt", prompt_digest=hashlib.sha256(b"fixture prompt").hexdigest(),
        capability_digest=catalog["catalog_digest"], execution_policy_digest=value["policy_digest"], trace_id="trace")
    monkeypatch.setattr(peer, "verify_room_grant", Mock(return_value={"permissions": [], "grant_id": "grant"}))
    monkeypatch.setattr(attachments, "verify_room_grant", peer.verify_room_grant)
    spool = Mock()
    spool.prepare.return_value = spool.put.return_value = {"idempotent": False}
    monkeypatch.setattr(attachments, "_default_spool", Mock(return_value=spool))
    validator = AsyncMock(side_effect=lambda body, **kw: (body, None))
    monkeypatch.setattr(attachments, "_validate_dispatch_attachments", validator)
    monkeypatch.setattr(session_api_turn, "bind_api_session", Mock(return_value=SimpleNamespace(session_id="hidden-session")))
    monkeypatch.setattr(session_api_turn, "check_api_turn", Mock())
    monkeypatch.setattr(session_api_turn, "admit_session_input", Mock(return_value={"admission_id": "fixture"}))
    from hermes_state_terminal import retry_terminal_admission
    assert callable(retry_terminal_admission)  # import only; replace before the bound method can reach it
    monkeypatch.setattr("hermes_state_terminal.retry_terminal_admission", Mock(return_value=None))
    api = SimpleNamespace(_openai_error=_error, _api_request_profile=ContextVar("request-profile", default="named"))
    return SimpleNamespace(adapter=adapter, authority=authority, dispatch=dispatch, api=api, mode=mode,
                           visited=visited, current=current, spool=spool, validator=validator)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["canonical", "unavailable"])
@pytest.mark.parametrize("with_file", [False, True])
@pytest.mark.parametrize("entry", ["runs", "normalize", "session", "manifest", "upload", "admission"])
async def test_canonical_peer_refused_before_any_modeled_side_effect(boundary, mode, with_file, entry):
    b = boundary
    b.mode[0] = mode
    b.visited.clear()
    manifest = [{"attachment_id": "file", "kind": "file", "name": "fixture.txt", "size": 7,
                 "mime": "text/plain", "sha256": hashlib.sha256(b"fixture").hexdigest()}]
    dispatch = b.dispatch.as_mapping()
    if with_file:
        dispatch["attachment_manifest_digest"] = peer.attachment_manifest_digest(manifest)
    body = {"input": b.dispatch.prompt, "hosted_room_dispatch": dispatch}
    request = SimpleNamespace(json=AsyncMock(return_value=body), headers={"Idempotency-Key": "room:task:3"},
                              match_info={"task_id": "task", "execution_generation": "3", "attachment_id": "file"})
    b.adapter._read_json_body = AsyncMock(return_value=({"hosted_room_dispatch": dispatch, "attachments": manifest}, None))
    async def chunks(_size):
        yield b"fixture"
    request.content = SimpleNamespace(iter_chunked=Mock(side_effect=chunks))
    # If the outer gate regresses, the nested normalizer is still inert and visibly called.
    b.adapter._normalize_room_dispatch = AsyncMock(return_value=(body, runs.web.json_response({"fixture": "late refusal"}, status=400)))
    if entry in {"session", "admission"}:
        with pytest.raises(RuntimeStoreError) as exc:
            if entry == "session":
                await dispatches._ensure_hosted_member_session(b.adapter, b.dispatch)
            else:
                # Resolve the lower canonical ingress in its own target scope.
                with b.adapter._profile_scope("named"):
                    session_api_turn.admit_api_turn(b.adapter, user_message=b.dispatch.prompt,
                        conversation_history=[], session_id="hidden-session", active_run_id="run",
                        room_dispatch=dispatch)
        assert exc.value.reason == "canonical_room_peer_unsupported"
    else:
        if entry == "runs":
            response = await runs._handle_runs(b.adapter, request, _api_server=b.api)
        elif entry == "normalize":
            _, response = await dispatches._normalize_room_dispatch(b.adapter, request, body, _api_server=b.api)
        else:
            response = await getattr(attachments, "_handle_room_attachment_" + entry)(b.adapter, request,
                _openai_error=_error, _api_request_profile=b.api._api_request_profile)
        assert response is not None and response.status == 409
        assert json.loads(response.text)["error"]["code"] == "canonical_room_peer_unsupported"
    assert "named" in b.visited
    assert b.current.get() == "default"
    b.adapter._ensure_session_db_async.assert_not_called()
    b.adapter._ensure_session_db.assert_not_called()
    b.adapter._ensure_hosted_member_session.assert_not_called()
    b.adapter._normalize_room_dispatch.assert_not_called()
    b.adapter._run_idempotency_store.reserve.assert_not_called()
    b.adapter._activate_admitted_request.assert_not_called()
    b.spool.prepare.assert_not_called()
    b.spool.put.assert_not_called()
    request.content.iter_chunked.assert_not_called()
    b.validator.assert_not_called()
    session_api_turn.bind_api_session.assert_not_called()
    session_api_turn.admit_session_input.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["canonical", "unavailable", "legacy"])
async def test_catalog_is_honest_and_legacy_normalization_survives(boundary, mode):
    b = boundary
    b.mode[0] = mode
    if mode == "legacy":
        b.adapter.gateway_runner.session_authorities = None
    _, value = grants._local_room_catalog(b.adapter, "named", "target")
    catalog = peer.GatewayRoomCatalog.from_mapping(value)  # exact existing schema and digest
    assert catalog.text is (mode == "legacy")
    assert catalog.attachments is (mode == "legacy")
    assert catalog.protocol_versions == (peer.PROTOCOL_VERSION,)
    if mode == "legacy":
        body = {"input": b.dispatch.prompt, "hosted_room_dispatch": {
            **b.dispatch.as_mapping(), "capability_digest": catalog.catalog_digest}}
        request = SimpleNamespace(headers={"Idempotency-Key": "room:task:3"})
        normalized, error = await dispatches._normalize_room_dispatch(b.adapter, request, body, _api_server=b.api)
        assert error is None and normalized["session_id"] == "hidden-session"
        b.adapter._ensure_hosted_member_session.assert_awaited_once()
        b.validator.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("permission", ["stop", "approve", "status"])
@pytest.mark.parametrize("room_scoped", [False, True])
async def test_canonical_peer_control_refused_before_control_consumer(boundary, permission, room_scoped):
    b = boundary
    b.adapter._room_grant_token.return_value = "fixture-grant" if room_scoped else ""
    request = SimpleNamespace(match_info={"run_id": "run"}, method="POST")
    b.adapter._check_run_auth = Mock(return_value=None)
    b.adapter._request_owns_run = Mock(return_value=True)
    b.adapter._active_run_agents = {}
    b.adapter._active_run_tasks = {}
    b.adapter._durable_run_status = Mock(return_value={"status": "running"})
    _, _, _, _, error = runs._load_owned_run(b.adapter, request, _api_server=b.api,
                                           permission=permission, active_fallback=False)
    if not room_scoped or permission == "status":
        assert error is None
        b.adapter._durable_run_status.assert_called_once_with(request, "run")
        return
    assert error is not None and error.status == 409
    assert json.loads(error.text)["error"]["code"] == "canonical_room_peer_unsupported"
    b.adapter._durable_run_status.assert_not_called()
