"""Retained peer observations never require new-admission capability or inputs."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.hosted_room_driver import TaskIdentity
from tests.tui_gateway.test_hosted_room_peer_transport import BINDING, ROUTE
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
from tui_gateway.hosted_room_service import HostedRoomService


@pytest.mark.parametrize("receipt_present", [True, False])
def test_indeterminate_file_receipt_observation_does_not_stage_or_probe(monkeypatch, receipt_present):
    service = object.__new__(HostedRoomService)
    task = TaskIdentity(BINDING.room_id, "retained-task", "thread", "turn")
    service.db_path = "unused-state.db"
    service.peer_routes = {(BINDING.room_id, ROUTE.member_id): ROUTE}
    service.attachments = object()
    client = SimpleNamespace(bind_observation=Mock(), history=Mock(return_value=[{"status": "settled"}]),
                             stage_attachments=Mock(side_effect=AssertionError("read must not upload")))
    def recover(*, dispatch, grant, receipt_only=False):
        assert receipt_only is True
        assert dispatch["task_id"] == task.task_id and dispatch["execution_generation"] == 3
        if not receipt_present:
            raise PeerRunsHTTPError("accepted receipt unavailable", retryable=True, ambiguous=True)
        return {"run_id": "original-run", "status": "accepted"}
    client.recover_dispatch = Mock(side_effect=recover)
    service.peer_clients = {(BINDING.room_id, ROUTE.member_id): client}
    service._hydrate_persisted_peer_route = lambda *_: None
    service._tracked_peer_client = lambda *_args, **_kwargs: client
    service._refresh_peer_attachment_catalog = Mock(side_effect=AssertionError("observation must not probe admission capability"))
    service._load_task_attachments = Mock(side_effect=AssertionError("observation must not read/re-upload inputs"))
    monkeypatch.setattr("gateway.hosted_room_link_records.room_link_retirement_started", lambda *_args, **_kwargs: False)
    retained = {"identity": task, "status": "indeterminate", "execution_generation": 3,
                "payload": {"target_profile": "reviewer", "target_member_id": ROUTE.member_id,
                            "source_event_seq": 7, "prompt": "original", "attachments": [{"attachment_id": "retained"}]}}
    if receipt_present:
        transport = service._resolve_member_transport(BINDING, retained)
        assert transport.history(profile="reviewer", session_id="session", source="bot_room") == [{"status": "settled"}]
        client.history.assert_called_once()
    else:
        with pytest.raises(PeerRunsHTTPError) as error:
            service._resolve_member_transport(BINDING, retained)
        assert error.value.ambiguous is True and error.value.not_admitted is False
        client.history.assert_not_called()
    client.recover_dispatch.assert_called_once()
    service._refresh_peer_attachment_catalog.assert_not_called()
    service._load_task_attachments.assert_not_called()
    client.stage_attachments.assert_not_called()
