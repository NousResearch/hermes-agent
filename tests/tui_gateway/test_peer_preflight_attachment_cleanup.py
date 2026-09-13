"""Late/foreign attempt completion must not retire accepted peer inputs.

Only the real producer's submission segment runs, against a byte-retaining fake
peer and a temporary committed source store. All control/lifecycle edges are
mocked; no supervisor, recovery, wait loop or real peer is invoked.
"""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway import hosted_room_driver as state
from tests.tui_gateway.test_hosted_room_peer_transport import (
    BINDING,
    ROUTE,
    FailingPeerClient,
    _committed_attachments,
)
from tui_gateway.hosted_room_driver import HostedRoomRuntime, ROOM_SESSION_SOURCE
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
from tui_gateway.hosted_room_peer_transport import PeerHostedRoomTransport


class _RetainingPeerClient(FailingPeerClient):
    """Record fake admission custody so a discard would destroy the witness."""

    def __init__(self):
        super().__init__(method="stage_attachments")
        self.method = "none"
        self.uploads = {}
        self.accepted_inputs = {}

    def stage_attachments(self, **kwargs):
        result = super().stage_attachments(**kwargs)
        dispatch = kwargs["dispatch"]
        key = (dispatch["task_id"], dispatch["execution_generation"])
        self.uploads[key] = deepcopy(kwargs["attachments"])
        return result

    def dispatch(self, **kwargs):
        result = super().dispatch(**kwargs)
        dispatch = kwargs["dispatch"]
        key = (dispatch["task_id"], dispatch["execution_generation"])
        self.accepted_inputs[key] = deepcopy(self.uploads[key])
        return result

    def discard_attachments(self, **kwargs):
        key = (kwargs["task_id"], kwargs["execution_generation"])
        self.accepted_inputs.pop(key, None)
        self.uploads.pop(key, None)
        return super().discard_attachments(**kwargs)


@pytest.fixture
def accepted_peer(monkeypatch, tmp_path):
    client = _RetainingPeerClient()
    task = state.TaskIdentity("room-1", "task-files", "thread-1", "turn-files")
    store, manifests = _committed_attachments(tmp_path)
    transport = PeerHostedRoomTransport(
        binding=BINDING, route=replace(ROUTE, attachments=True), client=client,
        source_event_seq=7, task_id=task.task_id, execution_generation=4,
        attachment_store=store,
    )
    transport.create(
        profile="reviewer", title="Group: room-1", source=ROOM_SESSION_SOURCE,
    )
    terminal = Mock(side_effect=AssertionError("terminal processing is held"))
    assert transport.submit(
        profile="reviewer", session_id="group-session", source=ROOM_SESSION_SOURCE,
        prompt="Review the file", task=task, execution_generation=4,
        attachments=manifests, on_terminal=terminal,
    )["status"] == "accepted"
    terminal.assert_not_called()
    accepted = deepcopy(client.accepted_inputs)
    assert accepted[(task.task_id, 4)][0]["data"] == b"brief"
    client.method = "stage_attachments"

    # Same inert seams as the submission-contract helper, but with configurable
    # snapshots/identities to distinguish a fresh preflight from a late attempt.
    runtime = HostedRoomRuntime(
        db_path=tmp_path / "unused-runtime.db", rooms=[], rpc=object(),
        transport_resolver=lambda *_: transport, turn_lock=lambda _: nullcontext(),
        attachment_loader=Mock(side_effect=AssertionError("peer must resolve its manifest")),
    )
    runtime._resolve_or_create = Mock(return_value={"session_id": "group-session"})
    runtime._wait_for_terminal = Mock(return_value=None)
    runtime._on_terminal = terminal
    runtime._mark_ambiguous = Mock()
    runtime._settle_failure_if_current = Mock()
    runtime._defer_unavailable_route = Mock(return_value=1)
    runtime._finish_attachment_staging_after_error = Mock(
        side_effect=AssertionError("peer completion must not enter legacy finalization"),
    )
    monkeypatch.setattr(state, "require_active_lease", Mock())
    monkeypatch.setattr(state, "defer_not_admitted_task", Mock(return_value=None))
    for name in ("settle_task", "requeue_not_admitted_task", "release_lease"):
        monkeypatch.setattr(state, name, Mock(side_effect=AssertionError(f"{name} is held")))
    for name in ("start", "stop", "_worker_loop", "_drop_lease", "_inspect_abandoned_attempts"):
        # Patch existing lifecycle edges only; never install staging on the peer.
        monkeypatch.setattr(runtime, name, Mock(side_effect=AssertionError(f"{name} is held")))
    monkeypatch.setattr("threading.Thread.start", Mock(side_effect=AssertionError("workers are held")))

    failures = []
    submit = transport.submit

    def record_submit(**params):
        assert params["attachments"] is manifests
        try:
            return submit(**params)
        except Exception as exc:
            failures.append(exc)
            raise

    transport.submit = Mock(side_effect=record_submit)

    def run(*, identity=task, generation=3, profile="reviewer", fresh=False):
        attempt = state.TaskAttempt(
            identity,
            state.DriverLease("room-1", "gateway-home", 2, "process", 1, 999),
            generation, 0,
        )
        snapshot = {
            "identity": identity, "status": "queued" if fresh else "running",
            "execution_generation": generation - 1 if fresh else generation,
            "payload": {"target_profile": profile, "target_member_id": ROUTE.member_id,
                        "prompt": "Review the file", "attachments": manifests},
        }
        runtime._execute_attempt(BINDING, snapshot, attempt)
        return attempt

    def assert_retained():
        assert client.accepted_inputs == accepted
        assert not any(method == "discard_attachments" for method, _ in client.calls)
        runtime._finish_attachment_staging_after_error.assert_not_called()
        runtime.attachment_loader.assert_not_called()
        runtime._wait_for_terminal.assert_not_called()
        runtime._settle_failure_if_current.assert_not_called()
        terminal.assert_not_called()
        state.settle_task.assert_not_called()
        state.requeue_not_admitted_task.assert_not_called()
        state.release_lease.assert_not_called()
        assert not runtime.db_path.exists()
        # The canonical source is durable, not a pending-session scratch buffer.
        item = manifests[0]
        assert store.read(
            room_id=task.room_id, event_id=item["event_id"],
            attachment_id=item["attachment_id"], recipient_member_id=ROUTE.member_id,
        ).data == b"brief"

    return SimpleNamespace(
        client=client, transport=transport, task=task, runtime=runtime,
        run=run, assert_retained=assert_retained, failures=failures,
    )


@pytest.mark.parametrize("failure", [
    "preflight-fresh", "preflight-late", "dispatch-proven", "dispatch-ambiguous",
])
def test_driver_error_finalization_never_retires_an_accepted_successor(accepted_peer, failure):
    peer = accepted_peer
    if failure.startswith("dispatch"):
        peer.client.method = "dispatch"
        peer.client.error = PeerRunsHTTPError(
            "dispatch failed", retryable=True,
            not_admitted=failure == "dispatch-proven",
            ambiguous=failure == "dispatch-ambiguous",
        )
    attempt = peer.run(fresh=failure == "preflight-fresh")
    assert len(peer.failures) == 1
    error = peer.failures[0]
    assert isinstance(error, PeerRunsHTTPError)
    if failure.startswith("preflight"):
        assert error.dispatch_not_attempted is True
        assert error.not_admitted is False
        assert peer.client.error.ambiguous is True  # Original evidence is not rewritten.
    else:
        assert error is peer.client.error
    if failure in {"preflight-fresh", "dispatch-proven"}:
        state.defer_not_admitted_task.assert_called_once()
        assert state.defer_not_admitted_task.call_args.args[1] == attempt
        peer.runtime._mark_ambiguous.assert_not_called()
    else:
        state.defer_not_admitted_task.assert_not_called()
        peer.runtime._mark_ambiguous.assert_called_once_with(BINDING, attempt)
    methods = [method for method, _ in peer.client.calls]
    assert methods.count("stage_attachments") == 2
    assert methods.count("dispatch") == (2 if failure.startswith("dispatch") else 1)
    peer.assert_retained()


@pytest.mark.parametrize("wrong", ["generation", "profile"])
def test_wrong_generation_or_profile_cannot_retire_the_current_batch(accepted_peer, wrong):
    peer = accepted_peer
    attempt = peer.run(**({"generation": 5} if wrong == "generation" else {"profile": "another"}))
    assert len(peer.failures) == 1
    if wrong == "generation":
        assert peer.failures[0].dispatch_not_attempted is True
        assert peer.client.calls[-1][1]["dispatch"]["execution_generation"] == 5
    else:
        assert isinstance(peer.failures[0], ValueError)
        assert "profile does not match" in str(peer.failures[0])
        assert [method for method, _ in peer.client.calls] == ["prepare", "stage_attachments", "dispatch"]
    peer.runtime._mark_ambiguous.assert_called_once_with(BINDING, attempt)
    state.defer_not_admitted_task.assert_not_called()
    peer.assert_retained()


def test_another_task_cannot_retire_the_accepted_dispatch(accepted_peer):
    peer = accepted_peer
    other = replace(peer.task, task_id="another-task", turn_id="another-turn")
    attempt = peer.run(identity=other)
    assert len(peer.failures) == 1 and peer.failures[0].dispatch_not_attempted is True
    assert peer.client.calls[-1][1]["dispatch"]["task_id"] == other.task_id
    peer.runtime._mark_ambiguous.assert_called_once_with(BINDING, attempt)
    state.defer_not_admitted_task.assert_not_called()
    peer.assert_retained()


def test_duplicate_error_finalization_does_not_send_a_discard(accepted_peer):
    peer = accepted_peer
    for _ in range(2):
        peer.run()
        peer.assert_retained()
    assert len(peer.failures) == 2
    assert all(error.dispatch_not_attempted is True for error in peer.failures)
    assert peer.runtime._mark_ambiguous.call_count == 2
    state.defer_not_admitted_task.assert_not_called()
    methods = [method for method, _ in peer.client.calls]
    assert methods.count("stage_attachments") == 3
    assert methods.count("dispatch") == 1
