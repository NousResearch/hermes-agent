"""Legacy ``session.events.since`` replays a reconnect that includes ``replay_epoch``.

Desktop in URL, Cloud or SSH mode (no ``native_dial``) speaks the legacy
``/api/ws`` contract of a standalone ``hermes serve``. A session with no epoch
of its own used to send the process epoch from ``gateway.ready``. The legacy
params model refused that key, so the replay never ran and a pending approval
or clarify request was not delivered again. The process epoch is compared with
the answer's ``epoch``; it does not select events. Any other unknown key is
still refused.
"""

import pytest

from tui_gateway import event_replay


@pytest.fixture(autouse=True)
def _clean_replay_ring():
    event_replay.reset_replay_state()
    yield
    event_replay.reset_replay_state()


def _contract():
    # Importing the package registers every method contract, including session.events.since.
    from tui_gateway.contracts import registry as contracts

    return contracts, contracts.METHODS["session.events.since"]


def test_legacy_events_since_accepts_the_process_epoch_clients_send():
    contracts, contract = _contract()
    _, problem = contracts.validate_params(
        contract, {"session_id": "s1", "last_seen": 3, "replay_epoch": "proc-1"}
    )

    assert problem is None, problem


def test_legacy_events_since_still_refuses_other_unknown_keys():
    contracts, contract = _contract()
    _, problem = contracts.validate_params(
        contract, {"session_id": "s1", "last_seen": 3, "bogus": 1}
    )

    assert problem is not None and "bogus" in problem


@pytest.mark.parametrize(
    "params",
    [
        {"session_id": "s-legacy", "last_seen": 1, "replay_epoch": "proc-1"},
        {"session_id": "s-legacy", "last_seen": 1},
    ],
    ids=["with-replay-epoch", "without-replay-epoch"],
)
def test_legacy_events_since_replays_missed_events_and_open_requests(params):
    from tui_gateway import server, server_requests

    for n in (1, 2):
        event_replay._stamp_event(
            {
                "jsonrpc": "2.0",
                "method": "event",
                "params": {
                    "type": "message.delta",
                    "session_id": "s-legacy",
                    "payload": {"n": n},
                },
            }
        )
    req = server_requests.ServerRequest(
        "s-legacy", "clarify", {"question": "Ship it?", "choices": ["yes", "no"]}
    )
    server_requests._open[req.id] = req
    try:
        response = server.handle_request(
            {"id": "r1", "method": "session.events.since", "params": params}
        )
    finally:
        server_requests._open.pop(req.id, None)

    assert "error" not in response, response
    result = response["result"]
    assert [event["payload"]["n"] for event in result["events"]] == [2]
    assert [(entry["id"], entry["method"]) for entry in result["open_requests"]] == [
        (req.id, "clarify")
    ]
    assert result["epoch"] == event_replay.replay_epoch()
