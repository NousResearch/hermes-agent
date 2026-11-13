"""Admission classification for RoomLink peer runs: what proves a run was never admitted."""
import io
import json
import urllib.error

import pytest

from tests.tui_gateway import test_hosted_room_peer_http as peer_http
from hermes_agent.tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError

peer_server = peer_http.peer_server


_LOST = PeerRunsHTTPError("peer response was lost", retryable=True, ambiguous=True)
_REFUSED = PeerRunsHTTPError("peer endpoint refused the replay", retryable=True, not_admitted=True)


@pytest.mark.parametrize("method,first,replay", [
    ("dispatch", _LOST, _REFUSED),  # a refused replay says nothing about the lost first request
    ("recover_dispatch", _LOST, {}),  # nor does a 2xx without a run id
    ("recover_dispatch", {}, {}),  # a first 2xx without a run id is itself uncertain
], ids=["lost-then-refused", "lost-then-malformed", "malformed-then-malformed"])
def test_an_uncertain_admission_stays_uncertain_through_a_failed_replay(tmp_path, method, first, replay):
    """The identical replay of an uncertain admission can fail without proving the first request
    was not admitted: the attempt stays uncertain, and recovery then backs off."""
    client = PeerRunsHTTPClient(base_url="https://peer.example.test", api_key="",
                                receipt_db_path=tmp_path / "state.db", clock=lambda: 0.0)
    requests = []

    def answer(path, **kwargs):
        requests.append(kwargs)
        result = first if len(requests) == 1 else replay
        if isinstance(result, Exception):
            raise result
        return result

    client._request = answer
    with pytest.raises(PeerRunsHTTPError) as caught:
        getattr(client, method)(dispatch=peer_http._dispatch(), grant="signed.room.grant")
    assert (caught.value.retryable, caught.value.ambiguous, caught.value.not_admitted) == (True, True, False)
    assert len(requests) == 2 and requests[0]["headers"] == requests[1]["headers"]
    assert requests[0]["body"] == requests[1]["body"]
    if method == "recover_dispatch":
        with pytest.raises(PeerRunsHTTPError, match="backing off"):
            client.recover_dispatch(dispatch=peer_http._dispatch(), grant="signed.room.grant")
        assert len(requests) == 2


@pytest.mark.parametrize("status,code,definite", [
    *((status, "fixture_refusal", True) for status in (400, 401, 404, 408, 422, 425, 429)),
    (409, "permission_denied", True), (409, "admission_conflict", True),  # proven refusals
    (409, "idempotency_key_conflict", False),  # a run already exists under this key
    (200, b"not-json", False), (200, b"[]", False),  # a 2xx this client cannot read
])
def test_only_a_definite_refusal_marks_an_admission_not_admitted(monkeypatch, status, code, definite):
    def respond(*args, **kwargs):
        if isinstance(code, bytes):
            return io.BytesIO(code)
        raise urllib.error.HTTPError("https://peer.example.test/v1/runs", status, "refused", {},
                                     io.BytesIO(json.dumps({"error": {"code": code}}).encode()))
    monkeypatch.setattr("hermes_agent.hermes_cli.urllib_security.open_credentialed_url", respond)
    client = PeerRunsHTTPClient(base_url="https://peer.example.test", api_key="")
    with pytest.raises(PeerRunsHTTPError) as caught:
        client._request("/v1/runs", method="POST", body={"input": "frozen"}, room_grant="signed.room.grant")
    assert (caught.value.not_admitted, caught.value.ambiguous) == (definite, not definite)
    if not isinstance(code, bytes):
        assert (caught.value.status_code, caught.value.error_code) == (status, code)


def test_first_idempotency_conflict_replays_only_the_same_frozen_request(tmp_path, peer_server, monkeypatch):
    requests = []

    def do_POST(self):
        data = self.rfile.read(int(self.headers["Content-Length"]))
        requests.append((self.path, data, self.headers["Idempotency-Key"], self.headers["Authorization"]))
        if len(requests) == 1:
            return self._json({"error": {"code": "idempotency_key_conflict"}}, 409)
        return self._json({"run_id": "retained-run", "status": "started", "replayed": True}, 202)

    monkeypatch.setattr(peer_http.FakePeer, "do_POST", do_POST)
    client = PeerRunsHTTPClient(base_url=peer_server, api_key="",
                                receipt_db_path=tmp_path / "state.db")
    result = client.dispatch(dispatch=peer_http._dispatch(), grant="signed.room.grant")
    assert result["run_id"] == "retained-run" and result["replayed"]
    assert len(requests) == 2 and requests[0] == requests[1]
    assert requests[0][2] == "room:task-1:1"
    assert requests[0][3] == "HermesRoom signed.room.grant"
    saved = client._receipt("task-1", 1)
    assert saved["run_id"] == "retained-run" and saved["execution_generation"] == 1
