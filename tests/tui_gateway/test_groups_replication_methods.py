"""Tests for the fail-closed passive-replica JSON-RPC surface."""

from __future__ import annotations

import pytest

import tui_gateway.server as srv
from tui_gateway import methods_groups


@pytest.fixture
def home(tmp_path, monkeypatch):
    path = tmp_path / ".hermes"
    path.mkdir()
    (path / "profiles" / "ops").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(path))
    methods_groups.stop_hosted_room_service(timeout=1.0)
    methods_groups.start_hosted_room_service()
    yield path
    methods_groups.stop_hosted_room_service(timeout=1.0)


def _result(envelope):
    assert "error" not in envelope, envelope
    return envelope["result"]


def _error(envelope):
    assert "error" in envelope, envelope
    return envelope["error"]


def test_capabilities_do_not_advertise_unverified_replication(home):
    result = _result(srv._methods["groups.capabilities"](1, {}))
    assert "log_replication" not in result["features"]
    assert "authority_takeover" not in result["features"]
    assert "groups.replicate" not in result["methods"]
    assert "groups.replicate" not in srv._LONG_HANDLERS
    assert "groups.replica_state" in result["methods"]
    assert "groups.replica_state" in srv._LONG_HANDLERS
    blocked = srv._methods["groups.replicate"](
        2,
        {"room_id": "forged", "page": {"authority": {"epoch": 1}}},
    )
    assert _error(blocked)["data"]["reason"] == "replica_provenance_required"
    for name in ("groups.promote", "groups.demote"):
        assert name not in result["methods"]
        assert name not in srv._LONG_HANDLERS
        envelope = srv._methods[name](3, {"room_id": "room-1", "confirm": True})
        assert _error(envelope)["data"]["reason"] == "authority_takeover_disabled"
