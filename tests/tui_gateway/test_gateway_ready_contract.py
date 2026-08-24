"""The gateway.ready contract both transports must satisfy.

Three client-visible failures traced back to this one payload:

* "Background recovery unavailable - legacy transport" — the ``turn_recovery``
  capability block was missing, so the Android client fell back to the legacy
  REST transport.
* A dropped ``replay_epoch`` — reconnecting clients could not detect a backend
  restart and reset their per-session seq watermarks.
* Divergence between the stdio TUI and the WS sidecar, because each transport
  built its own payload independently.

These are behaviour contracts, not snapshots: they assert how the payload must
relate to the runtime capability, never that a particular literal is present.
"""

import tui_gateway.entry as entry
import tui_gateway.ws as ws


def _payloads():
    return {
        "stdio": entry.build_gateway_ready_payload({"name": "test"}),
        "ws": ws.build_gateway_ready_payload(
            {"name": "test"}, change_events=True, heartbeat=True
        ),
    }


class TestGatewayReadyPayload:
    def test_both_transports_share_one_implementation(self):
        # A second copy of the builder is how the WS sidecar silently drifted.
        stdio = entry.build_gateway_ready_payload({"name": "test"})
        via_ws = ws.build_gateway_ready_payload({"name": "test"})
        assert set(stdio) == set(via_ws)

    def test_every_transport_advertises_turn_recovery(self):
        for transport, payload in _payloads().items():
            capability = payload.get("capabilities", {}).get("turn_recovery")
            assert capability, (
                f"{transport} dropped turn_recovery; clients degrade to the "
                "legacy transport"
            )

    def test_protocol_is_announced_so_clients_do_not_guess(self):
        for transport, payload in _payloads().items():
            protocol = payload.get("protocol") or {}
            assert protocol.get("name") == "hermes-jsonrpc", transport
            assert protocol.get("major") == 2, transport

    def test_recovery_methods_the_client_calls_are_advertised(self):
        # The Android client refuses the v2 transport unless each of these is
        # named; a partial list is what produces "legacy transport".
        required = {"session.open", "turn.reconcile", "turn.interrupt"}
        for transport, payload in _payloads().items():
            methods = set(
                payload["capabilities"]["turn_recovery"].get("methods", [])
            )
            assert required <= methods, f"{transport} is missing {required - methods}"

    def test_advertised_capability_matches_the_runtime(self):
        from tui_gateway import server

        expected = server._turn_recovery_runtime.ready_payload({})
        for transport, payload in _payloads().items():
            assert payload.get("capabilities") == expected.get("capabilities"), (
                f"{transport} advertises a capability the runtime does not serve"
            )

    def test_every_transport_carries_replay_epoch(self):
        for transport, payload in _payloads().items():
            assert payload.get("replay_epoch"), (
                f"{transport} dropped replay_epoch; reconnecting clients cannot "
                "detect a backend restart"
            )

    def test_replay_epoch_is_stable_within_a_process(self):
        # Restart detection is meaningless if the value changes per call.
        first = entry.build_gateway_ready_payload({"name": "test"})
        second = ws.build_gateway_ready_payload({"name": "test"})
        assert first["replay_epoch"] == second["replay_epoch"]

    def test_heartbeat_is_opt_in_per_transport(self):
        # Only the WS sidecar actually sends heartbeats.
        assert "heartbeat" not in entry.build_gateway_ready_payload({"name": "t"})
        assert ws.build_gateway_ready_payload({"name": "t"}, heartbeat=True)[
            "heartbeat"
        ]

    def test_skin_and_change_events_survive(self):
        payload = entry.build_gateway_ready_payload(
            {"name": "custom"}, change_events=True
        )
        assert payload["skin"] == {"name": "custom"}
        assert payload["change_events"] is True
