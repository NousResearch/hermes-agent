"""message.complete forwards ``response_transformed`` (#96465).

A transform_llm_output hook may rewrite the final text after streaming; the desktop
renderer needs the flag to replace the streamed bubble even without a prefix match.
"""

import contextlib
from types import SimpleNamespace

import tui_gateway.server as srv


def _turn(result):
    return SimpleNamespace(
        result=result, agent=SimpleNamespace(_session_title_hint="Scratch"), terminal_callback=None,
        receipt_committed=True, receipt_attempted=False, marker_key="", error_retained=False,
        error_detail="", prompt_text="ping",
    )


def test_complete_turn_payload_forwards_response_transformed_only_when_set(monkeypatch):
    monkeypatch.setattr(srv, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(srv, "render_message", lambda _text, _cols: None)
    monkeypatch.setattr(srv, "_clear_inflight_turn", lambda _session: None)
    session = {"pending_title": None, "session_key": "k", "history_lock": contextlib.nullcontext(),
               "agent": SimpleNamespace(_session_title_hint="Scratch")}

    transformed = {"final_response": "example-service.internal", "response_transformed": True}
    payload, _, _ = srv._complete_turn_payload(session, _turn(transformed), None, 80)
    assert payload["text"] == "example-service.internal"
    assert payload.get("response_transformed") is True
    # The real emit path validates against the wire contract (extra="forbid"); under
    # HERMES_TEST_ISOLATION an undeclared key raises ContractViolation here.
    frame = srv._event_frame("message.complete", "sid", payload)
    assert frame["params"]["payload"]["response_transformed"] is True

    payload, _, _ = srv._complete_turn_payload(session, _turn({"final_response": "TOKEN_1"}), None, 80)
    assert "response_transformed" not in payload
