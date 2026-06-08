import json

from tui_gateway.ws import _ws_frame_diagnostics, _ws_parse_error_diagnostics


def test_ws_frame_diagnostics_describes_event_without_payload_values():
    frame = {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {
            "type": "message.delta",
            "session_id": "20260607_174833_7911aa",
            "payload": {"text": "do not log streamed content"},
        },
    }
    line = '{"jsonrpc":"2.0","method":"event","params":{"type":"message.delta"}}'

    diagnostics = _ws_frame_diagnostics(frame, line)

    assert diagnostics == {
        "frame_type": "event",
        "event_type": "message.delta",
        "frame_id": None,
        "session_id": "20260607_174833_7911aa",
        "byte_len": len(line.encode("utf-8")),
    }
    assert "do not log streamed content" not in repr(diagnostics)


def test_ws_frame_diagnostics_describes_response_without_result_values():
    frame = {
        "jsonrpc": "2.0",
        "id": 208,
        "result": {"token": "do-not-log"},
    }
    line = '{"jsonrpc":"2.0","id":208,"result":{"token":"do-not-log"}}'

    diagnostics = _ws_frame_diagnostics(frame, line)

    assert diagnostics == {
        "frame_type": "response",
        "event_type": None,
        "frame_id": 208,
        "session_id": None,
        "byte_len": len(line.encode("utf-8")),
    }
    assert "do-not-log" not in repr(diagnostics)


def test_ws_frame_diagnostics_sanitizes_nonprimitive_metadata():
    frame = {
        "jsonrpc": "2.0",
        "id": {"token": "SECRET"},
        "error": {"message": "do not log"},
    }
    line = '{"jsonrpc":"2.0","id":{"token":"SECRET"},"error":{"message":"do not log"}}'

    diagnostics = _ws_frame_diagnostics(frame, line)

    assert diagnostics["frame_type"] == "response"
    assert diagnostics["frame_id"] == "<dict>"
    assert "SECRET" not in repr(diagnostics)
    assert "do not log" not in repr(diagnostics)


def test_ws_frame_diagnostics_bounds_peer_controlled_strings():
    frame = {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {"type": "x" * 200, "session_id": ["SECRET"]},
    }
    line = "{}"

    diagnostics = _ws_frame_diagnostics(frame, line)

    assert diagnostics["event_type"] == f"{'x' * 128}…"
    assert diagnostics["session_id"] == "<list>"
    assert "SECRET" not in repr(diagnostics)


def test_ws_parse_error_diagnostics_do_not_include_invalid_payload_preview():
    line = '{"jsonrpc":"2.0","params":{"token":"SECRET"}'

    try:
        json.loads(line)
    except json.JSONDecodeError as exc:
        diagnostics = _ws_parse_error_diagnostics(line, exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("test fixture should be invalid JSON")

    assert diagnostics["byte_len"] == len(line.encode("utf-8"))
    assert diagnostics["pos"] > 0
    assert "SECRET" not in repr(diagnostics)
    assert line not in repr(diagnostics)
