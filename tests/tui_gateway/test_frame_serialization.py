"""``serialize_frame`` must emit strict JSON: a frame no parser accepts is a frame that never arrived.

``json.dumps`` writes a non-finite float as the bare token ``Infinity``/``NaN``, which strict parsers
reject. A dropped *response* strands its caller silently — the desktop transport returns ``null`` for
text that is not JSON, so a roster call whose frame carried one never settled and no roster ever
painted — the same shape of loss #92506 fixed for unserializable payloads and #97288 for text.
"""

from __future__ import annotations

import json
import logging

import pytest

from tui_gateway.transport import serialize_frame

LOGGER_NAME = "tui_gateway.transport"
LOGGER = logging.getLogger(LOGGER_NAME)


def _strict(text: str):
    """Parse *text* the way a client does: JSON has no ``Infinity``/``NaN`` token."""
    def _reject(constant: str) -> None:
        raise AssertionError(f"frame is not strict JSON: bare {constant} token")

    return json.loads(text, parse_constant=_reject)


def _transport_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == LOGGER_NAME]


def test_a_non_finite_number_is_coerced_not_emitted(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        line = serialize_frame({"jsonrpc": "2.0", "id": 7, "result": {"chat": float("inf")}},
                               "test", LOGGER)

    assert _strict(line)["result"]["chat"] == "inf"
    assert "Infinity" not in line
    assert "coerced to string" in caplog.text, "a coerced value is never silent"


def test_coercion_reaches_into_containers(caplog: pytest.LogCaptureFixture) -> None:
    payload = {"jsonrpc": "2.0", "id": 8,
               "result": {"list": [float("nan"), {"deep": float("-inf")}]}}

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        line = serialize_frame(payload, "test", LOGGER)

    result = _strict(line)["result"]
    assert result["list"][0] == "nan"
    assert result["list"][1]["deep"] == "-inf"


def test_an_ordinary_frame_is_byte_identical_and_silent(caplog: pytest.LogCaptureFixture) -> None:
    """The fast path must not change: no walk, no rewrite, and the *text* "Infinity" stays text."""
    payload = {"jsonrpc": "2.0", "id": 9,
               "result": {"n": 2.5, "huge": 10 ** 30, "s": "Infinity", "none": None, "t": True}}

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        line = serialize_frame(payload, "test", LOGGER)

    assert line == json.dumps(payload, ensure_ascii=False)
    assert _transport_records(caplog) == []


def test_an_unserializable_payload_still_becomes_an_error_frame(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        line = serialize_frame({"jsonrpc": "2.0", "id": 10, "result": object()}, "test", LOGGER)

    frame = _strict(line)
    assert frame["id"] == 10
    assert frame["error"]["code"] == -32603
    assert "response serialization error" in frame["error"]["message"]


def test_a_circular_reference_still_becomes_an_error_frame(caplog: pytest.LogCaptureFixture) -> None:
    """The coercion walk must not recurse into a cycle: same error frame, no hang."""
    payload: dict = {"jsonrpc": "2.0", "id": 11, "result": {}}
    payload["result"]["self"] = payload["result"]

    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        line = serialize_frame(payload, "test", LOGGER)

    assert _strict(line)["error"]["code"] == -32603
