"""Submit failures identify the error type without logging exception contents."""

import logging

from tests.tui_gateway import test_hosted_room_driver_runtime as runtime_fixtures
from tests.tui_gateway.test_hosted_room_driver_runtime import (
    BINDING, FakeSessionRPC, _admit, _identity, _runtime,
)

db = runtime_fixtures.db


def test_submit_failure_logs_exception_type_without_its_text(db, monkeypatch, caplog):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC()

    def fail_submit(**kwargs):
        raise TimeoutError("PRIVATE_PROMPT_SENTINEL PRIVATE_CREDENTIAL_SENTINEL")

    monkeypatch.setattr(rpc, "submit", fail_submit)
    runtime = _runtime(db, rpc)
    with caplog.at_level(logging.WARNING, logger="tui_gateway.hosted_room_driver"):
        runtime._process_room(BINDING)

    assert "PRIVATE_PROMPT_SENTINEL" not in caplog.text
    assert "PRIVATE_CREDENTIAL_SENTINEL" not in caplog.text
    assert f"hosted room task {identity.task_id}: attempt_failed (TimeoutError)" in caplog.messages
