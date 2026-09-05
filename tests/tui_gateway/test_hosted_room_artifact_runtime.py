"""Runtime tests for the hosted-room session adapter."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gateway import hosted_room_driver as state
from gateway import hosted_rooms
from tui_gateway.hosted_room_driver import (
    MAX_TERMINAL_TEXT_BYTES,
    ROOM_SESSION_SOURCE,
    HostedRoomBinding,
    HostedRoomRuntime,
    _find_terminal_receipt,
    room_session_title,
)
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
from tui_gateway.hosted_room_peer_transport import (
    PeerHostedRoomTransport,
    PeerMemberRoute,
)


ROOM_ID = "room-1"
PROFILE = "ops"
BINDING = HostedRoomBinding(
    room_id=ROOM_ID,
    gateway_id="gateway-a",
    authority_epoch=1,
)


from tests.tui_gateway.test_hosted_room_driver_runtime import (
    FakeSessionRPC,
    _admit,
    _identity,
    _runtime,
    _wait_for,
    db,
)


def test_terminal_receipt_preserves_artifact_manifest_and_remote_run_id():
    identity = _identity("task-artifact")
    manifest = {
        "version": 1,
        "manifest_digest": "a" * 64,
        "items": [
            {
                "artifact_id": "rart_0123456789abcdef0123456789abcdef",
                "kind": "file",
                "name": "handoff.md",
                "size": 8,
                "mime": "text/markdown",
                "sha256": "b" * 64,
            }
        ],
    }

    receipt = _find_terminal_receipt(
        [
            {
                "role": "assistant",
                "task_id": identity.task_id,
                "execution_generation": 3,
                "status": "settled",
                "message_id": "peer-run:run-artifact",
                "content": "Shared the handoff.",
                "artifacts": manifest,
                "run_id": "run-artifact",
            }
        ],
        identity,
        3,
    )

    assert receipt is not None
    assert receipt.result["artifacts"] == manifest
    assert receipt.result["run_id"] == "run-artifact"
