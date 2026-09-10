"""Reusable local and peer transports for hosted Discussion coordinator tests."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace


from gateway import hosted_rooms
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


def _append_room_event(db, **kwargs):
    if kwargs.get("kind") == "message.user":
        room = hosted_rooms.room_state(db, room_id=kwargs["room_id"])
        kwargs.setdefault(
            "authority_gateway_id", str(room["authority_gateway_id"])
        )
        kwargs.setdefault("authority_epoch", int(room["authority_epoch"]))
    return hosted_rooms.append_event(db, **kwargs)

class _FakeRPC:
    def __init__(self) -> None:
        self.sessions = {}
        self.approvals = []

    def resolve_exact(self, *, profile, title, source):
        return self.sessions.get((profile, title))

    def create(self, *, profile, title, source):
        session = {"session_id": f"{profile}-session", "title": title}
        self.sessions[(profile, title)] = session
        return session

    def resume(self, *, profile, session_id, source):
        return {"session_id": session_id}

    def submit(
        self,
        *,
        profile,
        session_id,
        prompt,
        source,
        task,
        execution_generation,
        on_terminal,
    ):
        on_terminal({"status": "settled", "text": f"reply from {profile}"})
        return {"accepted": True}

    def history(self, *, profile, session_id, source):
        return []

    def info(self, *, profile, session_id, source):
        return {"active": False, "task_id": None}

    def interrupt(self, *, profile, session_id, source, expected_task_id):
        return {"interrupted": True}

    def approve(self, **kwargs):
        self.approvals.append(dict(kwargs))
        return {"resolved": 1}

class _FakePeerClient:
    def __init__(self) -> None:
        self.dispatches = []
        self.revoked = []
        self.session = {"session_id": "peer-group-session"}

    def prepare(self, **kwargs):
        return (
            self.session
            if kwargs["create"] or kwargs.get("expected_session_id")
            else None
        )

    def dispatch(self, **kwargs):
        self.dispatches.append(kwargs["dispatch"])
        return {"status": "accepted", "task_id": kwargs["dispatch"]["task_id"]}

    def history(self, **kwargs):
        if not self.dispatches:
            return []
        dispatch = self.dispatches[-1]
        return [
            {
                "role": "assistant",
                "task_id": dispatch["task_id"],
                "execution_generation": dispatch["execution_generation"],
                "status": "settled",
                "message_id": f"peer:{dispatch['task_id']}",
                "content": "Remote review complete.",
            }
        ]

    def status(self, **kwargs):
        task_id = self.dispatches[-1]["task_id"] if self.dispatches else None
        return {"active": False, "task_id": task_id}

    def stop(self, **kwargs):
        return {"status": "cancelled"}

    def revoke_grant(self, **kwargs):
        self.revoked.append(kwargs["grant"])
        return {"revoked": True}

class _UnavailablePeerClient(_FakePeerClient):
    def prepare(self, **kwargs):
        raise RuntimeError("peer is offline before admission")

class _NotAdmittedPeerClient(_FakePeerClient):
    def __init__(self) -> None:
        super().__init__()
        self.offline = True

    def dispatch(self, **kwargs):
        if self.offline:
            raise PeerRunsHTTPError(
                "peer refused the connection",
                retryable=True,
                not_admitted=True,
            )
        return super().dispatch(**kwargs)

class _ExpiredGrantPeerClient(_FakePeerClient):
    def prepare(self, **kwargs):
        raise PeerRunsHTTPError(
            "peer room authorization needs renewal",
            status_code=401,
            error_code="invalid_room_grant",
        )

class _UnavailableRevokePeerClient(_FakePeerClient):
    def revoke_grant(self, **kwargs):
        raise RuntimeError("peer is offline during revocation")

class _ExpiredRevokePeerClient(_FakePeerClient):
    def revoke_grant(self, **kwargs):
        raise PeerRunsHTTPError(
            "peer room authorization needs renewal",
            status_code=401,
            error_code="invalid_room_grant",
        )

class _RefreshingPeerClient(_FakePeerClient):
    def __init__(self, replacement: str, catalog=None) -> None:
        super().__init__()
        self.replacement = replacement
        self.catalog = catalog
        self.refreshed = []
        self.refresh_arguments = []
        self.dispatched_grants = []

    def refresh_grant(self, **kwargs):
        self.refreshed.append(kwargs["grant"])
        self.refresh_arguments.append(dict(kwargs))
        return {
            "grant": self.replacement,
            **({"catalog": self.catalog} if self.catalog is not None else {}),
        }

    def dispatch(self, **kwargs):
        self.dispatched_grants.append(kwargs["grant"])
        return super().dispatch(**kwargs)

class _ApprovalPeerClient(_FakePeerClient):
    def __init__(self) -> None:
        super().__init__()
        self.approvals = []

    def status(self, **kwargs):
        task_id = self.dispatches[-1]["task_id"] if self.dispatches else "task-1"
        return {
            "status": "waiting_for_approval",
            "active": True,
            "task_id": task_id,
                "execution_generation": 2,
                "run_id": "run-peer-1",
                "session_id": "peer-group-session",
                "request_id": "req-peer-1",
                "approval": {
                "description": "Run the focused tests",
                "command": "pytest -q tests/focused",
                "choices": ["once", "deny"],
            },
        }

    def approve_receipt(self, **kwargs):
        self.approvals.append(dict(kwargs))
        return {"resolved": 1}

class _RecoveringPeerClient(_FakePeerClient):
    def __init__(self) -> None:
        super().__init__()
        self.recoveries = []

    def recover_dispatch(self, **kwargs):
        dispatch = dict(kwargs["dispatch"])
        self.recoveries.append({**kwargs, "dispatch": dispatch})
        self.dispatches.append(dispatch)
        return {
            "status": "accepted",
            "task_id": dispatch["task_id"],
            "execution_generation": dispatch["execution_generation"],
            "run_id": "run-recovered",
        }

class _PromptRecordingRPC(_FakeRPC):
    def __init__(self) -> None:
        super().__init__()
        self.prompts: list[tuple[str, str]] = []

    def submit(
        self,
        *,
        profile,
        session_id,
        prompt,
        source,
        task,
        execution_generation,
        on_terminal,
    ):
        self.prompts.append((profile, prompt))
        on_terminal({"status": "settled", "text": f"reply from {profile}"})
        return {"accepted": True}

class _BlockingFirstRPC(_PromptRecordingRPC):
    def __init__(self) -> None:
        super().__init__()
        self.first_started = threading.Event()
        self.release_first = threading.Event()

    def submit(self, **kwargs):
        self.prompts.append((kwargs["profile"], kwargs["prompt"]))
        if len(self.prompts) == 1:
            self.first_started.set()
            assert self.release_first.wait(timeout=2)
        kwargs["on_terminal"](
            {"status": "settled", "text": f"reply from {kwargs['profile']}"}
        )
        return {"accepted": True}

def _server():
    return SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())

def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached")
