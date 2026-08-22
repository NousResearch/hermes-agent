"""Session propagation tests for the client-required pre-prompt handler."""

import pytest
from types import SimpleNamespace

from tui_gateway import server


class _StoredSessionDB:
    def __init__(self, session_id: str, cwd: str):
        self.session_id = session_id
        self.cwd = cwd

    def get_session(self, session_id: str):
        if session_id != self.session_id:
            return None
        return {"id": session_id, "cwd": self.cwd, "message_count": 0}

    def get_session_by_title(self, _title: str):
        return None

    def resolve_resume_session_id(self, session_id: str) -> str:
        return session_id

    def assert_resume_safe(self, _session_id: str) -> None:
        return None

    def reopen_session(self, _session_id: str) -> None:
        return None

    def get_resume_conversations(self, _session_id: str):
        return [], []

    def get_messages_as_conversation(self, _session_id: str, **_kwargs):
        return []

    def get_ancestor_display_prefix(self, _session_id: str):
        return []


def _prepare_resume(monkeypatch, tmp_path, target: str) -> None:
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_get_db", lambda: _StoredSessionDB(target, str(tmp_path)))
    monkeypatch.setattr(server, "_profile_home", lambda _profile: None)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *_args: False)
    monkeypatch.setattr(server, "_schedule_resume_hydration", lambda *_args, **_kwargs: None)


def test_session_create_retains_required_prompt_handler(monkeypatch, tmp_path):
    """Dropping the create parameter would silently remove iOS fail-closed policy."""
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_completion_cwd", lambda _params=None: str(tmp_path))

    response = server._methods["session.create"](
        "create-required-handler",
        {
            "source": "ios",
            "profile": "router",
            "required_prompt_handler": "  hoppe_ocr_approval  ",
        },
    )
    sid = response["result"]["session_id"]
    try:
        assert server._sessions[sid]["required_prompt_handler"] == "hoppe_ocr_approval"
        assert response["result"]["required_prompt_handler"] == "hoppe_ocr_approval"
    finally:
        server._sessions.pop(sid, None)


def test_session_create_reports_no_required_prompt_handler(monkeypatch, tmp_path):
    """An omitted create policy must be acknowledged explicitly as absent."""
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_completion_cwd", lambda _params=None: str(tmp_path))

    response = server._methods["session.create"](
        "create-without-required-handler",
        {"source": "ios", "profile": "router"},
    )
    sid = response["result"]["session_id"]
    try:
        assert response["result"]["required_prompt_handler"] is None
    finally:
        server._sessions.pop(sid, None)


def test_cold_session_resume_retains_required_prompt_handler(monkeypatch, tmp_path):
    """A backend restart must not discard the iOS-required gate on resume."""
    target = "stored-ios"
    _prepare_resume(monkeypatch, tmp_path, target)

    response = server._methods["session.resume"](
        "resume-required-handler",
        {
            "session_id": target,
            "source": "ios",
            "profile": "router",
            "required_prompt_handler": "  hoppe_ocr_approval  ",
        },
    )

    sid = response["result"]["session_id"]
    assert server._sessions[sid]["required_prompt_handler"] == "hoppe_ocr_approval"
    assert response["result"]["required_prompt_handler"] == "hoppe_ocr_approval"


def test_cold_session_resume_reports_no_required_prompt_handler(monkeypatch, tmp_path):
    """Cold resume must distinguish an omitted policy from an old backend."""
    target = "stored-ios-without-handler"
    _prepare_resume(monkeypatch, tmp_path, target)

    response = server._methods["session.resume"](
        "resume-without-required-handler",
        {"session_id": target, "source": "ios", "profile": "router"},
    )

    assert response["result"]["required_prompt_handler"] is None


@pytest.mark.parametrize(
    ("resume_mode", "required_handler"),
    [
        ({"lazy": True}, "  hoppe_ocr_approval  "),
        ({"lazy": True}, None),
        ({"defer_history": True}, "  hoppe_ocr_approval  "),
        ({"defer_history": True}, None),
    ],
    ids=[
        "lazy-watch-handler",
        "lazy-watch-null",
        "deferred-history-handler",
        "deferred-history-null",
    ],
)
def test_alternate_session_resume_modes_retain_required_prompt_handler(
    monkeypatch,
    tmp_path,
    resume_mode,
    required_handler,
):
    """Alternate cold-resume branches must preserve the same iOS policy."""
    target = "stored-ios-alternate"
    _prepare_resume(monkeypatch, tmp_path, target)
    params = {
        "session_id": target,
        "source": "ios",
        "profile": "router",
        "required_prompt_handler": required_handler,
        **resume_mode,
    }

    response = server._methods["session.resume"]("resume-alternate", params)

    sid = response["result"]["session_id"]
    expected = "hoppe_ocr_approval" if required_handler else None
    assert server._sessions[sid]["required_prompt_handler"] == expected
    assert response["result"]["required_prompt_handler"] == expected


def test_live_session_resume_refreshes_required_prompt_handler(monkeypatch, tmp_path):
    """Reconnect must reassert policy even when Hermes reuses a live session."""
    target = "stored-ios-live"
    _prepare_resume(monkeypatch, tmp_path, target)
    record = server._deferred_session_record(
        target,
        cols=80,
        cwd=str(tmp_path),
        history=[],
        lease=None,
        source="ios",
    )
    server._sessions["live-ios-ui"] = record

    response = server._methods["session.resume"](
        "resume-live-required-handler",
        {
            "session_id": target,
            "source": "ios",
            "profile": "router",
            "required_prompt_handler": "  hoppe_ocr_approval  ",
        },
    )

    assert response["result"]["session_id"] == "live-ios-ui"
    assert record["required_prompt_handler"] == "hoppe_ocr_approval"
    assert response["result"]["required_prompt_handler"] == "hoppe_ocr_approval"


def test_live_session_resume_reports_no_required_prompt_handler(monkeypatch, tmp_path):
    """Live resume must explicitly confirm that no policy was requested."""
    target = "stored-ios-live-without-handler"
    _prepare_resume(monkeypatch, tmp_path, target)
    record = server._deferred_session_record(
        target,
        cols=80,
        cwd=str(tmp_path),
        history=[],
        lease=None,
        source="ios",
    )
    server._sessions["live-ios-ui"] = record

    response = server._methods["session.resume"](
        "resume-live-without-required-handler",
        {"session_id": target, "source": "ios", "profile": "router"},
    )

    assert response["result"]["required_prompt_handler"] is None


def test_eager_session_constructor_retains_required_prompt_handler(monkeypatch, tmp_path):
    """The eager resume constructor must carry the same gate as deferred records."""
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_start_notification_poller", lambda *_args: None)
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *_args: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *_args: None)

    server._init_session(
        "eager-ios-ui",
        "stored-ios-eager",
        SimpleNamespace(model="test"),
        [],
        cwd=str(tmp_path),
        source="ios",
        required_prompt_handler="hoppe_ocr_approval",
    )

    assert (
        server._sessions["eager-ios-ui"]["required_prompt_handler"]
        == "hoppe_ocr_approval"
    )


@pytest.mark.parametrize(
    ("required_handler", "expected"),
    [
        ("  hoppe_ocr_approval  ", "hoppe_ocr_approval"),
        (None, None),
    ],
    ids=["handler", "null"],
)
def test_eager_session_resume_passes_required_prompt_handler(
    monkeypatch, tmp_path, required_handler, expected
):
    """The real eager-resume branch must pass the client policy to its constructor."""
    target = "stored-ios-eager-resume"
    _prepare_resume(monkeypatch, tmp_path, target)
    monkeypatch.setattr(server, "_make_agent", lambda *_args, **_kwargs: SimpleNamespace(model="test"))
    monkeypatch.setattr(server, "_set_session_context", lambda _target: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_start_notification_poller", lambda *_args: None)
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *_args: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *_args: None)
    monkeypatch.setattr(server, "_transfer_db_to_agent", lambda *_args: None)
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})

    response = server._methods["session.resume"](
        "resume-eager-required-handler",
        {
            "session_id": target,
            "source": "ios",
            "profile": "router",
            "required_prompt_handler": required_handler,
            "eager_build": True,
        },
    )

    sid = response["result"]["session_id"]
    assert server._sessions[sid]["required_prompt_handler"] == expected
    assert response["result"]["required_prompt_handler"] == expected


def test_eager_resume_race_reasserts_handler_on_concurrent_live_winner(
    monkeypatch, tmp_path
):
    """A concurrent eager winner must use the same reconnect contract."""
    target = "stored-ios-eager-race"
    _prepare_resume(monkeypatch, tmp_path, target)
    winner = server._deferred_session_record(
        target,
        cols=80,
        cwd=str(tmp_path),
        history=[],
        lease=None,
        source="ios",
        required_prompt_handler=None,
    )
    closed = []

    class _RedundantAgent:
        def close(self):
            closed.append(True)

    def build_then_lose_race(*_args, **_kwargs):
        server._sessions["winner-ui"] = winner
        return _RedundantAgent()

    monkeypatch.setattr(server, "_make_agent", build_then_lose_race)
    monkeypatch.setattr(server, "_set_session_context", lambda _target: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)

    response = server._methods["session.resume"](
        "resume-eager-race",
        {
            "session_id": target,
            "source": "ios",
            "profile": "router",
            "required_prompt_handler": "  hoppe_ocr_approval  ",
            "eager_build": True,
        },
    )

    assert response["result"]["session_id"] == "winner-ui"
    assert response["result"]["required_prompt_handler"] == "hoppe_ocr_approval"
    assert winner["required_prompt_handler"] == "hoppe_ocr_approval"
    assert closed == [True]
