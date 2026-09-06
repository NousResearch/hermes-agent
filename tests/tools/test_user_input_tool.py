"""RED contracts for Hermes-native asynchronous structured user input."""

from __future__ import annotations

import json
import math
import threading
import time

import pytest

from hermes_state import SessionDB


QUESTIONS = [
    {
        "id": "api_version",
        "text": "Which API version should I target?",
        "options": ["v1 (stable)", "v2 (beta)"],
        "allow_free_text": False,
        "default": "v1 (stable)",
    }
]


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    database.create_session("session-1", source="cli")
    database.create_session("session-2", source="cli")
    try:
        yield database
    finally:
        database.close()


def test_user_input_tools_are_registered_in_clarify_toolset():
    from tools import user_input_tool  # noqa: F401
    from tools.registry import registry

    request = registry.get_entry("request_user_input")
    check = registry.get_entry("check_user_input")
    assert request is not None
    assert check is not None
    assert request.toolset == "clarify"
    assert check.toolset == "clarify"


def test_create_and_read_pending_request_round_trip(db):
    record = db.create_pending_user_input(
        request_id="request-1",
        session_id="session-1",
        questions=QUESTIONS,
        context="Choosing the compatibility target",
        expires_at=time.time() + 60,
        turn_id="turn-1",
    )

    assert record["request_id"] == "request-1"
    assert record["session_id"] == "session-1"
    assert record["status"] == "pending"
    assert record["questions"] == QUESTIONS
    assert record["context"] == "Choosing the compatibility target"
    assert record["answer"] is None

    loaded = db.get_pending_user_input("request-1", session_id="session-1")
    assert loaded == record


def test_pending_request_cannot_be_read_or_answered_from_another_session(db):
    db.create_pending_user_input(
        request_id="request-2",
        session_id="session-1",
        questions=QUESTIONS,
        context="",
        expires_at=time.time() + 60,
    )

    assert db.get_pending_user_input("request-2", session_id="session-2") is None
    result = db.answer_pending_user_input(
        "request-2", {"api_version": "v2 (beta)"}, session_id="session-2"
    )
    assert result["status"] == "not_found"
    assert db.get_pending_user_input("request-2", session_id="session-1")["status"] == "pending"


def test_answer_is_single_writer_and_returns_canonical_record(db):
    db.create_pending_user_input(
        request_id="request-3",
        session_id="session-1",
        questions=QUESTIONS,
        context="",
        expires_at=time.time() + 60,
    )
    barrier = threading.Barrier(3)
    results = []

    def answer(value):
        barrier.wait()
        results.append(
            db.answer_pending_user_input(
                "request-3", {"api_version": value}, session_id="session-1"
            )
        )

    first = threading.Thread(target=answer, args=("v1 (stable)",))
    second = threading.Thread(target=answer, args=("v2 (beta)",))
    first.start()
    second.start()
    barrier.wait()
    first.join()
    second.join()

    assert sorted(result["status"] for result in results) == ["answered", "answered"]
    assert sum(bool(result.get("accepted")) for result in results) == 1
    stored = db.get_pending_user_input("request-3", session_id="session-1")
    assert stored["status"] == "answered"
    assert stored["answer"]["api_version"] in {"v1 (stable)", "v2 (beta)"}


def test_expiry_is_settled_durably_and_uses_question_defaults(db):
    db.create_pending_user_input(
        request_id="request-4",
        session_id="session-1",
        questions=QUESTIONS,
        context="",
        expires_at=time.time() - 1,
    )

    result = db.get_pending_user_input("request-4", session_id="session-1")
    assert result["status"] == "expired"
    assert result["answer"] == {"api_version": "v1 (stable)"}

    import sqlite3

    with sqlite3.connect(db.db_path) as conn:
        row = conn.execute(
            "SELECT status, answer FROM pending_user_inputs WHERE request_id = ?",
            ("request-4",),
        ).fetchone()
    assert row[0] == "expired"
    assert json.loads(row[1]) == {"api_version": "v1 (stable)"}


def test_request_tool_allows_closed_choice_without_optional_default(monkeypatch, db):
    from tools import user_input_tool

    monkeypatch.setattr(user_input_tool, "_shared_session_db", db)
    result = json.loads(
        user_input_tool.request_user_input(
            questions=[
                {
                    "id": "drink",
                    "text": "Which drink?",
                    "options": ["Coffee", "Tea"],
                    "allow_free_text": False,
                }
            ],
            session_id="session-1",
        )
    )

    assert result["status"] == "pending"
    assert result["questions"][0]["default"] == ""


def test_request_tool_validates_shape_and_returns_immediately(monkeypatch, db):
    from tools import user_input_tool

    monkeypatch.setattr(user_input_tool, "_shared_session_db", db)
    result = json.loads(
        user_input_tool.request_user_input(
            questions=QUESTIONS,
            context="Need a compatibility decision",
            timeout_s=60,
            session_id="session-1",
        )
    )
    assert result["status"] == "pending"
    assert result["request_id"]
    assert "hint" in result
    assert db.list_pending_user_inputs("session-1")[0]["request_id"] == result["request_id"]

    invalid = json.loads(
        user_input_tool.request_user_input(
            questions=QUESTIONS * 6,
            session_id="session-1",
        )
    )
    assert "error" in invalid


def test_check_tool_is_session_scoped_and_reports_expiry(monkeypatch, db):
    from tools import user_input_tool

    db.create_pending_user_input(
        request_id="request-5",
        session_id="session-1",
        questions=QUESTIONS,
        context="",
        expires_at=time.time() - 1,
    )
    monkeypatch.setattr(user_input_tool, "_shared_session_db", db)

    wrong_session = json.loads(
        user_input_tool.check_user_input("request-5", session_id="session-2")
    )
    assert wrong_session["status"] == "not_found"

    expired = json.loads(
        user_input_tool.check_user_input("request-5", session_id="session-1")
    )
    assert expired["status"] == "expired"
    assert expired["answer"]["api_version"] == "v1 (stable)"


def test_dispatcher_passes_agent_session_db_to_user_input_handler(monkeypatch, db):
    from tools import user_input_tool
    import model_tools

    sentinel = object()
    monkeypatch.setattr(user_input_tool, "_shared_session_db", sentinel)
    result = json.loads(
        model_tools.handle_function_call(
            "request_user_input",
            {"questions": QUESTIONS, "timeout_s": 60},
            task_id="task-1",
            session_id="session-1",
            session_db=db,
        )
    )
    assert result["status"] == "pending"
    request = db.get_pending_user_input(result["request_id"], session_id="session-1")
    assert request is not None
    assert user_input_tool._shared_session_db is sentinel


def test_agent_runtime_executor_passes_session_db_to_user_input_handler(monkeypatch, db):
    from types import SimpleNamespace

    from agent.agent_runtime_helpers import invoke_tool
    from tools import user_input_tool

    sentinel = object()
    monkeypatch.setattr(user_input_tool, "_shared_session_db", sentinel)
    agent = SimpleNamespace(
        _session_db=db,
        session_id="session-1",
        valid_tool_names={"request_user_input"},
        _context_engine_tool_names=set(),
        _memory_manager=None,
        _current_turn_id="turn-1",
        _current_api_request_id="",
        enabled_toolsets=None,
        disabled_toolsets=None,
    )
    result = json.loads(
        invoke_tool(
            agent,
            "request_user_input",
            {"questions": QUESTIONS, "timeout_s": 60},
            "task-1",
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        )
    )
    assert result["status"] == "pending"
    assert db.get_pending_user_input(result["request_id"], session_id="session-1") is not None
    assert user_input_tool._shared_session_db is sentinel


@pytest.mark.parametrize(
    "answer",
    [
        {},
        {"api_version": "v1 (stable)", "unexpected": "value"},
        {"api_version": ""},
        {"api_version": "   "},
        {"api_version": ["v1 (stable)"]},
        {"api_version": {"value": "v1 (stable)"}},
        {"api_version": None},
        {"api_version": math.nan},
        {"api_version": math.inf},
        {"api_version": "not-an-option"},
        {"api_version": "x" * 2001},
        {"api_version": 42},
        {"api_version": False},
    ],
)
def test_invalid_answer_is_rejected_without_settling_request(db, answer):
    db.create_pending_user_input(
        request_id="request-invalid",
        session_id="session-1",
        questions=QUESTIONS,
        expires_at=time.time() + 60,
    )

    result = db.answer_pending_user_input(
        "request-invalid", answer, session_id="session-1"
    )

    assert result["status"] == "invalid"
    assert result["accepted"] is False
    assert result["error"]
    pending = db.get_pending_user_input("request-invalid", session_id="session-1")
    assert pending["status"] == "pending"
    assert pending["answer"] is None


def test_valid_answer_is_normalized_and_tool_delivers_canonical_answer(db):
    from types import SimpleNamespace

    from tools.user_input_tool import answer_user_input

    questions = [
        {
            "id": "choice",
            "text": "Which target?",
            "options": ["stable", "beta"],
            "allow_free_text": False,
        },
        {
            "id": "notes",
            "text": "Any notes?",
            "options": [],
            "allow_free_text": False,
        },
    ]
    db.create_pending_user_input(
        request_id="request-canonical",
        session_id="session-1",
        questions=questions,
        expires_at=time.time() + 60,
        turn_id="turn-1",
    )
    delivered = []
    agent = SimpleNamespace(
        _current_turn_id="turn-1",
        _executing_tools=True,
        steer=lambda text: delivered.append(text) or True,
    )

    result = answer_user_input(
        "request-canonical",
        {"choice": " beta ", "notes": "  free text  "},
        session_id="session-1",
        session_db=db,
        agent=agent,
    )

    assert result["status"] == "answered"
    assert result["accepted"] is True
    assert result["answer"] == {"choice": "beta", "notes": "free text"}
    assert delivered == [json.dumps(result["answer"], ensure_ascii=False)]


def test_invalid_and_valid_answers_race_without_invalid_writer_acceptance(db):
    db.create_pending_user_input(
        request_id="request-race",
        session_id="session-1",
        questions=QUESTIONS,
        expires_at=time.time() + 60,
    )
    barrier = threading.Barrier(3)
    results = []

    def answer(payload):
        barrier.wait()
        results.append(
            db.answer_pending_user_input(
                "request-race", payload, session_id="session-1"
            )
        )

    invalid = threading.Thread(
        target=answer,
        args=({"api_version": "v2 (beta)", "unexpected": "value"},),
    )
    valid = threading.Thread(
        target=answer,
        args=({"api_version": "v2 (beta)"},),
    )
    invalid.start()
    valid.start()
    barrier.wait()
    invalid.join()
    valid.join()

    assert all(result["status"] in {"answered", "invalid"} for result in results)
    assert sum(bool(result.get("accepted")) for result in results) == 1
    stored = db.get_pending_user_input("request-race", session_id="session-1")
    assert stored["status"] == "answered"
    assert stored["answer"] == {"api_version": "v2 (beta)"}
