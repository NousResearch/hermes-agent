"""Tests for the Roundtable tools (tools.roundtable_tools)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hermes_cli import roundtable_db as rdb


@pytest.fixture
def roundtable_env(tmp_path, monkeypatch):
    """Isolated environment for roundtable tool tests."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_ROUNDTABLE_DB", str(home / "roundtable.db"))
    rdb._INITIALIZED_PATHS.clear()
    return home


@pytest.fixture
def db_conn(roundtable_env):
    """A connected roundtable DB."""
    conn = rdb.connect()
    yield conn
    conn.close()


def _make_participants():
    return [
        {"profile": "alice", "role": "Engineer", "perspective": "Technical", "display_name": "Alice"},
        {"profile": "bob", "role": "Designer", "perspective": "UX", "display_name": "Bob"},
    ]


# ---------------------------------------------------------------------------
# Import tests (tool registration)
# ---------------------------------------------------------------------------


def test_tools_module_imports():
    """Verify the roundtable tools module imports cleanly."""
    from tools import roundtable_tools
    assert roundtable_tools is not None


def test_tool_schemas_are_valid():
    """Verify all 7 tool schemas have required fields."""
    from tools.roundtable_tools import (
        ROUNDTABLE_INIT_SCHEMA,
        ROUNDTABLE_SPEAK_SCHEMA,
        ROUNDTABLE_READ_SCHEMA,
        ROUNDTABLE_STATUS_SCHEMA,
        ROUNDTABLE_SUMMARIZE_SCHEMA,
        ROUNDTABLE_END_SCHEMA,
        ROUNDTABLE_LIST_SCHEMA,
    )
    schemas = [
        ROUNDTABLE_INIT_SCHEMA, ROUNDTABLE_SPEAK_SCHEMA,
        ROUNDTABLE_READ_SCHEMA, ROUNDTABLE_STATUS_SCHEMA,
        ROUNDTABLE_SUMMARIZE_SCHEMA, ROUNDTABLE_END_SCHEMA,
        ROUNDTABLE_LIST_SCHEMA,
    ]
    for s in schemas:
        assert "name" in s
        assert "description" in s
        assert "parameters" in s
        assert s["parameters"]["type"] == "object"


# ---------------------------------------------------------------------------
# Handler tests: roundtable_init
# ---------------------------------------------------------------------------


def test_handler_init_success(roundtable_env):
    from tools.roundtable_tools import _handle_init
    result = _handle_init({
        "topic": "Test topic",
        "participants": _make_participants(),
        "context": "Some context",
        "max_rounds": 3,
    })
    data = json.loads(result)
    assert data["ok"] is True
    assert data["discussion_id"].startswith("rt_")
    assert data["topic"] == "Test topic"
    assert data["participants"] == ["alice", "bob"]
    assert data["max_rounds"] == 3


def test_handler_init_missing_topic(roundtable_env):
    from tools.roundtable_tools import _handle_init
    result = _handle_init({"participants": _make_participants()})
    data = json.loads(result)
    assert "error" in data
    assert "topic" in data["error"].lower()


def test_handler_init_too_few_participants(roundtable_env):
    from tools.roundtable_tools import _handle_init
    result = _handle_init({
        "topic": "Test",
        "participants": [{"profile": "alice"}],
    })
    data = json.loads(result)
    assert "error" in data
    assert "2 participants" in data["error"]


# ---------------------------------------------------------------------------
# Handler tests: roundtable_speak
# ---------------------------------------------------------------------------


def test_handler_speak_success(roundtable_env):
    from tools.roundtable_tools import _handle_init, _handle_speak
    init_result = _handle_init({
        "topic": "Test",
        "participants": _make_participants(),
    })
    disc_id = json.loads(init_result)["discussion_id"]

    result = _handle_speak({
        "discussion_id": disc_id,
        "participant": "alice",
        "content": "Hello!",
    })
    data = json.loads(result)
    assert data["ok"] is True
    assert data["speech_id"] > 0
    assert data["round"] == 0
    assert data["participant"] == "alice"


def test_handler_speak_unknown_participant(roundtable_env):
    from tools.roundtable_tools import _handle_init, _handle_speak
    init_result = _handle_init({
        "topic": "Test",
        "participants": _make_participants(),
    })
    disc_id = json.loads(init_result)["discussion_id"]

    result = _handle_speak({
        "discussion_id": disc_id,
        "participant": "eve",
        "content": "Sneaky!",
    })
    data = json.loads(result)
    assert "error" in data
    assert "not an active member" in data["error"]


def test_handler_speak_missing_content(roundtable_env):
    from tools.roundtable_tools import _handle_init, _handle_speak
    init_result = _handle_init({
        "topic": "Test",
        "participants": _make_participants(),
    })
    disc_id = json.loads(init_result)["discussion_id"]

    result = _handle_speak({
        "discussion_id": disc_id,
        "participant": "alice",
    })
    data = json.loads(result)
    assert "error" in data
    assert "content" in data["error"].lower()


# ---------------------------------------------------------------------------
# Handler tests: roundtable_read
# ---------------------------------------------------------------------------


def test_handler_read_success(roundtable_env):
    from tools.roundtable_tools import _handle_init, _handle_speak, _handle_read
    init_result = _handle_init({
        "topic": "Test",
        "participants": _make_participants(),
    })
    disc_id = json.loads(init_result)["discussion_id"]

    _handle_speak({"discussion_id": disc_id, "participant": "alice", "content": "Hi"})
    _handle_speak({"discussion_id": disc_id, "participant": "bob", "content": "Hello"})

    result = _handle_read({"discussion_id": disc_id})
    data = json.loads(result)
    assert data["ok"] is True
    assert data["speech_count"] == 2
    assert len(data["speeches"]) == 2
    assert "formatted_history" in data


def test_handler_read_with_since_round(roundtable_env):
    from tools.roundtable_tools import _handle_init, _handle_speak, _handle_read
    init_result = _handle_init({
        "topic": "Test",
        "participants": _make_participants(),
    })
    disc_id = json.loads(init_result)["discussion_id"]

    # Round 0
    _handle_speak({"discussion_id": disc_id, "participant": "alice", "content": "r0s1"})
    _handle_speak({"discussion_id": disc_id, "participant": "bob", "content": "r0s2"})
    # Round 1
    _handle_speak({"discussion_id": disc_id, "participant": "alice", "content": "r1s1"})

    result = _handle_read({"discussion_id": disc_id, "since_round": 1})
    data = json.loads(result)
    assert data["speech_count"] == 1
    assert data["speeches"][0]["content"] == "r1s1"


# ---------------------------------------------------------------------------
# Handler tests: roundtable_status
# ---------------------------------------------------------------------------


def test_handler_status(roundtable_env):
    from tools.roundtable_tools import _handle_init, _handle_status
    init_result = _handle_init({
        "topic": "Test",
        "participants": _make_participants(),
    })
    disc_id = json.loads(init_result)["discussion_id"]

    result = _handle_status({"discussion_id": disc_id})
    data = json.loads(result)
    assert data["ok"] is True
    assert data["status"] == "active"
    assert data["current_round"] == 0
    assert data["speech_count"] == 0


# ---------------------------------------------------------------------------
# Handler tests: roundtable_end
# ---------------------------------------------------------------------------


def test_handler_end_conclude(roundtable_env):
    from tools.roundtable_tools import _handle_init, _handle_end
    init_result = _handle_init({
        "topic": "Test",
        "participants": _make_participants(),
    })
    disc_id = json.loads(init_result)["discussion_id"]

    result = _handle_end({"discussion_id": disc_id})
    data = json.loads(result)
    assert data["ok"] is True
    assert data["action"] == "concluded"


def test_handler_end_force_cancel(roundtable_env):
    from tools.roundtable_tools import _handle_init, _handle_end
    init_result = _handle_init({
        "topic": "Test",
        "participants": _make_participants(),
    })
    disc_id = json.loads(init_result)["discussion_id"]

    result = _handle_end({"discussion_id": disc_id, "force": True})
    data = json.loads(result)
    assert data["ok"] is True
    assert data["action"] == "cancelled"


# ---------------------------------------------------------------------------
# Handler tests: roundtable_list
# ---------------------------------------------------------------------------


def test_handler_list(roundtable_env):
    from tools.roundtable_tools import _handle_init, _handle_list
    for i in range(3):
        _handle_init({
            "topic": f"Topic {i}",
            "participants": _make_participants(),
        })

    result = _handle_list({})
    data = json.loads(result)
    assert data["ok"] is True
    assert data["count"] == 3


def test_handler_list_filter_status(roundtable_env):
    from tools.roundtable_tools import _handle_init, _handle_end, _handle_list
    init_result = _handle_init({
        "topic": "Test",
        "participants": _make_participants(),
    })
    disc_id = json.loads(init_result)["discussion_id"]
    _handle_end({"discussion_id": disc_id})
    _handle_init({"topic": "Test2", "participants": _make_participants()})

    result = _handle_list({"status": "active"})
    data = json.loads(result)
    assert data["count"] == 1

    result = _handle_list({"status": "concluded"})
    data = json.loads(result)
    assert data["count"] == 1


# ---------------------------------------------------------------------------
# Handler tests: roundtable_summarize
# ---------------------------------------------------------------------------


def test_handler_summarize(roundtable_env):
    from tools.roundtable_tools import _handle_init, _handle_speak, _handle_summarize
    init_result = _handle_init({
        "topic": "DB Selection",
        "participants": _make_participants(),
        "context": "We need a new database",
    })
    disc_id = json.loads(init_result)["discussion_id"]

    _handle_speak({"discussion_id": disc_id, "participant": "alice", "content": "PostgreSQL"})
    _handle_speak({"discussion_id": disc_id, "participant": "bob", "content": "MySQL"})

    result = _handle_summarize({"discussion_id": disc_id})
    data = json.loads(result)
    assert data["ok"] is True
    assert data["topic"] == "DB Selection"
    assert data["speech_count"] == 2
    assert "rounds" in data
    assert "participants" in data
    assert "consensus_points" in data
    assert "formatted_history" in data
