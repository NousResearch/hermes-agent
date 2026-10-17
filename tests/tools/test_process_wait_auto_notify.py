"""Repeated process_manage wait/poll on a silent running process promotes it to notify_on_complete."""

import json

import pytest

import tools.process_registry as pr


@pytest.fixture
def sleeper(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    session = pr.process_registry.spawn_local("sleep 30", cwd=str(tmp_path), task_id="t-autonotify")
    session.notify_on_complete = False
    yield home, session
    pr.process_registry.kill_process(session.id)


def _wait(sid):
    return json.loads(pr._handle_process({"action": "wait", "session_id": sid, "timeout": 1}))


def test_second_unchanged_wait_enables_notify_and_tells_model_to_stop(sleeper):
    _home, session = sleeper
    first = _wait(session.id)
    assert first["status"] == "timeout" and not session.notify_on_complete
    second = _wait(session.id)
    assert second["status"] == "timeout" and second["process_running"] is True
    assert session.notify_on_complete is True
    assert "Do NOT call process_manage wait or poll" in second["timeout_note"]


def test_opt_out_keeps_plain_polling(sleeper):
    home, session = sleeper
    (home / "config.yaml").write_text("terminal:\n  process_wait_auto_notify: false\n", encoding="utf-8")
    for _ in range(3):
        result = _wait(session.id)
        assert result["status"] == "timeout"
        assert "Poll again later" in result["timeout_note"] and "notify_on_complete" not in result
    assert session.notify_on_complete is False
