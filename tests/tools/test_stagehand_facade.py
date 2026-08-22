"""Behavior contracts for the Stagehand ``browser_exec`` backend."""

from __future__ import annotations

import json

import tools.browser_use_cli as browser_exec
import tools.stagehand_facade as stagehand
import tools.stagehand_facade_client as client


def _select_stagehand(monkeypatch):
    monkeypatch.setattr(
        browser_exec,
        "_read_browser_cfg",
        lambda: {"backend": "stagehand"},
    )


def test_stagehand_is_an_explicit_browser_exec_backend(monkeypatch):
    _select_stagehand(monkeypatch)

    assert browser_exec.is_stagehand_facade_mode() is True
    assert browser_exec.is_browser_use_cli_mode() is False
    assert browser_exec.is_browser_exec_mode() is True


def test_stagehand_schema_keeps_one_tool_and_switches_code_language(monkeypatch):
    _select_stagehand(monkeypatch)

    overrides = browser_exec._dynamic_schema_overrides()
    properties = overrides["parameters"]["properties"]

    assert "Playwright-shaped facade" in overrides["description"]
    assert "JavaScript" in properties["code"]["description"]
    assert "session" not in properties
    assert set(properties) == {"code", "timeout_s"}


def test_browser_exec_dispatches_to_stagehand_without_browser_use_cli(
    monkeypatch,
):
    _select_stagehand(monkeypatch)
    monkeypatch.setattr(browser_exec, "_blocked_url_in_code", lambda _code: None)
    captured = {}

    def fake_stagehand_browser_exec(**kwargs):
        captured.update(kwargs)
        return '{"success": true, "exit_code": 0, "output": "ok"}'

    monkeypatch.setattr(stagehand, "stagehand_browser_exec", fake_stagehand_browser_exec)

    result = browser_exec.browser_exec(
        code="return await page.title();",
        timeout_s=30,
        task_id="conversation-7",
    )

    assert json.loads(result) == {
        "success": True,
        "exit_code": 0,
        "output": "ok",
    }
    assert captured == {
        "code": "return await page.title();",
        "timeout_s": 30,
        "task_id": "conversation-7",
    }


def test_stagehand_success_matches_browser_use_model_visible_envelope(monkeypatch):
    monkeypatch.setattr(stagehand, "_stagehand_root", lambda: "/stagehand")
    monkeypatch.setattr(stagehand, "_node_executable", lambda: "/usr/bin/node")
    monkeypatch.setattr(
        stagehand,
        "call_stagehand_facade",
        lambda **_kwargs: {"success": True, "output": '{"title":"Example"}'},
    )

    result = json.loads(
        stagehand.stagehand_browser_exec(
            code="return { title: await page.title() };",
            timeout_s=60,
            task_id="task-a",
        )
    )

    assert result == {
        "success": True,
        "exit_code": 0,
        "output": '{"title":"Example"}',
    }


def test_stagehand_failure_preserves_same_envelope(monkeypatch):
    monkeypatch.setattr(stagehand, "_stagehand_root", lambda: "/stagehand")
    monkeypatch.setattr(stagehand, "_node_executable", lambda: "/usr/bin/node")
    monkeypatch.setattr(
        stagehand,
        "call_stagehand_facade",
        lambda **_kwargs: {"success": False, "error": "locator was not found"},
    )

    result = json.loads(
        stagehand.stagehand_browser_exec(
            code="await page.locator('#missing').click();",
            task_id="task-a",
        )
    )

    assert result == {
        "success": False,
        "exit_code": 1,
        "output": "",
        "stderr": "locator was not found",
    }


def test_worker_environment_is_least_privilege_and_scope_aware(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        client,
        "hermes_subprocess_env",
        lambda **_kwargs: {"PATH": "/usr/bin"},
    )
    secrets = {
        "BROWSERBASE_API_KEY": "bb_test_value",
        "BROWSERBASE_PROJECT_ID": "project-value",
        "OPENAI_API_KEY": "must-not-leak",
    }
    monkeypatch.setattr(
        client,
        "get_secret",
        lambda name, default="": secrets.get(name, default),
    )

    env = client._worker_environment(tmp_path)

    assert env == {
        "PATH": "/usr/bin",
        "BROWSERBASE_API_KEY": "bb_test_value",
        "BROWSERBASE_PROJECT_ID": "project-value",
        "STAGEHAND_FACADE_ROOT": str(tmp_path),
    }
