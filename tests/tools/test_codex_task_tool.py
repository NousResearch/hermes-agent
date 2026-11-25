import json

from tools import codex_task_tool as mod


def test_codex_task_wraps_delegate_task_with_default_acp(monkeypatch):
    captured = {}

    def fake_delegate_task(**kwargs):
        captured.update(kwargs)
        return json.dumps({"ok": True, "backend": "codex"})

    monkeypatch.setattr(mod, "delegate_task", fake_delegate_task)

    result = mod.codex_task(
        goal="Refactor auth flow",
        context="Repo: /tmp/demo",
        toolsets=["terminal", "file"],
    )

    assert json.loads(result) == {"ok": True, "backend": "codex"}
    assert captured["goal"] == "Refactor auth flow"
    assert captured["context"] == "Repo: /tmp/demo"
    assert captured["toolsets"] == ["terminal", "file"]
    assert captured["acp_command"] == "codex"
    assert captured["acp_args"] == ["--acp", "--stdio"]


def test_handle_codex_task_requires_goal():
    result = json.loads(mod._handle_codex_task({}))
    assert "error" in result
    assert "goal" in result["error"].lower()
