"""The real registered file handler never returns a quarantined auth store."""

import json

import pytest

from model_tools import handle_function_call
from tools import file_tools
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


@pytest.mark.parametrize("profile", ["", "profiles/full", "profiles/reviewer"])
def test_quarantined_auth_read_is_refused_without_changing_bytes(tmp_path, monkeypatch, profile):
    root = tmp_path / ".hermes"
    home = root / profile if profile else root
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    canary = "fixture-quarantined-auth-canary"
    protected = home / "auth.json.corrupt"
    protected.write_text(canary, encoding="utf-8")
    control = tmp_path / "normal.txt"
    control.write_text("fixture-plain-file-control", encoding="utf-8")
    environment = LocalEnvironment(cwd=str(tmp_path), timeout=15)
    operations = ShellFileOperations(environment)
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda task_id: operations)
    task = "quarantined-auth-fixture"
    try:
        result = json.loads(handle_function_call("read_file", {"path": str(protected)}, task_id=task))
        assert result.get("error")
        assert "credential store" in result["error"]
        assert canary not in json.dumps(result)
        assert protected.read_text(encoding="utf-8") == canary
        plain = handle_function_call("read_file", {"path": str(control)}, task_id=task)
        assert "fixture-plain-file-control" in plain
    finally:
        environment.cleanup()
