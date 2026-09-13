"""Literal local environment settings are validated and isolated per profile."""

import json
import shlex
import sys

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli.config import validate_config_structure
from tools import terminal_tool as terminal
from tools.terminal_scope import install_and_reset_profile_terminal_scope


@pytest.mark.parametrize("mapping, bad_key", [
    ({"BAD-NAME": "x"}, "BAD-NAME"),
    ({"1BAD": "x"}, "1BAD"),
    ({"BAD\n": "x"}, "BAD"),
    ({"BAD": []}, "BAD"),
    ({"BAD": {}}, "BAD"),
    ({"BAD": None}, "BAD"),
    ([], "terminal.env_vars"),
])
def test_invalid_env_vars_rejected_by_config_check_and_runtime(tmp_path, mapping, bad_key):
    config = {"terminal": {"backend": "local", "env_vars": mapping}}
    (tmp_path / "config.yaml").write_text(json.dumps(config))
    issues = validate_config_structure(config)
    errors = [issue.message for issue in issues if issue.severity == "error"]
    assert len(errors) == 1
    assert "terminal.env_vars" in errors[0]
    assert bad_key in errors[0]
    assert "config.yaml" in errors[0]
    with install_and_reset_profile_terminal_scope(tmp_path):
        with pytest.raises(ValueError) as exc:
            terminal._get_env_config()
    assert str(exc.value) == errors[0]


def test_local_env_vars_reach_descendants_without_profile_leaks(tmp_path, monkeypatch):
    monkeypatch.setattr(terminal, "_active_environments", {})
    monkeypatch.delenv("ONLY_PROFILE_A", raising=False)
    monkeypatch.setenv("HERMES_TEST_WORKERS", "99")
    profiles = [tmp_path / "a", tmp_path / "b"]
    mappings = [
        {"HERMES_TEST_WORKERS": 3, "ONLY_PROFILE_A": True, "FRACTION": 1.5},
        {"HERMES_TEST_WORKERS": "2"},
    ]
    for home, mapping in zip(profiles, mappings):
        home.mkdir()
        (home / "config.yaml").write_text(json.dumps({"terminal": {
            "backend": "local", "cwd": str(tmp_path), "env_vars": mapping,
        }}))
    probe = (
        "import os,json; print(json.dumps([os.environ.get(k) for k in "
        "['HERMES_TEST_WORKERS','ONLY_PROFILE_A','FRACTION']]))"
    )
    child = (
        "import subprocess,sys; " + probe + "; "
        f"subprocess.run([sys.executable, '-c', {probe!r}], check=True)"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(child)}"
    try:
        for index in (0, 1, 0, 1):
            tokens = set_session_vars(session_key=f"env-vars-profile-{index}", platform="discord")
            try:
                with install_and_reset_profile_terminal_scope(profiles[index]):
                    result = json.loads(terminal.terminal_tool(command, task_id=f"turn-{index}", force=True))
                assert result["exit_code"] == 0, result
                expected = ["3", "True", "1.5"] if index == 0 else ["2", None, None]
                assert [json.loads(line) for line in result["output"].splitlines()] == [expected, expected]
            finally:
                clear_session_vars(tokens)
        assert len(terminal._active_environments) == 2
    finally:
        terminal.cleanup_all_environments()
