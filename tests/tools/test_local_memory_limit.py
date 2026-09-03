import json
import os
import sys

import pytest


def test_terminal_config_bridges_local_memory_limit(monkeypatch):
    from hermes_cli.config import apply_terminal_config_to_env

    env = {}
    apply_terminal_config_to_env(
        env=env,
        config={"terminal": {"backend": "local", "local_memory_max_mb": 256}},
        override=True,
    )

    assert env["TERMINAL_LOCAL_MEMORY_MAX_MB"] == "256"


@pytest.mark.skipif(sys.platform == "win32", reason="resource limits are POSIX-only")
def test_local_terminal_foreground_inherits_memory_limit(monkeypatch, tmp_path):
    from tools import terminal_tool as tt

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "128")
    monkeypatch.setattr(tt, "_terminal_config_bridge_attempted", True)
    tt.cleanup_all_environments()

    command = (
        f"{sys.executable} -c "
        "'import resource; print(resource.getrlimit(resource.RLIMIT_AS)[0])'"
    )
    try:
        result = json.loads(tt.terminal_tool(command=command, timeout=10))
    finally:
        tt.cleanup_all_environments()

    assert result["exit_code"] == 0
    assert result["output"].strip().splitlines()[0] == str(128 * 1024 * 1024)


@pytest.mark.skipif(sys.platform == "win32", reason="resource limits are POSIX-only")
def test_local_terminal_memory_limit_zero_disables_cap(monkeypatch, tmp_path):
    from tools import terminal_tool as tt

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "0")
    monkeypatch.setattr(tt, "_terminal_config_bridge_attempted", True)
    tt.cleanup_all_environments()

    command = (
        f"{sys.executable} -c "
        "'import resource; print(resource.getrlimit(resource.RLIMIT_AS)[0])'"
    )
    try:
        result = json.loads(tt.terminal_tool(command=command, timeout=10))
    finally:
        tt.cleanup_all_environments()

    assert result["exit_code"] == 0
    assert result["output"].strip().splitlines()[0] == "-1"
