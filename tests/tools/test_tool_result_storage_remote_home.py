"""#154749: a generic tool-result spill on a home-relative backend (ssh/daytona/vercel) names the
synced ``~/.hermes/cache/spillover`` file, never a second ``<sandbox tmp>/hermes-results`` copy.

The backend here is a real bash with its own HOME (the "remote" home), so ``~`` is expanded by the
backend shell exactly as over SSH."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.tool_result_storage import extract_persisted_path, get_spillover_dir, maybe_persist_tool_result


class _RemoteHomeEnv:
    """A non-local backend: commands run in bash under *remote_home*; an optional sync manager
    mirrors the host spillover dir into ``<remote_home>/.hermes/cache/spillover``."""

    def __init__(self, remote_home: Path, with_sync: bool):
        self.remote_home = remote_home
        if with_sync:
            self._sync_manager = self

    def sync(self, force: bool = False):
        dst = self.remote_home / ".hermes" / "cache" / "spillover"
        dst.mkdir(parents=True, exist_ok=True)
        for item in get_spillover_dir().iterdir():
            shutil.copy2(item, dst / item.name)

    def get_temp_dir(self):
        return str(self.remote_home / "sandbox-tmp")

    def execute(self, command, timeout=None, stdin_data=None):
        proc = subprocess.run(["bash", "-c", command], input=stdin_data, capture_output=True, text=True, check=False,
                              env={"HOME": str(self.remote_home), "PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                              timeout=timeout)
        return {"output": proc.stdout + proc.stderr, "returncode": proc.returncode}


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("with_sync", [True, False], ids=["synced", "unsynced"])
def test_home_relative_backend_spill_names_the_synced_spillover_path(tmp_path, monkeypatch, with_sync):
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    remote_home = tmp_path / "remote-home"
    remote_home.mkdir()
    env = _RemoteHomeEnv(remote_home, with_sync)
    content = "\n".join(f"row-{i:05d}" for i in range(6000))

    result = maybe_persist_tool_result(content=content, tool_name="mcp_probe", tool_use_id="tc_ssh_1",
                                       env=env, threshold=30_000)

    path = extract_persisted_path(result)
    assert path == "~/.hermes/cache/spillover/tc_ssh_1.txt"
    assert "hermes-results" not in result
    # The named path is readable where the agent's tools run (the backend's own ~).
    assert (remote_home / ".hermes" / "cache" / "spillover" / "tc_ssh_1.txt").read_text() == content


def test_sandbox_start_error_keeps_the_finished_result(monkeypatch):
    """A terminal config that raises while starting the sandbox for a spill (bad TERMINAL_TIMEOUT)
    must not lose the completed tool result: storage gets env=None and decides from there."""
    from agent import tool_executor

    def broken(_task_id):
        raise ValueError("invalid TERMINAL_TIMEOUT")

    monkeypatch.setattr(tool_executor, "get_active_env", lambda _task_id: None)
    monkeypatch.setattr(tool_executor, "ensure_task_env", broken)
    assert tool_executor._result_storage_env("t1", needs_persistence=True) is None
