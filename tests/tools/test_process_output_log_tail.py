"""An adopted, completed worker command keeps the end of its output log."""

import json
import shlex
import subprocess
import sys

import pytest

from tools import process_registry as pr
from tools.process_registry_output_log import output_dir, record_exit_command


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("lines", [10, 350_000], ids=["small", "over-final-drain-limit"])
def test_recovered_worker_log_keeps_final_output(tmp_path, monkeypatch, lines):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    checkpoint = tmp_path / "processes.json"
    monkeypatch.setattr(pr, "CHECKPOINT_PATH", checkpoint)
    log = output_dir() / "proc_completed.log"
    log.parent.mkdir(parents=True)
    final_line = "BUILD FAILED: missing dependency"
    command = shlex.join([
        sys.executable, "-c",
        f"import sys; print('noise\\n' * {lines}, end=''); print({final_line!r}); sys.exit(7)",
    ])
    with log.open("wb") as output:
        child = subprocess.Popen(
            ["bash", "-c", record_exit_command(log, command)],
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
        )
        assert child.wait(timeout=10) == 7
    checkpoint.write_text(json.dumps([{
        "session_id": "proc_completed", "command": command, "pid": child.pid,
        "pid_scope": "host", "output_log": str(log),
    }]), encoding="utf-8")

    registry = pr.ProcessRegistry()
    assert registry.recover_from_checkpoint() == 1
    result = registry.wait("proc_completed", timeout=5)
    assert result["status"] == "exited" and result["exit_code"] == 7
    assert result["output"].endswith(final_line + "\n")
    assert registry.read_log("proc_completed", limit=1)["output"] == final_line
