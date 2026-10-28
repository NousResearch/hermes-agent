"""Library entry arguments are explicit; real CLI arguments remain strict."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("read_only", [False, True])
def test_library_main_ignores_host_argv_and_keeps_real_one_shot_policy(tmp_path, read_only):
    probe = r'''
import os
from pathlib import Path
import sys
from tui_gateway import entry
from agent.codex_auth_policy import codex_auth_read_only
from gateway.runtime_ownership import process_ownership
expected = sys.argv[1] == "yes"
# The embedding host owns sys.argv. No caller arguments means default-off.
sys.argv = ["embedding-host", "--host-only-option", "--codex-auth-read-only"]
served = []
entry._claim_launch_home = lambda: Path(os.environ["HERMES_HOME"])
process_ownership.release = lambda home: None
entry._serve = lambda: served.append(codex_auth_read_only())
if expected:
    entry.main(["--codex-auth-read-only"])
else:
    entry.main()
assert served == [expected]
try:
    entry.main([] if expected else ["--codex-auth-read-only"])
except RuntimeError as exc:
    assert "already bound" in str(exc)
else:
    raise AssertionError("process policy was rebound")
assert codex_auth_read_only() is expected
assert served == [expected]
'''
    result = subprocess.run(
        [sys.executable, "-c", probe, "yes" if read_only else "no"],
        env=_env(tmp_path), capture_output=True, text=True, timeout=45, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("arguments, code", [(["--help"], 0), (["--codex-auth-read-only", "--typo"], 2)])
def test_real_module_cli_forwards_arguments_and_rejects_unknown_flags(tmp_path, arguments, code):
    result = subprocess.run(
        [sys.executable, "-m", "tui_gateway.entry", *arguments],
        env=_env(tmp_path), input="", capture_output=True, text=True, timeout=45, check=False,
    )
    assert result.returncode == code, result.stdout + result.stderr
    if code == 0:
        assert "--codex-auth-read-only" in result.stdout + result.stderr
    else:
        assert "unrecognized arguments: --typo" in result.stderr


@pytest.mark.platforms("linux", "macos")
def test_real_fork_fails_closed_and_fresh_exec_has_default_policy(tmp_path):
    probe = r'''
import os
import subprocess
import sys
from agent.codex_auth_policy import bind_codex_auth_policy, codex_auth_read_only
bind_codex_auth_policy(read_only=True)
pid = os.fork()
if pid == 0:
    try:
        codex_auth_read_only()
    except RuntimeError:
        os._exit(0)
    os._exit(1)
_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0
result = subprocess.run([sys.executable, "-c", "from agent.codex_auth_policy import codex_auth_read_only; assert not codex_auth_read_only()"], check=False, timeout=30)
assert result.returncode == 0
assert codex_auth_read_only()
'''
    result = subprocess.run(
        [sys.executable, "-c", probe], env=_env(tmp_path),
        capture_output=True, text=True, timeout=45, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _env(tmp_path):
    return {**os.environ, "HOME": str(tmp_path), "HERMES_HOME": str(tmp_path / "hermes")}
