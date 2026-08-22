"""Regression test: the shell.exec RPC spawns children with a sanitized env.

The tui_gateway process carries API keys and DB passwords in os.environ
(provider keys, ``*_PASSWORD``, relay tokens). Passing no ``env=`` to
``subprocess.run`` would leak the gateway's full environment to every quick
shell command spawned via the shell.exec RPC. The handler must route through
``tools.environments.local.build_subprocess_env()`` — the same sanitized-env
factory the quick-command exec path uses — so children only receive the
scrubbed environment.

Test pattern follows tests/tui_gateway/test_subprocess_encoding.py
(kwargs-level assertion on the shell.exec subprocess.run call).
"""

from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock, patch

import tui_gateway.server as server


def _make_completed_process() -> MagicMock:
    """A CompletedProcess-like mock with str stdout/stderr (text=True contract)."""
    cp = MagicMock()
    cp.stdout = ""
    cp.stderr = ""
    cp.returncode = 0
    return cp


def test_shell_exec_spawns_with_sanitized_env():
    """shell.exec subprocess.run must pass env= from build_subprocess_env(),
    which scrubs provider credentials and *_PASSWORD vars from os.environ."""
    handler = server._methods["shell.exec"]
    with patch("subprocess.run", return_value=_make_completed_process()) as mock_run, \
         patch("tools.approval.detect_hardline_command", return_value=(False, "")), \
         patch("tools.approval.detect_dangerous_command", return_value=(False, None, "")), \
         patch.dict(os.environ, {"EMAIL_PASSWORD": "planted-secret"}):
        handler(1, {"command": "echo hello"})
    assert mock_run.called, "subprocess.run was not invoked"
    kwargs = mock_run.call_args[1]
    env = kwargs.get("env")
    assert env is not None, (
        "shell.exec subprocess.run must pass env= (sanitized env) — got None"
    )
    assert "EMAIL_PASSWORD" not in env, (
        "shell.exec subprocess.run env must scrub provider credentials "
        "(*_PASSWORD leaked into child env)"
    )
