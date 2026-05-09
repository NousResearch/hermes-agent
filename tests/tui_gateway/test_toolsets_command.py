"""Regression coverage for TUI /toolsets slash-worker parity.

The Ink TUI routes /toolsets through tui_gateway.slash_worker, not the
interactive CLI entrypoint. When /toolsets changed from a static catalog view to
an active/configured view, the worker also needed the session's effective
configured toolsets; otherwise the TUI showed an empty active list while the CLI
showed the correct config.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_worker(tmp_path: Path, command: str) -> dict:
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        textwrap.dedent(
            """\
            model:
              default: test/model
              provider: test
            platform_toolsets:
              cli:
                - web
                - terminal
            """
        )
    )

    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["HERMES_INTERACTIVE"] = "1"
    env.setdefault("OPENROUTER_API_KEY", "dummy")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tui_gateway.slash_worker",
            "--session-key",
            "tui-toolsets-test-session",
            "--model",
            "test/model",
        ],
        cwd=REPO_ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    request = json.dumps({"id": 1, "command": command}) + "\n"
    stdout, stderr = proc.communicate(request, timeout=20)

    assert proc.returncode == 0, stderr
    lines = [line for line in stdout.splitlines() if line.strip()]
    assert len(lines) == 1, stdout
    response = json.loads(lines[0])
    assert response["id"] == 1
    assert response.get("ok") is True, response
    return response


def test_tui_toolsets_worker_shows_configured_active_toolsets(tmp_path):
    response = _run_worker(tmp_path, "toolsets")

    output = response["output"]
    assert "Active Toolsets" in output
    assert "No toolsets are configured" not in output
    assert " web" in output
    assert " terminal" in output


def test_tui_toolsets_available_worker_still_shows_catalog(tmp_path):
    response = _run_worker(tmp_path, "toolsets available")

    output = response["output"]
    assert "Toolsets Available" in output
    assert "full catalog" in output
