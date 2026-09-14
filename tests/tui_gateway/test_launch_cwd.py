"""Regression for the local TUI wrapper's process-local launch-cwd pin."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_tui_entry_pins_wrapper_cwd_across_late_dotenv_reload(tmp_path):
    project = tmp_path / "project"
    configured = tmp_path / "profile-cwd"
    home = tmp_path / "hermes-home"
    project.mkdir()
    configured.mkdir()
    home.mkdir()
    (home / "config.yaml").write_text(
        f"terminal:\n  backend: local\n  cwd: {configured}\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    result_path = tmp_path / "result.txt"
    env.update(
        HERMES_CWD=str(project),
        HERMES_HOME=str(home),
        HERMES_TEST_ISOLATION=str(home),
        RESULT_PATH=str(result_path),
        TERMINAL_CWD=str(configured),
        TERMINAL_ENV="local",
    )
    code = (
        "import os; from pathlib import Path; import tui_gateway.entry; "
        "from hermes_cli.env_loader import load_hermes_dotenv; "
        "load_hermes_dotenv(load_external_secrets=False); "
        "Path(os.environ['RESULT_PATH']).write_text(os.environ['TERMINAL_CWD'], encoding='utf-8')"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result_path.read_text(encoding="utf-8") == str(project)
