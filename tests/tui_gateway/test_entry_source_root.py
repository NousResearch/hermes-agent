"""The TUI gateway child imports the tree it was launched from, whatever HERMES_PYTHON_SRC_ROOT says.

A checkout's dev shell exports HERMES_PYTHON_SRC_ROOT=<checkout>. An install started from that
shell must not run the checkout's agent code: its own CLI over another tree's ``run_agent``.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_entry_ignores_an_inherited_source_root(tmp_path):
    decoy = tmp_path / "decoy-checkout"
    decoy.mkdir()
    (decoy / "hermes_agent").mkdir()
    (decoy / "hermes_agent/__init__.py").write_text("raise SystemExit('imported the decoy tree')\n", encoding="utf-8")

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONPATH=str(PROJECT_ROOT), HERMES_PYTHON_SRC_ROOT=str(decoy),
               HERMES_HOME=str(tmp_path / "home"))
    report = tmp_path / "report.json"
    # tui_gateway routes stdout to stderr (the JSON-RPC channel), so the probe writes a file.
    result = subprocess.run(
        [sys.executable, "-c",
         "import json, sys, hermes_agent.tui_gateway.entry;from hermes_agent import run_agent;"
         f" open({str(report)!r}, 'w').write(json.dumps([sys.path[0], run_agent.__file__]))"],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=120, check=False,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    first_path, run_agent_file = json.loads(report.read_text(encoding="utf-8"))
    assert Path(first_path) == PROJECT_ROOT
    assert Path(run_agent_file).parent == PROJECT_ROOT / "hermes_agent"
