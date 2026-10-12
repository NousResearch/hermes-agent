"""The TUI backend's update-check prefetch must run under the OS trust store.

``tui_gateway.server`` starts the update-check prefetch at import time, which
``tui_gateway.entry`` triggers before ``main()`` sets up TLS trust. The import
runs in a fresh subprocess so this test process's own imports cannot hide it.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_IMPORT_ENTRY_AND_RECORD = """
import json, sys
import agent.ssl_verify
import hermes_cli.banner

calls = []
agent.ssl_verify.install_truststore = lambda: calls.append("truststore") or True
hermes_cli.banner.prefetch_update_check = lambda: calls.append("prefetch")

import tui_gateway.entry

# The server rebinds sys.stdout (its RPC channel), so report through a file.
with open(sys.argv[1], "w", encoding="utf-8") as out:
    json.dump(calls, out)
"""


def test_entry_import_installs_os_trust_before_update_prefetch(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "HERMES_PYTHON_SRC_ROOT"}
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["HERMES_HOME"] = str(tmp_path / "hermes_home")
    calls_path = tmp_path / "calls.json"

    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_ENTRY_AND_RECORD, str(calls_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(calls_path.read_text(encoding="utf-8-sig")) == ["truststore", "prefetch"]
