"""The profile gateway must consume dashboard capabilities before imports."""

from __future__ import annotations

import subprocess
import sys


def test_entry_removes_dashboard_capabilities_before_gateway_runtime_imports(tmp_path):
    env = {
        "HOME": str(tmp_path),
        "HERMES_HOME": str(tmp_path / "hermes"),
        "PATH": "/usr/bin:/bin",
        "PYTHONUTF8": "1",
    }
    env["HERMES_TUI_SIDECAR_URL"] = (
        "ws://dashboard.test/api/pub?internal=synthetic&channel=test"
    )
    env["HERMES_TUI_GATEWAY_URL"] = (
        "ws://dashboard.test/api/ws?internal=synthetic"
    )
    output_file = tmp_path / "probe.txt"
    probe = (
        "import os; import tui_gateway.entry; "
        f"open({str(output_file)!r}, 'w').write("
        "'present' if any(key in os.environ for key in "
        "('HERMES_TUI_GATEWAY_URL', 'HERMES_TUI_SIDECAR_URL')) else 'absent')"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        check=False,
        env=env,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert output_file.read_text(encoding="utf-8") == "absent"