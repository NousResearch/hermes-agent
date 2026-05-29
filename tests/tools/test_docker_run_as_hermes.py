from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "docker" / "run-as-hermes.sh"


def _run_shell(script: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sh", "-eu", "-c", script],
        cwd=REPO_ROOT,
        env={"PATH": os.environ.get("PATH", "")} | (env or {}),
        capture_output=True,
        text=True,
    )


def test_run_as_hermes_bypasses_s6_when_already_target_uid() -> None:
    script = f"""
    . "{HELPER}"
    id() {{
        if [ "$1" = "-u" ] && [ "$#" -eq 1 ]; then
            printf '10000\\n'
            return 0
        fi
        if [ "$1" = "-u" ] && [ "${{2:-}}" = "hermes" ]; then
            printf '10000\\n'
            return 0
        fi
        return 1
    }}
    run_as_hermes printf 'direct\\n'
    """
    proc = _run_shell(script)

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "direct\n"


def test_run_as_hermes_uses_s6_when_not_target_uid(tmp_path: Path) -> None:
    s6 = tmp_path / "s6-setuidgid"
    s6.write_text("#!/bin/sh\nprintf 's6:%s:%s\\n' \"$1\" \"$2\"\n")
    s6.chmod(0o755)

    script = f"""
    . "{HELPER}"
    id() {{
        if [ "$1" = "-u" ] && [ "$#" -eq 1 ]; then
            printf '0\\n'
            return 0
        fi
        if [ "$1" = "-u" ] && [ "${{2:-}}" = "hermes" ]; then
            printf '10000\\n'
            return 0
        fi
        return 1
    }}
    run_as_hermes printf 'dropped\\n'
    """
    proc = _run_shell(script, env={"PATH": f"{tmp_path}:{os.environ.get('PATH', '')}"})

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "s6:hermes:printf\n"


@pytest.mark.parametrize(
    ("path", "helper_call"),
    [
        ("docker/stage2-hook.sh", "run_as_hermes"),
        ("docker/main-wrapper.sh", "exec_as_hermes"),
        ("docker/s6-rc.d/dashboard/run", "exec_as_hermes"),
        ("docker/cont-init.d/02-reconcile-profiles", "exec_as_hermes"),
    ],
)
def test_container_scripts_use_hermes_drop_helper(path: str, helper_call: str) -> None:
    text = (REPO_ROOT / path).read_text()

    assert "run-as-hermes.sh" in text
    assert helper_call in text
