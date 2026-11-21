"""Execute the production pre-s6 dependency status mapper without a Docker image.

The shell block is run verbatim, not reimplemented or inspected for status strings.
Only its child process is replaced: PM behavior is covered by tests/pm instead.
Regression for #128185: baked-extra restoration must not refuse a healthy boot.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

STAGE2_HOOK = Path(__file__).resolve().parents[2] / "docker" / "stage2-hook.sh"
NOISE = (
    "arbitrary installer output $(echo should-not-execute)\n"
    "https://user:secret-index-credential-537@index.invalid/simple\n"
    "Traceback (most recent call last):\n"
)
WARNING = "[stage2] Warning: optional extras could not be refreshed; using image dependencies\n"
INVALID = "[stage2] ERROR: dependency refresh returned no valid state; refusing startup\n"
FAILED = (
    "[stage2] ERROR: dependency refresh failed; refusing startup "
    "(check network, index access, plugin requirements and volume permissions)\n"
)


@pytest.fixture(scope="module")
def dependency_block():
    # Like the other stage2 shell harnesses, execute just this bounded boot phase
    # so host UID remapping and absolute container paths are never touched.
    text = STAGE2_HOOK.read_text(encoding="utf-8")
    start = text.index("if dependency_output=$(")
    end = text.index("\n# auth.json: bootstrap", start)
    return text[start:end]


def _run_mapper(block: str, tmp_path: Path, stdout: str, exit_code: int):
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("sh required")
    (tmp_path / "stdout").write_text(stdout, encoding="utf-8")
    (tmp_path / "stderr").write_text(NOISE, encoding="utf-8")
    shim = tmp_path / "s6-setuidgid"
    shim.write_text(
        '#!/bin/sh\nset -eu\n'
        '[ "$#" = 4 ] && [ "$1" = hermes ] && '
        '[ "$2" = /opt/hermes/.venv/bin/python ] && [ "$3" = -c ] || exit 99\n'
        'cat "$PROBE_ROOT/stdout"\ncat "$PROBE_ROOT/stderr" >&2\n'
        'exit "$PROBE_EXIT"\n',
        encoding="utf-8",
    )
    shim.chmod(0o700)
    env = {
        "PATH": str(tmp_path) + os.pathsep + os.defpath,
        "HOME": str(tmp_path),
        "PROBE_ROOT": str(tmp_path),
        "PROBE_EXIT": str(exit_code),
    }
    return subprocess.run(
        [shell, "-c", 'set -eu\nINSTALL_DIR=/opt/hermes\n' + block +
         '\nprintf "mapper-complete\\n"\n'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10, check=False,
    )


@pytest.mark.parametrize("state, expected", [
    ("restored hindsight, messaging", "rebuilt"),
    ("current", "current"),
    ("base", "base"),
    ("rebuilt", "rebuilt"),
    ("fallback", "fallback"),
])
def test_known_state_emits_only_fixed_status(dependency_block, tmp_path, state, expected):
    output = NOISE + f"[stage2] dependency environment: {state}\n" + NOISE
    result = _run_mapper(dependency_block, tmp_path, output, 0)
    assert result.returncode == 0, result.stderr
    warning = WARNING if expected == "fallback" else ""
    assert result.stdout == f"[stage2] dependency environment: {expected}\n{warning}mapper-complete\n"
    assert result.stderr == ""


@pytest.mark.parametrize("output, exit_code, expected_error", [
    ("[stage2] dependency environment: unknown\n" + NOISE, 0, INVALID),
    ("", 0, INVALID),
    (" \n\t", 0, INVALID),
    (NOISE, 0, INVALID),
    (NOISE, 1, FAILED),
    ("[stage2] dependency environment: current\n" + NOISE, 7, FAILED),
])
def test_invalid_or_failed_child_refuses_without_relaying_output(
    dependency_block, tmp_path, output, exit_code, expected_error,
):
    result = _run_mapper(dependency_block, tmp_path, output, exit_code)
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == expected_error
