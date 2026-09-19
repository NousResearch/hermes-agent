"""Interrupt debug logging follows the documented environment flag semantics."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROBE = (
    "from tools import interrupt; "
    "from tools.environments import base; "
    "print(interrupt._DEBUG_INTERRUPT, base._DEBUG_INTERRUPT)"
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", "True True"),
        ("true", "True True"),
        ("0", "False False"),
        ("false", "False False"),
        ("off", "False False"),
    ],
)
def test_interrupt_debug_flag_uses_shared_truthy_values(value, expected):
    env = os.environ.copy()
    env["HERMES_DEBUG_INTERRUPT"] = value

    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == expected
